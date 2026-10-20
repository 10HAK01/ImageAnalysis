#!/usr/bin/env python3
"""
Analyze interleaved, multi-page fluorescence TIFF stacks.

Analysis workflow:
    1. Detect the number of interleaved channels in each TIFF.
    2. Maximum-project every channel across z, or optionally select one slice.
    3. Build an object mask from the background-corrected D channel.
    4. Estimate each channel's background outside the expanded D-object mask.
    5. Identify connected objects and apply optional size/border filters.
    6. Measure corrected D/F/T/C intensities and channel ratios.
    7. Save image-level data, object-level data, and visual QC montages.

TIFF page order:
    z1/channel1, z1/channel2, ..., z2/channel1, z2/channel2, ...

Channels 1-4 are treated as D, F, T, and C by default. The channel count is
detected per file from ImageJ/OME metadata when possible. Use --channels to
override automatic detection. Each channel is maximum-projected across z by
default; use --z-mode single to reproduce the earlier single-slice workflow.

Example:
    python3 fluorescence_tiff_analysis.py /path/to/tiff_folder

Default outputs inside the TIFF folder:
    IFData.csv       One summary row per TIFF.
    IFObjects.csv    One row per connected object.
    QC/              One visual QC montage per TIFF.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import deque
from pathlib import Path

# NumPy performs quantitative array calculations. Pillow reads TIFF pages and
# creates masks and QC images. The script stops with an installation hint if
# either dependency is unavailable.
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    sys.exit(
        "This script needs NumPy and Pillow.\n"
        "Install them with: python3 -m pip install numpy Pillow"
    )


# Column order for the object-level CSV. Keeping a fixed schema allows an empty
# object file to contain useful headers and makes results consistent between
# experiments with different numbers of channels.
OBJECT_FIELDNAMES = [
    "filename",
    "detected_channels",
    "z_method",
    "selected_z",
    "object_id",
    "included",
    "exclusion_reason",
    "touches_border",
    "area_pixels",
    "centroid_x_px_0based",
    "centroid_y_px_0based",
    "bbox_x_min_px_0based",
    "bbox_y_min_px_0based",
    "bbox_x_max_px_0based",
    "bbox_y_max_px_0based",
    "D_background",
    "F_background",
    "T_background",
    "C_background",
    "D_mean_corrected",
    "F_mean_corrected",
    "T_mean_corrected",
    "C_mean_corrected",
    "D_integrated_corrected",
    "F_integrated_corrected",
    "T_integrated_corrected",
    "C_integrated_corrected",
    "F_over_D",
    "T_over_D",
    "C_over_F",
]


# -----------------------------------------------------------------------------
# Basic mask and measurement utilities
# -----------------------------------------------------------------------------

def masked_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean of all masked pixels, including zero/negative values."""
    selected_pixels = image[mask]
    if selected_pixels.size == 0:
        return math.nan
    return float(selected_pixels.mean())


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a binary mask by a configurable square radius."""
    if radius <= 0:
        return mask.copy()
    kernel_size = 2 * radius + 1
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    expanded = mask_image.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(expanded) > 0


def connected_components_8(mask: np.ndarray) -> list[np.ndarray]:
    """Return flat pixel indices for each 8-connected foreground object."""
    height, width = mask.shape
    foreground = mask.ravel()
    visited = np.zeros(foreground.size, dtype=bool)
    components: list[np.ndarray] = []

    for seed_value in np.flatnonzero(foreground):
        seed = int(seed_value)
        if visited[seed]:
            continue

        visited[seed] = True
        queue: deque[int] = deque([seed])
        pixels: list[int] = []

        while queue:
            index = queue.popleft()
            pixels.append(index)
            row, column = divmod(index, width)

            for neighbor_row in range(max(0, row - 1), min(height, row + 2)):
                row_offset = neighbor_row * width
                for neighbor_column in range(
                    max(0, column - 1), min(width, column + 2)
                ):
                    neighbor = row_offset + neighbor_column
                    if (
                        neighbor != index
                        and foreground[neighbor]
                        and not visited[neighbor]
                    ):
                        visited[neighbor] = True
                        queue.append(neighbor)

        components.append(np.asarray(pixels, dtype=np.int64))

    return components


# -----------------------------------------------------------------------------
# QC visualization
#
# These functions scale images for display only. They never change the arrays
# used for numerical measurements. Green outlines are included objects, orange
# outlines are excluded objects, blue marks background, and red marks the full
# threshold-positive mask.
# -----------------------------------------------------------------------------

def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner outline of a binary mask."""
    interior = np.zeros_like(mask)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        center = mask[1:-1, 1:-1].copy()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                center &= mask[
                    1 + row_offset : mask.shape[0] - 1 + row_offset,
                    1 + column_offset : mask.shape[1] - 1 + column_offset,
                ]
        interior[1:-1, 1:-1] = center
    return mask & ~interior


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Scale an image robustly for QC display without changing measurements."""
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite_values, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        scaled = np.clip((image - low) / (high - low), 0, 1)
        scaled = np.nan_to_num(scaled, nan=0.0)
        scaled = np.asarray(np.rint(scaled * 255), dtype=np.uint8)
    return np.repeat(scaled[:, :, np.newaxis], 3, axis=2)


def tint_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Return an RGB copy with a translucent mask overlay."""
    result = rgb.astype(np.float64, copy=True)
    color_array = np.asarray(color, dtype=np.float64)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color_array
    return np.asarray(np.clip(result, 0, 255), dtype=np.uint8)


def make_qc_panel(
    rgb: np.ndarray,
    title: str,
    max_image_size: int,
    labels: list[tuple[float, float, str, tuple[int, int, int]]] | None = None,
) -> Image.Image:
    """Resize an RGB array and add a title bar."""
    image = Image.fromarray(rgb, mode="RGB")
    scale = max_image_size / max(image.width, image.height)
    target_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    if labels:
        label_draw = ImageDraw.Draw(image)
        for x_position, y_position, text, color in labels:
            label_draw.text(
                (x_position * scale, y_position * scale),
                text,
                fill=color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                anchor="mm",
            )

    title_height = 26
    panel = Image.new(
        "RGB", (image.width, image.height + title_height), (18, 18, 18)
    )
    panel.paste(image, (0, title_height))
    ImageDraw.Draw(panel).text((6, 7), title, fill=(255, 255, 255))
    return panel


def save_qc_image(
    output_path: Path,
    raw_images: dict[str, np.ndarray | None],
    corrected_images: dict[str, np.ndarray | None],
    threshold_object_mask: np.ndarray,
    included_object_mask: np.ndarray,
    background_mask: np.ndarray,
    result: dict[str, object],
    object_rows: list[dict[str, object]],
    max_image_size: int,
) -> None:
    """Save a two-row QC montage for one analyzed TIFF."""
    d_image = raw_images["D"]
    if d_image is None:
        raise ValueError("Cannot create QC output without the D image.")

    d_rgb = image_to_rgb(d_image)
    included_outline = mask_outline(included_object_mask)
    excluded_outline = mask_outline(
        threshold_object_mask & ~included_object_mask
    )
    reduction = max(d_image.shape) / max_image_size
    if reduction > 1:
        outline_radius = max(1, int(math.ceil(reduction / 2)))
        included_outline = dilate_mask(included_outline, outline_radius)
        excluded_outline = dilate_mask(excluded_outline, outline_radius)

    object_overlay = d_rgb.copy()
    object_overlay[included_outline] = (0, 255, 0)
    object_overlay[excluded_outline] = (255, 140, 0)
    object_labels = []
    for row in object_rows[:200]:
        label_color = (0, 255, 0) if row["included"] else (255, 140, 0)
        object_labels.append(
            (
                float(row["centroid_x_px_0based"]),
                float(row["centroid_y_px_0based"]),
                str(row["object_id"]),
                label_color,
            )
        )

    background_overlay = tint_mask(
        d_rgb, background_mask, color=(0, 100, 255), alpha=0.45
    )
    threshold_outline = mask_outline(threshold_object_mask)
    if reduction > 1:
        threshold_outline = dilate_mask(threshold_outline, outline_radius)
    background_overlay[threshold_outline] = (255, 0, 0)

    panels = [
        make_qc_panel(
            d_rgb,
            (
                "D maximum projection"
                if result["z_method"] == "max_projection"
                else "Selected D plane"
            ),
            max_image_size,
        ),
        make_qc_panel(
            object_overlay,
            "Objects (green=in, orange=out)",
            max_image_size,
            labels=object_labels,
        ),
        make_qc_panel(
            background_overlay,
            "Background blue; objects red",
            max_image_size,
        ),
    ]

    for name in ("D", "F", "T", "C"):
        corrected_image = corrected_images[name]
        if corrected_image is None:
            corrected_rgb = np.zeros_like(d_rgb)
            title = f"Corrected {name} unavailable"
        else:
            corrected_rgb = image_to_rgb(corrected_image)
            title = f"Corrected {name}"
        panels.append(make_qc_panel(corrected_rgb, title, max_image_size))

    panel_width = panels[0].width
    panel_height = panels[0].height
    summary = Image.new("RGB", (panel_width, panel_height), (25, 25, 25))
    summary_lines = [
        str(result["filename"]),
        f"channels={result['detected_channels']}",
        (
            f"z=max projection of {result['z_slices']} slices"
            if result["z_method"] == "max_projection"
            else f"selected z={result['selected_z']}/{result['z_slices']}"
        ),
        (
            f"objects={result['object_count_included']}/"
            f"{result['object_count_total']} included"
        ),
        "object labels match IFObjects.csv",
        (
            f"pixels={result['object_pixel_count']}  "
            f"background pixels={result['background_pixel_count']}"
        ),
        (
            "backgrounds D/F/T/C: "
            f"{result['D_background']:.4g}, "
            f"{result['F_background']:.4g}, "
            f"{result['T_background']:.4g}, "
            f"{result['C_background']:.4g}"
        ),
        f"background: {result['background_region']}",
    ]
    ImageDraw.Draw(summary).multiline_text(
        (8, 10), "\n".join(summary_lines), fill=(255, 255, 255), spacing=6
    )
    panels.append(summary)

    columns = 4
    rows = 2
    montage = Image.new(
        "RGB",
        (columns * panel_width, rows * panel_height),
        (0, 0, 0),
    )
    for index, panel in enumerate(panels):
        montage.paste(
            panel,
            ((index % columns) * panel_width, (index // columns) * panel_height),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, format="PNG")


# -----------------------------------------------------------------------------
# TIFF reading and channel detection
#
# TIFF pages are indexed from zero internally. Channel numbers supplied by the
# user are one-based, matching their acquisition order. Metadata is preferred
# for channel detection; page-count inference is used only as a fallback.
# -----------------------------------------------------------------------------

def read_grayscale_page(tiff: Image.Image, page_index: int) -> np.ndarray:
    """Read one zero-based TIFF page as a 2-D float64 array."""
    tiff.seek(page_index)
    image = np.asarray(tiff)
    if image.ndim != 2:
        raise ValueError(
            f"Page {page_index + 1} is not a 2-D grayscale image "
            f"(shape: {image.shape})."
        )
    return image.astype(np.float64, copy=False)


def image_description(tiff: Image.Image) -> str:
    """Return the TIFF ImageDescription tag as text, if it exists."""
    description = tiff.tag_v2.get(270, "")
    if isinstance(description, bytes):
        return description.decode("utf-8", errors="replace")
    if isinstance(description, (tuple, list)):
        return "\n".join(str(item) for item in description)
    return str(description)


def detect_channel_count(
    tiff: Image.Image, requested_channels: int | None
) -> tuple[int, str]:
    """
    Detect interleaved channels and report how the count was determined.

    ImageJ and OME metadata are authoritative. If metadata is absent, the
    largest page-count divisor among 2, 3, and 4 is used so processing can
    continue. The CSV and console output flag an ambiguous fallback.
    """
    page_count = getattr(tiff, "n_frames", 1)

    if requested_channels is not None:
        if page_count % requested_channels != 0:
            raise ValueError(
                f"{page_count} TIFF pages cannot be divided into the requested "
                f"{requested_channels} channels."
            )
        return requested_channels, "manual override"

    description = image_description(tiff)
    metadata_patterns = (
        ("ImageJ metadata", r"(?im)^\s*channels\s*=\s*(\d+)\s*$"),
        ("OME metadata", r"\bSizeC\s*=\s*[\"'](\d+)[\"']"),
    )
    for source, pattern in metadata_patterns:
        match = re.search(pattern, description)
        if match:
            channels = int(match.group(1))
            if channels < 1:
                continue
            if page_count % channels != 0:
                raise ValueError(
                    f"{source} reports {channels} channels, but {page_count} "
                    "pages are not divisible by that count."
                )
            return channels, source

    plausible_counts = [
        channels for channels in (2, 3, 4) if page_count % channels == 0
    ]
    if not plausible_counts:
        return 1, "page-count fallback"
    if len(plausible_counts) == 1:
        return plausible_counts[0], "page-count inference"

    channels = max(plausible_counts)
    candidates = "/".join(str(item) for item in plausible_counts)
    return channels, f"ambiguous page count ({candidates}); chose {channels}"


# -----------------------------------------------------------------------------
# Z-stack processing
#
# Maximum projection takes the brightest value at each x/y position across all
# z slices, separately for every channel. Single-slice mode instead scores the
# focus channel and uses the same selected z position for all channels.
# -----------------------------------------------------------------------------

def select_z_slice(
    tiff: Image.Image,
    total_channels: int,
    focus_channel: int,
    focus_threshold: float,
) -> tuple[int, list[float]]:
    """
    Return the selected zero-based z index and all focus scores.

    Each score is the mean of focus-channel pixels above focus_threshold.
    The first z slice with the largest score is selected.
    """
    page_count = getattr(tiff, "n_frames", 1)
    if page_count % total_channels != 0:
        raise ValueError(
            f"{page_count} TIFF pages cannot be divided into "
            f"{total_channels} channels."
        )

    z_count = page_count // total_channels
    scores: list[float] = []

    for z_index in range(z_count):
        page_index = z_index * total_channels + (focus_channel - 1)
        focus_image = read_grayscale_page(tiff, page_index)
        selected_pixels = focus_image[focus_image > focus_threshold]
        score = (
            float(selected_pixels.mean())
            if selected_pixels.size > 0
            else math.nan
        )
        scores.append(score)

    finite_scores = np.asarray(scores, dtype=np.float64)
    if np.isnan(finite_scores).all():
        # Selecting the first slice gives a deterministic result when every
        # focus score is unavailable, while the NaN score exposes the issue.
        return 0, scores

    # Ignore unavailable scores and select the first occurrence of the maximum.
    return int(np.nanargmax(finite_scores)), scores


def max_project_channel(
    tiff: Image.Image,
    total_channels: int,
    channel: int,
) -> np.ndarray:
    """Return the pixelwise maximum projection for one interleaved channel."""
    page_count = getattr(tiff, "n_frames", 1)
    if page_count % total_channels != 0:
        raise ValueError(
            f"{page_count} TIFF pages cannot be divided into "
            f"{total_channels} channels."
        )

    z_count = page_count // total_channels
    projection: np.ndarray | None = None
    for z_index in range(z_count):
        page_index = z_index * total_channels + (channel - 1)
        image = read_grayscale_page(tiff, page_index)
        if projection is None:
            projection = image.copy()
        else:
            if image.shape != projection.shape:
                raise ValueError(
                    f"Channel {channel} page {page_index + 1} shape "
                    f"{image.shape} does not match {projection.shape}."
                )
            np.maximum(projection, image, out=projection)

    if projection is None:
        raise ValueError(f"Channel {channel} has no z slices.")
    return projection


# -----------------------------------------------------------------------------
# Object-level measurements
#
# The D mask is separated into eight-connected objects. Every object is written
# to the object CSV. Objects that fail size or border rules remain in the file
# with included=False and an explicit exclusion reason.
# -----------------------------------------------------------------------------

def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio, using NaN for unavailable measurements."""
    if (
        math.isnan(numerator)
        or math.isnan(denominator)
        or denominator == 0
    ):
        return math.nan
    return numerator / denominator


def measure_objects(
    filename: str,
    detected_channels: int,
    z_method: str,
    selected_z: int | str,
    object_mask: np.ndarray,
    corrected_images: dict[str, np.ndarray | None],
    backgrounds: dict[str, float],
    min_object_size: int,
    max_object_size: int,
    exclude_border_objects: bool,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    """Measure every connected object and return the included analysis mask."""
    height, width = object_mask.shape
    included_mask = np.zeros_like(object_mask)
    rows: list[dict[str, object]] = []
    components = connected_components_8(object_mask)

    for object_id, indices in enumerate(components, start=1):
        y_coordinates = indices // width
        x_coordinates = indices % width
        area = int(indices.size)
        touches_border = bool(
            np.any(y_coordinates == 0)
            or np.any(y_coordinates == height - 1)
            or np.any(x_coordinates == 0)
            or np.any(x_coordinates == width - 1)
        )

        exclusion_reasons = []
        if area < min_object_size:
            exclusion_reasons.append(f"area<{min_object_size}")
        if max_object_size > 0 and area > max_object_size:
            exclusion_reasons.append(f"area>{max_object_size}")
        if exclude_border_objects and touches_border:
            exclusion_reasons.append("touches_border")
        included = not exclusion_reasons
        if included:
            included_mask.ravel()[indices] = True

        means: dict[str, float] = {}
        integrated: dict[str, float] = {}
        for name, image in corrected_images.items():
            if image is None:
                means[name] = math.nan
                integrated[name] = math.nan
            else:
                values = image.ravel()[indices]
                means[name] = float(values.mean())
                integrated[name] = float(values.sum(dtype=np.float64))

        row = {
            "filename": filename,
            "detected_channels": detected_channels,
            "z_method": z_method,
            "selected_z": selected_z,
            "object_id": object_id,
            "included": included,
            "exclusion_reason": ";".join(exclusion_reasons),
            "touches_border": touches_border,
            "area_pixels": area,
            "centroid_x_px_0based": float(x_coordinates.mean()),
            "centroid_y_px_0based": float(y_coordinates.mean()),
            "bbox_x_min_px_0based": int(x_coordinates.min()),
            "bbox_y_min_px_0based": int(y_coordinates.min()),
            "bbox_x_max_px_0based": int(x_coordinates.max()),
            "bbox_y_max_px_0based": int(y_coordinates.max()),
            "D_background": backgrounds["D"],
            "F_background": backgrounds["F"],
            "T_background": backgrounds["T"],
            "C_background": backgrounds["C"],
            "D_mean_corrected": means["D"],
            "F_mean_corrected": means["F"],
            "T_mean_corrected": means["T"],
            "C_mean_corrected": means["C"],
            "D_integrated_corrected": integrated["D"],
            "F_integrated_corrected": integrated["F"],
            "T_integrated_corrected": integrated["T"],
            "C_integrated_corrected": integrated["C"],
            "F_over_D": safe_ratio(means["F"], means["D"]),
            "T_over_D": safe_ratio(means["T"], means["D"]),
            "C_over_F": safe_ratio(means["C"], means["F"]),
        }
        rows.append(row)

    return included_mask, rows, len(components)


# -----------------------------------------------------------------------------
# Complete analysis of one TIFF
#
# This function loads or projects the requested channels, builds and refines
# the D-derived object/background masks, measures objects, and returns:
#   1. an image-level summary row,
#   2. all object-level rows,
#   3. arrays needed to build the QC montage.
# -----------------------------------------------------------------------------

def analyze_tiff(
    path: Path,
    requested_channels: int | None,
    z_mode: str,
    focus_channel: int,
    d_channel: int,
    f_channel: int,
    t_channel: int,
    c_channel: int,
    focus_threshold: float,
    background_threshold: float,
    object_threshold: float,
    background_dilation: int,
    min_object_size: int,
    max_object_size: int,
    exclude_border_objects: bool,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    """Analyze one interleaved, multi-channel TIFF stack."""
    # Read metadata and create one 2-D analysis image for each available
    # channel. Missing optional channels are represented by None and later
    # written as NaN rather than causing the TIFF to fail.
    with Image.open(path) as tiff:
        page_count = getattr(tiff, "n_frames", 1)
        total_channels, detection_method = detect_channel_count(
            tiff, requested_channels
        )
        if z_mode == "single" and focus_channel > total_channels:
            raise ValueError(
                f"Focus channel {focus_channel} is unavailable in a "
                f"{total_channels}-channel TIFF."
            )
        if d_channel > total_channels:
            raise ValueError(
                f"D/reference channel {d_channel} is unavailable in a "
                f"{total_channels}-channel TIFF."
            )

        channel_map = {
            "D": d_channel,
            "F": f_channel,
            "T": t_channel,
            "C": c_channel,
        }
        images: dict[str, np.ndarray | None] = {}
        if z_mode == "max":
            z_method = "max_projection"
            selected_z: int | str = ""
            z_score = math.nan
            projection_cache: dict[int, np.ndarray] = {}
            for name, channel in channel_map.items():
                if channel > total_channels:
                    images[name] = None
                    continue
                if channel not in projection_cache:
                    projection_cache[channel] = max_project_channel(
                        tiff,
                        total_channels=total_channels,
                        channel=channel,
                    )
                images[name] = projection_cache[channel]
        else:
            z_method = "single_slice"
            z_index, z_scores = select_z_slice(
                tiff,
                total_channels=total_channels,
                focus_channel=focus_channel,
                focus_threshold=focus_threshold,
            )
            selected_z = z_index + 1
            z_score = z_scores[z_index]
            page_cache: dict[int, np.ndarray] = {}
            for name, channel in channel_map.items():
                if channel > total_channels:
                    images[name] = None
                    continue
                if channel not in page_cache:
                    page_index = z_index * total_channels + (channel - 1)
                    page_cache[channel] = read_grayscale_page(tiff, page_index)
                images[name] = page_cache[channel]

    # All available channels must have the same x/y dimensions so the D mask
    # can be applied safely to every channel.
    d_image = images["D"]
    if d_image is None:
        raise RuntimeError("The D/reference image could not be read.")
    for name, image in images.items():
        if image is not None and image.shape != d_image.shape:
            raise ValueError(
                f"Analyzed {name} image shape {image.shape} does not match "
                f"the D/reference shape {d_image.shape}."
            )

    # First make a preliminary object mask using the low-intensity D pixels.
    # Calculate the initial median only from pixels selected as background.
    seed_background_mask = d_image < background_threshold
    if seed_background_mask.any():
        preliminary_d_background = float(
            np.median(d_image[seed_background_mask])
        )
        background_region_method = "outside dilated D object mask"
    else:
        # This keeps the batch running while making the fallback visible in
        # the CSV. The lowest 10% is only a pragmatic fallback, not a substitute
        # for a known object-free region.
        tenth_percentile = float(np.percentile(d_image, 10))
        seed_background_mask = d_image <= tenth_percentile
        preliminary_d_background = float(
            np.median(d_image[seed_background_mask])
        )
        background_region_method = (
            "outside dilated D object mask; initial lowest-10% fallback"
        )

    object_mask = (
        d_image - preliminary_d_background
    ) > object_threshold

    # Refine the object mask and per-channel backgrounds together:
    #   a. expand the current object mask,
    #   b. use pixels outside it as the background region,
    #   c. subtract each channel's median background,
    #   d. rebuild the D mask from corrected D intensities.
    # In typical images this converges after one or two passes.
    backgrounds: dict[str, float] = {}
    corrected: dict[str, np.ndarray | None] = {}
    background_mask = seed_background_mask
    for _ in range(10):
        expanded_object_mask = dilate_mask(object_mask, background_dilation)
        candidate_background_mask = ~expanded_object_mask

        if not candidate_background_mask.any():
            candidate_background_mask = ~object_mask
            if "dilation fallback" not in background_region_method:
                background_region_method += "; dilation fallback"
        if not candidate_background_mask.any():
            candidate_background_mask = seed_background_mask
            if "seed-mask fallback" not in background_region_method:
                background_region_method += "; seed-mask fallback"
        if not candidate_background_mask.any():
            raise ValueError("No usable background pixels were found.")

        background_mask = candidate_background_mask
        for name, image in images.items():
            if image is None:
                backgrounds[name] = math.nan
                corrected[name] = None
            else:
                background = float(np.median(image[background_mask]))
                backgrounds[name] = background
                corrected[name] = image - background

        d_corrected = corrected["D"]
        if d_corrected is None:
            raise RuntimeError("The D/reference image could not be corrected.")
        refined_object_mask = d_corrected > object_threshold
        if np.array_equal(refined_object_mask, object_mask):
            break
        object_mask = refined_object_mask

    # Separate the final threshold-positive mask into connected objects. The
    # returned included mask contains only objects that pass configured rules.
    included_object_mask, object_rows, object_count_total = measure_objects(
        filename=path.name,
        detected_channels=total_channels,
        z_method=z_method,
        selected_z=selected_z,
        object_mask=object_mask,
        corrected_images=corrected,
        backgrounds=backgrounds,
        min_object_size=min_object_size,
        max_object_size=max_object_size,
        exclude_border_objects=exclude_border_objects,
    )

    # Include every corrected pixel within included objects. Discarding
    # negative noise values would bias low-intensity channel means upward.
    means: dict[str, float] = {}
    for name, image in corrected.items():
        means[name] = (
            math.nan
            if image is None
            else masked_mean(image, included_object_mask)
        )

    object_count_included = sum(bool(row["included"]) for row in object_rows)

    # The image-level row summarizes all included object pixels. Detailed
    # object measurements remain available in IFObjects.csv.
    result = {
        "filename": path.name,
        "detected_channels": total_channels,
        "channel_detection": detection_method,
        "z_slices": page_count // total_channels,
        "z_method": z_method,
        "selected_z": selected_z,
        "z_score": z_score,
        "background_method": "per-channel object-free median",
        "background_region": background_region_method,
        "background_dilation_pixels": background_dilation,
        "background_pixel_count": int(background_mask.sum()),
        "preliminary_D_background": preliminary_d_background,
        "D_background": backgrounds["D"],
        "F_background": backgrounds["F"],
        "T_background": backgrounds["T"],
        "C_background": backgrounds["C"],
        "threshold_positive_pixel_count": int(object_mask.sum()),
        "object_pixel_count": int(included_object_mask.sum()),
        "object_count_total": object_count_total,
        "object_count_included": object_count_included,
        "object_count_excluded": object_count_total - object_count_included,
        "min_object_size_pixels": min_object_size,
        "max_object_size_pixels": max_object_size,
        "exclude_border_objects": exclude_border_objects,
        "qc_file": "",
        "IFData_1_D_mean": means["D"],
        "IFData_2_F_mean": means["F"],
        "IFData_3_T_mean": means["T"],
        "IFData_4_C_mean": means["C"],
        "IFData_5_F_over_D": safe_ratio(means["F"], means["D"]),
        "IFData_6_T_over_D": safe_ratio(means["T"], means["D"]),
        "IFData_7_C_over_F": safe_ratio(means["C"], means["F"]),
    }
    # QC arrays are returned separately so display scaling cannot accidentally
    # affect the quantitative arrays above.
    qc_payload: dict[str, object] = {
        "raw_images": images,
        "corrected_images": corrected,
        "threshold_object_mask": object_mask,
        "included_object_mask": included_object_mask,
        "background_mask": background_mask,
    }
    return result, object_rows, qc_payload


# -----------------------------------------------------------------------------
# Command-line option validation and definitions
#
# Small type-checking helpers give clear messages for invalid channel numbers,
# object sizes, dilation radii, and QC dimensions before analysis begins.
# -----------------------------------------------------------------------------

def channel_number(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("channel numbers must be 1 or greater")
    return number


def channel_setting(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return channel_number(value)


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be one or greater")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Maximum-project each TIFF channel across z, then analyze the "
            "D-derived object mask and fluorescence measurements."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="folder containing TIFF files (default: current folder)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV path (default: IFData.csv inside the input folder)",
    )
    parser.add_argument(
        "--objects-output",
        type=Path,
        help=(
            "object-level CSV path "
            "(default: IFObjects.csv inside the input folder)"
        ),
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        help="QC image directory (default: QC inside the input folder)",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="do not create per-image QC montages",
    )
    parser.add_argument(
        "--qc-max-size",
        type=positive_integer,
        default=600,
        metavar="PIXELS",
        help="maximum width/height of each QC panel (default: 600)",
    )
    parser.add_argument(
        "--channels",
        type=channel_setting,
        default=None,
        metavar="auto|N",
        help=(
            "detect channels per TIFF or force N channels "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--z-mode",
        choices=("max", "single"),
        default="max",
        help=(
            "maximum-project each channel or select one z slice "
            "(default: max)"
        ),
    )
    parser.add_argument(
        "--focus-channel",
        type=channel_number,
        default=1,
        help="channel used for --z-mode single selection (default: 1)",
    )
    parser.add_argument(
        "--d-channel",
        type=channel_number,
        default=1,
        help="D/reference channel (default: 1)",
    )
    parser.add_argument(
        "--f-channel",
        type=channel_number,
        default=2,
        help="F/signal channel (default: 2)",
    )
    parser.add_argument(
        "--t-channel",
        type=channel_number,
        default=3,
        help="T/signal channel (default: 3; omitted if unavailable)",
    )
    parser.add_argument(
        "--c-channel",
        type=channel_number,
        default=4,
        help="C/signal channel (default: 4; omitted if unavailable)",
    )
    parser.add_argument(
        "--focus-threshold",
        type=float,
        default=700.0,
        help="focus threshold used only by --z-mode single (default: 700)",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=500.0,
        help=(
            "initial D intensity cutoff used to seed background estimation "
            "(default: 500)"
        ),
    )
    parser.add_argument(
        "--object-threshold",
        type=float,
        default=3500.0,
        help="background-corrected D cutoff for the object mask (default: 3500)",
    )
    parser.add_argument(
        "--background-dilation",
        type=nonnegative_integer,
        default=5,
        metavar="PIXELS",
        help=(
            "expand the D object mask before selecting background pixels "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        "--min-object-size",
        type=positive_integer,
        default=1,
        metavar="PIXELS",
        help="minimum connected-object area to include (default: 1)",
    )
    parser.add_argument(
        "--max-object-size",
        type=nonnegative_integer,
        default=0,
        metavar="PIXELS",
        help="maximum object area; 0 disables the limit (default: 0)",
    )
    parser.add_argument(
        "--exclude-border-objects",
        action="store_true",
        help="exclude objects touching an image border from image-level means",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Batch execution and output writing
#
# Every TIFF is analyzed independently. A damaged or incompatible TIFF is
# reported and skipped so other files can still be processed. Image summaries,
# object rows, and QC montages are written only after successful analysis.
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()

    # Validate settings that apply to the entire batch before opening images.
    if not folder.is_dir():
        print(f"Input folder does not exist: {folder}", file=sys.stderr)
        return 2
    if (
        args.max_object_size > 0
        and args.max_object_size < args.min_object_size
    ):
        print(
            "--max-object-size must be zero or at least --min-object-size.",
            file=sys.stderr,
        )
        return 2

    # Analyze TIFF files directly inside the input folder. Subfolders, existing
    # CSV files, and QC images are not treated as inputs.
    tiff_paths = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.name,
    )
    if not tiff_paths:
        print(f"No .tif or .tiff files found in: {folder}", file=sys.stderr)
        return 1

    # Unless overridden, keep all results beside the source TIFF files.
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else folder / "IFData.csv"
    )
    objects_output_path = (
        args.objects_output.expanduser().resolve()
        if args.objects_output
        else folder / "IFObjects.csv"
    )
    qc_directory = (
        args.qc_dir.expanduser().resolve()
        if args.qc_dir
        else folder / "QC"
    )

    results = []
    all_object_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    qc_failures: list[tuple[str, str]] = []

    # Analyze files one at a time to limit memory use and isolate file errors.
    for path in tiff_paths:
        try:
            result, object_rows, qc_payload = analyze_tiff(
                path,
                requested_channels=args.channels,
                z_mode=args.z_mode,
                focus_channel=args.focus_channel,
                d_channel=args.d_channel,
                f_channel=args.f_channel,
                t_channel=args.t_channel,
                c_channel=args.c_channel,
                focus_threshold=args.focus_threshold,
                background_threshold=args.background_threshold,
                object_threshold=args.object_threshold,
                background_dilation=args.background_dilation,
                min_object_size=args.min_object_size,
                max_object_size=args.max_object_size,
                exclude_border_objects=args.exclude_border_objects,
            )
        except (OSError, ValueError, RuntimeError) as error:
            failures.append((path.name, str(error)))
            print(f"ERROR: {path.name}: {error}", file=sys.stderr)
            continue

        if not args.no_qc:
            qc_path = qc_directory / f"{path.name}_qc.png"
            try:
                save_qc_image(
                    qc_path,
                    raw_images=qc_payload["raw_images"],
                    corrected_images=qc_payload["corrected_images"],
                    threshold_object_mask=qc_payload[
                        "threshold_object_mask"
                    ],
                    included_object_mask=qc_payload[
                        "included_object_mask"
                    ],
                    background_mask=qc_payload["background_mask"],
                    result=result,
                    object_rows=object_rows,
                    max_image_size=args.qc_max_size,
                )
                result["qc_file"] = str(qc_path)
            except (OSError, ValueError) as error:
                qc_failures.append((path.name, str(error)))
                print(
                    f"WARNING: {path.name}: QC image was not saved: {error}",
                    file=sys.stderr,
                )

        results.append(result)
        all_object_rows.extend(object_rows)
        if str(result["channel_detection"]).startswith("ambiguous"):
            print(
                f"WARNING: {path.name}: {result['channel_detection']}. "
                "Use --channels N if this is incorrect.",
                file=sys.stderr,
            )
        if "fallback" in str(result["background_region"]):
            print(
                f"WARNING: {path.name}: background region used "
                f"{result['background_region']}.",
                file=sys.stderr,
            )
        if result["object_pixel_count"] == 0:
            print(
                f"WARNING: {path.name}: no objects passed the filters; "
                "review the object threshold and size/border settings.",
                file=sys.stderr,
            )
        z_description = (
            f"max({result['z_slices']})"
            if result["z_method"] == "max_projection"
            else str(result["selected_z"])
        )
        print(
            f"{path.name}: channels={result['detected_channels']}, "
            f"z={z_description}, "
            f"objects={result['object_count_included']}/"
            f"{result['object_count_total']}, "
            f"D={result['IFData_1_D_mean']:.6g}, "
            f"F={result['IFData_2_F_mean']:.6g}, "
            f"T={result['IFData_3_T_mean']:.6g}, "
            f"C={result['IFData_4_C_mean']:.6g}"
        )

    if not results:
        print("No TIFF files were successfully analyzed.", file=sys.stderr)
        return 1

    # Write the image-level table.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # Write every detected object, including objects marked as excluded.
    objects_output_path.parent.mkdir(parents=True, exist_ok=True)
    with objects_output_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OBJECT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_object_rows)

    print(f"\nAnalyzed {len(results)} TIFF file(s).")
    if failures:
        print(f"Skipped {len(failures)} unreadable or incompatible file(s).")
    if qc_failures:
        print(f"Failed to create {len(qc_failures)} QC image(s).")
    print(f"Results written to: {output_path}")
    print(f"Object measurements written to: {objects_output_path}")
    if not args.no_qc:
        print(f"QC images written to: {qc_directory}")
    return 1 if failures or qc_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
