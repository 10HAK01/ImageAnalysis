#!/usr/bin/env python3
"""
Measure channel-3 fluorescence inside a bulk channel-2 centromere mask.

Analysis workflow:
    1. Detect the number of interleaved channels in each TIFF.
    2. Maximum-project channels across z, or optionally select one z slice.
    3. Estimate the channel-2 background and threshold channel 2 to create one
       bulk mask containing every centromere-positive pixel. Clustered foci
       remain together and are not split or counted as individual objects.
    4. Expand the bulk mask to define an object-free background region.
    5. Subtract the median background independently from channels 2 and 3.
    6. Measure channel-3 mean and integrated fluorescence inside the bulk mask.
    7. Save one result row, one binary mask, and one QC montage per TIFF.

Expected TIFF page order:
    z1/channel1, z1/channel2, ..., z2/channel1, z2/channel2, ...

Default channel roles:
    channel 2 = centromere/mask channel
    channel 3 = FL measurement channel

Default outputs inside the TIFF folder:
    CentromereBulk_FL_Data.csv
    CentromereBulkMasks/
    CentromereBulk_QC/

Example:
    python3 centromere_bulk_fl_analysis.py /path/to/tiff_folder
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    sys.exit(
        "This script needs NumPy and Pillow.\n"
        "Install them with: python3 -m pip install numpy Pillow"
    )


# -----------------------------------------------------------------------------
# Basic image and mask utilities
# -----------------------------------------------------------------------------

def masked_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean of all masked pixels, including negative values."""
    values = image[mask]
    if values.size == 0:
        return math.nan
    return float(values.mean())


def masked_sum(image: np.ndarray, mask: np.ndarray) -> float:
    """Return the integrated intensity of all masked pixels."""
    values = image[mask]
    if values.size == 0:
        return math.nan
    return float(values.sum(dtype=np.float64))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio or NaN when either value is unavailable."""
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a binary mask using a square neighborhood."""
    if radius <= 0:
        return mask.copy()
    kernel_size = 2 * radius + 1
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    expanded = mask_image.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(expanded) > 0


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


# -----------------------------------------------------------------------------
# TIFF reading and channel detection
# -----------------------------------------------------------------------------

def read_grayscale_page(tiff: Image.Image, page_index: int) -> np.ndarray:
    """Read one zero-based TIFF page as a two-dimensional float array."""
    tiff.seek(page_index)
    image = np.asarray(tiff)
    if image.ndim != 2:
        raise ValueError(
            f"Page {page_index + 1} is not a 2-D grayscale image "
            f"(shape: {image.shape})."
        )
    return image.astype(np.float64, copy=False)


def image_description(tiff: Image.Image) -> str:
    """Return the TIFF ImageDescription metadata as text when available."""
    description = tiff.tag_v2.get(270, "")
    if isinstance(description, bytes):
        return description.decode("utf-8", errors="replace")
    if isinstance(description, (tuple, list)):
        return "\n".join(str(item) for item in description)
    return str(description)


def detect_channel_count(
    tiff: Image.Image, requested_channels: int | None
) -> tuple[int, str]:
    """Detect channel count from an override, metadata, or page divisibility."""
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
        if not match:
            continue
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
# -----------------------------------------------------------------------------

def select_z_slice(
    tiff: Image.Image,
    total_channels: int,
    focus_channel: int,
    focus_threshold: float,
) -> tuple[int, list[float]]:
    """Select the z slice with the highest focus-channel thresholded mean."""
    page_count = getattr(tiff, "n_frames", 1)
    z_count = page_count // total_channels
    scores: list[float] = []

    for z_index in range(z_count):
        page_index = z_index * total_channels + (focus_channel - 1)
        image = read_grayscale_page(tiff, page_index)
        values = image[image > focus_threshold]
        scores.append(float(values.mean()) if values.size else math.nan)

    score_array = np.asarray(scores, dtype=np.float64)
    if np.isnan(score_array).all():
        return 0, scores
    return int(np.nanargmax(score_array)), scores


def max_project_channel(
    tiff: Image.Image, total_channels: int, channel: int
) -> np.ndarray:
    """Maximum-project one interleaved channel across all z slices."""
    page_count = getattr(tiff, "n_frames", 1)
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
                    f"Channel {channel} page {page_index + 1} has shape "
                    f"{image.shape}; expected {projection.shape}."
                )
            np.maximum(projection, image, out=projection)

    if projection is None:
        raise ValueError(f"Channel {channel} has no z slices.")
    return projection


# -----------------------------------------------------------------------------
# Bulk-mask analysis
# -----------------------------------------------------------------------------

def analyze_tiff(
    path: Path,
    requested_channels: int | None,
    z_mode: str,
    mask_channel: int,
    fl_channel: int,
    focus_channel: int,
    focus_threshold: float,
    background_threshold: float,
    mask_threshold: float,
    background_dilation: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Analyze one TIFF and return its data row plus arrays for output/QC."""
    with Image.open(path) as tiff:
        page_count = getattr(tiff, "n_frames", 1)
        total_channels, detection_method = detect_channel_count(
            tiff, requested_channels
        )

        for role, channel in (
            ("mask", mask_channel),
            ("FL", fl_channel),
        ):
            if channel > total_channels:
                raise ValueError(
                    f"The requested {role} channel {channel} is unavailable "
                    f"in this {total_channels}-channel TIFF."
                )
        if z_mode == "single" and focus_channel > total_channels:
            raise ValueError(
                f"Focus channel {focus_channel} is unavailable in this "
                f"{total_channels}-channel TIFF."
            )

        z_slices = page_count // total_channels
        if z_mode == "max":
            z_method = "max_projection"
            selected_z: int | str = ""
            z_score = math.nan
            projections: dict[int, np.ndarray] = {}
            for channel in {mask_channel, fl_channel}:
                projections[channel] = max_project_channel(
                    tiff, total_channels, channel
                )
            mask_image = projections[mask_channel]
            fl_image = projections[fl_channel]
        else:
            z_method = "single_slice"
            z_index, z_scores = select_z_slice(
                tiff, total_channels, focus_channel, focus_threshold
            )
            selected_z = z_index + 1
            z_score = z_scores[z_index]
            pages: dict[int, np.ndarray] = {}
            for channel in {mask_channel, fl_channel}:
                page_index = z_index * total_channels + (channel - 1)
                pages[channel] = read_grayscale_page(tiff, page_index)
            mask_image = pages[mask_channel]
            fl_image = pages[fl_channel]

    if mask_image.shape != fl_image.shape:
        raise ValueError(
            f"Mask-channel shape {mask_image.shape} does not match "
            f"FL-channel shape {fl_image.shape}."
        )

    # Seed the first background estimate with low-intensity channel-2 pixels.
    seed_background_mask = mask_image < background_threshold
    if seed_background_mask.any():
        preliminary_background = float(
            np.median(mask_image[seed_background_mask])
        )
        background_region_method = "outside dilated bulk centromere mask"
    else:
        tenth_percentile = float(np.percentile(mask_image, 10))
        seed_background_mask = mask_image <= tenth_percentile
        preliminary_background = float(
            np.median(mask_image[seed_background_mask])
        )
        background_region_method = (
            "outside dilated bulk centromere mask; lowest-10% seed fallback"
        )

    # Every threshold-positive pixel belongs to one bulk mask. No connected-
    # component splitting or individual-focus counting is performed.
    bulk_mask = (mask_image - preliminary_background) > mask_threshold

    background_mask = seed_background_mask
    mask_background = math.nan
    fl_background = math.nan
    mask_corrected = mask_image.copy()
    fl_corrected = fl_image.copy()

    # Refine the background and bulk mask until the mask stops changing.
    for _ in range(10):
        background_mask = ~dilate_mask(bulk_mask, background_dilation)
        if not background_mask.any():
            background_mask = ~bulk_mask
            if "dilation fallback" not in background_region_method:
                background_region_method += "; dilation fallback"
        if not background_mask.any():
            background_mask = seed_background_mask
            if "seed-mask fallback" not in background_region_method:
                background_region_method += "; seed-mask fallback"
        if not background_mask.any():
            raise ValueError("No usable background pixels were found.")

        mask_background = float(np.median(mask_image[background_mask]))
        fl_background = float(np.median(fl_image[background_mask]))
        mask_corrected = mask_image - mask_background
        fl_corrected = fl_image - fl_background
        refined_mask = mask_corrected > mask_threshold
        if np.array_equal(refined_mask, bulk_mask):
            break
        bulk_mask = refined_mask

    mask_pixel_count = int(bulk_mask.sum())
    total_pixel_count = int(bulk_mask.size)
    mask_fraction = mask_pixel_count / total_pixel_count
    mask_mean = masked_mean(mask_corrected, bulk_mask)
    mask_integrated = masked_sum(mask_corrected, bulk_mask)
    fl_mean = masked_mean(fl_corrected, bulk_mask)
    fl_integrated = masked_sum(fl_corrected, bulk_mask)

    result = {
        "filename": path.name,
        "detected_channels": total_channels,
        "channel_detection": detection_method,
        "z_slices": z_slices,
        "z_method": z_method,
        "selected_z": selected_z,
        "z_score": z_score,
        "mask_channel": mask_channel,
        "fl_channel": fl_channel,
        "mask_threshold_corrected": mask_threshold,
        "background_seed_threshold": background_threshold,
        "background_method": "per-channel object-free median",
        "background_region": background_region_method,
        "background_dilation_pixels": background_dilation,
        "background_pixel_count": int(background_mask.sum()),
        "preliminary_mask_channel_background": preliminary_background,
        "mask_channel_background": mask_background,
        "fl_channel_background": fl_background,
        "bulk_mask_pixel_count": mask_pixel_count,
        "bulk_mask_fraction": mask_fraction,
        "mask_channel_mean_corrected": mask_mean,
        "mask_channel_integrated_corrected": mask_integrated,
        "fl_mean_corrected_in_bulk_mask": fl_mean,
        "fl_integrated_corrected_in_bulk_mask": fl_integrated,
        "fl_over_mask_channel_mean": safe_ratio(fl_mean, mask_mean),
        "mask_file": "",
        "qc_file": "",
    }
    arrays: dict[str, object] = {
        "mask_image": mask_image,
        "fl_image": fl_image,
        "mask_corrected": mask_corrected,
        "fl_corrected": fl_corrected,
        "bulk_mask": bulk_mask,
        "background_mask": background_mask,
    }
    return result, arrays


# -----------------------------------------------------------------------------
# Binary-mask and QC outputs
# -----------------------------------------------------------------------------

def image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Scale an image robustly for display without altering measurements."""
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
    """Add a translucent colored mask to an RGB display image."""
    result = rgb.astype(np.float64, copy=True)
    color_array = np.asarray(color, dtype=np.float64)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color_array
    return np.asarray(np.clip(result, 0, 255), dtype=np.uint8)


def make_panel(
    rgb: np.ndarray, title: str, max_image_size: int
) -> Image.Image:
    """Resize an RGB image and add a title bar."""
    image = Image.fromarray(rgb, mode="RGB")
    scale = max_image_size / max(image.width, image.height)
    target_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)

    title_height = 26
    panel = Image.new(
        "RGB", (image.width, image.height + title_height), (18, 18, 18)
    )
    panel.paste(image, (0, title_height))
    ImageDraw.Draw(panel).text((6, 7), title, fill=(255, 255, 255))
    return panel


def save_bulk_mask(output_path: Path, bulk_mask: np.ndarray) -> None:
    """Save the quantitative bulk mask as an 8-bit black-and-white PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bulk_mask.astype(np.uint8) * 255).save(
        output_path, format="PNG"
    )


def save_qc_image(
    output_path: Path,
    arrays: dict[str, object],
    result: dict[str, object],
    max_image_size: int,
) -> None:
    """Save a QC montage of the images, bulk mask, and background region."""
    mask_image = arrays["mask_image"]
    fl_image = arrays["fl_image"]
    mask_corrected = arrays["mask_corrected"]
    fl_corrected = arrays["fl_corrected"]
    bulk_mask = arrays["bulk_mask"]
    background_mask = arrays["background_mask"]

    mask_rgb = image_to_rgb(mask_image)
    fl_rgb = image_to_rgb(fl_image)
    bulk_overlay = tint_mask(mask_rgb, bulk_mask, (0, 255, 0), 0.55)
    bulk_overlay[mask_outline(bulk_mask)] = (255, 255, 0)
    background_overlay = tint_mask(
        mask_rgb, background_mask, (0, 100, 255), 0.5
    )
    background_overlay[mask_outline(bulk_mask)] = (255, 0, 0)

    projection_label = (
        "maximum projection"
        if result["z_method"] == "max_projection"
        else f"selected z {result['selected_z']}"
    )
    panels = [
        make_panel(
            mask_rgb,
            f"Channel {result['mask_channel']} ({projection_label})",
            max_image_size,
        ),
        make_panel(
            bulk_overlay,
            "Bulk centromere mask (green)",
            max_image_size,
        ),
        make_panel(
            background_overlay,
            "Background blue; mask red",
            max_image_size,
        ),
        make_panel(
            image_to_rgb(mask_corrected),
            f"Corrected channel {result['mask_channel']}",
            max_image_size,
        ),
        make_panel(
            fl_rgb,
            f"Channel {result['fl_channel']} FL ({projection_label})",
            max_image_size,
        ),
        make_panel(
            image_to_rgb(fl_corrected),
            f"Corrected channel {result['fl_channel']} FL",
            max_image_size,
        ),
    ]

    panel_width = panels[0].width
    panel_height = panels[0].height
    summary = Image.new("RGB", (panel_width, panel_height), (25, 25, 25))
    summary_lines = [
        str(result["filename"]),
        f"channels={result['detected_channels']}",
        f"z={projection_label}; slices={result['z_slices']}",
        f"bulk pixels={result['bulk_mask_pixel_count']}",
        f"bulk fraction={result['bulk_mask_fraction']:.4g}",
        (
            f"background ch{result['mask_channel']}/"
            f"ch{result['fl_channel']}: "
            f"{result['mask_channel_background']:.4g}, "
            f"{result['fl_channel_background']:.4g}"
        ),
        f"FL mean={result['fl_mean_corrected_in_bulk_mask']:.4g}",
        f"FL integrated={result['fl_integrated_corrected_in_bulk_mask']:.4g}",
        f"background: {result['background_region']}",
    ]
    ImageDraw.Draw(summary).multiline_text(
        (8, 10), "\n".join(summary_lines), fill=(255, 255, 255), spacing=5
    )
    panels.append(summary)

    blank = Image.new("RGB", (panel_width, panel_height), (0, 0, 0))
    panels.append(blank)

    montage = Image.new(
        "RGB", (4 * panel_width, 2 * panel_height), (0, 0, 0)
    )
    for index, panel in enumerate(panels):
        montage.paste(
            panel,
            ((index % 4) * panel_width, (index // 4) * panel_height),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, format="PNG")


# -----------------------------------------------------------------------------
# Command-line options
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
            "Measure channel-3 FL intensity inside a bulk channel-2 "
            "centromere mask."
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
        help=(
            "result CSV path "
            "(default: CentromereBulk_FL_Data.csv in the TIFF folder)"
        ),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="binary-mask folder (default: CentromereBulkMasks)",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        help="QC montage folder (default: CentromereBulk_QC)",
    )
    parser.add_argument(
        "--no-mask-files",
        action="store_true",
        help="do not save separate binary mask PNG files",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="do not save QC montage PNG files",
    )
    parser.add_argument(
        "--qc-max-size",
        type=positive_integer,
        default=600,
        metavar="PIXELS",
        help="width/height of each QC panel (default: 600)",
    )
    parser.add_argument(
        "--channels",
        type=channel_setting,
        default=None,
        metavar="auto|N",
        help="detect channels per TIFF or force N channels (default: auto)",
    )
    parser.add_argument(
        "--z-mode",
        choices=("max", "single"),
        default="max",
        help="maximum-project channels or select one z slice (default: max)",
    )
    parser.add_argument(
        "--mask-channel",
        type=channel_number,
        default=2,
        help="centromere channel used to make the bulk mask (default: 2)",
    )
    parser.add_argument(
        "--fl-channel",
        type=channel_number,
        default=3,
        help="channel measured inside the bulk mask (default: 3)",
    )
    parser.add_argument(
        "--focus-channel",
        type=channel_number,
        default=2,
        help="channel used only for single-slice selection (default: 2)",
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
        help="initial mask-channel background cutoff (default: 500)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=3500.0,
        help="corrected mask-channel cutoff for the bulk mask (default: 3500)",
    )
    parser.add_argument(
        "--background-dilation",
        type=nonnegative_integer,
        default=5,
        metavar="PIXELS",
        help="expand the bulk mask before selecting background (default: 5)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Batch execution and output writing
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Input folder does not exist: {folder}", file=sys.stderr)
        return 2

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

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else folder / "CentromereBulk_FL_Data.csv"
    )
    mask_directory = (
        args.mask_dir.expanduser().resolve()
        if args.mask_dir
        else folder / "CentromereBulkMasks"
    )
    qc_directory = (
        args.qc_dir.expanduser().resolve()
        if args.qc_dir
        else folder / "CentromereBulk_QC"
    )

    results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    output_failures: list[tuple[str, str]] = []

    for path in tiff_paths:
        try:
            result, arrays = analyze_tiff(
                path=path,
                requested_channels=args.channels,
                z_mode=args.z_mode,
                mask_channel=args.mask_channel,
                fl_channel=args.fl_channel,
                focus_channel=args.focus_channel,
                focus_threshold=args.focus_threshold,
                background_threshold=args.background_threshold,
                mask_threshold=args.mask_threshold,
                background_dilation=args.background_dilation,
            )
        except (OSError, ValueError, RuntimeError) as error:
            failures.append((path.name, str(error)))
            print(f"ERROR: {path.name}: {error}", file=sys.stderr)
            continue

        if not args.no_mask_files:
            mask_path = mask_directory / f"{path.name}_bulk_mask.png"
            try:
                save_bulk_mask(mask_path, arrays["bulk_mask"])
                result["mask_file"] = str(mask_path)
            except OSError as error:
                output_failures.append((path.name, str(error)))
                print(
                    f"WARNING: {path.name}: mask file was not saved: {error}",
                    file=sys.stderr,
                )

        if not args.no_qc:
            qc_path = qc_directory / f"{path.name}_qc.png"
            try:
                save_qc_image(qc_path, arrays, result, args.qc_max_size)
                result["qc_file"] = str(qc_path)
            except (OSError, ValueError) as error:
                output_failures.append((path.name, str(error)))
                print(
                    f"WARNING: {path.name}: QC image was not saved: {error}",
                    file=sys.stderr,
                )

        results.append(result)
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
        if result["bulk_mask_pixel_count"] == 0:
            print(
                f"WARNING: {path.name}: the bulk centromere mask is empty; "
                "review --mask-threshold.",
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
            f"bulk pixels={result['bulk_mask_pixel_count']}, "
            f"FL mean={result['fl_mean_corrected_in_bulk_mask']:.6g}, "
            f"FL integrated="
            f"{result['fl_integrated_corrected_in_bulk_mask']:.6g}"
        )

    if not results:
        print("No TIFF files were successfully analyzed.", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAnalyzed {len(results)} TIFF file(s).")
    if failures:
        print(f"Skipped {len(failures)} incompatible file(s).")
    if output_failures:
        print(f"Could not save {len(output_failures)} mask/QC output(s).")
    print(f"Results written to: {output_path}")
    if not args.no_mask_files:
        print(f"Bulk masks written to: {mask_directory}")
    if not args.no_qc:
        print(f"QC images written to: {qc_directory}")
    return 1 if failures or output_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
