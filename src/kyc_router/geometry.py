"""Known-layout geometry only: no image loading, identity lookup or text labels."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .schema import RAW_TO_CANONICAL, REQUIRED_FIELDS


def check_rgb(image: np.ndarray, size: tuple[int, int] | None = None) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 RGB NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 pixels, got {image.dtype}.")
    if not image.size:
        raise ValueError("Image must not be empty.")
    if size is not None and image.shape[:2] != (size[1], size[0]):
        raise ValueError(f"Expected canonical RGB size {size}, got {image.shape}.")


def _check_rectangle(shape: Mapping[str, int]) -> None:
    for key in ("x", "y", "width", "height"):
        if key not in shape or isinstance(shape[key], bool) or not isinstance(shape[key], Integral):
            raise ValueError(f"Rectangle requires integer {key!r}.")
    if shape["width"] <= 0 or shape["height"] <= 0:
        raise ValueError(f"Invalid field shape: {shape}")


def scale_field_shape(shape, source_width, source_height, target_width, target_height) -> dict[str, int]:
    """Match notebook rounding of each coordinate/extent independently."""
    _check_rectangle(shape)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive.")
    return {
        "x": round(shape["x"] * target_width / source_width),
        "y": round(shape["y"] * target_height / source_height),
        "width": round(shape["width"] * target_width / source_width),
        "height": round(shape["height"] * target_height / source_height),
    }


def rectangles_from_template(
    record: Mapping,
    source_width: int,
    source_height: int,
    target_width: int = 1084,
    target_height: int = 680,
) -> dict[str, dict[str, int]]:
    """Small MIDV/VIA geometry adapter; never reads region_attributes['value']."""
    rectangles = {}
    for raw_name, canonical_name in RAW_TO_CANONICAL.items():
        regions = [region for region in record["regions"]
                   if region["region_attributes"].get("field_name") == raw_name]
        if len(regions) != 1:
            raise ValueError(f"Expected exactly one region named {raw_name!r}, found {len(regions)}.")
        shape = regions[0]["shape_attributes"]
        if shape.get("name") != "rect":
            raise ValueError(f"Field {raw_name!r} must be a rectangle.")
        rectangles[canonical_name] = scale_field_shape(
            shape, source_width, source_height, target_width, target_height)
    return rectangles


def crop_field(
    image_rgb: np.ndarray,
    shape: Mapping[str, int],
    *,
    padding_x: float = 0.1,
    minimum_padding_x: int = 12,
    padding_top: float = 0.1,
    padding_bottom: float = 0.4,
) -> tuple[np.ndarray, dict[str, int]]:
    check_rgb(image_rgb)
    _check_rectangle(shape)
    if min(padding_x, minimum_padding_x, padding_top, padding_bottom) < 0:
        raise ValueError("Crop padding must be nonnegative.")
    x, y, width, height = (int(shape[k]) for k in ("x", "y", "width", "height"))
    image_height, image_width = image_rgb.shape[:2]
    horizontal = max(round(width * padding_x), minimum_padding_x)
    bounds = {
        "x1": max(0, x - horizontal),
        "y1": max(0, y - round(height * padding_top)),
        "x2": min(image_width, x + width + horizontal),
        "y2": min(image_height, y + height + round(height * padding_bottom)),
    }
    if bounds["x2"] <= bounds["x1"] or bounds["y2"] <= bounds["y1"]:
        raise ValueError(f"Empty crop produced from shape {shape}.")
    crop = image_rgb[bounds["y1"]:bounds["y2"], bounds["x1"]:bounds["x2"]].copy()
    return crop, bounds


def extract_field_crops(image_rgb: np.ndarray, rectangles: Mapping, crop_config: Mapping) -> dict:
    if set(rectangles) != set(REQUIRED_FIELDS):
        raise ValueError("Rectangles must contain exactly the seven required canonical fields.")
    # Build all crops before invoking the model: invalid geometry fails early.
    return {field: crop_field(image_rgb, rectangles[field], **crop_config) for field in REQUIRED_FIELDS}
