"""Backward-compatible import path for the consolidated preprocessing pipeline.

The implementation now lives in :mod:`inputpreprocessing`.  This module remains
so older experiments importing ``inputpreprocessing_v2`` keep working.
"""

from inputpreprocessing import (
    CropInfo,
    LearnedRectifier,
    PerspectiveInfo,
    PreprocessConfig,
    SurfaceInfo,
    binarize,
    crop_equation_region,
    letterbox_resize,
    locate_equation_region,
    pad_mobilenet_batch,
    perspective_correct,
    preprocess_image,
    preprocess_pipeline,
    resize_to_height,
    to_mobilenet_input,
)

__all__ = [
    "CropInfo",
    "LearnedRectifier",
    "PerspectiveInfo",
    "PreprocessConfig",
    "SurfaceInfo",
    "binarize",
    "crop_equation_region",
    "letterbox_resize",
    "locate_equation_region",
    "pad_mobilenet_batch",
    "perspective_correct",
    "preprocess_image",
    "preprocess_pipeline",
    "resize_to_height",
    "to_mobilenet_input",
]


if __name__ == "__main__":
    from inputpreprocessing import _main

    _main()
