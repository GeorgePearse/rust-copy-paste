"""Albumentations-compatible copy-paste augmentation using Rust implementation."""

import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
from loguru import logger

try:
    from copy_paste._core import CopyPasteTransform
    RUST_AVAILABLE = True
except ImportError:
    logger.warning("copy_paste Rust module not available. Build with: maturin develop")
    RUST_AVAILABLE = False


class CopyPasteAugmentation(A.DualTransform):
    """Copy-paste augmentation using Rust implementation with Albumentations interface.

    This transform applies copy-paste augmentation to images and masks, utilizing
    a Rust implementation for high performance. It maintains full compatibility with
    Albumentations' DualTransform interface.

    Args:
        image_width: Width of output images
        image_height: Height of output images
        max_paste_objects: Maximum number of objects to paste per image
        use_rotation: Whether to apply random rotation
        use_scaling: Whether to apply random scaling
        rotation_range: Range of rotation in degrees (min, max)
        scale_range: Range of scaling factors (min, max)
        use_random_background: Whether to generate random background
        blend_mode: Blending mode ('normal' or 'xray')
        p: Probability of applying the transform (0.0 to 1.0)
    """

    def __init__(
        self,
        image_width: int = 512,
        image_height: int = 512,
        max_paste_objects: int = 1,
        use_rotation: bool = True,
        use_scaling: bool = True,
        rotation_range: tuple[float, float] = (-30.0, 30.0),
        scale_range: tuple[float, float] = (0.8, 1.2),
        use_random_background: bool = False,
        blend_mode: str = "normal",
        p: float = 1.0,
    ):
        """Initialize the CopyPasteAugmentation transform."""
        super().__init__(p=p)

        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust implementation not available. "
                "Build with: pip install -e . or maturin develop"
            )

        self.image_width = image_width
        self.image_height = image_height
        self.max_paste_objects = max_paste_objects
        self.use_rotation = use_rotation
        self.use_scaling = use_scaling
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.use_random_background = use_random_background
        self.blend_mode = blend_mode

        # Initialize Rust transform
        self.rust_transform = CopyPasteTransform(
            image_width=image_width,
            image_height=image_height,
            max_paste_objects=max_paste_objects,
            use_rotation=use_rotation,
            use_scaling=use_scaling,
            use_random_background=use_random_background,
            blend_mode=blend_mode,
        )

        logger.info(
            f"Initialized CopyPasteAugmentation | "
            f"size=({image_width}x{image_height}), "
            f"max_objects={max_paste_objects}, "
            f"rotation={use_rotation}, scaling={use_scaling}"
        )

    def apply(
        self,
        img: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply copy-paste augmentation to image.

        Args:
            img: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Augmented image
        """
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # Ensure image is BGR
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {img.shape}")

        # For now, return image as-is (placeholder)
        # Full implementation would call Rust methods here
        return img

    def apply_to_masks(
        self,
        masks: list[np.ndarray],
        **params: Any,
    ) -> list[np.ndarray]:
        """Apply transformations to masks.

        Args:
            masks: List of binary masks

        Returns:
            Transformed masks
        """
        # For now, return masks as-is (placeholder)
        return masks

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply transformations to bounding boxes.

        Args:
            bboxes: Bounding boxes in normalized format [x_min, y_min, x_max, y_max, class_label]

        Returns:
            Transformed bounding boxes
        """
        # For now, return bboxes as-is (placeholder)
        return bboxes

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return names of arguments used to initialize the transform."""
        return (
            "image_width",
            "image_height",
            "max_paste_objects",
            "use_rotation",
            "use_scaling",
            "rotation_range",
            "scale_range",
            "use_random_background",
            "blend_mode",
            "p",
        )


# Alias for backwards compatibility
SimpleCopyPaste = CopyPasteAugmentation


__all__ = ["CopyPasteAugmentation", "SimpleCopyPaste"]
