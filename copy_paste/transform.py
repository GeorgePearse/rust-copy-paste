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
        object_counts: Dict mapping class_id to exact count of objects to paste per class
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
        object_counts: dict[int, int] | None = None,
        p: float = 1.0,
    ):
        """Initialize the CopyPasteAugmentation transform.

        Args:
            object_counts: Optional dict mapping class_id to exact number of objects to paste.
                          If None, no per-class count constraint is applied.
        """
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
        self.object_counts = object_counts or {}

        # Initialize Rust transform
        self.rust_transform = CopyPasteTransform(
            image_width=image_width,
            image_height=image_height,
            max_paste_objects=max_paste_objects,
            use_rotation=use_rotation,
            use_scaling=use_scaling,
            use_random_background=use_random_background,
            blend_mode=blend_mode,
            object_counts=self.object_counts if self.object_counts else None,
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

        # Call Rust implementation
        # Note: The Rust apply() method currently returns data unchanged
        # Full implementation would perform the copy-paste operations
        try:
            augmented_image, _ = self.rust_transform.apply(
                img,
                np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
                np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
            )
            return augmented_image
        except Exception as e:
            logger.warning(f"Rust augmentation failed: {e}, returning original image")
            return img

    def apply_to_masks(
        self,
        masks: list[np.ndarray],
        **params: Any,
    ) -> list[np.ndarray]:
        """Apply transformations to masks.

        Args:
            masks: List of binary masks (each H x W)

        Returns:
            Transformed masks
        """
        # Masks are typically modified during copy-paste to reflect new object positions
        # For now, return unchanged as the Rust implementation needs full algorithm
        # In production: convert masks to Rust format, apply transformations, convert back
        if not masks:
            return masks

        result = []
        for mask in masks:
            if mask.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            result.append(mask)
        return result

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply transformations to bounding boxes.

        Args:
            bboxes: Bounding boxes in normalized format [x_min, y_min, x_max, y_max, class_label]
                   Values should be in range [0, 1]

        Returns:
            Transformed bounding boxes in same format
        """
        if len(bboxes) == 0:
            return bboxes

        # Validate input format
        if bboxes.shape[1] < 4:
            raise ValueError(
                f"Bboxes must have at least 4 columns [x_min, y_min, x_max, y_max], got {bboxes.shape}"
            )

        # For copy-paste: bboxes may be added/removed/moved based on pasted objects
        # The actual transformation happens in the Rust layer
        # Currently returns unchanged bboxes
        try:
            # Convert normalized coords to pixel coords for Rust
            pixel_bboxes = bboxes.copy()
            pixel_bboxes[:, 0] *= self.image_width  # x_min
            pixel_bboxes[:, 1] *= self.image_height  # y_min
            pixel_bboxes[:, 2] *= self.image_width  # x_max
            pixel_bboxes[:, 3] *= self.image_height  # y_max

            # Call Rust transformation
            transformed = self.rust_transform.apply_to_bboxes(pixel_bboxes[:, :4].astype(np.float32))

            # Convert back to normalized coords
            result = bboxes.copy()
            if len(transformed) > 0:
                result[:, 0] = transformed[:, 0] / self.image_width
                result[:, 1] = transformed[:, 1] / self.image_height
                result[:, 2] = transformed[:, 2] / self.image_width
                result[:, 3] = transformed[:, 3] / self.image_height

            return result
        except Exception as e:
            logger.warning(f"Bbox transformation failed: {e}, returning original bboxes")
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
            "object_counts",
            "p",
        )


# Alias for backwards compatibility
SimpleCopyPaste = CopyPasteAugmentation


__all__ = ["CopyPasteAugmentation", "SimpleCopyPaste"]
