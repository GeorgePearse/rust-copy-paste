#!/usr/bin/env python3
"""Generate labeled visualizations with segmentation masks and class names.

This script creates side-by-side comparisons showing:
1. Original images with mask overlay and class label
2. Augmented images with mask overlay and class labels
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from copy_paste import CopyPasteAugmentation


def create_mask_overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Create an overlay of the mask on the image.

    Args:
        image: RGB image (H, W, 3)
        mask: Grayscale mask (H, W) with class IDs
        alpha: Transparency of overlay (0-1)

    Returns:
        Image with mask overlay
    """
    overlay = image.copy()

    # Create colored mask (use distinct colors per class)
    color_map = {
        1: (255, 0, 0),  # Triangle - Red
        2: (0, 255, 0),  # Circle - Green
        3: (0, 0, 255),  # Square - Blue
    }

    # Apply colors based on mask values
    for class_id, color in color_map.items():
        mask_region = mask == class_id
        overlay[mask_region] = color

    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def add_label(image: np.ndarray, class_name: str, position: str = "top") -> np.ndarray:
    """Add class name label to image.

    Args:
        image: Input image
        class_name: Name of the class to display
        position: 'top' or 'bottom'

    Returns:
        Image with label added
    """
    img_copy = image.copy()
    height, width = img_copy.shape[:2]

    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        class_name, font, font_scale, thickness
    )

    # Calculate position
    padding = 10
    if position == "top":
        text_x = padding
        text_y = padding + text_height
        rect_y1 = 0
        rect_y2 = text_height + 2 * padding
    else:  # bottom
        text_x = padding
        text_y = height - padding
        rect_y1 = height - text_height - 2 * padding
        rect_y2 = height

    # Draw background rectangle
    cv2.rectangle(
        img_copy,
        (0, rect_y1),
        (text_width + 2 * padding, rect_y2),
        bg_color,
        -1,
    )

    # Draw text
    cv2.putText(
        img_copy,
        class_name,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return img_copy


def extract_class_name_from_filename(filename: str) -> str:
    """Extract class name from image filename.

    Args:
        filename: e.g., 'triangle_000.png'

    Returns:
        Class name, e.g., 'triangle'
    """
    return filename.split("_")[0]


def get_mask_from_image(
    coco_data: dict[str, Any], img_info: dict[str, Any], masks_dir: Path
) -> tuple[np.ndarray | None, int]:
    """Load mask for an image and extract class ID.

    Args:
        coco_data: COCO annotation data
        img_info: Image information from COCO
        masks_dir: Directory containing masks

    Returns:
        Tuple of (mask array, class_id)
    """
    # Load mask
    mask_filename = img_info["file_name"].replace(".png", "_mask.png")
    mask_path = masks_dir / mask_filename

    if not mask_path.exists():
        return None, 0

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, 0

    # Get class ID from annotation
    img_id = img_info["id"]
    annotations = [a for a in coco_data["annotations"] if a["image_id"] == img_id]

    if not annotations:
        return mask, 0

    class_id = annotations[0]["category_id"]
    return mask, class_id


def main() -> bool:
    """Generate labeled visualization outputs."""
    # Setup paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dummy_dataset"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    annotations_file = dataset_dir / "annotations" / "annotations.json"
    output_dir = script_dir / "augmented_outputs"
    labeled_dir = output_dir / "labeled"

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    labeled_dir.mkdir(exist_ok=True)

    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        return False

    print(f"📊 Generating labeled visualizations from {dataset_dir}")

    # Load annotations
    with open(annotations_file) as f:
        coco_data = json.load(f)

    # Create class name mapping
    class_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    print(f"🏷️  Classes: {list(class_names.values())}")

    # Create transform
    try:
        transform = CopyPasteAugmentation(
            image_width=512,
            image_height=512,
            max_paste_objects=2,
            scale_range=(1.0, 1.0),
            rotation_range=(0, 360),
            p=1.0,
        )
        print("🦀 Using CopyPasteAugmentation transform")
    except (RuntimeError, TypeError) as e:
        print(f"⚠️  CopyPasteAugmentation not available: {e}")
        return False

    # Process one example per class
    examples_per_class = {}
    for img_info in coco_data["images"]:
        class_name = extract_class_name_from_filename(img_info["file_name"])
        if class_name not in examples_per_class:
            examples_per_class[class_name] = img_info["file_name"]

    print(f"\n📸 Processing {len(examples_per_class)} examples (one per class)")

    for class_name, filename in examples_per_class.items():
        print(f"\n  Processing {class_name}: {filename}")

        # Find image info
        img_info = next(
            (img for img in coco_data["images"] if img["file_name"] == filename), None
        )
        if not img_info:
            print("    ⚠️  Image info not found")
            continue

        # Load original image
        image_path = images_dir / filename
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print("    ❌ Failed to load image")
            continue

        # Load mask
        mask, class_id = get_mask_from_image(coco_data, img_info, masks_dir)
        if mask is None:
            print("    ⚠️  Mask not found")
            continue

        # Create original with label and mask overlay
        original_with_mask = create_mask_overlay(original_img, mask)
        original_labeled = add_label(original_with_mask, class_name.upper(), "top")

        # Apply augmentation
        try:
            augmented_img = transform.apply(original_img.copy(), mask=mask)
            if augmented_img is None:
                print("    ⚠️  Transform returned None")
                continue

            # For augmented, we don't have the output mask, so just add label
            augmented_labeled = add_label(
                augmented_img, f"{class_name.upper()} (Augmented)", "top"
            )

            # Save individual labeled images
            cv2.imwrite(
                str(labeled_dir / f"{class_name}_original_labeled.png"),
                original_labeled,
            )
            cv2.imwrite(
                str(labeled_dir / f"{class_name}_augmented_labeled.png"),
                augmented_labeled,
            )

            # Create side-by-side comparison
            comparison = np.hstack([original_labeled, augmented_labeled])
            cv2.imwrite(
                str(labeled_dir / f"{class_name}_comparison.png"),
                comparison,
            )

            print("    ✅ Saved labeled outputs")

        except Exception as e:
            print(f"    ❌ Error during augmentation: {e}")
            continue

    print(f"\n✨ Generated labeled visualizations in {labeled_dir}")
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"⚠️  Error generating labeled outputs: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
