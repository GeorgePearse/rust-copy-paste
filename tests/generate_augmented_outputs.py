#!/usr/bin/env python3
"""Generate augmented outputs from the dummy dataset for CI/CD validation.

This script tests the copy_paste transform by:
1. Loading dummy dataset images
2. Applying the RustCopyPaste transform
3. Saving augmented outputs and metadata
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from copy_paste import RustCopyPaste


def main():
    """Generate augmented outputs from dummy dataset."""
    # Setup paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dummy_dataset"
    images_dir = dataset_dir / "images"
    annotations_file = dataset_dir / "annotations" / "annotations.json"
    output_dir = script_dir / "augmented_outputs"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        return False

    print(f"📊 Dummy dataset found at {dataset_dir}")

    # Load annotations
    with open(annotations_file) as f:
        coco_data = json.load(f)

    print(f"📋 Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

    # Extract class names
    class_list = [cat["name"] for cat in coco_data["categories"]]
    print(f"🏷️  Classes: {class_list}")

    # Create transform
    try:
        transform = RustCopyPaste(
            target_image_width=512,
            target_image_height=512,
            mm_class_list=class_list,
            annotation_file=str(annotations_file),
            paste_prob=1.0,
            max_paste_objects=2,
            scale_range=(1.0, 1.0),
            rotation_range=(0, 360),
            verbose=True,
        )
        print("🦀 Using RustCopyPaste transform")
    except RuntimeError:
        # Fallback to CustomCopyPaste if Rust not available
        from copy_paste import CustomCopyPaste

        transform = CustomCopyPaste(
            target_image_width=512,
            target_image_height=512,
            mm_class_list=class_list,
            annotation_file=str(annotations_file),
            paste_prob=1.0,
            max_paste_objects=2,
            scale_range=(1.0, 1.0),
            rotation_range=(0, 360),
            verbose=True,
        )
        print("🐍 Using CustomCopyPaste transform (Rust not available)")

    # Process up to 5 images
    results = []
    num_to_process = min(5, len(coco_data["images"]))

    for img_idx, img_info in enumerate(coco_data["images"][:num_to_process]):
        print(f"\n📸 Processing image {img_idx + 1}/{num_to_process}: {img_info['file_name']}")

        # Load image
        image_path = images_dir / img_info["file_name"]
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue

        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print(f"  ❌ Failed to load image")
            continue

        height, width = original_img.shape[:2]

        # Create minimal results dict for transform
        results_dict = {
            "img": original_img.copy(),
            "gt_bboxes": torch.zeros((0, 4), dtype=torch.float32),
            "gt_bboxes_labels": np.array([], dtype=np.int64),
            "gt_masks": np.zeros((0, height, width), dtype=np.uint8),
            "gt_ignore_flags": np.array([], dtype=bool),
            "img_shape": (height, width),
        }

        # Apply transform
        try:
            augmented = transform.transform(results_dict)

            if augmented is None:
                print(f"  ⚠️  Transform returned None")
                continue

            # Save augmented image
            output_filename = f"augmented_{img_idx:03d}.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), augmented["img"])

            num_objects = len(augmented["gt_bboxes_labels"])
            print(f"  ✅ Augmented with {num_objects} objects")

            results.append({
                "original": img_info["file_name"],
                "augmented": output_filename,
                "width": width,
                "height": height,
                "num_objects": int(num_objects),
            })

        except Exception as e:
            print(f"  ❌ Error during augmentation: {e}")
            continue

    # Save metadata
    metadata = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "num_processed": len(results),
        "images": results,
        "classes": class_list,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✨ Generated {len(results)} augmented outputs")
    print(f"📁 Outputs saved to {output_dir}")
    print(f"📄 Metadata saved to {metadata_path}")

    return len(results) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
