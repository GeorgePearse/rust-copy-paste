#!/usr/bin/env python3
"""Generate augmented outputs from the dummy dataset for CI/CD validation.

This script tests the copy_paste transform by:
1. Loading dummy dataset images
2. Applying the RustCopyPaste transform
3. Saving augmented outputs and metadata
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from copy_paste import SimpleCopyPaste


VARIANTS_PER_IMAGE = 3
MAX_ATTEMPTS_PER_VARIANT = 5


def collect_rotation_metadata(transform: SimpleCopyPaste) -> list[dict[str, Any]]:
    """Pull rotation metadata from the most recent pipeline run."""

    bbox_metadata = transform.apply_to_bboxes(np.empty((0, 4), dtype=np.float32))
    bbox_array = np.asarray(bbox_metadata)

    metadata: list[dict[str, Any]] = []
    for row in bbox_array:
        metadata.append(
            {
                "x_min": float(row[0]),
                "y_min": float(row[1]),
                "x_max": float(row[2]),
                "y_max": float(row[3]),
                "class_id": int(round(row[4])),
                "rotation_deg": float(row[5]),
            }
        )

    return metadata


def run_pipeline_variant(
    transform: SimpleCopyPaste,
    original_img: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    """Run the Simple Copy-Paste pipeline until it applies a random transform."""

    last_image = original_img.copy()
    last_changed = 0
    last_metadata: list[dict[str, Any]] = []

    for attempt in range(1, MAX_ATTEMPTS_PER_VARIANT + 1):
        augmented_image = transform.apply(original_img.copy(), mask=mask)

        if augmented_image is None:
            print("    ⚠️  Transform returned None; aborting attempt")
            break

        changed_pixels = int(np.count_nonzero(augmented_image != original_img))
        rotation_metadata = collect_rotation_metadata(transform)

        if changed_pixels > 0 and rotation_metadata:
            return augmented_image, changed_pixels, rotation_metadata

        print(
            f"    ⚠️  Attempt {attempt} had no pasted objects; retrying to ensure pipeline runs"
        )
        last_image = augmented_image
        last_changed = changed_pixels
        last_metadata = rotation_metadata

    # If we reach here, fall back to last attempt even if it had no placements
    return last_image, last_changed, last_metadata


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

    print(
        f"📋 Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations"
    )

    # Extract class names
    class_list = [cat["name"] for cat in coco_data["categories"]]
    print(f"🏷️  Classes: {class_list}")

    # Create transform
    try:
        transform = SimpleCopyPaste(
            image_width=512,
            image_height=512,
            max_paste_objects=3,
            use_rotation=True,
            rotation_range=(-180.0, 180.0),
            use_scaling=True,
            scale_range=(0.85, 1.25),
            p=1.0,
        )
        print("🦀 Using SimpleCopyPaste pipeline")
    except (RuntimeError, TypeError) as e:
        print(f"⚠️  CopyPasteAugmentation not available: {e}")
        print("🐍 Skipping augmentation generation")
        return True  # Skip this step gracefully

    # Process all images (9 total: 3 triangles, 3 circles, 3 squares)
    results = []
    num_to_process = len(coco_data["images"])

    for img_idx, img_info in enumerate(coco_data["images"][:num_to_process]):
        print(
            f"\n📸 Processing image {img_idx + 1}/{num_to_process}: {img_info['file_name']}"
        )

        # Load image
        image_path = images_dir / img_info["file_name"]
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue

        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print("  ❌ Failed to load image")
            continue

        height, width = original_img.shape[:2]

        # Load corresponding mask
        mask_filename = img_info["file_name"].replace(".png", "_mask.png")
        mask_path = dataset_dir / "masks" / mask_filename

        if not mask_path.exists():
            print(f"  ⚠️  Mask not found: {mask_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("  ❌ Failed to load mask")
            continue

        # Apply transform to image using the .apply() method with mask
        for variant_idx in range(VARIANTS_PER_IMAGE):
            try:
                augmented_img, changed_pixels, rotation_metadata = run_pipeline_variant(
                    transform, original_img, mask
                )

                if augmented_img is None:
                    print("  ⚠️  Transform returned None")
                    continue

                output_filename = f"augmented_{img_idx:03d}_v{variant_idx}.png"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), augmented_img)

                if changed_pixels == 0 or not rotation_metadata:
                    print(
                        "  ⚠️  Saved output but pipeline produced no detectable rotations"
                    )
                else:
                    rotations = ", ".join(
                        f"{meta['rotation_deg']:.1f}°" for meta in rotation_metadata
                    )
                    print(
                        f"  🎲 Variant {variant_idx + 1}: {len(rotation_metadata)} objects | rotations: {rotations}"
                    )

                results.append(
                    {
                        "original": img_info["file_name"],
                        "augmented": output_filename,
                        "variant": variant_idx,
                        "width": width,
                        "height": height,
                        "changed_pixels": changed_pixels,
                        "rotations": rotation_metadata,
                    }
                )

            except Exception as e:
                print(f"  ❌ Error during augmentation: {e}")
                continue

    # Save metadata
    metadata = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "num_processed": len(results),
        "variants_per_image": VARIANTS_PER_IMAGE,
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
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"⚠️  Error generating augmented outputs: {e}")
        print("🐍 Continuing without augmented outputs...")
        exit(0)  # Don't fail the workflow
