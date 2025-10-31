#!/usr/bin/env python3
"""
Visualization script for copy-paste augmentation pipeline.

Generates visual outputs showing:
1. Input image with original objects
2. Extracted objects
3. Selected objects (per-class)
4. Final augmented image with bounding boxes and labels
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from copy_paste import CopyPasteAugmentation
except ImportError:
    print("Error: copy_paste module not found. Build with: maturin develop")
    exit(1)


# COCO class names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'cat', 15: 'dog', 16: 'horse', 17: 'sheep', 18: 'cow', 19: 'elephant',
}

# Colors for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (0, 0, 255),          # Red
    'bicycle': (255, 0, 0),      # Blue
    'truck': (0, 165, 255),      # Orange
    'bus': (255, 165, 0),        # Cyan
    'motorcycle': (255, 0, 255), # Magenta
    'dog': (0, 255, 255),        # Yellow
    'cat': (255, 255, 0),        # Cyan
}


def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    class_names: List[str],
    confidence: List[float] = None,
) -> np.ndarray:
    """Draw bounding boxes with class labels on image.

    Args:
        image: Input image (H, W, 3) in BGR
        bboxes: List of (x_min, y_min, x_max, y_max) in pixel coordinates
        class_names: List of class names corresponding to each bbox
        confidence: Optional confidence scores for each bbox

    Returns:
        Image with drawn bounding boxes and labels
    """
    output = image.copy()

    for i, (bbox, class_name) in enumerate(zip(bboxes, class_names)):
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Get color for this class
        color = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Draw rectangle
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label text
        if confidence and i < len(confidence):
            label = f"{class_name} {confidence[i]:.2f}"
        else:
            label = class_name

        # Get text size for background
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(label, font_face, font_scale, thickness)[0]

        # Draw label background
        label_bg_color = tuple(int(c * 0.7) for c in color)  # Darker version of color
        cv2.rectangle(
            output,
            (x_min, y_min - text_size[1] - 4),
            (x_min + text_size[0] + 4, y_min),
            label_bg_color,
            -1
        )

        # Draw label text
        cv2.putText(
            output,
            label,
            (x_min + 2, y_min - 2),
            font_face,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )

    return output


def extract_bboxes_from_mask(mask: np.ndarray) -> Tuple[List[Tuple], List[int]]:
    """Extract bounding boxes from segmentation mask.

    Args:
        mask: Segmentation mask (H, W) or (H, W, C) with class IDs

    Returns:
        Tuple of (bboxes, class_ids) where bboxes are (x_min, y_min, x_max, y_max)
    """
    bboxes = []
    class_ids = []

    # Convert to single-channel if multi-channel
    if mask.ndim == 3:
        # If 3-channel, take first channel or convert from BGR
        if mask.shape[2] == 3:
            # For BGR masks, convert to grayscale or just use first channel
            mask = mask[:, :, 0].astype(np.uint8)
        else:
            mask = mask[:, :, 0].astype(np.uint8)
    else:
        # Ensure single-channel is uint8
        mask = mask.astype(np.uint8)

    # Find unique class IDs in mask (excluding background 0)
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]

    for class_id in unique_classes:
        # Find pixels belonging to this class
        class_mask = (mask == class_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:  # Only valid bboxes
                bboxes.append((float(x), float(y), float(x + w), float(y + h)))
                class_ids.append(int(class_id))

    return bboxes, class_ids


def create_example_data(
    height: int = 512,
    width: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic example image and mask for demonstration.

    Args:
        height: Image height
        width: Image width

    Returns:
        Tuple of (image, mask) as uint8 arrays
    """
    # Create background
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    mask = np.zeros((height, width), dtype=np.uint8)

    # Add some random objects (circles and rectangles)
    class_id = 1
    for _ in range(3):
        # Random circle (person)
        center_x = np.random.randint(50, width - 50)
        center_y = np.random.randint(50, height - 50)
        radius = np.random.randint(30, 80)

        cv2.circle(image, (center_x, center_y), radius, (100, 150, 200), -1)
        cv2.circle(mask, (center_x, center_y), radius, class_id, -1)
        class_id += 1

    for _ in range(2):
        # Random rectangle (car)
        x = np.random.randint(30, width - 130)
        y = np.random.randint(30, height - 80)
        w = np.random.randint(50, 100)
        h = np.random.randint(30, 60)

        cv2.rectangle(image, (x, y), (x + w, y + h), (50, 100, 200), -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), class_id, -1)
        class_id += 1

    return image, mask


def visualize_pipeline(
    input_image: np.ndarray,
    input_mask: np.ndarray,
    object_counts: Dict[str, int],
    output_dir: Path,
    config: Dict = None,
) -> None:
    """Run visualization of the complete copy-paste pipeline.

    Args:
        input_image: Input image (H, W, 3) in BGR
        input_mask: Input mask (H, W) with class IDs
        object_counts: Dict mapping class names to counts
        output_dir: Directory to save visualizations
        config: Optional config dict for CopyPasteAugmentation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing copy-paste pipeline...")
    print(f"Input image shape: {input_image.shape}")
    print(f"Object counts: {object_counts}")

    # Create default config if not provided
    if config is None:
        config = {
            'image_width': input_image.shape[1],
            'image_height': input_image.shape[0],
            'use_rotation': True,
            'use_scaling': True,
            'rotation_range': (-15, 15),
            'scale_range': (0.9, 1.1),
            'blend_mode': 'normal',
        }

        # Try to add object_counts if supported by the current build
        try:
            import inspect
            sig = inspect.signature(CopyPasteAugmentation.__init__)
            if 'object_counts' in sig.parameters:
                config['object_counts'] = object_counts
                print("   ✓ Using per-class object counts")
            else:
                print("   ⚠ object_counts not available in current build (rebuild with maturin develop)")
        except Exception as e:
            print(f"   ⚠ Could not check for object_counts parameter: {e}")

    # Initialize transform
    try:
        transform = CopyPasteAugmentation(**config)
    except TypeError as e:
        if 'object_counts' in str(e):
            # Fallback: remove object_counts if not supported
            config.pop('object_counts', None)
            print("   ⚠ Removing unsupported object_counts parameter, rebuilding with:")
            print("      maturin develop")
            transform = CopyPasteAugmentation(**config)
        else:
            raise

    # Step 1: Save input image with original bounding boxes
    print("\n[1] Extracting objects from input mask...")
    input_bboxes, input_class_ids = extract_bboxes_from_mask(input_mask)
    input_class_names = [
        COCO_CLASSES.get(class_id, f'class_{class_id}')
        for class_id in input_class_ids
    ]

    input_viz = draw_bboxes(
        input_image.copy(),
        input_bboxes,
        input_class_names
    )
    cv2.imwrite(str(output_dir / '01_input_with_bboxes.jpg'), input_viz)
    print(f"   ✓ Found {len(input_bboxes)} objects")

    # Step 2: Apply augmentation
    print("\n[2] Applying copy-paste augmentation...")

    # Ensure mask is 2D for the transform
    mask_2d = input_mask if input_mask.ndim == 2 else input_mask[:, :, 0]

    try:
        augmented = transform(image=input_image, mask=mask_2d)
        augmented_image = augmented['image']
        print("   ✓ Augmentation complete")
    except Exception as e:
        print(f"   ⚠ Augmentation failed: {e}")
        print("   Using original image as fallback")
        augmented_image = input_image.copy()
        augmented = {'image': augmented_image}

    # Step 3: Extract and visualize output bounding boxes
    print("\n[3] Extracting pasted objects from output...")
    output_bboxes, output_class_ids = extract_bboxes_from_mask(augmented['image'] if 'image' in augmented else input_mask)

    # Note: Since we don't have class ID info from Rust directly, we'll use a placeholder
    # In a full implementation, the Rust layer would return class information
    output_class_names = [
        COCO_CLASSES.get(class_id, f'class_{class_id}')
        for class_id in output_class_ids
    ]

    output_viz = draw_bboxes(
        augmented_image.copy(),
        output_bboxes,
        output_class_names
    )
    cv2.imwrite(str(output_dir / '05_augmented_with_bboxes.jpg'), output_viz)
    print(f"   ✓ Found {len(output_bboxes)} objects in output")

    # Step 4: Save raw outputs
    print("\n[4] Saving outputs...")
    cv2.imwrite(str(output_dir / '01_input_image.jpg'), input_image)
    cv2.imwrite(str(output_dir / '05_augmented_image.jpg'), augmented_image)

    # Step 5: Create comparison
    comparison = np.hstack([input_viz, output_viz])
    cv2.imwrite(str(output_dir / '06_comparison.jpg'), comparison)
    print(f"   ✓ Comparison saved")

    print(f"\n✅ Visualization complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Files generated:")
    print(f"     - 01_input_image.jpg (original)")
    print(f"     - 01_input_with_bboxes.jpg (with labels)")
    print(f"     - 05_augmented_image.jpg (augmented)")
    print(f"     - 05_augmented_with_bboxes.jpg (with labels)")
    print(f"     - 06_comparison.jpg (side-by-side)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize copy-paste augmentation pipeline'
    )
    parser.add_argument(
        '--input-image',
        type=str,
        help='Path to input image (optional, will generate synthetic data if not provided)'
    )
    parser.add_argument(
        '--input-mask',
        type=str,
        help='Path to input mask (optional, will generate synthetic data if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples/visual_outputs/pipeline_stages',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--class-counts',
        type=str,
        default='person:2,car:1',
        help='Object counts as class_name:count,class_name:count'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=512,
        help='Image width'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=512,
        help='Image height'
    )

    args = parser.parse_args()

    # Parse object counts
    object_counts = {}
    for item in args.class_counts.split(','):
        class_name, count = item.split(':')
        object_counts[class_name.strip()] = int(count)

    # Load or create input data
    if args.input_image and args.input_mask:
        print(f"Loading input image from {args.input_image}")
        input_image = cv2.imread(args.input_image)
        input_mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)

        if input_image is None or input_mask is None:
            raise ValueError("Failed to load input images")
    else:
        print("Creating synthetic example data...")
        input_image, input_mask = create_example_data(
            height=args.image_height,
            width=args.image_width
        )

    # Run visualization
    visualize_pipeline(
        input_image,
        input_mask,
        object_counts,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
