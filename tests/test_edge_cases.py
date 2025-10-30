"""Edge case tests for CustomCopyPaste transform."""

import numpy as np
import torch
from copy_paste import CustomCopyPaste
from visdet.structures.bbox import HorizontalBoxes
from visdet.structures.mask import BitmapMasks


def test_empty_annotations():
    """Test transform with empty annotations."""
    transform = CustomCopyPaste(target_image_width=256, target_image_height=256, mm_class_list=["test"], paste_prob=1.0)

    height, width = 256, 256
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    results = {
        "img": img,
        "gt_bboxes": HorizontalBoxes(torch.zeros((0, 4), dtype=torch.float32)),
        "gt_bboxes_labels": np.array([], dtype=np.int64),
        "gt_masks": BitmapMasks(np.zeros((0, height, width), dtype=np.uint8), height, width),
        "gt_ignore_flags": np.array([], dtype=bool),
        "img_shape": (height, width),
    }

    output = transform.transform(results)

    assert output is not None
    assert output["img"].shape == img.shape
    assert len(output["gt_bboxes_labels"]) == 0
    assert output["gt_masks"].masks.shape[0] == 0


def test_zero_paste_probability():
    """Test transform with paste_prob=0 (should not paste anything)."""
    transform = CustomCopyPaste(
        target_image_width=512,
        target_image_height=512,
        mm_class_list=["test"],
        paste_prob=0.0,  # Never paste
        use_random_background=False,
    )

    height, width = 512, 512
    original_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    results = {
        "img": original_img.copy(),
        "gt_bboxes": HorizontalBoxes(torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)),
        "gt_bboxes_labels": np.array([0], dtype=np.int64),
        "gt_masks": BitmapMasks(np.ones((1, height, width), dtype=np.uint8), height, width),
        "gt_ignore_flags": np.array([False], dtype=bool),
        "img_shape": (height, width),
    }

    output = transform.transform(results)

    assert output is not None
    # With paste_prob=0, image should be unchanged
    assert np.array_equal(output["img"], original_img)
    # Annotations should be unchanged
    assert len(output["gt_bboxes_labels"]) == 1


def test_max_paste_objects_zero():
    """Test transform with max_paste_objects=0."""
    transform = CustomCopyPaste(
        target_image_width=256,
        target_image_height=256,
        mm_class_list=["test"],
        paste_prob=1.0,
        max_paste_objects=0,  # Don't paste any objects
    )

    height, width = 256, 256
    original_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    results = {
        "img": original_img.copy(),
        "gt_bboxes": HorizontalBoxes(torch.zeros((0, 4), dtype=torch.float32)),
        "gt_bboxes_labels": np.array([], dtype=np.int64),
        "gt_masks": BitmapMasks(np.zeros((0, height, width), dtype=np.uint8), height, width),
        "gt_ignore_flags": np.array([], dtype=bool),
        "img_shape": (height, width),
    }

    output = transform.transform(results)

    assert output is not None
    # With max_paste_objects=0, nothing should be pasted
    assert len(output["gt_bboxes_labels"]) == 0


def test_mismatched_dimensions():
    """Test handling of mismatched image dimensions."""
    transform = CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["test"], paste_prob=1.0)

    # Input image with different dimensions
    height, width = 256, 256
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    results = {
        "img": img,
        "gt_bboxes": HorizontalBoxes(torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)),
        "gt_bboxes_labels": np.array([0], dtype=np.int64),
        "gt_masks": BitmapMasks(np.ones((1, height, width), dtype=np.uint8), height, width),
        "gt_ignore_flags": np.array([False], dtype=bool),
        "img_shape": (height, width),
    }

    output = transform.transform(results)

    assert output is not None
    # Should handle dimension mismatch gracefully
    assert output["img_shape"] == (height, width)
    assert output["gt_masks"].height == height
    assert output["gt_masks"].width == width


def test_extreme_scale_ranges():
    """Test transform with extreme scale ranges."""
    # Very small scale
    transform = CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["test"], scale_range=(0.1, 0.1))
    assert transform.scale_range == (0.1, 0.1)

    # Very large scale
    transform = CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["test"], scale_range=(10.0, 10.0))
    assert transform.scale_range == (10.0, 10.0)


def test_extreme_rotation_ranges():
    """Test transform with extreme rotation ranges."""
    # Full rotation
    transform = CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["test"], rotation_range=(0, 360))
    assert transform.rotation_range == (0, 360)

    # Negative rotation
    transform = CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["test"], rotation_range=(-180, 180))
    assert transform.rotation_range == (-180, 180)


def test_blend_modes():
    """Test different blend modes."""
    # Normal blend mode
    transform = CustomCopyPaste(target_image_width=256, target_image_height=256, mm_class_list=["test"], blend_mode="normal")
    assert transform.blend_mode == "normal"
    assert transform.xray_paste is False

    # X-ray blend mode
    transform = CustomCopyPaste(target_image_width=256, target_image_height=256, mm_class_list=["test"], blend_mode="xray")
    assert transform.blend_mode == "xray"
    assert transform.xray_paste is True


def test_class_name_mapping():
    """Test class name mapping functionality."""
    mapping = {"old_class": "new_class", "another": "mapped"}

    transform = CustomCopyPaste(
        target_image_width=256, target_image_height=256, mm_class_list=["new_class", "mapped", "unmapped"], class_name_mapping=mapping
    )

    # Test the mapping method
    assert transform._map_class_name("old_class") == "new_class"
    assert transform._map_class_name("another") == "mapped"
    assert transform._map_class_name("unmapped") == "unmapped"  # No mapping
    assert transform._map_class_name("unknown") == "unknown"  # No mapping
