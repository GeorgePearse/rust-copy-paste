"""Basic functionality tests for CustomCopyPaste transform."""

import numpy as np
import pytest
import torch
from copy_paste import CustomCopyPaste


@pytest.fixture
def sample_transform():
    """Create a basic CustomCopyPaste transform."""
    return CustomCopyPaste(
        target_image_width=512,
        target_image_height=512,
        mm_class_list=["class1", "class2", "class3"],
        paste_prob=1.0,  # Always paste for testing
        max_paste_objects=2,
        scale_range=(1.0, 1.0),  # No scaling for predictable tests
        rotation_range=(0, 0),  # No rotation for predictable tests
        use_random_background=False,
    )


@pytest.fixture
def sample_results():
    """Create sample input data for transform."""
    height, width = 512, 512
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Create sample annotations (using plain tensors and arrays)
    gt_bboxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)
    gt_labels = np.array([0, 1], dtype=np.int64)

    # Create sample masks
    masks = np.zeros((2, height, width), dtype=np.uint8)
    masks[0, 10:50, 10:50] = 1
    masks[1, 100:150, 100:150] = 1

    return {
        "img": img,
        "gt_bboxes": gt_bboxes,
        "gt_bboxes_labels": gt_labels,
        "gt_masks": masks,
        "gt_ignore_flags": np.array([False, False], dtype=bool),
        "img_shape": (height, width),
    }


def test_transform_initialization(sample_transform):
    """Test that transform initializes correctly."""
    assert sample_transform.target_image_width == 512
    assert sample_transform.target_image_height == 512
    assert len(sample_transform.mm_class_list) == 3
    assert sample_transform.paste_prob == 1.0


def test_transform_preserves_structure(sample_transform, sample_results):
    """Test that transform preserves the structure of results dict."""
    # Note: Without annotation file, transform won't paste anything
    # but should still preserve the structure
    output = sample_transform.transform(sample_results)

    assert output is not None
    assert "img" in output
    assert "gt_bboxes" in output
    assert "gt_bboxes_labels" in output
    assert "gt_masks" in output
    assert "gt_ignore_flags" in output
    assert "img_shape" in output

    # Check types are preserved (plain tensors and arrays)
    assert isinstance(output["gt_bboxes"], torch.Tensor)
    assert isinstance(output["gt_masks"], np.ndarray)
    assert isinstance(output["gt_bboxes_labels"], np.ndarray)
    assert isinstance(output["gt_ignore_flags"], np.ndarray)


def test_random_background_generation():
    """Test random background generation feature."""
    transform = CustomCopyPaste(
        target_image_width=256,
        target_image_height=256,
        mm_class_list=["test"],
        use_random_background=True,
        random_background_prob=1.0,  # Always generate for testing
    )

    height, width = 256, 256
    img = np.zeros((height, width, 3), dtype=np.uint8)

    results = {
        "img": img,
        "gt_bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "gt_bboxes_labels": np.array([], dtype=np.int64),
        "gt_masks": np.zeros((0, height, width), dtype=np.uint8),
        "gt_ignore_flags": np.array([], dtype=bool),
        "img_shape": (height, width),
    }

    output = transform.transform(results)

    # Check that image was replaced with random background
    assert output is not None
    assert not np.all(output["img"] == 0)  # Should not be all zeros
    assert output["img"].shape == (256, 256, 3)

    # When using random background, annotations should be empty
    assert len(output["gt_bboxes_labels"]) == 0
    assert output["gt_masks"].shape[0] == 0


def test_object_counts_configuration():
    """Test object_counts parameter configuration."""
    # Test with exact counts
    transform = CustomCopyPaste(
        target_image_width=512, target_image_height=512, mm_class_list=["class1", "class2"], object_counts={"class1": 2, "class2": 3}
    )
    assert transform.object_counts["class1"] == 2
    assert transform.object_counts["class2"] == 3

    # Test with probabilities
    transform = CustomCopyPaste(
        target_image_width=512, target_image_height=512, mm_class_list=["class1", "class2"], object_counts={"class1": 0.5, "class2": 0.8}
    )
    assert transform.object_counts["class1"] == 0.5
    assert transform.object_counts["class2"] == 0.8

    # Test with 'all' keyword
    transform = CustomCopyPaste(
        target_image_width=512, target_image_height=512, mm_class_list=["class1", "class2", "class3"], object_counts={"all": 2, "class1": 5}
    )
    assert transform.object_counts["all"] == 2
    assert transform.object_counts["class1"] == 5


def test_invalid_object_counts_raises_error():
    """Test that invalid object_counts values raise appropriate errors."""
    with pytest.raises(ValueError, match="must be >= 0"):
        CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["class1"], object_counts={"class1": -1})

    with pytest.raises(TypeError, match="must be int or float"):
        CustomCopyPaste(target_image_width=512, target_image_height=512, mm_class_list=["class1"], object_counts={"class1": "invalid"})  # type: ignore
