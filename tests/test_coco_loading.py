"""Test COCO annotations loading functionality."""

import numpy as np
import pytest
from pathlib import Path

from simple_copy_paste import CopyPasteAugmentation


@pytest.fixture
def coco_annotation_file():
    """Path to test COCO annotations file."""
    return "tests/dummy_dataset/annotations/annotations.json"


@pytest.fixture
def sample_background_image():
    """Create a sample background image for testing."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


def test_coco_loading_initialization(coco_annotation_file):
    """Test that transform can load COCO annotations."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=3,
        annotation_file=coco_annotation_file,
        p=1.0,
    )
    assert transform is not None
    assert transform.annotation_file == coco_annotation_file


def test_coco_loading_with_specific_classes(coco_annotation_file):
    """Test loading only specific classes from COCO."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=2,
        annotation_file=coco_annotation_file,
        class_list=["triangle", "circle"],
        p=1.0,
    )
    assert transform is not None


def test_coco_loading_with_object_counts(coco_annotation_file):
    """Test loading COCO with per-class object counts."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=5,
        annotation_file=coco_annotation_file,
        object_counts={"triangle": 2.0, "circle": 1.0},
        mm_class_list=["background", "triangle", "circle", "square"],
        p=1.0,
    )
    assert transform is not None


def test_coco_augmentation_applies(coco_annotation_file, sample_background_image):
    """Test that augmentation using COCO objects actually works."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=3,
        annotation_file=coco_annotation_file,
        use_rotation=True,
        use_scaling=True,
        p=1.0,
    )

    # Apply transform
    result = transform.apply(sample_background_image)

    assert result.shape == sample_background_image.shape
    assert result.dtype == sample_background_image.dtype

    # Result should be different from background (objects were pasted)
    # Note: This could fail with low probability if objects happen to match background
    assert not np.array_equal(result, sample_background_image)


def test_coco_with_bboxes(coco_annotation_file, sample_background_image):
    """Test that bounding boxes are generated from COCO objects."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=2,
        annotation_file=coco_annotation_file,
        use_rotation=False,
        use_scaling=False,
        p=1.0,
    )

    # Apply transform
    _ = transform.apply(sample_background_image)

    # Get bounding boxes
    bboxes = transform.apply_to_bboxes(np.empty((0, 5), dtype=np.float32))

    # Should have generated bboxes from pasted objects
    assert bboxes.shape[1] == 6  # [x_min, y_min, x_max, y_max, class_id, rotation]
    assert bboxes.shape[0] > 0  # At least one object should be pasted

    # Bboxes should be normalized
    assert np.all(bboxes[:, :4] >= 0.0)
    assert np.all(bboxes[:, :4] <= 1.0)


def test_coco_loading_invalid_file():
    """Test that invalid annotation file raises error."""
    with pytest.raises(Exception) as exc_info:
        transform = CopyPasteAugmentation(
            image_width=512,
            image_height=512,
            max_paste_objects=1,
            annotation_file="nonexistent_file.json",
            p=1.0,
        )

    # Should raise an error about the file not existing
    assert "Failed to load COCO annotations" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


def test_coco_no_annotation_file_uses_mask():
    """Test that without annotation_file, transform uses mask extraction (original behavior)."""
    transform = CopyPasteAugmentation(
        image_width=64,
        image_height=64,
        max_paste_objects=1,
        p=1.0,
    )

    # Create image and mask with an object
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:36, 20:36] = 255

    # Should extract object from mask
    result = transform.apply(image, mask=mask)
    assert result.shape == image.shape


def test_coco_end_to_end_pipeline(coco_annotation_file):
    """Test complete pipeline with COCO loading."""
    import albumentations as A

    transform = A.Compose([
        CopyPasteAugmentation(
            image_width=512,
            image_height=512,
            max_paste_objects=3,
            annotation_file=coco_annotation_file,
            use_rotation=True,
            use_scaling=True,
            rotation_range=(-45.0, 45.0),
            scale_range=(0.8, 1.2),
            p=1.0,
        )
    ], bbox_params=A.BboxParams(format="albumentations"))

    # Create background image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    mask = np.zeros((512, 512), dtype=np.uint8)

    # Apply transform
    result = transform(image=image, mask=mask, bboxes=[])

    assert result["image"].shape == image.shape
    assert len(result["bboxes"]) > 0  # Should have pasted objects with bboxes

    # Check bbox format
    for bbox in result["bboxes"]:
        assert len(bbox) == 6  # [x_min, y_min, x_max, y_max, class_id, rotation]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
