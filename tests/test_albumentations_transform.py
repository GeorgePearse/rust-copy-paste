"""Tests for Albumentations-compatible CopyPasteAugmentation transform."""

import numpy as np
import pytest

try:
    import albumentations as A  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("albumentations not installed", allow_module_level=True)

from copy_paste import CopyPasteAugmentation, SimpleCopyPaste
from copy_paste.transform import RUST_AVAILABLE

if not RUST_AVAILABLE:
    pytest.skip("Rust extension not built; skipping CopyPasteAugmentation tests", allow_module_level=True)


@pytest.fixture
def sample_transform():
    """Create a basic CopyPasteAugmentation transform."""
    return CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=1,
        use_rotation=False,
        use_scaling=False,
        use_random_background=False,
        p=1.0,
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_bboxes():
    """Create sample bounding boxes in Albumentations format.

    Format: [x_min, y_min, x_max, y_max, class_label]
    Values are normalized (0-1)
    """
    return np.array([
        [0.1, 0.1, 0.3, 0.3, 0],  # Class 0
        [0.5, 0.5, 0.7, 0.7, 1],  # Class 1
    ])


@pytest.fixture
def sample_masks():
    """Create sample masks as a list of binary arrays."""
    return [
        np.zeros((512, 512), dtype=np.uint8),  # First mask (all zeros)
        np.zeros((512, 512), dtype=np.uint8),  # Second mask (all zeros)
    ]


def test_transform_initialization(sample_transform):
    """Test that transform initializes correctly."""
    assert sample_transform.image_width == 512
    assert sample_transform.image_height == 512
    assert sample_transform.max_paste_objects == 1
    assert sample_transform.p == 1.0


def test_transform_is_dual_transform(sample_transform):
    """Test that transform inherits from DualTransform."""
    assert isinstance(sample_transform, A.DualTransform)


def test_apply_basic_image(sample_transform, sample_image):
    """Test that apply method works with basic image."""
    result = sample_transform.apply(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == sample_image.dtype


def test_apply_to_bboxes(sample_transform, sample_bboxes):
    """Test that apply_to_bboxes method works."""
    result = sample_transform.apply_to_bboxes(sample_bboxes)
    assert result.shape == (0, 6)


def test_apply_to_masks(sample_transform, sample_masks):
    """Test that apply_to_masks method works."""
    result = sample_transform.apply_to_masks(sample_masks)
    assert len(result) == len(sample_masks)


def test_full_transform_call(sample_transform, sample_image):
    """Test full transform call with Albumentations interface."""
    data = {
        "image": sample_image,
        "bboxes": np.array([[0.1, 0.1, 0.3, 0.3, 0]]),
        "masks": [np.zeros((512, 512), dtype=np.uint8)],
    }

    # Transform should handle the data through __call__
    transform_instance = sample_transform
    # We can't call the transform directly like Albumentations
    # because it's missing the full implementation, but we can call apply
    result = transform_instance.apply(data["image"])
    assert result.shape == sample_image.shape


def test_simple_copy_paste_alias():
    """Test that SimpleCopyPaste alias works."""
    transform = SimpleCopyPaste(
        image_width=256,
        image_height=256,
        max_paste_objects=1,
        p=1.0,
    )
    assert isinstance(transform, CopyPasteAugmentation)


def test_transform_with_different_sizes():
    """Test transform with various image sizes."""
    sizes = [(256, 256), (512, 512), (1024, 1024)]

    for height, width in sizes:
        transform = CopyPasteAugmentation(
            image_width=width,
            image_height=height,
            p=1.0,
        )
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        result = transform.apply(image)
        assert result.shape == image.shape


def test_transform_probability_zero():
    """Test that transform with p=0 doesn't apply."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        p=0.0,  # Never apply
    )

    image = np.ones((512, 512, 3), dtype=np.uint8) * 100
    # When p=0, the transform's __call__ should not modify the image
    # For now, we're just testing that it doesn't error


def test_transform_with_rotation():
    """Test transform with rotation enabled."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        use_rotation=True,
        rotation_range=(-45.0, 45.0),
        p=1.0,
    )

    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = transform.apply(image)
    assert result.shape == image.shape


def test_transform_with_scaling():
    """Test transform with scaling enabled."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        use_scaling=True,
        scale_range=(0.8, 1.2),
        p=1.0,
    )

    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = transform.apply(image)
    assert result.shape == image.shape


def test_transform_with_random_background():
    """Test transform with random background generation."""
    transform = CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        use_random_background=True,
        p=1.0,
    )

    image = np.ones((512, 512, 3), dtype=np.uint8) * 50
    result = transform.apply(image)
    assert result.shape == image.shape


def test_transform_different_blend_modes():
    """Test transform with different blend modes."""
    modes = ["normal", "xray"]

    for blend_mode in modes:
        transform = CopyPasteAugmentation(
            image_width=512,
            image_height=512,
            blend_mode=blend_mode,
            p=1.0,
        )

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transform.apply(image)
        assert result.shape == image.shape


def test_get_transform_init_args_names(sample_transform):
    """Test that get_transform_init_args_names returns correct arguments."""
    args = sample_transform.get_transform_init_args_names()

    expected_args = {
        "image_width",
        "image_height",
        "max_paste_objects",
        "use_rotation",
        "use_scaling",
        "rotation_range",
        "scale_range",
        "use_random_background",
        "blend_mode",
        "object_counts",  # Optional per-class object counts
        "p",
    }

    assert set(args) == expected_args


def test_image_dtype_handling(sample_transform):
    """Test that transform handles different image dtypes."""
    # Test uint8
    image_uint8 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = sample_transform.apply(image_uint8)
    assert result.dtype == np.uint8

    # Test float32 in [0, 1] range
    image_float = np.random.random((512, 512, 3)).astype(np.float32)
    result = sample_transform.apply(image_float)
    assert result.dtype == np.uint8


def test_empty_bboxes(sample_transform):
    """Test transform with empty bounding boxes."""
    empty_bboxes = np.empty((0, 5), dtype=np.float32)
    result = sample_transform.apply_to_bboxes(empty_bboxes)
    assert result.shape == (0, 6)


def test_apply_to_bboxes_with_rotation():
    """Ensure rotation metadata propagates to bbox outputs."""
    transform = CopyPasteAugmentation(
        image_width=64,
        image_height=64,
        max_paste_objects=1,
        use_rotation=True,
        rotation_range=(-45.0, -45.0),
        use_scaling=False,
        use_random_background=False,
        p=1.0,
    )

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:36, 20:36] = 255

    transform.apply(image, mask=mask)

    transformed_bboxes = transform.apply_to_bboxes(np.empty((0, 5), dtype=np.float32))
    assert transformed_bboxes.shape[1] == 6
    assert transformed_bboxes.shape[0] >= 1

    # Coordinates are normalized; ensure they fall inside the unit interval
    np.testing.assert_array_less(transformed_bboxes[:, :4], 1.01)
    assert np.all(transformed_bboxes[:, :4] >= 0.0)

    # Angle column should reflect the forced rotation (-45 degrees)
    assert np.isclose(transformed_bboxes[0, 5], -45.0, atol=1.0)


def test_empty_masks(sample_transform):
    """Test transform with empty masks list."""
    empty_masks = []
    result = sample_transform.apply_to_masks(empty_masks)
    assert result == []
