"""Comprehensive tests for the CopyPasteAugmentation interface.

These tests focus on the Python-side logic, integration with Albumentations,
state management, and edge cases.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import albumentations as A
import numpy as np
import pytest

# Try to import the class to test
try:
    from simple_copy_paste.transform import CopyPasteAugmentation, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not available")
class TestCopyPasteAugmentationComprehensive:
    """Comprehensive tests for CopyPasteAugmentation."""

    @pytest.fixture
    def basic_transform(self):
        """Standard transform for testing."""
        return CopyPasteAugmentation(
            image_width=100,
            image_height=100,
            max_paste_objects=1,
            p=1.0,
        )

    @pytest.fixture
    def mocked_backend_transform(self):
        """Transform with mocked Rust backend."""
        with patch("simple_copy_paste.transform.CopyPasteTransform") as mock_cls:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            
            transform = CopyPasteAugmentation(
                image_width=100,
                image_height=100,
                max_paste_objects=1,
                p=1.0,
            )
            # Ensure the mock is attached
            transform.rust_transform = mock_instance
            yield transform

    def test_init_coco_loading_success(self):
        """Test loading COCO categories from annotation file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {"id": 1, "name": "cat"},
                        {"id": 2, "name": "dog"},
                    ]
                },
                f,
            )
            fname = f.name

        try:
            # Mock the rust backend so we don't trigger actual loading/validation in Rust
            with patch("simple_copy_paste.transform.CopyPasteTransform") as mock_rust:
                CopyPasteAugmentation(
                    object_counts={"cat": 1, "dog": 2},
                    annotation_file=fname,
                    p=1.0,
                )
                args, kwargs = mock_rust.call_args
                # Check object_counts in kwargs
                # 1 and 2 are float because they are counts
                expected_counts = {1: 1.0, 2: 2.0}
                assert kwargs["object_counts"] == expected_counts

        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_init_coco_loading_fallback(self):
        """Test fallback to mm_class_list when annotation file is missing/invalid."""
        mm_classes = ["bg", "cat", "dog"]  # indices: 0, 1, 2
        
        with patch("simple_copy_paste.transform.CopyPasteTransform") as mock_rust:
            CopyPasteAugmentation(
                object_counts={"cat": 5, "dog": 10},
                mm_class_list=mm_classes,
                p=1.0,
            )
            args, kwargs = mock_rust.call_args
            # Should map using list indices
            expected_counts = {1: 5.0, 2: 10.0}
            assert kwargs["object_counts"] == expected_counts

    def test_init_numeric_keys(self):
        """Test initialization with numeric keys in object_counts."""
        with patch("simple_copy_paste.transform.CopyPasteTransform") as mock_rust:
            CopyPasteAugmentation(
                object_counts={1: 5, "2": 10, "class3": 15},
                p=1.0,
            )
            args, kwargs = mock_rust.call_args
            expected_counts = {1: 5.0, 2: 10.0, 3: 15.0}
            assert kwargs["object_counts"] == expected_counts

    def test_state_management_cleanup(self, mocked_backend_transform):
        """Test that state is cleaned up even after exceptions."""
        
        # Mock rust_transform to raise exception
        mocked_backend_transform.rust_transform.apply.side_effect = RuntimeError("Rust Boom")
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Call via __call__ which sets state
        with pytest.raises(RuntimeError, match="Rust Boom"):
            mocked_backend_transform(image=image, mask=mask)
            
        # Verify state is cleared
        assert mocked_backend_transform._cached_source_mask is None
        assert mocked_backend_transform._last_mask_output is None

    def test_apply_mask_caching(self, mocked_backend_transform):
        """Test that mask is correctly cached and passed to apply."""
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Mock result
        mocked_backend_transform.rust_transform.apply.return_value = (image, mask)
        
        # 1. Call via __call__ (normal Albumentations usage)
        mocked_backend_transform(image=image, mask=mask)
        
        # Verify rust apply was called with the mask
        call_args = mocked_backend_transform.rust_transform.apply.call_args
        passed_mask = call_args[0][1] # second arg
        
        # Should match the input mask (after ensure_uint8 processing)
        assert np.array_equal(passed_mask[:, :, 0], mask)
        
        # 2. Call apply() directly without cached mask (should fallback to kwargs if provided)
        mocked_backend_transform.rust_transform.apply.reset_mock()
        mocked_backend_transform.apply(image, mask=mask)
        
        call_args = mocked_backend_transform.rust_transform.apply.call_args
        passed_mask = call_args[0][1]
        assert np.array_equal(passed_mask[:, :, 0], mask)
        
        # 3. Call apply() without any mask
        mocked_backend_transform.rust_transform.apply.reset_mock()
        mocked_backend_transform.apply(image)
        
        call_args = mocked_backend_transform.rust_transform.apply.call_args
        passed_mask = call_args[0][1]
        # Should be all zeros
        assert np.all(passed_mask == 0)

    def test_mask_resizing(self, basic_transform):
        """Test internal _resize_mask_if_needed method."""
        # 10x10 mask
        mask = np.ones((10, 10), dtype=np.uint8)
        
        # Resize to 20x20
        resized = basic_transform._resize_mask_if_needed(mask, (20, 20))
        assert resized.shape == (20, 20)
        assert np.all(resized == 1)
        
        # Resize to 5x5
        resized_small = basic_transform._resize_mask_if_needed(mask, (5, 5))
        assert resized_small.shape == (5, 5)

    def test_albumentations_compose_integration(self):
        """Test full integration in an Albumentations Compose pipeline."""
        pipeline = A.Compose([
            CopyPasteAugmentation(
                image_width=64, 
                image_height=64,
                p=1.0
            )
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]))
        
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        bboxes = [[10, 10, 20, 20]] # COCO format: x, y, w, h
        category_ids = [1]
        
        # Should not crash
        result = pipeline(image=image, mask=mask, bboxes=bboxes, category_ids=category_ids)
        
        assert "image" in result
        assert "mask" in result
        assert "bboxes" in result
        assert len(result["bboxes"]) >= 0

    def test_invalid_inputs(self, basic_transform):
        """Test handling of invalid inputs."""
        # Non-image input
        with pytest.raises(Exception): # numpy/cv2 might raise generic errors
             basic_transform.apply(np.array(["not", "an", "image"]))

        # Wrong channels
        image_wrong_channels = np.zeros((100, 100), dtype=np.uint8) # 1 channel
        with pytest.raises(ValueError, match="Expected BGR"):
             basic_transform.apply(image_wrong_channels)
             
    def test_apply_to_bboxes_metadata(self, mocked_backend_transform):
        """Test that apply_to_bboxes handles the metadata flow correctly."""
        # Mock rust response
        # 1 bbox: x, y, x, y, class, angle
        mock_bboxes = [10.0, 10.0, 20.0, 20.0, 1.0, 45.0]
        mocked_backend_transform.rust_transform.apply_to_bboxes.return_value = mock_bboxes
        
        # Input doesn't matter much as rust implementation generates new bboxes
        result = mocked_backend_transform.apply_to_bboxes(np.zeros((0, 5)))
        
        # Check normalization (divided by 100)
        assert result[0, 0] == 10.0 / 100
        assert result[0, 5] == 45.0 # Angle not normalized

    def test_reuse_instance(self, basic_transform):
        """Test reusing the same instance multiple times."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        for _ in range(5):
            res = basic_transform(image=image, mask=mask)
            assert res["image"].shape == (100, 100, 3)
            # Ensure no state leakage
            assert basic_transform._cached_source_mask is None
            assert basic_transform._last_mask_output is None

    def test_object_counts_validation(self):
        """Test validation of object_counts parameter."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CopyPasteAugmentation(object_counts={"cat": -1})
            
    def test_ranges_validation(self):
        """Test validation of range parameters."""
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            CopyPasteAugmentation(rotation_range=(1.0,))
            
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            CopyPasteAugmentation(scale_range=(1.0, 2.0, 3.0))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
