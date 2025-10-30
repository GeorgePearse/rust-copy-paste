"""Integration tests for RustCopyPaste on the real vmi_mrf dataset.

This module tests the full copy-paste augmentation pipeline on real CMR
(Center for Material Recycling) data with the vmi_mrf configuration.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from copy_paste.rust_copy_paste import RustCopyPaste
from visdet.structures.bbox import HorizontalBoxes
from visdet.structures.mask import BitmapMasks


# Test data paths as per CLAUDE.md guidelines
TRAIN_ANNOTATIONS = Path("/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json")
VAL_ANNOTATIONS = Path("/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json")
IMAGES_DIR = Path("/home/georgepearse/data/images")

# VMI MRF classes as per vmi_mrf.yaml
VMI_MRF_CLASSES = [
    "Phone",
    "Tablet",
    "Vape",
    "Propane Tank",
    "N2O Tank",
    "Co2 Tank",
    "Lithium Ion Battery",
    "Helium Tank",
    "Laptop",
    "Laptop Battery",
    "Electric Motor",
    "Lead Acid Battery",
    "Large Blade",
    "Other Battery",
    "E-Waste With Battery",
    "E-Waste",
]

# VMI MRF object counts from vmi_mrf.yaml config
VMI_MRF_OBJECT_COUNTS = {
    "Phone": 5,
    "Tablet": 1,
    "Vape": 10,
    "Propane Tank": 2,
    "N2O Tank": 2,
    "Co2 Tank": 2,
    "Lithium Ion Battery": 10,
    "Helium Tank": 2,
    "Laptop": 2,
    "Laptop Battery": 2,
    "Electric Motor": 2,
    "Lead Acid Battery": 2,
    "Large Blade": 1,
    "Other Battery": 10,
    "E-Waste With Battery": 5,
    "E-Waste": 2,
}


@pytest.fixture
def annotation_data() -> dict[str, Any]:
    """Load annotation data from CMR dataset."""
    if not TRAIN_ANNOTATIONS.exists():
        pytest.skip(f"Test data not available at {TRAIN_ANNOTATIONS}")

    with open(TRAIN_ANNOTATIONS) as f:
        data = json.load(f)

    return data


@pytest.fixture
def vmi_mrf_transform(annotation_data: dict[str, Any]) -> RustCopyPaste:
    """Create RustCopyPaste transform with vmi_mrf configuration."""
    return RustCopyPaste(
        target_image_width=1024,
        target_image_height=1024,
        mm_class_list=VMI_MRF_CLASSES,
        annotation_file=str(TRAIN_ANNOTATIONS),
        paste_prob=1.0,  # Always paste for testing
        max_paste_objects=999,  # Allow many objects
        object_counts=VMI_MRF_OBJECT_COUNTS,
        use_random_background=False,
        scale_range=(1.0, 1.0),  # No scaling for controlled tests
        rotation_range=(-180, 180),
        min_visibility=0.3,
        overlap_thresh=0.3,
    )


class TestVMIMRFDatasetInfo:
    """Test basic information about the VMI MRF dataset."""

    def test_annotation_file_exists(self) -> None:
        """Test that the annotation file exists."""
        assert TRAIN_ANNOTATIONS.exists(), f"Annotation file not found: {TRAIN_ANNOTATIONS}"
        assert TRAIN_ANNOTATIONS.stat().st_size > 0, "Annotation file is empty"

    def test_images_directory_exists(self) -> None:
        """Test that images directory exists."""
        assert IMAGES_DIR.exists(), f"Images directory not found: {IMAGES_DIR}"
        assert IMAGES_DIR.is_dir(), f"Path is not a directory: {IMAGES_DIR}"

    def test_annotation_structure(self, annotation_data: dict[str, Any]) -> None:
        """Test that annotation data has correct structure."""
        assert "images" in annotation_data, "Missing 'images' key"
        assert "annotations" in annotation_data, "Missing 'annotations' key"
        assert "categories" in annotation_data, "Missing 'categories' key"

        assert isinstance(annotation_data["images"], list), "'images' should be a list"
        assert isinstance(annotation_data["annotations"], list), "'annotations' should be a list"
        assert isinstance(annotation_data["categories"], list), "'categories' should be a list"

    def test_dataset_sizes(self, annotation_data: dict[str, Any]) -> None:
        """Test dataset sizes and statistics."""
        num_images = len(annotation_data["images"])
        num_annotations = len(annotation_data["annotations"])
        num_categories = len(annotation_data["categories"])

        print(f"\n📊 Dataset Statistics:")
        print(f"   Images: {num_images}")
        print(f"   Annotations: {num_annotations}")
        print(f"   Categories: {num_categories}")

        assert num_images > 0, "Dataset should have images"
        assert num_annotations > 0, "Dataset should have annotations"
        assert num_categories > 0, "Dataset should have categories"

    def test_category_structure(self, annotation_data: dict[str, Any]) -> None:
        """Test that categories are properly formatted."""
        for category in annotation_data["categories"]:
            assert "id" in category, "Category missing 'id'"
            assert "name" in category, "Category missing 'name'"
            assert isinstance(category["id"], int), "Category id should be int"
            assert isinstance(category["name"], str), "Category name should be str"


class TestRustCopyPasteBasicFunctionality:
    """Test basic functionality of RustCopyPaste with vmi_mrf config."""

    def test_transform_initialization(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that transform initializes correctly."""
        assert vmi_mrf_transform.target_image_width == 1024
        assert vmi_mrf_transform.target_image_height == 1024
        assert len(vmi_mrf_transform.mm_class_list) == len(VMI_MRF_CLASSES)
        assert vmi_mrf_transform.paste_prob == 1.0

    def test_coco_data_loading(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that COCO data is properly loaded."""
        vmi_mrf_transform._lazy_init()

        assert vmi_mrf_transform._coco_data is not None, "COCO data should be loaded"
        assert "images" in vmi_mrf_transform._coco_data, "COCO data should have images"
        assert "annotations" in vmi_mrf_transform._coco_data, "COCO data should have annotations"

    def test_random_object_sampling(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that random objects can be sampled from COCO data."""
        vmi_mrf_transform._lazy_init()

        # Try to sample a few random objects
        for _ in range(5):
            obj_info = vmi_mrf_transform._get_random_object_from_coco()

            assert obj_info is not None, "Should get an object info dict"
            assert isinstance(obj_info, dict), "Object info should be a dictionary"
            assert "image_path" in obj_info, "Object info should have image_path"
            assert "annotation" in obj_info, "Object info should have annotation"
            assert "category" in obj_info, "Object info should have category"
            assert isinstance(obj_info["image_path"], str), "Image path should be a string"
            assert isinstance(obj_info["category"], str), "Category should be a string"


class TestAnnotationRetention:
    """Test that annotations are properly retained and merged."""

    def test_single_transform_preserves_structure(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that a single transform preserves the output structure."""
        # Create minimal test data in Albumentations format (as expected by RustCopyPaste)
        height, width = 512, 512
        results = {
            "image": np.ones((height, width, 3), dtype=np.uint8) * 128,
            "bboxes": [],
            "masks": [],
            "gt_bboxes_labels": np.array([], dtype=np.int64),
        }

        output = vmi_mrf_transform.transform(results)

        assert output is not None, "Transform should not return None"
        assert "image" in output, "Output should have image"
        assert "bboxes" in output, "Output should have bboxes"
        assert "masks" in output, "Output should have masks"
        assert "gt_bboxes_labels" in output, "Output should have gt_bboxes_labels"

    def test_annotation_count_consistency(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that annotation counts are consistent after transformation."""
        height, width = 512, 512
        results = {
            "image": np.ones((height, width, 3), dtype=np.uint8) * 128,
            "bboxes": [],
            "masks": [],
            "gt_bboxes_labels": np.array([], dtype=np.int64),
        }

        output = vmi_mrf_transform.transform(results)

        # Check consistency between annotations
        num_bboxes = len(output["bboxes"])
        num_masks = len(output["masks"])
        num_labels = len(output["gt_bboxes_labels"])

        assert num_bboxes == num_masks, f"Bbox count ({num_bboxes}) != mask count ({num_masks})"
        assert num_bboxes == num_labels, f"Bbox count ({num_bboxes}) != label count ({num_labels})"

    def test_mask_format_correctness(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that masks are in correct format (list of numpy arrays)."""
        height, width = 512, 512
        results = {
            "image": np.ones((height, width, 3), dtype=np.uint8) * 128,
            "bboxes": [],
            "masks": [],
            "gt_bboxes_labels": np.array([], dtype=np.int64),
        }

        output = vmi_mrf_transform.transform(results)

        # Check that masks are in list format (Albumentations style)
        assert isinstance(output["masks"], list), "Masks should be list of numpy arrays"

        # When there are masks, check they are numpy arrays
        if len(output["masks"]) > 0:
            for i, mask in enumerate(output["masks"]):
                assert isinstance(mask, np.ndarray), f"Mask {i} should be numpy array"
                assert len(mask.shape) == 2, f"Mask {i} should be 2D (h, w)"


class TestRustImplementationPerformance:
    """Test performance characteristics of Rust implementation."""

    def test_paste_operation_completes(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that paste operation completes successfully."""
        height, width = 512, 512
        results = {
            "image": np.ones((height, width, 3), dtype=np.uint8) * 128,
            "bboxes": [],
            "masks": [],
            "gt_bboxes_labels": np.array([], dtype=np.int64),
        }

        # Set paste_prob to 1.0 to ensure pasting happens
        original_prob = vmi_mrf_transform.paste_prob
        vmi_mrf_transform.paste_prob = 1.0

        try:
            output = vmi_mrf_transform.transform(results)
            assert output is not None, "Transform should complete and return result"
        finally:
            vmi_mrf_transform.paste_prob = original_prob


class TestVMIMRFSpecificConfiguration:
    """Test behavior with vmi_mrf-specific configuration."""

    def test_class_list_matches_vmi_mrf(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that all VMI MRF classes are properly configured."""
        for class_name in VMI_MRF_CLASSES:
            assert class_name in vmi_mrf_transform.mm_class_list, f"Missing class: {class_name}"

    def test_object_counts_configured(self, vmi_mrf_transform: RustCopyPaste) -> None:
        """Test that object counts are properly configured."""
        for class_name, count in VMI_MRF_OBJECT_COUNTS.items():
            assert class_name in vmi_mrf_transform.object_counts, f"Missing object count for: {class_name}"
            assert vmi_mrf_transform.object_counts[class_name] == count, (
                f"Object count mismatch for {class_name}: "
                f"expected {count}, got {vmi_mrf_transform.object_counts[class_name]}"
            )

    def test_augmentation_config_loaded(self) -> None:
        """Test that vmi_mrf.yaml augmentation config can be loaded."""
        config_path = (
            Path("/home/georgepearse/core-worktrees/rust-copy-paste/machine_learning/packages/augmentation_configs")
            / "augmentation_configs/vmi_mrf.yaml"
        )

        assert config_path.exists(), f"Augmentation config not found: {config_path}"


def test_summary(annotation_data: dict[str, Any]) -> None:
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("VMI MRF Dataset Integration Test Summary")
    print("=" * 60)
    print(f"✅ Annotation file: {TRAIN_ANNOTATIONS}")
    print(f"✅ Total images: {len(annotation_data['images'])}")
    print(f"✅ Total annotations: {len(annotation_data['annotations'])}")
    print(f"✅ Total categories: {len(annotation_data['categories'])}")
    print(f"✅ Classes configured: {len(VMI_MRF_CLASSES)}")
    print(f"✅ Images directory: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
