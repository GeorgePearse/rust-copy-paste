"""Pytest benchmarks for copy-paste augmentation transforms."""

from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
import torch

from benchmarks.scenarios import SCENARIOS, create_dummy_results
from simple_copy_paste import CopyPasteAugmentation


@pytest.fixture
def dummy_dataset_path():
    """Get path to dummy dataset."""
    return Path(__file__).parent.parent / "tests" / "dummy_dataset"


@pytest.fixture
def annotation_file(dummy_dataset_path):
    """Get path to annotations file."""
    return dummy_dataset_path / "annotations" / "annotations.json"


def create_transform(scenario, annotation_file):
    """Create a transform for the given scenario."""
    return CopyPasteAugmentation(
        image_width=scenario.image_width,
        image_height=scenario.image_height,
        mm_class_list=["triangle", "circle", "square"],
        annotation_file=str(annotation_file),
        p=1.0,
        max_paste_objects=scenario.max_paste_objects,
        scale_range=scenario.scale_range,
        rotation_range=scenario.rotation_range,
        use_random_background=scenario.use_random_background,
        blend_mode=scenario.blend_mode,
        object_counts={"triangle": 2.0, "circle": 2.0, "square": 1.0},
    )


def call_transform(transform, results: Dict[str, Any]):
    """Helper to call transform with dict results (like MMDetection pipeline)."""
    # Extract image and mask from results
    img = results.get("img")
    if img is None:
        raise ValueError("Image not found in results")

    mask = results.get("gt_masks")

    # If mask is in CHW format (MMDet), convert to HWC for Albumentations
    if mask is not None and mask.ndim == 3 and mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
        # Simple heuristic: assuming channel first if first dim is small
        # Actually, CopyPasteAugmentation expects mask passed as 'mask' or 'masks'
        # If it's instance masks (N, H, W), we might need to merge them or pass as list
        # For benchmarking, we can just pass a dummy single channel mask
        pass

    # Create a dummy mask if needed, as create_dummy_results provides (0, H, W) empty mask
    h, w = img.shape[:2]
    dummy_mask = np.zeros((h, w), dtype=np.uint8)

    # Call the transform
    # We pass bboxes to trigger apply_to_bboxes
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2, 1]], dtype=np.float32)

    return transform(image=img, mask=dummy_mask, bboxes=bboxes, category_ids=[1])


# Dynamically create benchmark tests for each scenario
class TestCopyPasteBenchmarks:
    """Benchmarks for CopyPasteAugmentation transform."""

    @pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
    def test_transform(self, benchmark, scenario_name, annotation_file):
        """Benchmark transform execution."""
        scenario = SCENARIOS[scenario_name]

        # Create transform
        transform = create_transform(scenario, annotation_file)

        # Create test data
        results = create_dummy_results(scenario.image_width, scenario.image_height)

        # Benchmark the transform
        def run_transform():
            return call_transform(transform, results)

        benchmark(run_transform)


class TestTransformThroughput:
    """Throughput benchmarks (images per second)."""

    @pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
    def test_throughput(self, benchmark, scenario_name, annotation_file):
        """Benchmark images per second."""
        scenario = SCENARIOS[scenario_name]

        # Create transform
        transform = create_transform(scenario, annotation_file)

        # Create test data
        test_images = [create_dummy_results(scenario.image_width, scenario.image_height) for _ in range(5)]

        def run_batch():
            for results in test_images:
                call_transform(transform, results)

        # Measure time for batch
        benchmark(run_batch)


# Direct benchmark functions (not parameterized)
def test_baseline_512_1obj_direct(benchmark, annotation_file):
    """Direct benchmark: baseline 512x512, 1 object."""
    scenario = SCENARIOS["baseline_512x512_1obj"]
    transform = create_transform(scenario, annotation_file)
    results = create_dummy_results(scenario.image_width, scenario.image_height)

    benchmark(lambda: call_transform(transform, results))


def test_large_2048_1obj_direct(benchmark, annotation_file):
    """Direct benchmark: large 2048x2048, 1 object."""
    scenario = SCENARIOS["large_2048x2048_1obj"]
    transform = create_transform(scenario, annotation_file)
    results = create_dummy_results(scenario.image_width, scenario.image_height)

    benchmark(lambda: call_transform(transform, results))


def test_heavy_10obj_direct(benchmark, annotation_file):
    """Direct benchmark: 10 objects."""
    scenario = SCENARIOS["multi_object_10obj"]
    transform = create_transform(scenario, annotation_file)
    results = create_dummy_results(scenario.image_width, scenario.image_height)

    benchmark(lambda: call_transform(transform, results))
