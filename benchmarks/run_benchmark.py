import time
import numpy as np
import os
import sys
import logging
import argparse
import cProfile
import pstats
from pathlib import Path

# Add the package root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from simple_copy_paste import CopyPasteAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def create_dummy_image(width, height):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_dummy_mask(width, height):
    # Create a simple mask with a rectangle
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    return mask


def run_benchmark(iterations=10, profile=False, scenario="standard"):
    logger.info(f"Starting benchmark '{scenario}' with {iterations} iterations...")

    # Setup paths
    base_path = Path(__file__).parent.parent
    annotation_file = base_path / "tests" / "dummy_dataset" / "annotations" / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"Annotation file not found at {annotation_file}")
        return

    # Scenario configuration
    if scenario == "heavy":
        # Simulate user case: many existing objects, add a few
        n_existing_bboxes = 120
        image_size = 1024
        n_paste = 5
        obj_counts = {"triangle": 2.0, "circle": 2.0, "square": 1.0}
    else:
        # Standard
        n_existing_bboxes = 1
        image_size = 512
        n_paste = 5
        obj_counts = {"triangle": 2.0, "circle": 2.0, "square": 1.0}

    # Initialize transform
    transform = CopyPasteAugmentation(
        image_width=image_size,
        image_height=image_size,
        max_paste_objects=n_paste,
        use_rotation=True,
        use_scaling=True,
        annotation_file=str(annotation_file),
        object_counts=obj_counts,
        p=1.0,
    )

    logger.info(f"Transform initialized (size={image_size}, paste={n_paste}).")

    # Data setup
    image = create_dummy_image(image_size, image_size)
    mask = create_dummy_mask(image_size, image_size)

    # Create existing bboxes
    bboxes = []
    for i in range(n_existing_bboxes):
        # Random small boxes
        x = np.random.uniform(0, 0.9)
        y = np.random.uniform(0, 0.9)
        w = np.random.uniform(0.01, 0.1)
        h = np.random.uniform(0.01, 0.1)
        bboxes.append([x, y, min(x + w, 1.0), min(y + h, 1.0), 1])

    bboxes = np.array(bboxes, dtype=np.float32)

    # Warmup
    logger.info("Warming up...")
    transform(image=image, mask=mask, bboxes=bboxes, category_ids=[1] * n_existing_bboxes)

    logger.info("Running benchmark loop...")

    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    start_time = time.time()
    latencies = []

    for i in range(iterations):
        iter_start = time.time()

        # Albumentations call
        transform(image=image, mask=mask, bboxes=bboxes, category_ids=[1] * n_existing_bboxes)

        iter_end = time.time()
        latencies.append(iter_end - iter_start)

        if (i + 1) % 5 == 0:
            logger.info(f"Completed {i + 1}/{iterations} iterations")

    end_time = time.time()

    if profile and profiler:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime")
        stats.print_stats(20)
        # Also save to file
        stats.dump_stats("benchmark_profile.prof")
        logger.info("Profile saved to benchmark_profile.prof")

    total_time = end_time - start_time
    avg_latency = total_time / iterations

    logger.info(f"Benchmark completed.")
    logger.info(f"Total time: {total_time:.4f}s")
    logger.info(f"Average latency: {avg_latency:.4f}s")
    logger.info(f"Min latency: {min(latencies):.4f}s")
    logger.info(f"Max latency: {max(latencies):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CopyPasteAugmentation")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile")
    parser.add_argument("--scenario", type=str, default="standard", choices=["standard", "heavy"], help="Benchmark scenario")

    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, profile=args.profile, scenario=args.scenario)
