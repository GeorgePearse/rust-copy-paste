import time
import numpy as np
import os
import sys
import logging
import shutil
from pathlib import Path

# Add the package root to sys.path
# sys.path.append(str(Path(__file__).parent.parent))

from simple_copy_paste import CopyPasteAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def create_dummy_image(width, height):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_dummy_mask(width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    return mask


def run_caching_benchmark(iterations=10):
    logger.info(f"Starting caching benchmark with {iterations} iterations...")

    # Setup paths
    base_path = Path(__file__).parent.parent
    annotation_file = base_path / "tests" / "dummy_dataset" / "annotations" / "annotations.json"
    cache_dir = base_path / "benchmarks" / "cache_test"

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    if not annotation_file.exists():
        logger.error(f"Annotation file not found at {annotation_file}")
        return

    logger.info("Precaching objects...")
    start_precache = time.time()
    CopyPasteAugmentation.precache(annotation_file=str(annotation_file), cache_dir=str(cache_dir))
    precache_time = time.time() - start_precache
    logger.info(f"Precaching took {precache_time:.4f}s")

    logger.info("Initializing transform...")
    start_init = time.time()

    # Initialize transform with cache_dir
    # This should trigger precaching logic inside Rust if cache files exist
    # We pass cache_dir so it knows where to load from.
    transform = CopyPasteAugmentation(
        image_width=1024,
        image_height=1024,
        max_paste_objects=5,
        use_rotation=True,
        use_scaling=True,
        annotation_file=str(annotation_file),
        object_counts={"triangle": 2.0, "circle": 2.0, "square": 1.0},
        cache_dir=str(cache_dir),
        p=1.0,
    )

    init_time = time.time() - start_init
    logger.info(f"Initialization took {init_time:.4f}s")

    # Verify cache directory
    if cache_dir.exists():
        files = list(cache_dir.glob("*.png"))
        logger.info(f"Cache directory created with {len(files)} files")
    else:
        logger.error("Cache directory NOT created!")

    # Data setup
    image = create_dummy_image(1024, 1024)
    mask = create_dummy_mask(1024, 1024)

    # Bboxes
    bboxes = np.array([[0.2, 0.2, 0.4, 0.4, 1]], dtype=np.float32)

    logger.info("Running benchmark loop (using cached objects)...")

    start_time = time.time()
    latencies = []

    for i in range(iterations):
        iter_start = time.time()

        transform(image=image, mask=mask, bboxes=bboxes, category_ids=[1])

        iter_end = time.time()
        latencies.append(iter_end - iter_start)

        if (i + 1) % 5 == 0:
            logger.info(f"Completed {i + 1}/{iterations} iterations")

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = total_time / iterations

    logger.info(f"Benchmark completed.")
    logger.info(f"Total time: {total_time:.4f}s")
    logger.info(f"Average latency: {avg_latency:.4f}s")
    logger.info(f"Min latency: {min(latencies):.4f}s")
    logger.info(f"Max latency: {max(latencies):.4f}s")


if __name__ == "__main__":
    run_caching_benchmark(iterations=10)
