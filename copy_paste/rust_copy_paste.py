"""Rust-based SimpleCopyPaste augmentation using rusty_paste for performance."""

import json
import os
import random
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pycocotools import mask as mask_utils

try:
    from rusty_paste import ObjectPaster, PasteConfig

    RUST_AVAILABLE = True
except ImportError:
    logger.warning("rusty_paste not available, falling back to Python implementation")
    RUST_AVAILABLE = False


# Temporary hardcoded dataset root while paths are stabilizing.
_HARDCODED_IMAGE_ROOT = "/home/georgepearse/data/images"


# Global cached function for loading COCO annotation data
# Cache size of 4 allows for multiple annotation files (train/val/test)
@lru_cache(maxsize=4)
def _load_coco_data(annotation_file: str) -> dict:
    """Load and cache COCO annotation data from a file."""
    with open(annotation_file) as f:
        return json.load(f)


# Global cached function for loading images and masks
# This is defined at module level so lru_cache can work efficiently
# Fixed cache size of 128 images (module-level constant, not configurable per-instance)
@lru_cache(maxsize=128)
def _cached_load_image_and_mask(
    image_path: str,
    ann_id: int,
    annotation_file: str,
    coco_data_id: int,
    image_root: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and cache an image and its mask.

    Args:
        image_path: Path to the image file
        ann_id: Annotation ID for cache key uniqueness
        annotation_file: Path to annotation file (for constructing full path)
        coco_data_id: Unique ID for COCO data (for cache invalidation)
        image_root: Optional root directory containing source images

    Returns:
        Tuple of (image, mask) or None if loading failed
    """
    # Construct full path
    if os.path.isabs(image_path):
        full_path = image_path
    else:
        base_dir = image_root or (os.path.dirname(annotation_file) if annotation_file else "")
        full_path = os.path.join(base_dir, image_path)

    # Load image
    img = cv2.imread(full_path)
    if img is None:
        logger.warning(f"Failed to load image: {full_path}")
        return None

    # Load COCO data to decode mask (cached at module level)
    coco_data = _load_coco_data(annotation_file)

    # Find annotation
    ann = None
    for a in coco_data["annotations"]:
        if a["id"] == ann_id:
            ann = a
            break

    if ann is None:
        logger.warning(f"Annotation {ann_id} not found in {annotation_file}")
        return None

    # Decode mask
    if isinstance(ann["segmentation"], list):
        # Polygon format
        from pycocotools import mask as cocomask

        rles = cocomask.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1])
        rle = cocomask.merge(rles)
        mask = cocomask.decode(rle)
    elif isinstance(ann["segmentation"], dict):
        # RLE format
        mask = mask_utils.decode(ann["segmentation"])
    else:
        logger.warning(f"Unknown segmentation format for annotation {ann_id}")
        return None

    return img, mask


class RustCopyPaste:
    """Rust-accelerated copy-paste augmentation using rusty_paste.

    This transform provides a high-performance copy-paste augmentation using the Rust
    implementation (rusty_paste) for the core paste operations. It maintains the same
    interface as CustomCopyPaste but delegates geometric transformations and collision
    detection to the optimized Rust backend.

    Parameters match CustomCopyPaste for compatibility, but some features may have
    different behavior due to the Rust implementation.
    """

    DEFAULT_IMAGE_ROOT = _HARDCODED_IMAGE_ROOT

    def __init__(
        self,
        target_image_width: int,
        target_image_height: int,
        mm_class_list: list[str],
        annotation_file: str | None = None,
        paste_prob: float = 1.0,
        max_paste_objects: int = 1,
        object_counts: dict[str, int | float] | None = None,
        scale_range: tuple[float, float] = (1.0, 1.0),
        rotation_range: tuple[float, float] = (0, 360),
        min_visibility: float = 0.3,
        overlap_thresh: float = 0.3,
        blend_mode: str = "normal",
        class_list: list[str] | None = None,
        class_name_mapping: dict[str, str] | None = None,
        verbose: bool = False,
        use_random_background: bool = False,
        random_background_prob: float = 0.5,
        p: float = 1.0,  # Albumentations standard probability parameter
        image_root: str | None = None,
    ):
        """Initialize RustCopyPaste augmentation.

        Args:
            target_image_width: Target width for resized images.
            target_image_height: Target height for resized images.
            mm_class_list: List of class names in MMDetection format.
            annotation_file: Path to COCO annotation file.
            paste_prob: Probability of applying the transform.
            max_paste_objects: Maximum number of objects to paste per image.
            object_counts: Target counts per class (see CustomCopyPaste docs).
            scale_range: Scale range for objects (min, max). Currently only (1, 1) supported.
            rotation_range: Rotation range in degrees.
            min_visibility: Minimum visibility for pasted objects.
            overlap_thresh: Maximum IoU overlap allowed (used for collision detection).
            blend_mode: Blending mode ('normal' or 'xray'). Currently only 'normal' supported.
            use_random_background: Generate random backgrounds instead of using input image.
            random_background_prob: Probability of using random background when enabled.
            p: Albumentations standard probability parameter (overrides paste_prob if provided).
            image_root: Optional root directory containing COCO images (defaults to /home/georgepearse/data/images).

        Note:
            Image/mask loading is cached at the module level with a fixed cache size of 128.
            This cannot be configured per-instance to maintain compatibility with multiprocessing.
        """
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "RustCopyPaste requires rusty_paste to be installed. "
                "Please install it with: cd machine_learning/packages/rusty_paste && maturin develop"
            )

        # Initialize basic attributes
        self.target_image_width = target_image_width
        self.target_image_height = target_image_height
        self.mm_class_list = mm_class_list
        self.annotation_file = annotation_file
        resolved_root = image_root or self.DEFAULT_IMAGE_ROOT
        self.image_root = os.path.abspath(resolved_root)
        if not os.path.isdir(self.image_root):
            logger.warning(f"Image root directory does not exist: {self.image_root}")
        # Use Albumentations 'p' parameter if provided, otherwise use paste_prob
        self.paste_prob = p if p != 1.0 else paste_prob
        self.max_paste_objects = max_paste_objects
        self.object_counts = object_counts or {}
        # Convert lists to tuples if needed (YAML config may provide lists)
        self.scale_range = tuple(scale_range) if isinstance(scale_range, list) else scale_range
        self.rotation_range = tuple(rotation_range) if isinstance(rotation_range, list) else rotation_range
        self.min_visibility = min_visibility
        self.overlap_thresh = overlap_thresh
        self.blend_mode = blend_mode
        self.use_random_background = use_random_background
        self.random_background_prob = random_background_prob
        self.verbose = verbose

        # Validate configuration
        if self.scale_range != (1.0, 1.0):
            logger.warning(f"RustCopyPaste currently only supports scale_range=(1, 1), got {self.scale_range}. Scaling will be ignored.")

        if blend_mode != "normal":
            logger.warning(f"RustCopyPaste currently only supports blend_mode='normal', got '{blend_mode}'. Normal blending will be used.")

        # Create Rust ObjectPaster with configuration
        self.paster = ObjectPaster(  # type: ignore[possibly-unbound]
            PasteConfig(  # type: ignore[possibly-unbound]
                resize_factor=1.0,  # No scaling yet
                rotation_range=self.rotation_range,
                allow_partial=True,  # Allow partial visibility
                min_visibility=self.min_visibility,
                max_attempts=100,  # Use reasonable default
                overlap_thresh=self.overlap_thresh,  # IoU threshold for collision detection
            )
        )

        # COCO annotation loading (lazy initialization)
        self._initialized = False
        self._images_by_category: dict[str, list[dict]] = {}
        self._coco_data: dict | None = None
        # Unique ID for cache invalidation (based on object id, changes per instance)
        self._coco_data_id = id(self)

        # Error tracking
        self._error_count = 0
        self._max_errors_before_warning = 10

        logger.info(
            "🦀 RUST COPY-PASTE ACTIVE 🦀 | "
            f"Initialized RustCopyPaste with {max_paste_objects} max objects, "
            f"rotation={rotation_range}, min_visibility={min_visibility}, "
            f"overlap_thresh={overlap_thresh} (cache_size=128 fixed)"
        )

    @property
    def available_keys(self) -> set[str]:
        """Return keys that this transform can process (Albumentations interface)."""
        # Note: CustomAlbumentations maps MMDetection keys to Albumentations keys:
        # "img" -> "image", "gt_bboxes" -> "bboxes", "gt_masks" -> "masks"
        return {"image", "bboxes", "masks", "gt_bboxes_labels"}

    def __call__(self, results: dict | None = None, **kwargs: Any) -> dict | None:
        """Override to handle both MMDetection and Albumentations calling conventions.

        MMDetection calls: transform(results_dict)
        Albumentations calls: transform(**data_dict)

        Args:
            results: The result dict (MMDetection style) or None (Albumentations style)
            **kwargs: Keyword arguments (Albumentations style) or extra args (MMDetection)

        Returns:
            Modified results dict or None
        """
        # If we have kwargs, we're being called Albumentations-style with **kwargs
        # In this case, results might be None OR it might be one of the keyword args
        if kwargs:
            # Albumentations style: all data passed as kwargs
            return self.transform(kwargs)
        elif results is not None:
            # MMDetection style: data passed as results dict
            return self.transform(results)
        else:
            # Edge case: called with no arguments
            raise ValueError("RustCopyPaste requires either results dict or keyword arguments")

    def _lazy_init(self) -> None:
        """Lazy initialization of COCO data to support multiprocessing."""
        if self._initialized:
            return

        if not self.annotation_file or not os.path.exists(self.annotation_file):
            logger.warning(f"Annotation file not found: {self.annotation_file}")
            self._initialized = True
            return

        # Load COCO annotations
        with open(self.annotation_file) as f:
            self._coco_data = json.load(f)

        assert self._coco_data is not None, "Failed to load COCO data"

        # Build category name to ID mapping
        cat_name_to_id = {cat["name"]: cat["id"] for cat in self._coco_data["categories"]}

        # Build image ID to image info mapping
        img_id_to_info = {img["id"]: img for img in self._coco_data["images"]}

        # Group annotations by category
        self._images_by_category = {cat: [] for cat in cat_name_to_id}

        for ann in self._coco_data["annotations"]:
            img_id = ann["image_id"]
            img_info = img_id_to_info.get(img_id)
            if not img_info:
                continue

            # Find category name
            cat_name = None
            for name, cid in cat_name_to_id.items():
                if cid == ann["category_id"]:
                    cat_name = name
                    break

            if cat_name:
                self._images_by_category[cat_name].append(
                    {
                        "image_path": img_info["file_name"],
                        "annotation": ann,
                        "category": cat_name,
                    }
                )

        logger.info(f"Loaded {len(self._coco_data['annotations'])} annotations from {len(self._coco_data['images'])} images")
        for cat, objs in self._images_by_category.items():
            if objs:
                logger.debug(f"  {cat}: {len(objs)} objects")

        self._initialized = True

    def _load_image_and_mask(self, image_path: str, ann_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Load and cache an image and its mask using the global cache.

        Args:
            image_path: Path to the image file
            ann_id: Annotation ID for cache key uniqueness

        Returns:
            Tuple of (image, mask) or None if loading failed
        """
        if not self.annotation_file:
            logger.warning("No annotation file specified")
            return None

        return _cached_load_image_and_mask(
            image_path,
            ann_id,
            self.annotation_file,
            self._coco_data_id,
            self.image_root,
        )

    def _get_random_object_from_coco(self) -> dict[str, Any] | None:
        """Get a random object from COCO annotations based on object_counts."""
        if not self._initialized:
            self._lazy_init()

        if not self._images_by_category:
            return None

        # Choose category based on object_counts
        if self.object_counts:
            categories = []
            weights = []
            for cat, count in self.object_counts.items():
                if cat == "all":
                    continue
                if self._images_by_category.get(cat):
                    categories.append(cat)
                    weights.append(count)

            if not categories:
                return None

            category = random.choices(categories, weights=weights)[0]
        else:
            available_categories = [cat for cat in self._images_by_category if self._images_by_category[cat]]
            if not available_categories:
                return None
            category = random.choice(available_categories)

        if not self._images_by_category[category]:
            return None

        return random.choice(self._images_by_category[category])

    def transform(self, results: dict) -> dict | None:
        """Apply Rust-accelerated copy-paste augmentation.

        Args:
            results: Dictionary with Albumentations keys: 'image', 'bboxes', 'masks', 'gt_bboxes_labels'

        Returns:
            Modified results dictionary or None if augmentation fails
        """
        if not self._initialized:
            self._lazy_init()

        # Apply with probability
        if random.random() > self.paste_prob:
            return results

        # Handle random background selection with annotation retention
        background_annotations_count = 0
        if self.use_random_background and random.random() < self.random_background_prob:
            try:
                # Get random image with ALL its annotations from COCO
                from rusty_paste import CopyPasteAugmenter as RustAugmenter  # type: ignore[import-not-found]

                # Create a Rust augmenter if we don't have one
                if not hasattr(self, "_rust_augmenter"):
                    if not self.annotation_file or not os.path.exists(self.annotation_file):
                        logger.warning("Cannot use random background without annotation file")
                        background = results["image"].copy()
                    else:
                        from rusty_paste import AugmenterConfig  # type: ignore[import-not-found]

                        image_dir = self.image_root or os.path.dirname(self.annotation_file)
                        config = AugmenterConfig(
                            annotation_file=self.annotation_file,
                            image_dir=image_dir,
                        )
                        self._rust_augmenter = RustAugmenter(config)

                if hasattr(self, "_rust_augmenter"):
                    # Get random image ID and load with all annotations
                    random_image_id = self._rust_augmenter.get_random_image_id()
                    bg_img, bg_masks, bg_bboxes, bg_categories, bg_ann_ids = self._rust_augmenter.load_image_with_all_annotations(random_image_id)

                    # Update background image
                    background = bg_img.copy() if isinstance(bg_img, np.ndarray) else np.array(bg_img)

                    # CRITICAL: Validate mask dimensions match background image dimensions
                    background_annotations_count = len(bg_masks)
                    if len(bg_masks) > 0:
                        bg_height, bg_width = background.shape[:2]
                        mismatched_masks = []
                        for i, mask in enumerate(bg_masks):
                            mask_array = np.array(mask)
                            if mask_array.shape != (bg_height, bg_width):
                                mismatched_masks.append(
                                    f"mask_idx={i}, ann_id={bg_ann_ids[i]}, expected=({bg_height}, {bg_width}), got={mask_array.shape}"
                                )

                        if mismatched_masks:
                            logger.error(
                                f"❌ CRITICAL: Mask dimension mismatch detected for {len(mismatched_masks)}/{len(bg_masks)} masks. "
                                f"Background image: {bg_width}x{bg_height}. Details: [{', '.join(mismatched_masks[:3])}]"
                            )
                            # Filter out mismatched masks
                            valid_mask_indices = [i for i, mask in enumerate(bg_masks) if np.array(mask).shape == (bg_height, bg_width)]
                            if valid_mask_indices:
                                bg_bboxes = [bg_bboxes[i] for i in valid_mask_indices]
                                bg_masks = [bg_masks[i] for i in valid_mask_indices]
                                bg_categories = [bg_categories[i] for i in valid_mask_indices]
                                bg_ann_ids = [bg_ann_ids[i] for i in valid_mask_indices]
                                background_annotations_count = len(valid_mask_indices)
                                logger.warning(
                                    f"⚠️ Filtered to {background_annotations_count}/{len(bg_bboxes)} valid masks after removing dimension mismatches"
                                )
                            else:
                                logger.error("❌ All background masks had dimension mismatches, clearing annotations")
                                bg_bboxes = []
                                bg_masks = []
                                bg_categories = []
                                bg_ann_ids = []
                                background_annotations_count = 0

                    if background_annotations_count > 0:
                        # Convert background annotations to format expected by Albumentations
                        # bboxes are in x1, y1, x2, y2 format from Rust
                        results["bboxes"] = [list(bbox) for bbox in bg_bboxes]
                        results["masks"] = [np.array(mask) for mask in bg_masks]

                        # Validate all background categories exist in class list
                        unmapped_categories = [cat for cat in bg_categories if cat not in self.mm_class_list]
                        if unmapped_categories:
                            logger.error(
                                f"❌ CRITICAL: Background image has unmapped categories: {set(unmapped_categories)}. "
                                f"Available classes: {self.mm_class_list}. "
                                f"This indicates data/configuration mismatch. "
                                f"Annotation IDs: {bg_ann_ids}"
                            )
                            # Filter out unmapped annotations instead of silently corrupting them
                            valid_indices = [i for i, cat in enumerate(bg_categories) if cat in self.mm_class_list]
                            if valid_indices:
                                results["bboxes"] = [results["bboxes"][i] for i in valid_indices]
                                results["masks"] = [results["masks"][i] for i in valid_indices]
                                bg_categories = [bg_categories[i] for i in valid_indices]
                                background_annotations_count = len(valid_indices)
                                logger.warning(
                                    f"⚠️ Filtered to {background_annotations_count}/{len(bg_masks)} valid annotations "
                                    f"after removing unmapped categories"
                                )
                            else:
                                # All annotations were unmapped, clear results
                                results["bboxes"] = []
                                results["masks"] = []
                                background_annotations_count = 0
                                logger.warning("⚠️ No valid background annotations after filtering unmapped categories")

                        results["gt_bboxes_labels"] = np.array(
                            [self.mm_class_list.index(cat) for cat in bg_categories],
                            dtype=np.int64,
                        )
                    else:
                        # Background image has no annotations
                        results["bboxes"] = []
                        results["masks"] = []
                        results["gt_bboxes_labels"] = np.array([], dtype=np.int64)

                    if self.verbose:
                        logger.debug(f"✨ Using random background image with {background_annotations_count} background annotations")
                else:
                    background = results["image"].copy()
            except Exception as e:
                logger.warning(f"Failed to load random background: {e}, using input image instead")
                background = results["image"].copy()
        else:
            # Get background image - use Albumentations key 'image' not 'img'
            background = results["image"].copy()

        # Collect objects to paste
        objects_to_paste = []
        categories_to_paste = []

        # Determine how many objects to paste per category
        paste_counts: dict[str, int] = {}
        if self.object_counts:
            for cat, count in self.object_counts.items():
                if cat == "all":
                    continue
                if count >= 1:
                    paste_counts[cat] = round(count)
                elif 0 < count < 1 and random.random() < count:
                    paste_counts[cat] = 1
        else:
            # No object_counts specified, paste random objects up to max
            paste_counts = {}

        # Load objects for each category
        for _category, count in paste_counts.items():
            for _ in range(count):
                if len(objects_to_paste) >= self.max_paste_objects:
                    break

                obj_info = self._get_random_object_from_coco()
                if obj_info is None:
                    continue

                # Load image and mask
                loaded = self._load_image_and_mask(obj_info["image_path"], obj_info["annotation"]["id"])
                if loaded is None:
                    continue

                img, mask = loaded
                objects_to_paste.append((img, mask))
                categories_to_paste.append(obj_info["category"])

        if not objects_to_paste:
            return results

        # Use Rust implementation to paste objects
        try:
            result_img, result_masks, bboxes = self.paster.paste_objects(background, objects_to_paste)

            # Track skip rate (objects that failed to paste)
            attempted_count = len(objects_to_paste)
            successful_count = len(bboxes)
            skipped_count = attempted_count - successful_count

            if skipped_count > 0:
                skip_rate = 100 * skipped_count / attempted_count
                logger.warning(
                    f"⚠️ PASTE SKIP RATE: {skipped_count}/{attempted_count} objects failed to paste ({skip_rate:.1f}% skip rate). "
                    f"This usually indicates collision or visibility constraints. "
                    f"Categories attempted: {categories_to_paste}"
                )

            # Update results with pasted objects
            if len(bboxes) > 0:
                # Convert to Albumentations format (list of bboxes)
                new_bboxes = [list(bbox) for bbox in bboxes]  # (N, 4) in x1, y1, x2, y2 format

                # FIX #4: Add detailed logging for merge logic
                existing_bbox_count = len(results.get("bboxes", []))
                existing_mask_count = len(results.get("masks", []))
                new_bbox_count = len(new_bboxes)
                new_mask_count = len(result_masks)

                logger.debug(
                    f"🔍 MERGE LOGIC DETAILS: "
                    f"existing_bboxes={existing_bbox_count}, new_bboxes={new_bbox_count} → expected_total={existing_bbox_count + new_bbox_count} | "
                    f"existing_masks={existing_mask_count}, new_masks={new_mask_count} → expected_total={existing_mask_count + new_mask_count}"
                )

                # Add new objects to existing ones
                if results.get("bboxes"):
                    # CustomAlbumentations has already converted to list
                    existing_bboxes = results["bboxes"]
                    all_bboxes = existing_bboxes + new_bboxes
                    # FIX #4: Assert merge preserves count
                    assert len(all_bboxes) == existing_bbox_count + new_bbox_count, \
                        f"Bbox merge failed: expected {existing_bbox_count + new_bbox_count}, got {len(all_bboxes)}"
                else:
                    all_bboxes = new_bboxes

                results["bboxes"] = all_bboxes

                # Add masks (keep as list for Albumentations)
                if results.get("masks"):
                    existing_masks = results["masks"]
                    # Ensure result_masks is a list of 2D arrays
                    new_mask_list = [result_masks[i] for i in range(len(result_masks))]
                    all_masks = existing_masks + new_mask_list
                    # FIX #4: Assert merge preserves count
                    assert len(all_masks) == existing_mask_count + new_mask_count, \
                        f"Mask merge failed: expected {existing_mask_count + new_mask_count}, got {len(all_masks)}"
                else:
                    all_masks = [result_masks[i] for i in range(len(result_masks))]

                results["masks"] = all_masks

                # Add labels - use gt_bboxes_labels key (the mapped version of gt_labels)
                # Map category names to label indices
                new_labels = []
                unmapped_paste_categories = set()
                for cat in categories_to_paste[: len(bboxes)]:
                    if cat in self.mm_class_list:
                        new_labels.append(self.mm_class_list.index(cat))
                    else:
                        unmapped_paste_categories.add(cat)
                        logger.error(
                            f"❌ CRITICAL: Pasted object has unmapped category '{cat}'. "
                            f"Available classes: {self.mm_class_list}. "
                            f"This annotation cannot be correctly labeled and will be filtered out."
                        )

                # Filter out unmapped pasted annotations
                if unmapped_paste_categories:
                    valid_paste_indices = [i for i, cat in enumerate(categories_to_paste[: len(bboxes)]) if cat in self.mm_class_list]
                    if valid_paste_indices:
                        all_bboxes = all_bboxes[: len(results.get("bboxes", [])) - len(bboxes)] + [
                            all_bboxes[len(results.get("bboxes", [])) - len(bboxes) + i] for i in valid_paste_indices
                        ]
                        all_masks = all_masks[: len(results.get("masks", [])) - len(bboxes)] + [
                            all_masks[len(results.get("masks", [])) - len(bboxes) + i] for i in valid_paste_indices
                        ]
                        new_labels = [new_labels[i] for i in valid_paste_indices]
                        logger.warning(
                            f"⚠️ Filtered pasted objects: {len(valid_paste_indices)}/{len(bboxes)} valid. "
                            f"Removed {len(bboxes) - len(valid_paste_indices)} with unmapped categories {unmapped_paste_categories}"
                        )
                    else:
                        # All pasted annotations were unmapped, remove them all
                        all_bboxes = all_bboxes[: len(results.get("bboxes", [])) - len(bboxes)]
                        all_masks = all_masks[: len(results.get("masks", [])) - len(bboxes)]
                        new_labels = []
                        logger.warning(
                            f"⚠️ All {len(bboxes)} pasted objects had unmapped categories {unmapped_paste_categories}. They have been removed."
                        )

                if "gt_bboxes_labels" in results and results["gt_bboxes_labels"] is not None:
                    existing_labels = list(results["gt_bboxes_labels"])
                    all_labels = existing_labels + new_labels
                    # FIX #4: Assert labels sync with bboxes
                    assert len(all_labels) == len(all_bboxes), \
                        f"Label count mismatch: {len(all_labels)} labels vs {len(all_bboxes)} bboxes"
                else:
                    all_labels = new_labels
                    # FIX #4: Assert labels sync with bboxes
                    assert len(all_labels) == len(all_bboxes), \
                        f"Label count mismatch: {len(all_labels)} labels vs {len(all_bboxes)} bboxes"

                results["gt_bboxes_labels"] = all_labels

                # FIX #4: Log final merge state
                logger.debug(
                    f"✅ MERGE COMPLETE: bboxes={len(all_bboxes)}, masks={len(all_masks)}, labels={len(all_labels)}"
                )

                # Update image
                results["image"] = result_img

                if self.verbose:
                    logger.debug(f"🦀 Pasted {len(bboxes)} objects using Rust implementation")

                # Always log on first successful paste to confirm Rust is working
                if not hasattr(self, "_first_paste_logged"):
                    logger.info(f"🦀 RUST BACKEND CONFIRMED: Successfully pasted {len(bboxes)} objects using rusty_paste")
                    self._first_paste_logged = True

        except Exception as e:
            self._error_count += 1
            if self._error_count <= self._max_errors_before_warning:
                logger.warning(f"Rust paste_objects failed: {e}, returning original image")
            elif self._error_count == self._max_errors_before_warning + 1:
                logger.error(
                    f"🦀 RUST BACKEND ERROR: {self._error_count} consecutive failures detected. "
                    f"This may indicate a systemic issue. Further errors will be suppressed. "
                    f"Last error: {e}"
                )
            # Reset error count on successful paste (done in the try block)
            return results

        # COMPREHENSIVE VALIDATION: Verify annotation counts and track which annotations were retained
        final_annotations_count = len(results.get("gt_bboxes_labels", []))
        expected_count = background_annotations_count + len(bboxes)

        if final_annotations_count != expected_count:
            logger.error(
                f"❌ CRITICAL: Annotation count mismatch! "
                f"Expected {expected_count} (background={background_annotations_count} + pasted={len(bboxes)}), "
                f"but got {final_annotations_count}. "
                f"This indicates {expected_count - final_annotations_count} annotations were lost in the pipeline. "
                f"Background annotation IDs: {bg_ann_ids if hasattr(self, '_bg_ann_ids') else 'unknown'}"
            )
        else:
            if background_annotations_count > 0 or len(bboxes) > 0:
                logger.info(
                    f"✅ Annotation retention verified: {final_annotations_count} total annotations retained "
                    f"({background_annotations_count} background + {len(bboxes)} pasted). "
                    f"Pipeline integrity: {100 * final_annotations_count / max(expected_count, 1):.0f}%"
                )

        # Store annotation IDs for future tracking (if available from background)
        if "bg_ann_ids" in locals():
            self._bg_ann_ids = bg_ann_ids

        # Reset error count on successful execution
        self._error_count = 0
        return results
