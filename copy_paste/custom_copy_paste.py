import json
import os
import random
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from pycocotools import mask as mask_utils

# SimpleCopyPaste augmentation that loads objects from COCO annotation files
# instead of database queries for better performance and portability.


class CustomCopyPaste:
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
        cache_size: int = 64,
        class_list: list[str] | None = None,
        class_name_mapping: dict[str, str] | None = None,
        verbose: bool = False,
        use_random_background: bool = False,
        random_background_prob: float = 0.5,
    ):
        """Initialize the CustomCopyPaste augmentation.

        This transform pastes foreground objects from COCO annotation files onto background images
        to create augmented training data. It can also generate random backgrounds.

        Args:
            target_image_width (int): Target width for resized images.
            target_image_height (int): Target height for resized images.
            mm_class_list (List[str]): List of class names in MMDetection format.
            annotation_file (Optional[str]): Path to COCO annotation file.
            paste_prob (float): Probability of applying the transform.
            max_paste_objects (int): Maximum number of objects to paste per image.
            object_counts (Optional[Dict[str, Union[int, float]]]): Target counts per class.
                Values >= 1 are treated as exact counts (rounded to int if float).
                Values < 1 are treated as probabilities of pasting one object.
                Special key 'all' sets the count for all available classes (can be overridden
                by specific class counts). Example: {'all': 3} pastes 3 of each class,
                {'all': 3, 'Metal Can': 5} pastes 3 of each class except Metal Can which gets 5.
            scale_range (Tuple[float, float]): Scale range for objects (min, max).
            rotation_range (Tuple[float, float]): Rotation range in degrees.
            min_visibility (float): Minimum visibility for pasted objects.
            overlap_thresh (float): Maximum IoU overlap allowed.
            blend_mode (str): Blending mode ('normal' or 'xray').
            cache_size (int): Size of LRU cache for images/masks.
            use_database (bool): Use database loading (backward compatibility).
        """
        # Initialize basic attributes
        self.target_image_width = target_image_width
        self.target_image_height = target_image_height
        self.mm_class_list = mm_class_list
        self.annotation_file = annotation_file
        self.paste_prob = paste_prob
        self.max_paste_objects = max_paste_objects
        self.object_counts = object_counts or {}
        self.scale_range = scale_range

        # Validate object_counts
        if self.object_counts:
            for category, value in self.object_counts.items():
                if not isinstance(value, int | float):
                    raise TypeError(f"object_counts values must be int or float, got {type(value)} for '{category}'")
                if value < 0:
                    raise ValueError(f"object_counts values must be >= 0, got {value} for '{category}'")
                if category == "all":
                    if value >= 1:
                        logger.info(f"All available categories will have {round(value)} objects pasted (unless overridden)")
                    elif 0 < value < 1:
                        logger.info(f"All available categories have probability {value:.2%} of being pasted (unless overridden)")
                elif 0 < value < 1:
                    logger.info(f"Category '{category}' has probability {value:.2%} of being pasted")
                elif value >= 1:
                    logger.info(f"Category '{category}' will have {round(value)} objects pasted")
        self.rotation_range = rotation_range
        self.min_visibility = min_visibility
        self.overlap_thresh = overlap_thresh
        self.blend_mode = blend_mode
        self.cache_size = cache_size
        self.verbose = verbose
        self.class_name_mapping = class_name_mapping
        self.use_random_background = use_random_background
        self.random_background_prob = random_background_prob

        # Set up class list
        if class_list is None:
            self.class_list = set(mm_class_list)
        else:
            self.class_list = set(class_list)

        # Create class mapping
        self.class_mapping = {item: index for index, item in enumerate(mm_class_list)}

        # Initialize data - will be populated in _lazy_init
        self._initialized = False
        self._coco_data = None
        self._category_to_id = {}
        self._id_to_category = {}
        self._images_by_category = {}
        self._image_cache = None

        # For compatibility with existing code that expects these attributes
        self.foreground_prob = paste_prob
        self.random_subselect = max_paste_objects
        self.xray_paste = blend_mode == "xray"

        # Initialize empty data structures
        self.foreground_images = []
        self.all_indices = []
        self.potential_nested_foreground_objects = None
        self.background_images = []

        # Pipeline stage tracking for consolidated logging
        self._stage_counts: dict[str, dict[str, int]] = {}

    def _lazy_init(self) -> None:
        """Lazy initialization for multi-worker safety."""
        if self._initialized:
            return

        # Set up image cache
        self._image_cache = lru_cache(maxsize=self.cache_size)(self._load_image_cached)

        if self.annotation_file and os.path.exists(self.annotation_file):
            # Load COCO annotations
            self._load_coco_annotations()
        else:
            logger.warning(f"Annotation file not found: {self.annotation_file}")

        self._initialized = True

    def _load_coco_annotations(self) -> None:
        """Load and parse COCO annotation file."""
        if self.annotation_file is None:
            logger.warning("No annotation file provided")
            return
        with open(self.annotation_file) as f:
            self._coco_data = json.load(f)

        # Build category mappings
        for cat in self._coco_data["categories"]:
            cat_name = cat["name"]
            cat_id = cat["id"]
            self._category_to_id[cat_name] = cat_id
            self._id_to_category[cat_id] = cat_name

        # Build image index by category
        for cat_name in self.class_list:
            self._images_by_category[cat_name] = []

        # Index annotations by category
        image_id_to_image = {img["id"]: img for img in self._coco_data["images"]}

        for ann in self._coco_data["annotations"]:
            cat_id = ann["category_id"]
            cat_name = self._id_to_category.get(cat_id)

            if cat_name and cat_name in self.class_list:
                image_id = ann["image_id"]
                if image_id in image_id_to_image:
                    # Store annotation with image info
                    ann_with_image = {"annotation": ann, "image": image_id_to_image[image_id], "category": cat_name}
                    self._images_by_category[cat_name].append(ann_with_image)

        # Log statistics
        for cat_name, anns in self._images_by_category.items():
            logger.info(f"Loaded {len(anns)} annotations for category '{cat_name}'")

    def _load_image_cached(self, image_path: str) -> np.ndarray:
        """Load an image from disk (this will be wrapped by lru_cache)."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return np.zeros((self.target_image_height, self.target_image_width, 3), dtype=np.uint8)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return np.zeros((self.target_image_height, self.target_image_width, 3), dtype=np.uint8)

    def _get_random_object_from_coco(self) -> dict[str, Any] | None:
        """Get a random object from COCO annotations based on object_counts."""
        if not self._initialized:
            self._lazy_init()

        if not self._images_by_category:
            return None

        # Choose category based on object_counts
        if self.object_counts:
            # Weighted selection based on counts
            categories = []
            weights = []
            for cat, count in self.object_counts.items():
                if self._images_by_category.get(cat):
                    categories.append(cat)
                    weights.append(count)

            if not categories:
                return None

            category = random.choices(categories, weights=weights)[0]
        else:
            # Random selection from available categories
            available_categories = [cat for cat in self._images_by_category if self._images_by_category[cat]]
            if not available_categories:
                return None
            category = random.choice(available_categories)

        # Get random annotation from chosen category
        if not self._images_by_category[category]:
            return None

        return random.choice(self._images_by_category[category])

    def _get_specific_object_from_coco(self, category: str) -> dict[str, Any] | None:
        """Get a random object from a specific category."""
        if not self._initialized:
            self._lazy_init()

        if not self._images_by_category or category not in self._images_by_category:
            return None

        # Get random annotation from specified category
        if not self._images_by_category[category]:
            return None

        return random.choice(self._images_by_category[category])

    def _generate_random_background(self, height: int, width: int) -> np.ndarray:
        """Generate a random background image.

        Args:
            height (int): Height of the background image
            width (int): Width of the background image

        Returns:
            np.ndarray: Random background image
        """
        background_type = random.choice(
            ["rgb_noise", "red_dominated", "blue_dominated", "green_dominated", "gradient", "perlin_noise", "checkered", "striped"]
        )

        background = np.zeros((height, width, 3), dtype=np.uint8)

        if background_type == "rgb_noise":
            # Create random RGB noise
            background = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        elif background_type == "red_dominated":
            # Red-dominated noise (high R channel in BGR)
            background[:, :, 0] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # B
            background[:, :, 1] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # G
            background[:, :, 2] = np.random.randint(150, 256, (height, width), dtype=np.uint8)  # R

        elif background_type == "blue_dominated":
            # Blue-dominated noise (high B channel in BGR)
            background[:, :, 0] = np.random.randint(150, 256, (height, width), dtype=np.uint8)  # B
            background[:, :, 1] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # G
            background[:, :, 2] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # R

        elif background_type == "green_dominated":
            # Green-dominated noise
            background[:, :, 0] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # B
            background[:, :, 1] = np.random.randint(150, 256, (height, width), dtype=np.uint8)  # G
            background[:, :, 2] = np.random.randint(0, 100, (height, width), dtype=np.uint8)  # R

        elif background_type == "gradient":
            # Create a gradient background

            # Choose gradient direction and colors
            if random.choice([True, False]):
                # Horizontal gradient
                x = np.linspace(0, 1, width)
                gradient = np.tile(x, (height, 1))
            else:
                # Vertical gradient
                y = np.linspace(0, 1, height)
                gradient = np.tile(y.reshape(-1, 1), (1, width))

            # Random start and end colors for each channel
            for c in range(3):
                start = random.randint(0, 255)
                end = random.randint(0, 255)
                background[:, :, c] = (gradient * (end - start) + start).astype(np.uint8)

        elif background_type == "perlin_noise":
            # Approximate perlin noise with smoothed random values

            # Create base noise at lower resolution
            scale = 10  # Scale factor for noise
            small_h, small_w = height // scale, width // scale

            for c in range(3):
                # Create small random noise image
                small_noise = np.random.randint(0, 256, (small_h, small_w), dtype=np.uint8)

                # Resize with interpolation for smooth noise effect
                background[:, :, c] = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_CUBIC)

        elif background_type == "checkered":
            # Create a checkered pattern

            # Choose random colors for the pattern
            color1 = [random.randint(0, 255) for _ in range(3)]
            color2 = [random.randint(0, 255) for _ in range(3)]

            # Choose random checker size
            checker_size = random.randint(20, 100)

            # Create checker pattern
            y_grid, x_grid = np.indices((height, width))
            checker = ((y_grid // checker_size) % 2) ^ ((x_grid // checker_size) % 2)

            for c in range(3):
                background[:, :, c] = np.where(checker, color1[c], color2[c])

        elif background_type == "striped":
            # Create striped pattern

            # Choose random colors for stripes
            color1 = [random.randint(0, 255) for _ in range(3)]
            color2 = [random.randint(0, 255) for _ in range(3)]

            # Choose stripe width and orientation
            stripe_width = random.randint(10, 50)
            if random.choice([True, False]):
                # Horizontal stripes
                y_grid = np.indices((height, width))[0]
                stripes = (y_grid // stripe_width) % 2
            else:
                # Vertical stripes
                x_grid = np.indices((height, width))[1]
                stripes = (x_grid // stripe_width) % 2

            for c in range(3):
                background[:, :, c] = np.where(stripes, color1[c], color2[c])

        return background

    def transform(self, results: dict) -> dict | None:
        """Apply SimpleCopyPaste augmentation."""
        # Initialize on first use (for multi-worker safety)
        if not self._initialized:
            self._lazy_init()

        # Generate random background if enabled
        if self.use_random_background and random.random() < self.random_background_prob:
            height, width = results["img"].shape[:2]
            # Replace the original image with a randomly generated background
            results["img"] = self._generate_random_background(height, width)

            # Create empty annotations when using a random background
            # (no annotations for the background itself)
            results["gt_bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
            results["gt_bboxes_labels"] = np.array([], dtype=np.int64)
            results["gt_masks"] = np.zeros((0, height, width), dtype=np.uint8)
            results["gt_ignore_flags"] = np.array([], dtype=bool)

        # Apply foreground pasting if probability check passes
        if random.random() < self.paste_prob:
            results = self._paste_new_foreground(results)

        if results is None:
            return None

        results["img_shape"] = results["img"].shape[:2]
        return results

    @staticmethod
    def _crop_mask_with_bbox(mask):
        # Find the bounding box of non-zero values
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Check if mask is empty
        if not np.any(rows) or not np.any(cols):
            # Return empty mask and invalid bbox indicator
            return np.zeros((0, 0), dtype=mask.dtype), None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Crop the mask
        cropped_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

        # Return the cropped mask and bounding box
        return cropped_mask, [x1, y1, x2, y2]

    @staticmethod
    def _compute_bbox_from_mask(mask: np.ndarray) -> list[float] | None:
        """
        Compute tight bounding box from binary mask.

        Args:
            mask: Binary mask array (H, W) with values 0 or 1

        Returns:
            bbox: [x1, y1, x2, y2] coordinates of tight bounding box, or None if mask is empty
        """
        # Find all non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return [float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)]
        else:
            # Empty mask - return None instead of invalid bbox
            return None

    def _paste_new_foreground(self, results: dict) -> dict:
        img = results["img"]
        height, width = img.shape[:2]

        pasted_bboxes = []
        pasted_labels = []
        pasted_masks = []

        # Use deterministic counts if object_counts is specified
        if self.object_counts:
            # Create a list of required objects based on counts/probabilities
            required_objects = []

            # Check if 'all' key is present for setting counts for all categories
            if "all" in self.object_counts:
                all_value = self.object_counts["all"]
                # Need to ensure we're initialized to get available categories
                if not self._initialized:
                    self._lazy_init()

                # Get all available categories from the COCO data
                if self._images_by_category:
                    available_categories = [cat for cat in self._images_by_category if self._images_by_category[cat]]
                    for category in available_categories:
                        # Skip if category has its own specific count (specific counts override 'all')
                        if category in self.object_counts and category != "all":
                            continue

                        if all_value >= 1:
                            count = round(all_value)
                            required_objects.extend([category] * count)
                        elif 0 < all_value < 1:
                            if random.random() < all_value:
                                required_objects.append(category)

            # Process individual category counts (these override 'all' setting)
            for category, value in self.object_counts.items():
                if category == "all":
                    continue  # Already handled above

                if value >= 1:
                    # Treat as exact count (round if float)
                    count = round(value)
                    required_objects.extend([category] * count)
                elif 0 < value < 1:
                    # Treat as probability of pasting one object
                    if random.random() < value:
                        required_objects.append(category)
                elif value == 0:
                    # Skip this category
                    continue
                else:
                    logger.warning(f"Invalid value {value} for category {category}. Values must be >= 0.")

            # Shuffle to randomize order
            random.shuffle(required_objects)

            # Paste each required object
            for category in required_objects:
                coco_obj = self._get_specific_object_from_coco(category)
                if coco_obj is None:
                    if self.verbose:
                        logger.warning(f"Could not find object of category {category}")
                    continue

                # Process the COCO object
                paste_result = self._process_coco_object(coco_obj, img, height, width)

                if paste_result is None:
                    continue

                # Unpack the paste results
                new_bbox, mm_class_index, full_mask, mapped_class_name = paste_result

                # Add to our collection of pasted objects
                pasted_bboxes.append(new_bbox)
                pasted_labels.append(mm_class_index)
                pasted_masks.append(full_mask)

                if self.verbose:
                    logger.info(f"Successfully pasted object of class {mapped_class_name}")
        else:
            # Original behavior: random selection up to max_paste_objects
            for _ in range(self.max_paste_objects):
                # Get random object from COCO annotations
                coco_obj = self._get_random_object_from_coco()
                if coco_obj is None:
                    continue

                # Process the COCO object
                paste_result = self._process_coco_object(coco_obj, img, height, width)

                if paste_result is None:
                    continue

                # Unpack the paste results
                new_bbox, mm_class_index, full_mask, mapped_class_name = paste_result

                # Add to our collection of pasted objects
                pasted_bboxes.append(new_bbox)
                pasted_labels.append(mm_class_index)
                pasted_masks.append(full_mask)

                if self.verbose:
                    logger.info(f"Successfully pasted object of class {mapped_class_name}")

        # Log what objects were successfully pasted before combining
        if pasted_labels:
            logger.info(f"📋 Objects successfully pasted: {len(pasted_labels)} objects")
            self._log_object_counts(np.array(pasted_labels), stage="Pasted")

        # Combine new and existing annotations
        results = self._combine_annotations(results, pasted_bboxes, pasted_labels, pasted_masks, height, width)

        return results

    def _process_coco_object(
        self, coco_obj: dict[str, Any], img: np.ndarray, height: int, width: int
    ) -> tuple[list[float], int, np.ndarray, str] | None:
        """Process a COCO object for pasting."""
        annotation = coco_obj["annotation"]
        image_info = coco_obj["image"]
        category = coco_obj["category"]

        # Build image path
        image_filename = image_info["file_name"]

        # Check if image_filename is already an absolute path or needs to be resolved
        if os.path.isabs(image_filename):
            image_path = image_filename
        elif self.annotation_file:
            # Try to find the data directory relative to annotation file
            ann_dir = os.path.dirname(self.annotation_file)
            # Navigate up to find the data directory, then go to images
            data_dir = ann_dir
            while data_dir != "/" and not data_dir.endswith("/data"):
                data_dir = os.path.dirname(data_dir)
            if data_dir.endswith("/data"):
                image_path = os.path.join(data_dir, "images", image_filename)
            else:
                # Use environment variable or construct a default relative to the module
                image_base_dir = os.environ.get("IMAGE_BASE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "images"))
                image_path = os.path.join(image_base_dir, image_filename)
        else:
            # Use environment variable or construct a default relative to the module
            image_base_dir = os.environ.get("IMAGE_BASE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "images"))
            image_path = os.path.join(image_base_dir, image_filename)

        # Load the source image
        if hasattr(self, "_image_cache") and self._image_cache is not None:
            source_img = self._image_cache(image_path)
        else:
            source_img = self._load_image_cached(image_path)

        if source_img is None or source_img.size == 0:
            return None

        # Get mask from annotation
        original_mask = self._annotation_to_mask(annotation, image_info["height"], image_info["width"])
        if original_mask is None:
            return None

        # Crop the mask and foreground image
        cropped_mask, bbox = self._crop_mask_with_bbox(original_mask)
        if bbox is None:
            logger.warning(f"Empty mask for object of class {category}, skipping")
            return None

        x1, y1, x2, y2 = bbox
        cropped_foreground = source_img[y1 : y2 + 1, x1 : x2 + 1]

        # Resize and rotate the object
        processed_object = self._resize_and_rotate_object(cropped_mask, cropped_foreground, height, width)
        if processed_object is None:
            return None

        cropped_mask, cropped_foreground, new_height, new_width = processed_object

        # Calculate the offset for pasting
        y_offset = random.randint(0, max(0, height - new_height))
        x_offset = random.randint(0, max(0, width - new_width))

        # Paste the object onto the image
        paste_success, modified_img = self._paste_object_onto_image(img, cropped_mask, cropped_foreground, y_offset, x_offset, new_height, new_width)

        if not paste_success:
            return None

        # Update the image with the modified version
        img[:] = modified_img[:]

        # Map the class name if necessary
        mapped_class_name = self._map_class_name(category)

        # Get the class index
        if mapped_class_name is None:
            logger.warning("Mapped class name is None. Skipping this object.")
            return None
        mm_class_index = self.class_mapping.get(mapped_class_name)
        if mm_class_index is None:
            logger.warning(f"Class {mapped_class_name} not found in MMDetection classes. Skipping this object.")
            return None

        # Create a full-sized mask with boundary checking
        full_mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the actual region we can use (handle edge cases)
        actual_height = min(new_height, height - y_offset)
        actual_width = min(new_width, width - x_offset)

        # Ensure we don't have invalid dimensions
        if actual_height > 0 and actual_width > 0:
            # Only use the portion of cropped_mask that fits
            full_mask[y_offset : y_offset + actual_height, x_offset : x_offset + actual_width] = cropped_mask[:actual_height, :actual_width]

        # Compute tight bounding box from the actual transformed mask
        new_bbox = self._compute_bbox_from_mask(full_mask)

        # Skip if bbox is invalid (empty mask)
        if new_bbox is None:
            logger.warning(f"Skipping object of class {mapped_class_name} - empty mask after transformation")
            return None

        return new_bbox, mm_class_index, full_mask, mapped_class_name

    def _annotation_to_mask(self, annotation: dict[str, Any], img_height: int, img_width: int) -> np.ndarray | None:
        """Convert COCO annotation to binary mask."""
        try:
            if "segmentation" in annotation:
                seg = annotation["segmentation"]
                if isinstance(seg, list):
                    # Polygon format
                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    for polygon in seg:
                        if len(polygon) >= 6:  # Need at least 3 points
                            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [points], (1,))
                    return mask
                elif isinstance(seg, dict):
                    # RLE format
                    if "counts" in seg and "size" in seg:
                        # Create proper RLE object for pycocotools
                        rle = {"counts": seg["counts"], "size": seg["size"]}
                        mask = mask_utils.decode(rle)  # pyright: ignore[reportArgumentType]
                        return mask.astype(np.uint8)

            # If no segmentation, create mask from bbox
            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                mask[y1:y2, x1:x2] = 1
                return mask

        except Exception as e:
            logger.error(f"Error converting annotation to mask: {e}")

        return None

    def _resize_and_rotate_object(self, cropped_mask, cropped_foreground, height, width):
        """Resize and rotate a foreground object."""
        new_height = None
        new_width = None
        try:
            # Apply random scale from scale_range
            orig_height, orig_width = cropped_mask.shape[:2]
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            new_height = int(orig_height * scale_factor)
            new_width = int(orig_width * scale_factor)

            # Ensure the new size doesn't exceed the image dimensions
            new_height = min(new_height, height)
            new_width = min(new_width, width)

            cropped_mask = cv2.resize(cropped_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            cropped_foreground = cv2.resize(cropped_foreground, (new_width, new_height))

            # Apply random rotation from rotation_range
            rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), rotation_angle, 1.0)

            # Calculate new bounding rect after rotation to ensure rotated image fits
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_width_rot = int((new_height * sin) + (new_width * cos))
            new_height_rot = int((new_height * cos) + (new_width * sin))

            # Adjust translation to ensure the entire rotated image is visible
            rotation_matrix[0, 2] += (new_width_rot / 2) - new_width / 2
            rotation_matrix[1, 2] += (new_height_rot / 2) - new_height / 2

            # Apply rotation to both mask and foreground
            cropped_mask = cv2.warpAffine(
                cropped_mask,
                rotation_matrix,
                (new_width_rot, new_height_rot),
                flags=cv2.INTER_NEAREST,
                borderValue=(0,),
            )
            cropped_foreground = cv2.warpAffine(
                cropped_foreground,
                rotation_matrix,
                (new_width_rot, new_height_rot),
                flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0),
            )

            # Update dimensions after rotation
            new_height, new_width = cropped_mask.shape[:2]

            return cropped_mask, cropped_foreground, new_height, new_width

        except cv2.error as e:
            logger.warning(f"Exception: {e}")
            logger.warning(f"Invalid target dimensions: target_height={new_height}, target_width={new_width}")
            return None

    def remap_add(self, x, y):
        # this function can be any function, f: (0,1)x(0,1) -> (0,1) where  f(x,0) = x; f(x,y) = f(y,x); and f(x,y) >= max(x,y)
        #
        # a nice pattern for generating these is to pick a monotonically increasing function g: (0,1) -> (0,inf)
        # then f(x,y) := g^-1 (g(x) + g(y))

        # rx = mapf(x)
        # ry = mapf(y)

        # rz = rx + ry

        # z = mapinv(rz)

        # return z
        return y + x - x * y

    def _paste_object_onto_image(
        self, img: np.ndarray, mask: np.ndarray, foreground: np.ndarray, y_offset: int, x_offset: int, height: int, width: int
    ) -> tuple[bool, np.ndarray]:
        """Paste a foreground object onto the background image.

        Args:
            img: The background image to paste onto
            mask: The mask of the foreground object
            foreground: The foreground object image
            y_offset: Vertical offset for pasting
            x_offset: Horizontal offset for pasting
            height: Height of the foreground object
            width: Width of the foreground object

        Returns:
            tuple: (success, modified_image) where success is a boolean indicating if the paste was successful,
                   and modified_image is the resulting image after pasting
        """
        # Create a copy of the image to avoid modifying the original
        result_img = img.copy()

        # Get image dimensions
        img_height, img_width = img.shape[:2]

        # Calculate the actual region we can paste (handle edge cases)
        actual_height = min(height, img_height - y_offset)
        actual_width = min(width, img_width - x_offset)

        # Ensure we don't have negative dimensions
        if actual_height <= 0 or actual_width <= 0:
            logger.warning(f"Invalid paste region: offset ({y_offset}, {x_offset}), size ({height}, {width}), image ({img_height}, {img_width})")
            return False, img

        # Crop mask and foreground to the actual region
        mask_cropped = mask[:actual_height, :actual_width]
        foreground_cropped = foreground[:actual_height, :actual_width]

        try:
            if self.blend_mode == "xray":
                # X-ray paste mode
                normalized_mask = mask_cropped / 255
                normalized_foreground = foreground_cropped / 255
                invert_foreground = 1 - normalized_foreground

                # Ensure mask has correct shape for broadcasting
                if normalized_mask.ndim == 2:
                    normalized_mask = normalized_mask[:, :, np.newaxis]

                masked_battery = normalized_mask * invert_foreground

                # Get the target region from the image
                target_region = result_img[y_offset : y_offset + actual_height, x_offset : x_offset + actual_width]
                invert_base = 1 - target_region / 255

                # Apply the xray blending
                added = self.remap_add(masked_battery, invert_base)
                result_img[y_offset : y_offset + actual_height, x_offset : x_offset + actual_width] = (1 - added) * 255
                return True, result_img
            else:
                # Standard paste mode (normal)
                valid_mask = mask_cropped > 0

                # Get the target region
                target_region = result_img[y_offset : y_offset + actual_height, x_offset : x_offset + actual_width]

                # Ensure mask has correct shape for broadcasting
                if valid_mask.ndim == 2:
                    valid_mask = valid_mask[:, :, np.newaxis]

                # Apply the mask
                result_img[y_offset : y_offset + actual_height, x_offset : x_offset + actual_width] = np.where(
                    valid_mask,
                    foreground_cropped,
                    target_region,
                )
                return True, result_img
        except ValueError as e:
            logger.error(f"Pasting error: {e}")
            logger.error(f"Shapes - mask: {mask_cropped.shape}, foreground: {foreground_cropped.shape}")
            return False, img  # Return original image on failure

    def _map_class_name(self, original_class_name: str) -> str:
        """Map a class name using the class_name_mapping if provided."""
        if self.class_name_mapping:
            return self.class_name_mapping.get(original_class_name, original_class_name)
        return original_class_name

    def _track_stage_counts(self, labels: np.ndarray, stage: str) -> None:
        """Track object counts for a specific pipeline stage."""
        if len(labels) == 0:
            self._stage_counts[stage] = {}
            return

        # Count objects by class
        unique_labels, counts = np.unique(labels, return_counts=True)
        stage_counts = {}

        for label_idx, count in zip(unique_labels, counts, strict=False):
            if 0 <= label_idx < len(self.mm_class_list):
                class_name = self.mm_class_list[label_idx]
                stage_counts[class_name] = count

        self._stage_counts[stage] = stage_counts

    def _log_consolidated_counts(self) -> None:
        """Log a consolidated table showing counts across all pipeline stages."""
        if not self._stage_counts:
            logger.info("CustomCopyPaste: No object counts to display")
            return

        # Collect all class names that appear in any stage
        all_classes = set()
        for stage_counts in self._stage_counts.values():
            all_classes.update(stage_counts.keys())

        if not all_classes:
            logger.info("CustomCopyPaste: No objects found in pipeline")
            return

        # Sort classes alphabetically for consistent display
        sorted_classes = sorted(all_classes)
        stages = list(self._stage_counts.keys())

        # Calculate column widths
        class_width = max(25, max(len(cls) for cls in sorted_classes) + 2)
        stage_width = 10

        # Create consolidated table
        table_rows = ["CustomCopyPaste Pipeline Object Counts:"]
        table_rows.append("=" * (class_width + len(stages) * stage_width + 15))

        # Header row
        header = f"{'Class Name':<{class_width}}"
        for stage in stages:
            header += f"{stage[:8]:<{stage_width}}"
        header += f"{'Class ID':<8}"
        table_rows.append(header)
        table_rows.append("-" * (class_width + len(stages) * stage_width + 15))

        # Data rows
        stage_totals = dict.fromkeys(stages, 0)
        for class_name in sorted_classes:
            # Get class ID
            class_id = self.class_mapping.get(class_name, -1)

            row = f"{class_name:<{class_width}}"
            for stage in stages:
                count = self._stage_counts[stage].get(class_name, 0)
                row += f"{count:<{stage_width}}"
                stage_totals[stage] += count
            row += f"{class_id:<8}"
            table_rows.append(row)

        # Total row
        table_rows.append("-" * (class_width + len(stages) * stage_width + 15))
        total_row = f"{'TOTAL':<{class_width}}"
        for stage in stages:
            total_row += f"{stage_totals[stage]:<{stage_width}}"
        table_rows.append(total_row)
        table_rows.append("=" * (class_width + len(stages) * stage_width + 15))

        # Log the complete table
        # Shows how many examples of different objects end up in the image
        logger.debug("\n".join(table_rows))

    def _log_object_counts(self, labels: np.ndarray, stage: str = "Final Output") -> None:
        """Log the count of each object class at different pipeline stages."""
        # Track counts for consolidated logging
        self._track_stage_counts(labels, stage)

        # For backward compatibility, still log individual stage if verbose
        if self.verbose and len(labels) > 0:
            # Count objects by class
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Create a table format for clear visibility
            table_rows = [f"CustomCopyPaste {stage} Object Counts:"]
            table_rows.append("=" * 60)
            table_rows.append(f"{'Class Name':<25} {'Count':<8} {'Class ID':<10}")
            table_rows.append("-" * 60)

            total_count = 0

            for label_idx, count in zip(unique_labels, counts, strict=False):
                if 0 <= label_idx < len(self.mm_class_list):
                    class_name = self.mm_class_list[label_idx]
                    table_rows.append(f"{class_name:<25} {count:<8} {label_idx:<10}")
                    total_count += count
                else:
                    table_rows.append(f"Unknown class:<25 {count:<8} {label_idx:<10}")
                    total_count += count

            table_rows.append("-" * 60)
            table_rows.append(f"{'TOTAL':<25} {total_count:<8}")
            table_rows.append("=" * 60)

            # Log the complete table
            logger.info("\n".join(table_rows))

    def _check_overlap(self, new_bbox: list[float], existing_bboxes: list[list[float]]) -> bool:
        """Check if new bbox overlaps too much with existing ones."""
        if not existing_bboxes or self.overlap_thresh >= 1.0:
            return False

        new_x1, new_y1, new_x2, new_y2 = new_bbox
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)

        for bbox in existing_bboxes:
            x1, y1, x2, y2 = bbox

            # Calculate intersection
            inter_x1 = max(new_x1, x1)
            inter_y1 = max(new_y1, y1)
            inter_x2 = min(new_x2, x2)
            inter_y2 = min(new_y2, y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                bbox_area = (x2 - x1) * (y2 - y1)

                # Calculate IoU
                union_area = new_area + bbox_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou > self.overlap_thresh:
                    return True

        return False

    def _combine_annotations(self, results, pasted_bboxes, pasted_labels, pasted_masks, height, width):
        """Combine existing and new annotations."""
        # Handle both visdet structures and plain tensors for backward compatibility
        if hasattr(results["gt_bboxes"], "tensor"):
            existing_bboxes = results["gt_bboxes"].tensor
        else:
            existing_bboxes = results["gt_bboxes"]

        existing_labels = results["gt_bboxes_labels"]

        if hasattr(results["gt_masks"], "masks"):
            existing_masks = results["gt_masks"].masks
        else:
            existing_masks = results["gt_masks"]

        existing_ignore_flags = results["gt_ignore_flags"]

        # Log existing objects before combining
        if len(existing_labels) > 0:
            self._log_object_counts(existing_labels, stage="Existing")

        if len(pasted_bboxes) > 0:
            # Filter out invalid bboxes (zero width or height)
            valid_indices = []
            for i, bbox in enumerate(pasted_bboxes):
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:  # Valid bbox
                    valid_indices.append(i)
                else:
                    logger.warning(f"Filtering out invalid bbox: {bbox}")

            if len(valid_indices) > 0:
                # Use only valid bboxes, labels, and masks
                valid_bboxes = [pasted_bboxes[i] for i in valid_indices]
                valid_labels = [pasted_labels[i] for i in valid_indices]
                valid_masks = [pasted_masks[i] for i in valid_indices]

                new_bboxes = torch.tensor(valid_bboxes, dtype=torch.float32)
                combined_bboxes = torch.cat([existing_bboxes, new_bboxes])
                combined_labels = np.concatenate([existing_labels, np.array(valid_labels, dtype=np.int64)])
            else:
                # All pasted bboxes were invalid
                logger.warning("All pasted bboxes were invalid, keeping only existing annotations")
                combined_bboxes = existing_bboxes
                combined_labels = existing_labels
                combined_masks = existing_masks
                combined_ignore_flags = existing_ignore_flags

                results["gt_bboxes"] = combined_bboxes
                results["gt_bboxes_labels"] = combined_labels
                results["gt_masks"] = combined_masks
                results["gt_ignore_flags"] = combined_ignore_flags

                # Log final combined object counts and consolidated view
                # Always call these methods - they will use appropriate log levels internally
                if os.environ.get("LOG_LEVEL", None) == "DEBUG":
                    self._log_object_counts(combined_labels, stage="Final")
                    self._log_consolidated_counts()

                # Reset stage counts for next image
                self._stage_counts.clear()

                return results

            # Convert valid_masks to numpy array with proper shape
            valid_masks_array = np.array(valid_masks)
            if valid_masks_array.ndim == 1:
                # If valid_masks is a list of 2D arrays, stack them
                valid_masks_array = np.stack(valid_masks)

            combined_masks = np.concatenate([existing_masks, valid_masks_array])
            combined_ignore_flags = np.concatenate([existing_ignore_flags, np.zeros(len(valid_bboxes), dtype=bool)])
        else:
            # No new objects pasted, keep existing annotations
            combined_bboxes = existing_bboxes
            combined_labels = existing_labels
            combined_masks = existing_masks
            combined_ignore_flags = existing_ignore_flags

        results["gt_bboxes"] = combined_bboxes
        results["gt_bboxes_labels"] = combined_labels
        results["gt_masks"] = combined_masks
        results["gt_ignore_flags"] = combined_ignore_flags

        # Log final combined object counts and consolidated view
        self._log_object_counts(combined_labels, stage="Final")
        self._log_consolidated_counts()

        # Reset stage counts for next image
        self._stage_counts.clear()

        return results

    @property
    def _annotation_file(self) -> str | None:
        """Property to allow annotation_file to be set externally."""
        return self.annotation_file

    @_annotation_file.setter
    def _annotation_file(self, value: str) -> None:
        """Set annotation file and reset initialization."""
        self.annotation_file = value
        self._initialized = False
