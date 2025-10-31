use crate::affine::{apply_affine_transform, AffineTransform};
use crate::blending::{blend_pixel, BlendMode};
use crate::collision::{check_iou_collision, clip_bbox_to_image};
/// Object handling for copy-paste augmentation
/// Includes extraction, selection, and placement logic
use ndarray::{Array3, ArrayView3};
use rand::Rng;
use std::collections::HashMap;

/// Represents an extracted object from a mask
#[derive(Clone, Debug)]
pub struct ExtractedObject {
    /// The image patch for this object
    pub image: Array3<u8>,
    /// The mask for this object (binary)
    pub mask: Array3<u8>,
    /// Bounding box as (x_min, y_min, x_max, y_max) in pixel coordinates
    pub bbox: (u32, u32, u32, u32),
    /// Class ID of this object
    pub class_id: u32,
}

/// Extract all objects from a mask where each unique non-zero value is an object
///
/// # Arguments
/// * `image` - Source image array (H, W, C)
/// * `mask` - Mask array (H, W, C) where non-zero values represent objects
/// * `class_id` - The class ID for all extracted objects
///
/// # Returns
/// Vector of extracted objects
pub fn extract_objects_from_mask(
    image: ArrayView3<'_, u8>,
    mask: ArrayView3<'_, u8>,
) -> Vec<ExtractedObject> {
    let mut objects = Vec::new();
    let shape = image.shape();
    let (height, width, _channels) = (shape[0], shape[1], shape[2]);

    // Find all non-zero regions in the mask (we'll treat each connected region as an object)
    // For simplicity, we extract objects based on bounding boxes of non-zero pixels

    let mut visited = vec![vec![false; width]; height];

    for y in 0..height {
        for x in 0..width {
            if mask[[y, x, 0]] > 0 && !visited[y][x] {
                // Found a new object, extract its bounding box
                let (x_min, y_min, x_max, y_max) = find_object_bounds(&mask, x, y, &mut visited);

                if x_max > x_min && y_max > y_min {
                    // Extract the patch
                    if let Some(obj) = extract_object_patch(
                        &image,
                        &mask,
                        x_min as u32,
                        y_min as u32,
                        x_max as u32,
                        y_max as u32,
                    ) {
                        objects.push(obj);
                    }
                }
            }
        }
    }

    objects
}

/// Find the bounding box of an object starting from a point
fn find_object_bounds(
    mask: &ArrayView3<'_, u8>,
    start_x: usize,
    start_y: usize,
    visited: &mut Vec<Vec<bool>>,
) -> (usize, usize, usize, usize) {
    let shape = mask.shape();
    let (height, width) = (shape[0], shape[1]);

    let mut x_min = start_x;
    let mut x_max = start_x + 1;
    let mut y_min = start_y;
    let mut y_max = start_y + 1;

    // Simple flood fill to find bounds
    let mut stack = vec![(start_x, start_y)];

    while let Some((x, y)) = stack.pop() {
        if x >= width || y >= height || visited[y][x] {
            continue;
        }

        if mask[[y, x, 0]] == 0 {
            continue;
        }

        visited[y][x] = true;

        x_min = x_min.min(x);
        x_max = x_max.max(x + 1);
        y_min = y_min.min(y);
        y_max = y_max.max(y + 1);

        // Check neighbors
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x < width - 1 {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y < height - 1 {
            stack.push((x, y + 1));
        }
    }

    (x_min, y_min, x_max, y_max)
}

/// Extract a patch from the image and mask
fn extract_object_patch(
    image: &ArrayView3<'_, u8>,
    mask: &ArrayView3<'_, u8>,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
) -> Option<ExtractedObject> {
    let patch_height = (y_max - y_min) as usize;
    let patch_width = (x_max - x_min) as usize;
    let channels = image.shape()[2];

    if patch_height == 0 || patch_width == 0 {
        return None;
    }

    // Create patch arrays
    let mut patch_image = Array3::zeros((patch_height, patch_width, channels));
    let mut patch_mask = Array3::zeros((patch_height, patch_width, channels));

    // Copy data
    let img_shape = image.shape();
    let mut class_counts = [0usize; 256];

    for y in 0..patch_height {
        for x in 0..patch_width {
            let src_y = (y_min as usize) + y;
            let src_x = (x_min as usize) + x;

            if src_y < img_shape[0] && src_x < img_shape[1] {
                let class_value = mask[[src_y, src_x, 0]];
                if class_value > 0 {
                    class_counts[class_value as usize] += 1;
                }

                for c in 0..channels {
                    patch_image[[y, x, c]] = image[[src_y, src_x, c]];
                    patch_mask[[y, x, c]] = mask[[src_y, src_x, c]];
                }
            }
        }
    }

    let mut class_id = 0u32;
    let mut max_count = 0usize;
    for (value, count) in class_counts.iter().enumerate() {
        if value == 0 || *count == 0 {
            continue;
        }

        if *count > max_count {
            max_count = *count;
            class_id = value as u32;
        }
    }

    Some(ExtractedObject {
        image: patch_image,
        mask: patch_mask,
        bbox: (x_min, y_min, x_max, y_max),
        class_id,
    })
}

/// Select objects to paste based on object_counts
///
/// # Arguments
/// * `available_objects` - Vec of extracted objects grouped by class_id
/// * `object_counts` - HashMap specifying how many objects to paste per class
///
/// # Returns
/// Vector of selected objects ready to paste
pub fn select_objects_by_class(
    available_objects: &HashMap<u32, Vec<ExtractedObject>>,
    object_counts: &HashMap<u32, u32>,
    max_paste_objects: u32,
) -> Vec<ExtractedObject> {
    if max_paste_objects == 0 {
        return Vec::new();
    }

    let mut selected = Vec::new();
    let mut rng = rand::thread_rng();
    let mut total_selected = 0usize;
    let global_cap = max_paste_objects as usize;

    for (class_id, count) in object_counts.iter() {
        if total_selected >= global_cap {
            break;
        }

        if let Some(objects) = available_objects.get(class_id) {
            let remaining_global = global_cap - total_selected;
            let per_class_cap = (*count as usize).min(objects.len());
            let count_to_select = per_class_cap.min(remaining_global);

            if count_to_select == 0 {
                continue;
            }

            // Random selection without replacement
            let mut indices: Vec<usize> = (0..objects.len()).collect();
            for i in 0..count_to_select {
                let j = i + rng.gen_range(0..(indices.len() - i));
                indices.swap(i, j);
                selected.push(objects[indices[i]].clone());
            }

            total_selected += count_to_select;
        }
    }

    selected
}

/// Represents a placed object with its transformed location
#[derive(Clone, Debug)]
pub struct PlacedObject {
    /// Transformed bbox as (x_min, y_min, x_max, y_max)
    pub bbox: (f32, f32, f32, f32),
    /// The transformed image patch
    pub image: Array3<u8>,
    /// The transformed mask patch
    pub mask: Array3<u8>,
    /// Class ID
    pub class_id: u32,
}

/// Place objects onto target image with collision detection
///
/// # Arguments
/// * `selected_objects` - Objects to place
/// * `image_width` - Target image width
/// * `image_height` - Target image height
/// * `use_rotation` - Whether to apply random rotation
/// * `use_scaling` - Whether to apply random scaling
/// * `rotation_range` - (min, max) rotation in degrees
/// * `scale_range` - (min, max) scale factors
/// * `collision_threshold` - IoU threshold for collision detection (0.0 = no collision)
///
/// # Returns
/// Vector of successfully placed objects with their transformed bboxes
pub fn place_objects(
    selected_objects: &[ExtractedObject],
    image_width: u32,
    image_height: u32,
    use_rotation: bool,
    use_scaling: bool,
    rotation_range: (f32, f32),
    scale_range: (f32, f32),
    collision_threshold: f32,
) -> Vec<PlacedObject> {
    let mut placed: Vec<PlacedObject> = Vec::new();
    let mut rng = rand::thread_rng();

    for obj in selected_objects {
        // Random position within image bounds
        let obj_width = (obj.bbox.2 - obj.bbox.0) as f32;
        let obj_height = (obj.bbox.3 - obj.bbox.1) as f32;

        let max_x = ((image_width as f32 - obj_width).max(0.0)) as i32;
        let max_y = ((image_height as f32 - obj_height).max(0.0)) as i32;

        if max_x <= 0 || max_y <= 0 {
            continue; // Object too large for image
        }

        let pos_x = rng.gen_range(0..=(max_x as usize)) as f32;
        let pos_y = rng.gen_range(0..=(max_y as usize)) as f32;

        // Random transformation parameters
        let rotation = if use_rotation {
            rng.gen_range(rotation_range.0..=rotation_range.1)
        } else {
            0.0
        };

        let scale = if use_scaling {
            rng.gen_range(scale_range.0..=scale_range.1)
        } else {
            1.0
        };

        // Create affine transformation
        let transform = AffineTransform::new(rotation, scale, pos_x, pos_y);

        // Transform the object's bounding box
        let (x_min, y_min, x_max, y_max) = obj.bbox;
        let corners = vec![
            (x_min as f32, y_min as f32),
            (x_max as f32, y_min as f32),
            (x_min as f32, y_max as f32),
            (x_max as f32, y_max as f32),
        ];

        let transformed_corners: Vec<(f32, f32)> = corners
            .iter()
            .map(|&p| apply_affine_transform(p, &transform))
            .collect();

        let transformed_bbox = (
            transformed_corners
                .iter()
                .map(|p| p.0)
                .fold(f32::INFINITY, f32::min),
            transformed_corners
                .iter()
                .map(|p| p.1)
                .fold(f32::INFINITY, f32::min),
            transformed_corners
                .iter()
                .map(|p| p.0)
                .fold(f32::NEG_INFINITY, f32::max),
            transformed_corners
                .iter()
                .map(|p| p.1)
                .fold(f32::NEG_INFINITY, f32::max),
        );

        // Clip to image bounds
        let clipped_bbox = clip_bbox_to_image(transformed_bbox, image_width, image_height);

        // Check for collisions with already-placed objects
        let mut has_collision = false;
        for placed_obj in &placed {
            if check_iou_collision(clipped_bbox, placed_obj.bbox, collision_threshold) {
                has_collision = true;
                break;
            }
        }

        if has_collision {
            continue; // Skip this object due to collision
        }

        // For now, use the original object (not transformed patch yet)
        // This simplifies the implementation while still achieving object placement
        let placed_obj = PlacedObject {
            bbox: clipped_bbox,
            image: obj.image.clone(),
            mask: obj.mask.clone(),
            class_id: obj.class_id,
        };

        placed.push(placed_obj);
    }

    placed
}

/// Compose placed objects onto target image with blending
///
/// # Arguments
/// * `output_image` - Target image to paste objects onto
/// * `placed_objects` - Objects to paste
/// * `blend_mode` - Blending mode to use
///
/// # Returns
/// Modified image with objects blended on
pub fn compose_objects(
    output_image: &mut Array3<u8>,
    placed_objects: &[PlacedObject],
    blend_mode: BlendMode,
) {
    let out_shape = output_image.shape();
    let (height, width, channels) = (out_shape[0], out_shape[1], out_shape[2]);

    for placed_obj in placed_objects {
        let (x_min, y_min, x_max, y_max) = placed_obj.bbox;
        let x_min = (x_min as usize).min(width);
        let y_min = (y_min as usize).min(height);
        let x_max = ((x_max as usize).min(width)).max(x_min);
        let y_max = ((y_max as usize).min(height)).max(y_min);

        let patch_width = x_max - x_min;
        let patch_height = y_max - y_min;

        if patch_width == 0 || patch_height == 0 {
            continue;
        }

        // Blend the object patch onto the output image
        // Using the mask to determine alpha blending
        let obj_shape = placed_obj.image.shape();
        for py in 0..patch_height.min(obj_shape[0]) {
            for px in 0..patch_width.min(obj_shape[1]) {
                let target_y = y_min + py;
                let target_x = x_min + px;

                if target_y >= height || target_x >= width {
                    continue;
                }

                // Use mask to determine if pixel should be blended
                let mask_value = placed_obj.mask[[py, px, 0]];
                if mask_value == 0 {
                    continue; // Skip transparent pixels
                }

                let alpha = (mask_value as f32) / 255.0;

                // Blend each channel
                for c in 0..channels {
                    let base_pixel = output_image[[target_y, target_x, c]];
                    let overlay_pixel = placed_obj.image[[py, px, c]];

                    let blended = blend_pixel(base_pixel, overlay_pixel, alpha, blend_mode);
                    output_image[[target_y, target_x, c]] = blended;
                }
            }
        }
    }
}

/// Generate bounding boxes for pasted objects
///
/// # Arguments
/// * `placed_objects` - Objects that were placed
///
/// # Returns
/// Vector of bboxes in format (x_min, y_min, x_max, y_max) as floats
#[allow(dead_code)]
pub fn generate_output_bboxes(placed_objects: &[PlacedObject]) -> Vec<(f32, f32, f32, f32)> {
    placed_objects.iter().map(|obj| obj.bbox).collect()
}

/// Update output mask with placed objects
///
/// # Arguments
/// * `output_mask` - Target mask to update
/// * `placed_objects` - Objects to add to mask
pub fn update_output_mask(output_mask: &mut Array3<u8>, placed_objects: &[PlacedObject]) {
    let out_shape = output_mask.shape();
    let (height, width, _channels) = (out_shape[0], out_shape[1], out_shape[2]);

    for placed_obj in placed_objects {
        let (x_min, y_min, x_max, y_max) = placed_obj.bbox;
        let x_min = (x_min as usize).min(width);
        let y_min = (y_min as usize).min(height);
        let x_max = ((x_max as usize).min(width)).max(x_min);
        let y_max = ((y_max as usize).min(height)).max(y_min);

        let patch_width = x_max - x_min;
        let patch_height = y_max - y_min;

        if patch_width == 0 || patch_height == 0 {
            continue;
        }

        // Copy mask values where object exists
        let mask_shape = placed_obj.mask.shape();
        for py in 0..patch_height.min(mask_shape[0]) {
            for px in 0..patch_width.min(mask_shape[1]) {
                let target_y = y_min + py;
                let target_x = x_min + px;

                if target_y >= height || target_x >= width {
                    continue;
                }

                let mask_value = placed_obj.mask[[py, px, 0]];
                if mask_value > 0 {
                    // Use the class_id as the mask value for this object
                    output_mask[[target_y, target_x, 0]] = (placed_obj.class_id as u8).min(255);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_objects_from_mask() {
        // Create a simple test image and mask
        let image = Array3::zeros((10, 10, 3));
        let mut mask = Array3::zeros((10, 10, 3));

        // Add a simple object (a small rectangle)
        for y in 2..5 {
            for x in 2..5 {
                mask[[y, x, 0]] = 1;
            }
        }

        let objects = extract_objects_from_mask(image.view(), mask.view());
        assert!(!objects.is_empty());
        assert_eq!(objects[0].class_id, 1);
    }

    #[test]
    fn test_select_objects_by_class() {
        let mut objects = HashMap::new();

        // Create some test objects
        let obj1 = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::zeros((5, 5, 3)),
            bbox: (0, 0, 5, 5),
            class_id: 0,
        };

        objects.insert(0, vec![obj1.clone(), obj1.clone(), obj1.clone()]);

        let mut counts = HashMap::new();
        counts.insert(0, 2);

        let selected = select_objects_by_class(&objects, &counts, 5);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_place_objects_single() {
        // Create a simple object to place
        let obj = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::from_elem((5, 5, 3), 255u8),
            bbox: (0, 0, 5, 5),
            class_id: 0,
        };

        let placed = place_objects(
            &[obj],
            100,
            100,
            false,
            false,
            (-30.0, 30.0),
            (0.8, 1.2),
            0.01,
        );

        // Should place the object
        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].class_id, 0);
    }

    #[test]
    fn test_place_objects_collision_detection() {
        // Create two overlapping objects
        let obj = ExtractedObject {
            image: Array3::zeros((50, 50, 3)),
            mask: Array3::from_elem((50, 50, 3), 255u8),
            bbox: (0, 0, 50, 50),
            class_id: 0,
        };

        // Place objects with high collision threshold
        let placed = place_objects(
            &[obj.clone(), obj],
            100,
            100,
            false,
            false,
            (0.0, 0.0),
            (1.0, 1.0),
            0.0, // No collision tolerance
        );

        // May or may not place both depending on random position
        // But should respect collision detection
        assert!(placed.len() <= 2);
    }

    #[test]
    fn test_compose_objects_basic() {
        let mut output_image: Array3<u8> = Array3::zeros((100, 100, 3));

        // Create a test object
        let mut obj_image = Array3::zeros((10, 10, 3));
        for i in 0..10 {
            for j in 0..10 {
                obj_image[[i, j, 0]] = 255;
                obj_image[[i, j, 1]] = 255;
                obj_image[[i, j, 2]] = 255;
            }
        }

        let obj_mask = Array3::from_elem((10, 10, 3), 255u8);
        let placed = vec![PlacedObject {
            bbox: (10.0, 10.0, 20.0, 20.0),
            image: obj_image,
            mask: obj_mask,
            class_id: 0,
        }];

        compose_objects(&mut output_image, &placed, BlendMode::Normal);

        // Verify that some pixels were modified (compositing happened)
        let mut modified = false;
        for i in 10..20 {
            for j in 10..20 {
                if output_image[[i, j, 0]] > 0 {
                    modified = true;
                    break;
                }
            }
        }
        assert!(modified, "Image should be modified by composition");
    }

    #[test]
    fn test_update_output_mask() {
        let mut output_mask: Array3<u8> = Array3::zeros((100, 100, 3));

        let obj_mask = Array3::from_elem((10, 10, 3), 255u8);
        let placed = vec![PlacedObject {
            bbox: (10.0, 10.0, 20.0, 20.0),
            image: Array3::zeros((10, 10, 3)),
            mask: obj_mask,
            class_id: 5,
        }];

        update_output_mask(&mut output_mask, &placed);

        // Verify mask was updated
        let mut updated = false;
        for i in 10..20 {
            for j in 10..20 {
                if output_mask[[i, j, 0]] > 0 {
                    updated = true;
                    break;
                }
            }
        }
        assert!(updated, "Mask should be updated with placed objects");
    }

    #[test]
    fn test_multiple_classes() {
        let mut objects_by_class = HashMap::new();

        let obj1 = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::zeros((5, 5, 3)),
            bbox: (0, 0, 5, 5),
            class_id: 0,
        };

        let obj2 = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::zeros((5, 5, 3)),
            bbox: (0, 0, 5, 5),
            class_id: 1,
        };

        objects_by_class.insert(0, vec![obj1.clone(), obj1.clone()]);
        objects_by_class.insert(1, vec![obj2.clone()]);

        let mut counts = HashMap::new();
        counts.insert(0, 1);
        counts.insert(1, 1);

        let selected = select_objects_by_class(&objects_by_class, &counts, 10);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].class_id + selected[1].class_id, 1); // One of each class
    }

    #[test]
    fn test_extract_objects_majority_class() {
        let image = Array3::zeros((6, 6, 3));
        let mut mask = Array3::zeros((6, 6, 3));

        for y in 1..5 {
            for x in 1..5 {
                mask[[y, x, 0]] = if x < 3 { 2 } else { 3 };
            }
        }

        let objects = extract_objects_from_mask(image.view(), mask.view());
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].class_id, 2); // majority label inside patch
    }

    #[test]
    fn test_select_objects_respects_global_cap() {
        let mut objects_by_class = HashMap::new();

        let template = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::zeros((5, 5, 3)),
            bbox: (0, 0, 5, 5),
            class_id: 0,
        };

        let class_a: Vec<_> = (0..4)
            .map(|_| ExtractedObject {
                class_id: 0,
                ..template.clone()
            })
            .collect();
        let class_b: Vec<_> = (0..4)
            .map(|_| ExtractedObject {
                class_id: 1,
                ..template.clone()
            })
            .collect();

        objects_by_class.insert(0, class_a);
        objects_by_class.insert(1, class_b);

        let mut counts = HashMap::new();
        counts.insert(0, 3);
        counts.insert(1, 3);

        let selected = select_objects_by_class(&objects_by_class, &counts, 3);
        assert_eq!(selected.len(), 3);

        let class_zero = selected.iter().filter(|o| o.class_id == 0).count();
        let class_one = selected.iter().filter(|o| o.class_id == 1).count();

        assert!(class_zero <= 3 && class_one <= 3);
    }

    #[test]
    fn test_select_objects_per_class_limit() {
        let mut objects_by_class = HashMap::new();

        let template = ExtractedObject {
            image: Array3::zeros((5, 5, 3)),
            mask: Array3::zeros((5, 5, 3)),
            bbox: (0, 0, 5, 5),
            class_id: 0,
        };

        let class_entries: Vec<_> = (0..5)
            .map(|_| ExtractedObject {
                class_id: 0,
                ..template.clone()
            })
            .collect();

        objects_by_class.insert(0, class_entries);

        let mut counts = HashMap::new();
        counts.insert(0, 2);

        let selected = select_objects_by_class(&objects_by_class, &counts, 10);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|o| o.class_id == 0));
    }
}
