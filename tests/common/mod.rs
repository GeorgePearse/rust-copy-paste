//! Common test utilities for integration tests
//! Provides helpers for generating synthetic test data and verifying object counts

/// Test image with ground truth information
#[derive(Clone, Debug)]
pub struct TestImage {
    /// Image data: width * height * 3 (BGR)
    pub image: Vec<u8>,
    /// Mask data: width * height (255 for object, 0 for background)
    pub mask: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Expected number of objects
    pub expected_objects: usize,
}

impl TestImage {
    /// Create a new test image
    pub fn new(
        image: Vec<u8>,
        mask: Vec<u8>,
        width: u32,
        height: u32,
        expected_objects: usize,
    ) -> Self {
        TestImage {
            image,
            mask,
            width,
            height,
            expected_objects,
        }
    }

    /// Get image as flat array for ndarray
    pub fn image_array(&self) -> Vec<u8> {
        self.image.clone()
    }

    /// Get mask as flat array for ndarray
    pub fn mask_array(&self) -> Vec<u8> {
        self.mask.clone()
    }
}

/// Generate a synthetic test image with distinct objects
///
/// Creates a test image with colored circles, each representing a distinct object.
/// Objects are spaced apart to avoid overlap.
///
/// # Arguments
/// * `num_objects` - Number of objects to draw
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
/// `TestImage` with generated image, mask, and expected object count
pub fn generate_test_image_with_objects(num_objects: usize, width: u32, height: u32) -> TestImage {
    let w = width as usize;
    let h = height as usize;
    let mut image = vec![128u8; w * h * 3]; // Gray background
    let mut mask = vec![0u8; w * h];

    if num_objects == 0 {
        return TestImage::new(image, mask, width, height, 0);
    }

    // Calculate object radius based on number of objects
    let min_radius = 10;
    let max_radius = std::cmp::min(width, height) as usize / (num_objects.max(2) * 3);
    let radius = std::cmp::max(min_radius, max_radius);

    // Grid-based placement to avoid overlap
    let grid_cols = (width as usize / (radius * 3)).max(1);
    let grid_rows = (height as usize / (radius * 3)).max(1);

    let mut obj_count = 0;
    for row in 0..grid_rows {
        for col in 0..grid_cols {
            if obj_count >= num_objects {
                break;
            }

            // Center position in grid cell
            let cell_width = w / grid_cols;
            let cell_height = h / grid_rows;
            let center_x = (col * cell_width + cell_width / 2) as i32;
            let center_y = (row * cell_height + cell_height / 2) as i32;

            // Draw circle in image and mask
            draw_circle(
                &mut image, &mut mask, w, h, center_x, center_y, radius, obj_count,
            );

            obj_count += 1;
        }
    }

    TestImage::new(image, mask, width, height, obj_count)
}

/// Draw a circle on image and mask
fn draw_circle(
    image: &mut [u8],
    mask: &mut [u8],
    width: usize,
    height: usize,
    center_x: i32,
    center_y: i32,
    radius: usize,
    object_id: usize,
) {
    let r_squared = (radius * radius) as i32;
    let radius_i32 = radius as i32;

    // Color based on object ID (cycling through distinct colors)
    let colors = [
        (255u8, 0u8, 0u8),     // Red
        (0u8, 255u8, 0u8),     // Green
        (0u8, 0u8, 255u8),     // Blue
        (255u8, 255u8, 0u8),   // Yellow
        (255u8, 0u8, 255u8),   // Magenta
        (0u8, 255u8, 255u8),   // Cyan
        (192u8, 192u8, 192u8), // Silver
        (128u8, 0u8, 0u8),     // Maroon
    ];

    let (b, g, r) = colors[object_id % colors.len()];

    for y in -radius_i32..=radius_i32 {
        for x in -radius_i32..=radius_i32 {
            if x * x + y * y <= r_squared {
                let px = center_x + x;
                let py = center_y + y;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = (py as usize * width + px as usize) * 3;

                    // Draw on image (BGR format)
                    if idx + 2 < image.len() {
                        image[idx] = b;
                        image[idx + 1] = g;
                        image[idx + 2] = r;
                    }

                    // Draw on mask
                    if (py as usize * width + px as usize) < mask.len() {
                        mask[py as usize * width + px as usize] = 255;
                    }
                }
            }
        }
    }
}

/// Count objects in a mask using connected components
///
/// Uses a simple flood-fill algorithm to count connected white regions
/// in the mask. Each region is considered a separate object.
///
/// # Arguments
/// * `mask` - Flat mask array (height * width)
/// * `width` - Mask width
/// * `height` - Mask height
///
/// # Returns
/// Number of distinct objects (connected components) found
pub fn count_objects_simple(mask: &[u8], width: u32, height: u32) -> usize {
    let w = width as usize;
    let h = height as usize;

    if mask.len() != w * h {
        return 0;
    }

    let mut visited = vec![false; w * h];
    let mut count = 0;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] > 128 && !visited[idx] {
                // Found a new object, flood fill it
                flood_fill(&mask, &mut visited, w, h, x, y);
                count += 1;
            }
        }
    }

    count
}

/// Flood fill algorithm for connected component analysis
fn flood_fill(
    mask: &[u8],
    visited: &mut [bool],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) {
    let mut queue = vec![(start_x, start_y)];
    let mut head = 0;

    while head < queue.len() {
        let (x, y) = queue[head];
        head += 1;

        if x >= width || y >= height {
            continue;
        }

        let idx = y * width + x;
        if visited[idx] {
            continue;
        }

        if mask[idx] <= 128 {
            continue;
        }

        visited[idx] = true;

        // Add neighbors (4-connectivity)
        if x > 0 {
            queue.push((x - 1, y));
        }
        if x < width - 1 {
            queue.push((x + 1, y));
        }
        if y > 0 {
            queue.push((x, y - 1));
        }
        if y < height - 1 {
            queue.push((x, y + 1));
        }
    }
}

/// Count objects using connected components
///
/// For now, uses the simple flood-fill algorithm.
/// Future versions can integrate OpenCV for more robust counting.
///
/// # Arguments
/// * `mask` - Flat mask array
/// * `width` - Mask width
/// * `height` - Mask height
///
/// # Returns
/// Number of distinct objects found
pub fn count_objects_opencv(mask: &[u8], width: u32, height: u32) -> usize {
    // Currently uses simple method - can be extended to use OpenCV's
    // connectedComponents function for more robust object counting
    count_objects_simple(mask, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_single_object() {
        let img = generate_test_image_with_objects(1, 256, 256);
        assert_eq!(img.expected_objects, 1);
        assert_eq!(img.image.len(), 256 * 256 * 3);
        assert_eq!(img.mask.len(), 256 * 256);
    }

    #[test]
    fn test_generate_multiple_objects() {
        for num_objects in [1, 3, 5, 8] {
            let img = generate_test_image_with_objects(num_objects, 512, 512);
            assert_eq!(img.expected_objects, num_objects);
        }
    }

    #[test]
    fn test_generate_empty() {
        let img = generate_test_image_with_objects(0, 256, 256);
        assert_eq!(img.expected_objects, 0);
        assert!(img.mask.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_count_objects_in_generated_image() {
        let img = generate_test_image_with_objects(3, 512, 512);
        let count = count_objects_simple(&img.mask, img.width, img.height);
        assert_eq!(
            count, img.expected_objects,
            "Counted {} objects, expected {}",
            count, img.expected_objects
        );
    }

    #[test]
    fn test_count_objects_empty_mask() {
        let mask = vec![0u8; 256 * 256];
        let count = count_objects_simple(&mask, 256, 256);
        assert_eq!(count, 0);
    }
}
