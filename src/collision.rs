/// Bounding box represented as (x_min, y_min, x_max, y_max)
pub type BBox = (f32, f32, f32, f32);

/// Calculate Intersection over Union (IoU) between two bounding boxes
pub fn calculate_iou(bbox1: BBox, bbox2: BBox) -> f32 {
    let (x1_min, y1_min, x1_max, y1_max) = bbox1;
    let (x2_min, y2_min, x2_max, y2_max) = bbox2;

    // Calculate intersection area
    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
        return 0.0;
    }

    let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);

    // Calculate union area
    let area1 = (x1_max - x1_min) * (y1_max - y1_min);
    let area2 = (x2_max - x2_min) * (y2_max - y2_min);
    let union_area = area1 + area2 - inter_area;

    if union_area < 1e-10 {
        return 0.0;
    }

    inter_area / union_area
}

/// Check if two bounding boxes have collision (IoU > threshold)
pub fn check_iou_collision(bbox1: BBox, bbox2: BBox, threshold: f32) -> bool {
    calculate_iou(bbox1, bbox2) > threshold
}

/// Get the intersection box of two bounding boxes
#[allow(dead_code)]
pub fn get_intersection_box(bbox1: BBox, bbox2: BBox) -> Option<BBox> {
    let (x1_min, y1_min, x1_max, y1_max) = bbox1;
    let (x2_min, y2_min, x2_max, y2_max) = bbox2;

    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max > inter_x_min && inter_y_max > inter_y_min {
        Some((inter_x_min, inter_y_min, inter_x_max, inter_y_max))
    } else {
        None
    }
}

/// Get the union (bounding box that contains both boxes)
#[allow(dead_code)]
pub fn get_union_box(bbox1: BBox, bbox2: BBox) -> BBox {
    let (x1_min, y1_min, x1_max, y1_max) = bbox1;
    let (x2_min, y2_min, x2_max, y2_max) = bbox2;

    (
        x1_min.min(x2_min),
        y1_min.min(y2_min),
        x1_max.max(x2_max),
        y1_max.max(y2_max),
    )
}

/// Clip bounding box to be within image bounds
pub fn clip_bbox_to_image(bbox: BBox, image_width: u32, image_height: u32) -> BBox {
    let (mut x_min, mut y_min, mut x_max, mut y_max) = bbox;

    x_min = x_min.max(0.0).min(image_width as f32);
    y_min = y_min.max(0.0).min(image_height as f32);
    x_max = x_max.max(0.0).min(image_width as f32);
    y_max = y_max.max(0.0).min(image_height as f32);

    (x_min, y_min, x_max, y_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_identical_boxes() {
        let bbox = (0.0, 0.0, 10.0, 10.0);
        let iou = calculate_iou(bbox, bbox);
        assert!((iou - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let bbox1 = (0.0, 0.0, 10.0, 10.0);
        let bbox2 = (20.0, 20.0, 30.0, 30.0);
        let iou = calculate_iou(bbox1, bbox2);
        assert!(iou.abs() < 1e-5);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let bbox1 = (0.0, 0.0, 10.0, 10.0);
        let bbox2 = (5.0, 5.0, 15.0, 15.0);
        let iou = calculate_iou(bbox1, bbox2);
        // Intersection area = 5*5 = 25
        // Union area = 100 + 100 - 25 = 175
        // IoU = 25/175 ≈ 0.1429
        assert!((iou - (25.0 / 175.0)).abs() < 1e-5);
    }

    #[test]
    fn test_collision_detection() {
        let bbox1 = (0.0, 0.0, 10.0, 10.0);
        let bbox2 = (5.0, 5.0, 15.0, 15.0);
        assert!(check_iou_collision(bbox1, bbox2, 0.1));
        assert!(!check_iou_collision(bbox1, bbox2, 0.2));
    }

    #[test]
    fn test_clip_bbox() {
        let bbox = (-5.0, -5.0, 105.0, 105.0);
        let clipped = clip_bbox_to_image(bbox, 100, 100);
        assert_eq!(clipped, (0.0, 0.0, 100.0, 100.0));
    }
}
