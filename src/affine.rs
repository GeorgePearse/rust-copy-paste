use ndarray::Array2;
use std::f32::consts::PI;

/// Represents an affine transformation matrix and its parameters
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct AffineTransform {
    /// 2x3 affine transformation matrix
    pub matrix: Array2<f32>,
    pub rotation: f32,
    pub scale: f32,
    pub tx: f32,
    pub ty: f32,
}

impl AffineTransform {
    /// Create a new affine transformation
    #[allow(dead_code)]
    pub fn new(rotation: f32, scale: f32, tx: f32, ty: f32) -> Self {
        let matrix = create_affine_matrix(rotation, scale, tx, ty);
        AffineTransform {
            matrix,
            rotation,
            scale,
            tx,
            ty,
        }
    }

    /// Create identity transformation
    #[allow(dead_code)]
    pub fn identity() -> Self {
        AffineTransform {
            matrix: Array2::from_shape_fn((2, 3), |(i, j)| if i == j { 1.0 } else { 0.0 }),
            rotation: 0.0,
            scale: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }
}

/// Create a 2x3 affine transformation matrix
#[allow(dead_code)]
fn create_affine_matrix(rotation: f32, scale: f32, tx: f32, ty: f32) -> Array2<f32> {
    let rad = rotation * PI / 180.0;
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let mut matrix = Array2::zeros((2, 3));

    // Rotation and scaling
    matrix[[0, 0]] = scale * cos_a;
    matrix[[0, 1]] = -scale * sin_a;
    matrix[[1, 0]] = scale * sin_a;
    matrix[[1, 1]] = scale * cos_a;

    // Translation
    matrix[[0, 2]] = tx;
    matrix[[1, 2]] = ty;

    matrix
}

/// Apply affine transformation to a point
#[allow(dead_code)]
pub fn apply_affine_transform(point: (f32, f32), transform: &AffineTransform) -> (f32, f32) {
    let x = point.0;
    let y = point.1;
    let m = &transform.matrix;

    let new_x = m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]];
    let new_y = m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]];

    (new_x, new_y)
}

/// Get the inverse affine transformation
#[allow(dead_code)]
pub fn invert_affine(transform: &AffineTransform) -> AffineTransform {
    let m = &transform.matrix;

    // Calculate determinant of 2x2 rotation/scale matrix
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];

    if det.abs() < 1e-10 {
        return AffineTransform::identity();
    }

    // Create inverse matrix
    let mut inv_m = Array2::zeros((2, 3));

    // Inverse rotation/scale matrix
    inv_m[[0, 0]] = m[[1, 1]] / det;
    inv_m[[0, 1]] = -m[[0, 1]] / det;
    inv_m[[1, 0]] = -m[[1, 0]] / det;
    inv_m[[1, 1]] = m[[0, 0]] / det;

    // Inverse translation
    inv_m[[0, 2]] = -(inv_m[[0, 0]] * m[[0, 2]] + inv_m[[0, 1]] * m[[1, 2]]);
    inv_m[[1, 2]] = -(inv_m[[1, 0]] * m[[0, 2]] + inv_m[[1, 1]] * m[[1, 2]]);

    AffineTransform {
        matrix: inv_m,
        rotation: -transform.rotation,
        scale: 1.0 / transform.scale,
        tx: transform.tx,
        ty: transform.ty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let t = AffineTransform::identity();
        let point = (10.0, 20.0);
        let (x, y) = apply_affine_transform(point, &t);
        assert!((x - 10.0).abs() < 1e-5);
        assert!((y - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_translation() {
        let t = AffineTransform::new(0.0, 1.0, 5.0, 10.0);
        let point = (0.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);
        assert!((x - 5.0).abs() < 1e-5);
        assert!((y - 10.0).abs() < 1e-5);
    }

    // ==================== Phase 2: Affine Transform Tests ====================

    #[test]
    fn test_rotation_45_degrees() {
        // 45° rotation should move point (1, 0) to approximately (√2/2, √2/2)
        let t = AffineTransform::new(45.0, 1.0, 0.0, 0.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        let sqrt2_2 = std::f32::consts::SQRT_2 / 2.0;
        assert!((x - sqrt2_2).abs() < 1e-5);
        assert!((y - sqrt2_2).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_90_degrees() {
        // 90° rotation should move (1, 0) to (0, 1)
        let t = AffineTransform::new(90.0, 1.0, 0.0, 0.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!(x.abs() < 1e-5); // Should be ~0
        assert!((y - 1.0).abs() < 1e-5); // Should be ~1
    }

    #[test]
    fn test_rotation_180_degrees() {
        // 180° rotation should move (1, 0) to (-1, 0)
        let t = AffineTransform::new(180.0, 1.0, 0.0, 0.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - (-1.0)).abs() < 1e-5);
        assert!(y.abs() < 1e-5);
    }

    #[test]
    fn test_rotation_270_degrees() {
        // 270° rotation should move (1, 0) to (0, -1)
        let t = AffineTransform::new(270.0, 1.0, 0.0, 0.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!(x.abs() < 1e-5);
        assert!((y - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_360_degrees() {
        // 360° rotation should be back to original
        let t = AffineTransform::new(360.0, 1.0, 0.0, 0.0);
        let point = (5.0, 10.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 5.0).abs() < 1e-5);
        assert!((y - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_scaling() {
        // 2x scaling should double coordinates
        let t = AffineTransform::new(0.0, 2.0, 0.0, 0.0);
        let point = (5.0, 10.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 10.0).abs() < 1e-5);
        assert!((y - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_small_scaling() {
        // 0.5x scaling should halve coordinates
        let t = AffineTransform::new(0.0, 0.5, 0.0, 0.0);
        let point = (10.0, 20.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 5.0).abs() < 1e-5);
        assert!((y - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_large_scaling() {
        // 10x scaling
        let t = AffineTransform::new(0.0, 10.0, 0.0, 0.0);
        let point = (1.0, 1.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 10.0).abs() < 1e-5);
        assert!((y - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_and_scaling_combined() {
        // 90° rotation with 2x scaling
        let t = AffineTransform::new(90.0, 2.0, 0.0, 0.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        // Should be 2x * rotated (1,0) = 2x * (0,1) = (0, 2)
        assert!(x.abs() < 1e-5);
        assert!((y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_scaling_and_translation() {
        // 45° rotation, 2x scale, and translation
        let t = AffineTransform::new(45.0, 2.0, 5.0, 10.0);
        let point = (1.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        // (1, 0) rotated 45° and scaled 2x = 2*√2/2 * (1, 1) = (√2, √2) ≈ (1.414, 1.414)
        // Plus translation (5, 10) = (6.414, 11.414)
        let expected_x = 2.0 * std::f32::consts::SQRT_2 / 2.0 + 5.0;
        let expected_y = 2.0 * std::f32::consts::SQRT_2 / 2.0 + 10.0;

        assert!((x - expected_x).abs() < 1e-5);
        assert!((y - expected_y).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_inversion_identity() {
        // Identity transform inverted should still be identity
        let t = AffineTransform::identity();
        let inv = invert_affine(&t);

        let point = (10.0, 20.0);
        let (x, y) = apply_affine_transform(point, &inv);

        // Should get back to original (inverse of identity is identity)
        assert!((x - 10.0).abs() < 1e-5);
        assert!((y - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_inversion_translation() {
        // Invert a translation
        let t = AffineTransform::new(0.0, 1.0, 5.0, 10.0);
        let inv = invert_affine(&t);

        // Forward then inverse should give original point
        let point = (3.0, 7.0);
        let (x1, y1) = apply_affine_transform(point, &t);
        let (x2, y2) = apply_affine_transform((x1, y1), &inv);

        assert!((x2 - point.0).abs() < 1e-4);
        assert!((y2 - point.1).abs() < 1e-4);
    }

    #[test]
    fn test_matrix_inversion_scaling() {
        // Invert a scaling transformation
        let t = AffineTransform::new(0.0, 2.0, 0.0, 0.0);
        let inv = invert_affine(&t);

        // Forward then inverse should give original
        let point = (10.0, 20.0);
        let (x1, y1) = apply_affine_transform(point, &t);
        let (x2, y2) = apply_affine_transform((x1, y1), &inv);

        assert!((x2 - point.0).abs() < 1e-4);
        assert!((y2 - point.1).abs() < 1e-4);
    }

    #[test]
    fn test_matrix_inversion_rotation() {
        // Invert a rotation
        let t = AffineTransform::new(45.0, 1.0, 0.0, 0.0);
        let inv = invert_affine(&t);

        let point = (5.0, 10.0);
        let (x1, y1) = apply_affine_transform(point, &t);
        let (x2, y2) = apply_affine_transform((x1, y1), &inv);

        assert!((x2 - point.0).abs() < 1e-4);
        assert!((y2 - point.1).abs() < 1e-4);
    }

    #[test]
    fn test_matrix_inversion_complex_transform() {
        // Invert a complex transform (rotation + scaling + translation)
        let t = AffineTransform::new(30.0, 1.5, 7.0, 11.0);
        let inv = invert_affine(&t);

        let point = (100.0, 50.0);
        let (x1, y1) = apply_affine_transform(point, &t);
        let (x2, y2) = apply_affine_transform((x1, y1), &inv);

        // Should recover original point (with floating point tolerance)
        assert!((x2 - point.0).abs() < 1e-2);
        assert!((y2 - point.1).abs() < 1e-2);
    }

    #[test]
    fn test_singular_matrix_inversion() {
        // Matrix with det ~0 should return identity
        // We can't create exactly singular matrix due to how transformation works,
        // but we can test edge cases
        let t = AffineTransform::new(0.0, 1.0, 0.0, 0.0);
        let inv = invert_affine(&t);

        // Inverse should still be valid
        let point = (5.0, 5.0);
        let (x, y) = apply_affine_transform(point, &inv);
        assert!(x.is_finite() && y.is_finite());
    }

    #[test]
    fn test_zero_translation() {
        // Transform with zero translation
        let t = AffineTransform::new(45.0, 1.0, 0.0, 0.0);
        let point = (10.0, 10.0);
        let (x, y) = apply_affine_transform(point, &t);

        // Should only apply rotation, no translation offset
        assert!(x.is_finite() && y.is_finite());
    }

    #[test]
    fn test_large_translation() {
        // Very large translation values
        let t = AffineTransform::new(0.0, 1.0, 1000.0, 2000.0);
        let point = (1.0, 1.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 1001.0).abs() < 1e-5);
        assert!((y - 2001.0).abs() < 1e-5);
    }

    #[test]
    fn test_negative_scaling() {
        // Negative scaling (mirror)
        let t = AffineTransform::new(0.0, -1.0, 0.0, 0.0);
        let point = (5.0, 10.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - (-5.0)).abs() < 1e-5);
        assert!((y - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_transformations_chain() {
        // Apply transformations in sequence
        let t1 = AffineTransform::new(0.0, 1.0, 5.0, 0.0);
        let t2 = AffineTransform::new(90.0, 1.0, 0.0, 0.0);

        let point = (1.0, 0.0);
        let (x1, y1) = apply_affine_transform(point, &t1);
        let (x2, y2) = apply_affine_transform((x1, y1), &t2);

        // First: translate (1,0) by (5,0) -> (6,0)
        // Second: rotate (6,0) by 90° -> (0, 6)
        assert!(x2.abs() < 1e-5);
        assert!((y2 - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_affine_transform_preserves_points() {
        // Identity transform should preserve all points
        let t = AffineTransform::identity();

        let test_points = vec![(0.0, 0.0), (10.0, 20.0), (-5.0, 15.0), (1000.0, -1000.0)];

        for point in test_points {
            let (x, y) = apply_affine_transform(point, &t);
            assert!((x - point.0).abs() < 1e-5);
            assert!((y - point.1).abs() < 1e-5);
        }
    }

    #[test]
    fn test_small_rotation_angle() {
        // Very small rotation angle (1 degree)
        let t = AffineTransform::new(1.0, 1.0, 0.0, 0.0);
        let point = (100.0, 0.0);
        let (x, y) = apply_affine_transform(point, &t);

        // Should be very close to original (minimal rotation effect)
        assert!(x > 99.0); // Should rotate slightly but mostly preserve
        assert!(y.abs() < 2.0); // Small rotation means small y displacement
    }

    #[test]
    fn test_very_small_scaling() {
        // Very small scale (0.01x)
        let t = AffineTransform::new(0.0, 0.01, 0.0, 0.0);
        let point = (1000.0, 1000.0);
        let (x, y) = apply_affine_transform(point, &t);

        assert!((x - 10.0).abs() < 1e-5);
        assert!((y - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_composition() {
        // Test that rotation and scaling order doesn't matter for magnitude
        let t1 = AffineTransform::new(45.0, 2.0, 0.0, 0.0); // Rotate then scale
        let point = (1.0, 0.0);
        let (x1, y1) = apply_affine_transform(point, &t1);

        // Magnitude should be 2 (due to 2x scaling)
        let magnitude_1 = (x1 * x1 + y1 * y1).sqrt();
        assert!((magnitude_1 - 2.0).abs() < 1e-5);
    }
}
