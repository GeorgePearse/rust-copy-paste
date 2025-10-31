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
            matrix: Array2::from_shape_fn((2, 3), |(i, j)| {
                if i == j {
                    1.0
                } else if i == 1 && j == 2 {
                    0.0
                } else {
                    0.0
                }
            }),
            rotation: 0.0,
            scale: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }
}

/// Create a 2x3 affine transformation matrix
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
}
