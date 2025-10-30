use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};

mod affine;
mod blending;
mod collision;

/// Python module for copy-paste augmentation
#[pymodule]
fn copy_paste(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CopyPasteTransform>()?;
    m.add_class::<ObjectPaste>()?;
    Ok(())
}

/// Configuration for a copy-paste augmentation operation
#[pyclass]
#[derive(Clone, Debug)]
pub struct AugmentationConfig {
    #[pyo3(get, set)]
    pub image_width: u32,
    #[pyo3(get, set)]
    pub image_height: u32,
    #[pyo3(get, set)]
    pub max_paste_objects: u32,
    #[pyo3(get, set)]
    pub use_rotation: bool,
    #[pyo3(get, set)]
    pub use_scaling: bool,
    #[pyo3(get, set)]
    pub rotation_range: (f32, f32),
    #[pyo3(get, set)]
    pub scale_range: (f32, f32),
    #[pyo3(get, set)]
    pub use_random_background: bool,
    #[pyo3(get, set)]
    pub blend_mode: String,
}

#[pymethods]
impl AugmentationConfig {
    #[new]
    pub fn new(
        image_width: u32,
        image_height: u32,
        max_paste_objects: u32,
        use_rotation: bool,
        use_scaling: bool,
        rotation_range: (f32, f32),
        scale_range: (f32, f32),
        use_random_background: bool,
        blend_mode: String,
    ) -> Self {
        AugmentationConfig {
            image_width,
            image_height,
            max_paste_objects,
            use_rotation,
            use_scaling,
            rotation_range,
            scale_range,
            use_random_background,
            blend_mode,
        }
    }
}

/// A single object to be pasted
#[pyclass]
#[derive(Clone, Debug)]
pub struct ObjectPaste {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub scale: f32,
    #[pyo3(get, set)]
    pub rotation: f32,
    #[pyo3(get, set)]
    pub class_id: u32,
}

#[pymethods]
impl ObjectPaste {
    #[new]
    pub fn new(x: f32, y: f32, scale: f32, rotation: f32, class_id: u32) -> Self {
        ObjectPaste {
            x,
            y,
            scale,
            rotation,
            class_id,
        }
    }
}

/// Main copy-paste augmentation transform
#[pyclass]
pub struct CopyPasteTransform {
    config: AugmentationConfig,
}

#[pymethods]
impl CopyPasteTransform {
    #[new]
    pub fn new(
        image_width: u32,
        image_height: u32,
        max_paste_objects: u32,
        use_rotation: bool,
        use_scaling: bool,
        use_random_background: bool,
        blend_mode: String,
    ) -> Self {
        CopyPasteTransform {
            config: AugmentationConfig {
                image_width,
                image_height,
                max_paste_objects,
                use_rotation,
                use_scaling,
                rotation_range: (-30.0, 30.0),
                scale_range: (0.8, 1.2),
                use_random_background,
                blend_mode,
            },
        }
    }

    /// Apply copy-paste augmentation to image and masks
    pub fn apply(
        &self,
        py: Python<'_>,
        image: PyReadonlyArray3<u8>,
        mask: PyReadonlyArray3<u8>,
        target_mask: PyReadonlyArray3<u8>,
    ) -> PyResult<(Py<PyArray3<u8>>, Py<PyArray3<u8>>)> {
        let _img_shape = image.shape();
        let img_array = image.as_array().to_owned();
        let _mask_array = mask.as_array().to_owned();
        let mut output_mask = target_mask.as_array().to_owned();
        let mut output_image = img_array.clone();

        // Get dimensions
        let _height = _img_shape[0] as u32;
        let _width = _img_shape[1] as u32;

        // Simple placeholder: just return inputs (full algorithm requires more complex handling)
        // In a production implementation, this would:
        // 1. Find all objects in mask_array
        // 2. Randomly select which objects to paste
        // 3. Apply affine transformations
        // 4. Check for collisions between objects
        // 5. Blend selected objects onto output_image
        // 6. Update output_mask with new object locations

        Ok((
            output_image.into_pyarray_bound(py).unbind(),
            output_mask.into_pyarray_bound(py).unbind(),
        ))
    }

    /// Apply augmentation with bounding boxes (Albumentations format)
    pub fn apply_to_bboxes(
        &self,
        py: Python<'_>,
        bboxes: PyReadonlyArray1<f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let bboxes_array = bboxes.as_array().to_owned();
        // This is a placeholder - in full implementation, would update bboxes based on paste operations
        Ok(bboxes_array.into_pyarray_bound(py).unbind())
    }

    /// Get configuration
    pub fn get_config(&self) -> AugmentationConfig {
        self.config.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_paste_creation() {
        let obj = ObjectPaste::new(100.0, 100.0, 1.0, 0.0, 0);
        assert_eq!(obj.x, 100.0);
        assert_eq!(obj.y, 100.0);
    }

    #[test]
    fn test_config_creation() {
        let config = AugmentationConfig::new(
            512, 512, 5, true, true, (-30.0, 30.0), (0.8, 1.2), true, "normal".to_string(),
        );
        assert_eq!(config.image_width, 512);
    }
}
