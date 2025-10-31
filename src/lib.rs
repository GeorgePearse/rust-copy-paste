use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

mod affine;
mod blending;
mod collision;
mod objects;

/// Python module for copy-paste augmentation (_core submodule)
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    #[pyo3(get)]
    pub object_counts: HashMap<u32, u32>, // class_id -> exact count to paste
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
    #[pyo3(signature = (image_width, image_height, max_paste_objects, use_rotation, use_scaling, rotation_range, scale_range, use_random_background, blend_mode, object_counts=None))]
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
        object_counts: Option<HashMap<u32, u32>>,
    ) -> Self {
        AugmentationConfig {
            image_width,
            image_height,
            max_paste_objects,
            object_counts: object_counts.unwrap_or_default(),
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
    last_placed: RefCell<Vec<objects::PlacedObject>>,
}

#[pymethods]
impl CopyPasteTransform {
    #[new]
    #[pyo3(signature = (image_width, image_height, max_paste_objects, use_rotation, use_scaling, use_random_background, blend_mode, object_counts=None))]
    pub fn new(
        image_width: u32,
        image_height: u32,
        max_paste_objects: u32,
        use_rotation: bool,
        use_scaling: bool,
        use_random_background: bool,
        blend_mode: String,
        object_counts: Option<HashMap<u32, u32>>,
    ) -> Self {
        CopyPasteTransform {
            config: AugmentationConfig {
                image_width,
                image_height,
                max_paste_objects,
                object_counts: object_counts.unwrap_or_default(),
                use_rotation,
                use_scaling,
                rotation_range: (-30.0, 30.0),
                scale_range: (0.8, 1.2),
                use_random_background,
                blend_mode,
            },
            last_placed: RefCell::new(Vec::new()),
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
        let image_shape = image.shape();
        let mask_shape = mask.shape();
        let target_mask_shape = target_mask.shape();

        if image_shape.len() != 3 {
            return Err(PyValueError::new_err("image must have shape (H, W, C)"));
        }
        if mask_shape.len() != 3 {
            return Err(PyValueError::new_err("mask must have shape (H, W, 1)"));
        }
        if target_mask_shape.len() != 3 {
            return Err(PyValueError::new_err(
                "target_mask must have shape (H, W, 1)",
            ));
        }

        let mut output_image = image.as_array().to_owned();
        let mask_array = mask.as_array().to_owned();
        let mut output_mask = target_mask.as_array().to_owned();

        let height = image_shape[0] as u32;
        let width = image_shape[1] as u32;
        let config = self.config.clone();

        let placed_objects = py.allow_threads(|| {
            let extracted_objects =
                objects::extract_objects_from_mask(output_image.view(), mask_array.view());

            let mut objects_by_class: HashMap<u32, Vec<objects::ExtractedObject>> = HashMap::new();
            for obj in extracted_objects {
                objects_by_class
                    .entry(obj.class_id)
                    .or_insert_with(Vec::new)
                    .push(obj);
            }

            let selected_objects = objects::select_objects_by_class(
                &objects_by_class,
                &config.object_counts,
                config.max_paste_objects,
            );

            let placed_objects = objects::place_objects(
                &selected_objects,
                width,
                height,
                config.use_rotation,
                config.use_scaling,
                config.rotation_range,
                config.scale_range,
                0.01,
            );

            let blend_mode = blending::BlendMode::from_string(&config.blend_mode);
            objects::compose_objects(&mut output_image, &placed_objects, blend_mode);
            objects::update_output_mask(&mut output_mask, &placed_objects);

            placed_objects
        });

        self.last_placed.replace(placed_objects.clone());

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
        let placed_objects = self.last_placed.borrow();

        if placed_objects.is_empty() {
            // No placement has happened yet; return original input unchanged
            let bboxes_array = bboxes.as_array().to_owned();
            return Ok(bboxes_array.into_pyarray_bound(py).unbind());
        }

        let metadata = objects::generate_output_bboxes_with_rotation(&placed_objects);
        let flat: Vec<f32> = metadata
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect();

        Ok(PyArray1::from_vec_bound(py, flat).unbind())
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
            512,
            512,
            5,
            true,
            true,
            (-30.0, 30.0),
            (0.8, 1.2),
            true,
            "normal".to_string(),
            None,
        );
        assert_eq!(config.image_width, 512);
    }
}
