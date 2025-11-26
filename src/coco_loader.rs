/// COCO dataset loader for copy-paste augmentation
/// Loads objects directly from COCO annotation files
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::objects::SourceObject;

/// Custom deserializers to handle empty strings in JSON numeric fields
///
/// Deserialize Option<i32> from either a number or string (empty string becomes None)
fn deserialize_option_i32<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Deserialize};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrInt {
        String(String),
        Int(i32),
    }

    match Option::<StringOrInt>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrInt::Int(i)) => Ok(Some(i)),
        Some(StringOrInt::String(s)) => {
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse::<i32>()
                    .map(Some)
                    .map_err(de::Error::custom)
            }
        }
    }
}

/// Deserialize u32 from either a number or string (empty string becomes error, skipped)
fn deserialize_u32<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Deserialize};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNum {
        String(String),
        Num(u32),
    }

    match StringOrNum::deserialize(deserializer)? {
        StringOrNum::Num(n) => Ok(n),
        StringOrNum::String(s) => {
            if s.is_empty() {
                Err(de::Error::custom("empty string for required u32 field"))
            } else {
                s.parse::<u32>().map_err(de::Error::custom)
            }
        }
    }
}

/// Deserialize u64 from either a number or string (empty string becomes error, skipped)
fn deserialize_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Deserialize};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNum {
        String(String),
        Num(u64),
    }

    match StringOrNum::deserialize(deserializer)? {
        StringOrNum::Num(n) => Ok(n),
        StringOrNum::String(s) => {
            if s.is_empty() {
                Err(de::Error::custom("empty string for required u64 field"))
            } else {
                s.parse::<u64>().map_err(de::Error::custom)
            }
        }
    }
}

/// COCO annotation file structure
#[derive(Debug, Deserialize, Serialize)]
pub struct CocoAnnotations {
    pub info: Option<CocoInfo>,
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotation>,
    pub categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CocoInfo {
    pub description: Option<String>,
    pub version: Option<String>,
    #[serde(default, deserialize_with = "deserialize_option_i32")]
    pub year: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CocoImage {
    #[serde(deserialize_with = "deserialize_u64")]
    pub id: u64,
    pub file_name: String,
    #[serde(deserialize_with = "deserialize_u32")]
    pub width: u32,
    #[serde(deserialize_with = "deserialize_u32")]
    pub height: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CocoAnnotation {
    #[serde(deserialize_with = "deserialize_u64")]
    pub id: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    pub image_id: u64,
    #[serde(deserialize_with = "deserialize_u32")]
    pub category_id: u32,
    pub bbox: Vec<f64>,
    pub area: Option<f64>,
    pub iscrowd: Option<u8>,
    pub segmentation: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CocoCategory {
    #[serde(deserialize_with = "deserialize_u32")]
    pub id: u32,
    pub name: String,
    pub supercategory: Option<String>,
}

/// Object bank that stores objects loaded from COCO annotations
pub struct CocoObjectBank {
    /// Objects grouped by class ID
    pub objects: HashMap<u32, Vec<SourceObject>>,
    /// Category ID to name mapping
    #[allow(dead_code)]
    pub category_map: HashMap<u32, String>,
    /// Category name to ID mapping
    #[allow(dead_code)]
    pub name_to_id: HashMap<String, u32>,
}

impl CocoObjectBank {
    /// Create a new COCO object bank from annotation file
    pub fn from_file(
        annotation_path: impl AsRef<Path>,
        images_root: Option<impl AsRef<Path>>,
        class_filter: Option<&[String]>,
        max_objects_per_class: Option<usize>,
    ) -> Result<Self, String> {
        let annotation_path = annotation_path.as_ref();

        // Load COCO JSON
        let file = File::open(annotation_path)
            .map_err(|e| format!("Failed to open annotation file: {e}"))?;
        let reader = BufReader::new(file);
        let coco_data: CocoAnnotations = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse COCO annotations: {e}"))?;

        // Determine images root directory
        let images_root = if let Some(root) = images_root {
            PathBuf::from(root.as_ref())
        } else {
            // Default: go up 1 level from annotation file (to annotations/ parent), then join "images"
            // e.g., /data/annotations/file.json -> /data/images/
            annotation_path
                .parent() // annotations/
                .and_then(|p| p.parent()) // parent of annotations/
                .map_or_else(|| PathBuf::from("images"), |p| p.join("images"))
        };

        // Build category mappings
        let mut category_map = HashMap::new();
        let mut name_to_id = HashMap::new();
        for cat in &coco_data.categories {
            category_map.insert(cat.id, cat.name.clone());
            name_to_id.insert(cat.name.clone(), cat.id);
        }

        // Filter categories if class_filter is provided
        let valid_cat_ids: Vec<u32> = if let Some(filter) = class_filter {
            filter
                .iter()
                .filter_map(|name| name_to_id.get(name).copied())
                .collect()
        } else {
            category_map.keys().copied().collect()
        };

        // Build image_id -> image_info mapping
        let image_map: HashMap<u64, &CocoImage> = coco_data
            .images
            .iter()
            .map(|img| (img.id, img))
            .collect();

        // Group annotations by image_id and filter by category
        let mut annotations_by_image: HashMap<u64, Vec<&CocoAnnotation>> = HashMap::new();
        for ann in &coco_data.annotations {
            if valid_cat_ids.contains(&ann.category_id) {
                annotations_by_image
                    .entry(ann.image_id)
                    .or_default()
                    .push(ann);
            }
        }

        // Initialize object storage
        let mut objects: HashMap<u32, Vec<SourceObject>> = HashMap::new();
        for cat_id in &valid_cat_ids {
            objects.insert(*cat_id, Vec::new());
        }

        // Track objects per class for limiting
        let mut objects_per_class: HashMap<u32, usize> = HashMap::new();

        println!("Scanning annotations for {} images...", annotations_by_image.len());
        let pb = ProgressBar::new(annotations_by_image.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        // Process annotations (lazy load, just check existence)
        for (img_id, annotations) in &annotations_by_image {
            pb.inc(1);
            let Some(img_info) = image_map.get(img_id) else {
                eprintln!("Warning: Image ID {img_id} not found in images list");
                continue;
            };

            let img_path = images_root.join(&img_info.file_name);
            if !img_path.exists() {
                // Update message to show warning without breaking the bar
                pb.set_message(format!("Missing: {}", img_info.file_name));
                continue;
            }

            // Create SourceObject for each annotation
            for ann in annotations {
                let cat_id = ann.category_id;

                // Check per-class limit
                if let Some(max_per_class) = max_objects_per_class {
                    let count = objects_per_class.get(&cat_id).copied().unwrap_or(0);
                    if count >= max_per_class {
                        continue;
                    }
                }

                // Parse bounding box [x, y, width, height]
                if ann.bbox.len() != 4 {
                    continue;
                }

                let x = ann.bbox[0].max(0.0) as u32;
                let y = ann.bbox[1].max(0.0) as u32;
                let w = ann.bbox[2].max(0.0) as u32;
                let h = ann.bbox[3].max(0.0) as u32;

                if w == 0 || h == 0 {
                    continue;
                }

                // Clamp bbox to image bounds using metadata
                let img_w = img_info.width;
                let img_h = img_info.height;

                let x_safe = x.min(img_w.saturating_sub(1));
                let y_safe = y.min(img_h.saturating_sub(1));
                let w_safe = w.min(img_w - x_safe);
                let h_safe = h.min(img_h - y_safe);

                if w_safe == 0 || h_safe == 0 {
                    continue;
                }

                let source_obj = SourceObject {
                    image_path: img_path.clone(),
                    bbox: (x_safe, y_safe, w_safe, h_safe),
                    class_id: cat_id,
                };

                if let Some(obj_list) = objects.get_mut(&cat_id) {
                    obj_list.push(source_obj);
                    *objects_per_class.entry(cat_id).or_insert(0) += 1;
                }
            }
        }
        pb.finish_with_message("Done scanning annotations");

        // Log summary
        let total_objects: usize = objects.values().map(Vec::len).sum();
        println!(
            "Found {} objects from {} images",
            total_objects,
            annotations_by_image.len()
        );
        for (cat_id, objs) in &objects {
            let cat_name = category_map.get(cat_id).map_or("unknown", |s| s.as_str());
            println!("  {} (id={}): {} objects", cat_name, cat_id, objs.len());
        }

        Ok(CocoObjectBank {
            objects,
            category_map,
            name_to_id,
        })
    }

    /// Get all objects for a specific class
    #[allow(dead_code)]
    pub fn get_objects_by_class(&self, class_id: u32) -> &[SourceObject] {
        self.objects.get(&class_id).map_or(&[], Vec::as_slice)
    }

    /// Get number of objects for a specific class
    #[allow(dead_code)]
    pub fn get_num_objects(&self, class_id: Option<u32>) -> usize {
        if let Some(id) = class_id {
            self.objects.get(&id).map_or(0, Vec::len)
        } else {
            self.objects.values().map(Vec::len).sum()
        }
    }

    /// Get category name from ID
    #[allow(dead_code)]
    pub fn get_category_name(&self, class_id: u32) -> Option<&str> {
        self.category_map.get(&class_id).map(String::as_str)
    }

    /// Get category ID from name
    #[allow(dead_code)]
    pub fn get_category_id(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    /// Get all objects (consuming the bank)
    pub fn into_objects(self) -> HashMap<u32, Vec<SourceObject>> {
        self.objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_annotations_parsing() {
        let json_str = r#"{
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200}
            ],
            "categories": [{"id": 1, "name": "cat"}]
        }"#;

        let coco: CocoAnnotations = serde_json::from_str(json_str).unwrap();
        assert_eq!(coco.images.len(), 1);
        assert_eq!(coco.annotations.len(), 1);
        assert_eq!(coco.categories.len(), 1);
        assert_eq!(coco.categories[0].name, "cat");
    }
}
