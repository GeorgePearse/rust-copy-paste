/// Blending modes for combining images
#[derive(Clone, Copy, Debug)]
pub enum BlendMode {
    /// Standard alpha blending
    Normal,
    /// X-ray style blending (for overlays)
    XRay,
}

impl BlendMode {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "xray" => BlendMode::XRay,
            _ => BlendMode::Normal,
        }
    }
}

/// Blend two images with the specified mode
pub fn blend_images(
    base: &[u8],
    overlay: &[u8],
    alpha: f32,
    mode: BlendMode,
    _width: u32,
    _height: u32,
) -> Vec<u8> {
    let mut result = base.to_vec();
    let len = base.len();

    match mode {
        BlendMode::Normal => {
            for i in 0..len {
                let base_val = base[i] as f32;
                let overlay_val = overlay[i] as f32;
                let blended = base_val * (1.0 - alpha) + overlay_val * alpha;
                result[i] = blended.clamp(0.0, 255.0) as u8;
            }
        }
        BlendMode::XRay => {
            // X-ray blending: weighted sum that makes overlaid areas more visible
            for i in 0..len {
                let base_val = base[i] as f32;
                let overlay_val = overlay[i] as f32;
                // X-ray blend formula: creates a lighter blend
                let blended = (base_val + overlay_val * alpha).min(255.0);
                result[i] = blended as u8;
            }
        }
    }

    result
}

/// Blend two pixels
pub fn blend_pixel(base: u8, overlay: u8, alpha: f32, mode: BlendMode) -> u8 {
    match mode {
        BlendMode::Normal => {
            let base_val = base as f32;
            let overlay_val = overlay as f32;
            let blended = base_val * (1.0 - alpha) + overlay_val * alpha;
            blended.clamp(0.0, 255.0) as u8
        }
        BlendMode::XRay => {
            let base_val = base as f32;
            let overlay_val = overlay as f32;
            let blended = (base_val + overlay_val * alpha).min(255.0);
            blended as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_mode_from_string() {
        match BlendMode::from_string("normal") {
            BlendMode::Normal => (),
            _ => panic!("Expected Normal mode"),
        }

        match BlendMode::from_string("xray") {
            BlendMode::XRay => (),
            _ => panic!("Expected XRay mode"),
        }
    }

    #[test]
    fn test_blend_pixel_normal() {
        let result = blend_pixel(100, 200, 0.5, BlendMode::Normal);
        assert_eq!(result, 150);
    }

    #[test]
    fn test_blend_pixel_xray() {
        let result = blend_pixel(100, 200, 0.5, BlendMode::XRay);
        // 100 + 200*0.5 = 200
        assert_eq!(result, 200);
    }
}
