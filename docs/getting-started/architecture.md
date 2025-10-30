# Architecture

## System Overview

Copy-Paste Augmentation is built with a layered architecture:

```
┌─────────────────────────────────────────┐
│  Application Layer                      │
│  (Albumentations DualTransform)         │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Python Wrapper Layer                   │
│  (copy_paste/transform.py)              │
│  ├─ Configuration handling              │
│  ├─ Input validation                    │
│  ├─ Coordinate transformation           │
│  └─ Error handling                      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  PyO3 Bindings                          │
│  (Rust ↔ Python bridge)                 │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Rust Core Layer (src/)                 │
│  ├─ CopyPasteTransform (lib.rs)         │
│  ├─ Affine Transformations (affine.rs)  │
│  ├─ Image Blending (blending.rs)        │
│  └─ Collision Detection (collision.rs)  │
└──────────────────────────────────────────┘
```

## Component Details

### Python Wrapper (`copy_paste/transform.py`)

**Purpose**: Provide Albumentations-compatible interface

**Key Classes**:
- `CopyPasteAugmentation` - Main transform class
  - Inherits from `albumentations.DualTransform`
  - Implements required methods: `apply()`, `apply_to_bboxes()`, `apply_to_masks()`
  - Handles coordinate conversion (normalized ↔ pixel)

**Responsibilities**:
- Configuration validation
- Input format conversion
- Rust module invocation
- Output format conversion
- Error handling and logging

### Rust Core

#### CopyPasteTransform (`src/lib.rs`)

Main augmentation engine with PyO3 bindings.

**Interface**:
```rust
pub struct CopyPasteTransform {
    config: AugmentationConfig,
    rust_transform: CopyPasteTransform,
}

impl CopyPasteTransform {
    fn apply(&self, image, mask, target_mask) -> (image, mask);
    fn apply_to_bboxes(&self, bboxes) -> bboxes;
}
```

**Algorithm** (placeholder for full implementation):
1. Extract objects from source masks
2. Select random objects based on `max_paste_objects`
3. Apply affine transformations
4. Check collisions using IoU
5. Blend selected objects
6. Update masks

#### Affine Transformations (`src/affine.rs`)

Handles geometric transformations.

**Features**:
- Rotation (custom angle range)
- Scaling (custom scale range)
- Translation (automatic positioning)
- Inverse transformations for coordinate mapping

**Key Functions**:
- `create_affine_matrix()` - Build transformation matrix
- `apply_affine_transform()` - Transform points
- `invert_affine()` - Get inverse transformation

#### Image Blending (`src/blending.rs`)

Combines objects onto target images.

**Blend Modes**:
- **Normal**: Standard alpha blending
  ```
  output = base * (1 - alpha) + overlay * alpha
  ```
- **X-ray**: Weighted combination for visibility
  ```
  output = min(base + overlay * alpha, 255)
  ```

#### Collision Detection (`src/collision.rs`)

Ensures pasted objects don't overlap excessively.

**Key Algorithm**: Intersection over Union (IoU)
```
IoU = intersection_area / union_area
```

**Collision Check**:
```
collision = IoU(bbox1, bbox2) > threshold
```

## Data Flow

### Forward Pass

```
Input: {image, bboxes, masks}
  ↓
[Python Wrapper]
  ├─ Validate input formats
  ├─ Convert coordinates to pixel space
  ├─ Prepare Rust inputs
  ↓
[Rust Core]
  ├─ Extract objects from masks
  ├─ Sample random objects
  ├─ Apply transformations
  ├─ Check collisions
  ├─ Blend images
  ├─ Update masks
  ↓
[Python Wrapper]
  ├─ Convert coordinates back to normalized
  ├─ Validate output
  ├─ Handle errors
  ↓
Output: {image, bboxes, masks}
```

## Thread Safety

- **Python Wrapper**: Thread-safe through Albumentations
- **Rust Core**: No shared state, inherently thread-safe

## Performance Characteristics

### Complexity
- **Image Blending**: O(W × H) where W, H = image dimensions
- **Collision Detection**: O(N²) where N = number of objects
- **Affine Transform**: O(1) per point

### Memory Usage
- Input image: O(W × H × 3) for RGB
- Masks: O(W × H) per object
- Temporary buffers: O(W × H) during blending

### Optimization Opportunities
1. SIMD for blending operations
2. Parallelization across objects
3. Cache locality improvements
4. GPU acceleration for large batches

## Integration Points

### With Albumentations
- Inherits from `DualTransform`
- Implements standard interface methods
- Compatible with `Compose` and pipelines
- Works with bbox_params

### With Training Frameworks
- **PyTorch**: Works with DataLoader
- **TensorFlow**: Works with tf.data.Dataset
- **MLFrameworks**: Compatible with any framework using Albumentations

## Testing Strategy

### Unit Tests
- Affine transformations
- Blending modes
- Collision detection
- Coordinate conversions

### Integration Tests
- End-to-end transformation
- Format handling
- Error cases
- Edge cases (empty inputs, extreme values)

### Performance Tests
- Benchmark different image sizes
- Profile memory usage
- Track transformation time
- Compare against baselines

## Build System

### Cargo.toml
- PyO3 for Python bindings
- ndarray for numerical operations
- image for image processing
- serde for serialization

### pyproject.toml
- Maturin for building Python extensions
- Dependencies: numpy, albumentations, opencv-python-headless
- Extras for benchmarking

## Future Enhancements

1. **Multi-GPU Support**: CUDA/Metal acceleration
2. **Streaming**: Process large images in chunks
3. **Caching**: LRU cache for common transformations
4. **Profiling**: Built-in performance monitoring
5. **Advanced Algorithms**: More blending modes, collision handling

## Development Workflow

```
1. Modify Rust code (src/)
2. Run tests: cargo test
3. Rebuild Python extension: maturin develop
4. Test integration: pytest tests/
5. Benchmark: python benchmarks/collect_metrics.py
```
