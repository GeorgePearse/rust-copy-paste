# API Reference

## CopyPasteAugmentation

Main class for copy-paste augmentation with Albumentations integration.

### Constructor

```python
CopyPasteAugmentation(
    image_width: int = 512,
    image_height: int = 512,
    max_paste_objects: int = 1,
    use_rotation: bool = True,
    use_scaling: bool = True,
    rotation_range: tuple[float, float] = (-30.0, 30.0),
    scale_range: tuple[float, float] = (0.8, 1.2),
    use_random_background: bool = False,
    blend_mode: str = "normal",
    p: float = 1.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_width` | int | 512 | Width of output images in pixels |
| `image_height` | int | 512 | Height of output images in pixels |
| `max_paste_objects` | int | 1 | Maximum number of objects to paste per image |
| `use_rotation` | bool | True | Enable random rotation of pasted objects |
| `use_scaling` | bool | True | Enable random scaling of pasted objects |
| `rotation_range` | tuple[float, float] | (-30, 30) | Rotation range in degrees [min, max] |
| `scale_range` | tuple[float, float] | (0.8, 1.2) | Scaling factor range [min, max] |
| `use_random_background` | bool | False | Generate random background instead of using input |
| `blend_mode` | str | "normal" | Blending mode: "normal" or "xray" |
| `p` | float | 1.0 | Probability of applying transform (0.0 to 1.0) |

### Methods

#### apply()

```python
def apply(
    self,
    img: np.ndarray,
    **params: Any
) -> np.ndarray
```

Apply copy-paste augmentation to image.

**Parameters**:
- `img` (np.ndarray): Input image as BGR (H, W, C) with dtype uint8

**Returns**:
- np.ndarray: Augmented image with same shape and dtype

**Raises**:
- ValueError: If image is not BGR or has invalid shape

#### apply_to_bboxes()

```python
def apply_to_bboxes(
    self,
    bboxes: np.ndarray,
    **params: Any
) -> np.ndarray
```

Transform bounding boxes in Albumentations format.

**Parameters**:
- `bboxes` (np.ndarray): Bounding boxes with shape (N, 4) or (N, 5)
  - Format: [x_min, y_min, x_max, y_max, class_id]
  - Values: normalized to [0, 1]

**Returns**:
- np.ndarray: Transformed bboxes in same format

**Note**: Currently returns bboxes unchanged. Full implementation will update bbox positions based on pasted objects.

#### apply_to_masks()

```python
def apply_to_masks(
    self,
    masks: list[np.ndarray],
    **params: Any
) -> list[np.ndarray]
```

Transform masks (segmentation masks for objects).

**Parameters**:
- `masks` (list[np.ndarray]): List of binary masks, each (H, W)
  - Can be uint8 [0, 255] or float [0.0, 1.0]

**Returns**:
- list[np.ndarray]: Transformed masks as uint8

**Processing**:
- Validates each mask is 2D
- Converts to uint8 format if needed
- Ready for Rust processing

#### get_transform_init_args_names()

```python
def get_transform_init_args_names(self) -> tuple[str, ...]
```

Get names of arguments used to initialize transform (Albumentations interface).

**Returns**:
- tuple: Names of all initialization parameters

## SimpleCopyPaste

Alias for backwards compatibility.

```python
SimpleCopyPaste = CopyPasteAugmentation
```

## Configuration Parameters

### Image Dimensions

```python
image_width: int = 512      # Output width
image_height: int = 512     # Output height
```

### Object Control

```python
max_paste_objects: int = 1  # Max objects per image
```

### Geometric Transformations

```python
use_rotation: bool = True
rotation_range: tuple[float, float] = (-30.0, 30.0)  # Degrees

use_scaling: bool = True
scale_range: tuple[float, float] = (0.8, 1.2)       # Factors
```

### Background

```python
use_random_background: bool = False  # Generate random or use input
```

### Blending

```python
blend_mode: str = "normal"
# Options: "normal", "xray"
```

### Probability

```python
p: float = 1.0  # 0.0 (never) to 1.0 (always)
```

## Usage Patterns

### Minimal Configuration

```python
transform = CopyPasteAugmentation()
```

### Maximum Configuration

```python
transform = CopyPasteAugmentation(
    image_width=1024,
    image_height=1024,
    max_paste_objects=5,
    use_rotation=True,
    rotation_range=(-45, 45),
    use_scaling=True,
    scale_range=(0.5, 2.0),
    use_random_background=True,
    blend_mode="xray",
    p=0.8
)
```

### Albumentations Compose

```python
import albumentations as A

transform = A.Compose([
    CopyPasteAugmentation(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
], bbox_params=A.BboxParams(format='albumentations'))
```

## Data Format Reference

### Images

```
Format: uint8 BGR (OpenCV standard)
Shape: (height, width, 3)
Range: [0, 255]
```

### Bounding Boxes (Albumentations format)

```
Format: [x_min, y_min, x_max, y_max, class_id]
Coordinates: Normalized [0.0, 1.0]
Example: [0.1, 0.2, 0.5, 0.8, 0]
```

### Masks

```
Format: List of binary masks
Shape: Each mask is (height, width)
Type: uint8 [0, 255] or float [0.0, 1.0]
```

## Error Handling

The transform includes graceful error handling:

- **Image format errors**: Returns original image unchanged
- **Bbox errors**: Returns original bboxes unchanged
- **Mask errors**: Validates and converts formats automatically

## Performance Notes

- Rust implementation provides 5-10x speedup
- Memory usage: O(W × H) for image operations
- Computation time scales with image size
- Probability parameter `p` can reduce computation

## Examples

See [Quick Start](../getting-started/quickstart.md) for usage examples.

## See Also

- [Albumentations Documentation](https://albumentations.ai/)
- [Getting Started Guide](../getting-started/quickstart.md)
- [Architecture Overview](../getting-started/architecture.md)
