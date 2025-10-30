# Albumentations Integration

The Copy-Paste augmentation is fully integrated with Albumentations, allowing seamless integration into your existing augmentation pipelines.

## Basic Usage

```python
import albumentations as A
from copy_paste import CopyPasteAugmentation

# Create transform
transform = A.Compose([
    CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=3,
        p=0.5
    ),
], bbox_params=A.BboxParams(format='albumentations'))

# Apply to image data
result = transform(
    image=image,
    bboxes=bboxes,
    masks=masks
)
```

## Parameters

The `CopyPasteAugmentation` transform accepts the following parameters:

- `image_width` (int): Target image width
- `image_height` (int): Target image height
- `max_paste_objects` (int): Maximum number of objects to paste
- `use_rotation` (bool): Enable object rotation
- `use_scaling` (bool): Enable object scaling
- `blend_mode` (str): Blending mode ('normal' or 'xray')
- `p` (float): Probability of applying the transform (0-1)

## Bounding Box Support

The transform automatically handles bounding box transformations:

```python
bbox_params = A.BboxParams(
    format='albumentations',
    min_area=0,
    min_visibility=0
)

transform = A.Compose([
    CopyPasteAugmentation(...),
], bbox_params=bbox_params)
```

## Mask Support

Both binary masks and instance segmentation masks are supported:

```python
result = transform(
    image=image,
    bboxes=bboxes,
    masks=masks  # List of boolean arrays
)
```

## Pipeline Integration

Combine with other Albumentations transforms:

```python
transform = A.Compose([
    A.Resize(512, 512),
    CopyPasteAugmentation(
        image_width=512,
        image_height=512,
        max_paste_objects=3,
        p=0.8
    ),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
], bbox_params=bbox_params)
```

## Performance Considerations

- The Rust backend provides **5-10x speedup** compared to pure Python implementations
- Optimal performance with batch sizes of 32 or larger
- Memory usage scales linearly with object database size

## Troubleshooting

### Objects not being pasted
- Ensure `p > 0` to enable the transform
- Check that `max_paste_objects` is greater than 0
- Verify source images contain valid objects

### Bounding box issues
- Ensure bboxes are in the correct format (albumentations format: [x_min, y_min, x_max, y_max])
- Set appropriate `min_area` and `min_visibility` in `BboxParams`

### Performance issues
- Reduce `max_paste_objects` if processing is slow
- Consider reducing image dimensions if memory is constrained
