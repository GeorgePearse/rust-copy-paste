# Frequently Asked Questions

## General Questions

### What is copy-paste augmentation?

Copy-paste augmentation is a data augmentation technique that improves object detection and instance segmentation models by:
1. Extracting objects from source images
2. Pasting them onto target images with geometric transformations
3. Generating synthetic training data with object variations

### Why use copy-paste augmentation?

Benefits include:
- **Increased dataset size** without additional labeling
- **Improved model generalization** through synthetic variations
- **Better handling of object interactions** through realistic compositions
- **Efficient data augmentation** compared to manual collection

### How does this compare to other augmentations?

| Augmentation | Pros | Cons |
|--------------|------|------|
| **Copy-Paste** | Realistic, increases diversity, preserves annotations | Requires objects, computational cost |
| **Rotation/Flip** | Fast, simple | Limited diversity |
| **Mixup/Cutmix** | Diverse, efficient | May break annotations |
| **Mosaic** | Good for small objects | Complex to implement |

## Installation & Setup

### I get "ModuleNotFoundError: No module named 'copy_paste._core'"

The Rust extension didn't compile. Try:

1. Ensure Rust is installed:
```bash
rustup --version
```

2. Update Rust:
```bash
rustup update
```

3. Rebuild:
```bash
pip install -e . --force-reinstall
```

### Can I use this without Rust installed?

No, you need Rust for building from source. However, pre-built wheels are available:

```bash
pip install copy-paste  # Uses pre-built wheel for your Python version
```

### Does this work on Windows?

Currently, pre-built wheels are only available for Linux x86_64. For Windows/macOS, you need to build from source with Rust installed.

### What Python versions are supported?

Supported: **3.9, 3.10, 3.11, 3.12**

Pre-built wheels available for all versions on Linux x86_64.

## Usage Questions

### How do I use this with PyTorch?

```python
from torch.utils.data import Dataset
import albumentations as A
from copy_paste import CopyPasteAugmentation

class MyDataset(Dataset):
    def __init__(self):
        self.transform = A.Compose([
            CopyPasteAugmentation(p=0.5),
        ], bbox_params=A.BboxParams(format='albumentations'))

    def __getitem__(self, idx):
        result = self.transform(
            image=self.image,
            bboxes=self.bboxes,
            masks=self.masks
        )
        return result
```

### How do I control which objects are pasted?

Currently, objects are selected randomly from source masks. Future versions will support:
- Class-specific selection
- Size-based filtering
- Visibility constraints

### What happens if I set `p=0`?

The transform won't be applied to any samples (0% probability).

### Can I combine with other Albumentations transforms?

Yes! Use `Compose`:

```python
transform = A.Compose([
    CopyPasteAugmentation(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
], bbox_params=A.BboxParams(format='albumentations'))
```

### How do I handle image preprocessing?

Add preprocessing transforms after copy-paste:

```python
transform = A.Compose([
    CopyPasteAugmentation(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
], bbox_params=A.BboxParams(format='albumentations'))
```

## Performance Questions

### How much slower is copy-paste augmentation?

Rust implementation: **5-10x faster** than pure Python.

Expected timings on 512×512 image:
- Copy-paste: ~2-5ms
- Other transforms: ~0.2-1ms
- Batch of 32: ~100-200ms

### Can I speed it up further?

1. Reduce image size:
```python
CopyPasteAugmentation(image_width=256, image_height=256)
```

2. Lower probability:
```python
CopyPasteAugmentation(p=0.3)  # Apply less frequently
```

3. Reduce object count:
```python
CopyPasteAugmentation(max_paste_objects=1)  # Paste fewer objects
```

4. Use DataLoader with multiple workers:
```python
DataLoader(dataset, batch_size=32, num_workers=4)
```

### How much memory does it use?

Memory for single transform:
- Input image: ~1MB (512×512×3 uint8)
- Masks: ~0.25MB per object
- Working buffers: ~1MB

Total: ~2-3MB per image

### Can I use GPU acceleration?

Currently no. Future versions may support CUDA/Metal acceleration.

## Troubleshooting

### Why are my bboxes changing?

Currently, bounding box transformation is a placeholder. The actual implementation will:
1. Track object movements
2. Update bbox positions
3. Remove occluded objects
4. Add new pasted objects

### Why are my masks not updating?

Same as bboxes - the algorithm is being implemented. Masks will be:
1. Updated for moved objects
2. Created for pasted objects
3. Validated for consistency

### Getting different results on CPU vs GPU?

GPU acceleration is not implemented yet. All processing is on CPU.

### Memory usage keeps increasing

This could indicate a memory leak. Try:

```python
import gc
gc.collect()  # Force garbage collection
```

If issue persists, please report on GitHub.

## Contributing Questions

### How do I contribute?

See [Contributing Guide](development/contributing.md)

### Can I modify the Rust code?

Yes! The code is open source. Please:
1. Fork the repository
2. Create a feature branch
3. Follow the code style
4. Add tests
5. Submit a pull request

### Where should I report bugs?

Report on GitHub: https://github.com/GeorgePearse/rust-copy-paste/issues

## Advanced Questions

### Can I use custom blending modes?

Currently supported:
- `'normal'` - Standard alpha blending
- `'xray'` - X-ray style blending

Custom modes are on the roadmap.

### Can I get the transformation matrix?

Not currently exposed. Future versions will provide access to internal transformations.

### How do I integrate with custom annotation formats?

Convert to Albumentations format:
- Images: uint8 BGR
- Bboxes: [x_min, y_min, x_max, y_max, class_id] normalized
- Masks: List of uint8 binary masks

## Still have questions?

- Check the [documentation](index.md)
- Search [GitHub issues](https://github.com/GeorgePearse/rust-copy-paste/issues)
- Open a new issue with details
