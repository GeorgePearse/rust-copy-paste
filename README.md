# copy_paste

Custom Copy-Paste augmentation transform for object detection and instance segmentation.

## Overview

This package provides the `CustomCopyPaste` augmentation transform, which implements a sophisticated copy-paste augmentation technique for training object detection and instance segmentation models. The transform can:

- Load objects from COCO annotation files
- Paste objects onto background images with configurable parameters
- Generate random backgrounds for augmentation
- Control object placement, scaling, rotation, and blending
- Support class-specific pasting probabilities and counts

## Installation

This package is part of the workspace and can be installed via UV:

```bash
uv pip install -e ./copy_paste
```

## Usage

The transform is automatically registered with visdet's TRANSFORMS registry when imported:

```python
from copy_paste import CustomCopyPaste

# In your config file
transform = dict(
    type='CustomCopyPaste',
    target_image_width=1024,
    target_image_height=1024,
    mm_class_list=['class1', 'class2'],
    annotation_file='/path/to/coco/annotations.json',
    paste_prob=0.5,
    max_paste_objects=3,
    scale_range=(0.8, 1.2),
    rotation_range=(0, 360),
    use_random_background=True,
    random_background_prob=0.3
)
```

## Features

- **COCO Format Support**: Loads objects directly from COCO annotation files
- **Flexible Object Selection**: Configure which classes to use and how many objects to paste
- **Random Backgrounds**: Generate various types of random backgrounds (gradients, noise, patterns)
- **Advanced Blending**: Support for normal and x-ray blending modes
- **Performance Optimization**: LRU caching for efficient image loading
- **Detailed Logging**: Track object counts at each pipeline stage

## Parameters

- `target_image_width/height`: Target dimensions for resized images
- `mm_class_list`: List of class names in MMDetection format
- `annotation_file`: Path to COCO annotation file with source objects
- `paste_prob`: Probability of applying the transform
- `max_paste_objects`: Maximum number of objects to paste per image
- `object_counts`: Dictionary specifying exact counts or probabilities per class
- `scale_range`: Scale range for pasted objects (min, max)
- `rotation_range`: Rotation range in degrees
- `use_random_background`: Enable random background generation
- `blend_mode`: Blending mode ('normal' or 'xray')

## License

Part of the internal machine learning packages.
