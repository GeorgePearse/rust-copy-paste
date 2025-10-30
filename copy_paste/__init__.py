"""Custom Copy-Paste augmentation for object detection."""

from .custom_copy_paste import CustomCopyPaste
from .rust_copy_paste import RustCopyPaste

__version__ = "0.1.0"
__all__ = ["CustomCopyPaste", "RustCopyPaste"]

# Auto-registration happens when transforms are imported
# The @TRANSFORMS.register_module decorator automatically registers
# the classes with visdet's TRANSFORMS registry
