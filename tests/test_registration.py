"""Test that CustomCopyPaste is properly registered with visdet."""

from visdet.registry import TRANSFORMS


def test_transform_registered():
    """Test that CustomCopyPaste is registered in TRANSFORMS registry."""
    # Import should trigger registration
    from copy_paste import CustomCopyPaste

    # Check registration by name
    cls = TRANSFORMS.get("CustomCopyPaste")
    assert cls is not None
    assert cls is CustomCopyPaste


def test_transform_instantiation():
    """Test that we can instantiate the transform through the registry."""
    from copy_paste import CustomCopyPaste  # Ensure registration

    # Create through registry
    config = {"type": "CustomCopyPaste", "target_image_width": 512, "target_image_height": 512, "mm_class_list": ["test_class"], "paste_prob": 0.5}

    transform = TRANSFORMS.build(config)
    assert transform is not None
    assert isinstance(transform, CustomCopyPaste)
    assert transform.target_image_width == 512
    assert transform.target_image_height == 512
