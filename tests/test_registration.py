"""Test that CustomCopyPaste can be instantiated directly."""


def test_transform_instantiation():
    """Test that we can instantiate the transform."""
    from copy_paste import CustomCopyPaste

    # Create directly
    config = {"target_image_width": 512, "target_image_height": 512, "mm_class_list": ["test_class"], "paste_prob": 0.5}

    transform = CustomCopyPaste(**config)
    assert transform is not None
    assert isinstance(transform, CustomCopyPaste)
    assert transform.target_image_width == 512
    assert transform.target_image_height == 512


def test_rust_transform_instantiation():
    """Test that we can instantiate the RustCopyPaste transform."""
    try:
        from copy_paste import RustCopyPaste

        # Create directly
        config = {"target_image_width": 512, "target_image_height": 512, "mm_class_list": ["test_class"], "paste_prob": 0.5}

        transform = RustCopyPaste(**config)
        assert transform is not None
        assert isinstance(transform, RustCopyPaste)
        assert transform.target_image_width == 512
        assert transform.target_image_height == 512
    except RuntimeError as e:
        # RustCopyPaste requires rusty_paste which may not be installed
        if "rusty_paste" not in str(e):
            raise
