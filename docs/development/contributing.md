# Contributing

We welcome contributions! This guide will help you get started with developing the Copy-Paste augmentation project.

## Prerequisites

- Rust 1.70+
- Python 3.8+
- Git

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/GeorgePearse/rust-copy-paste.git
cd rust-copy-paste
```

2. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

3. Build the Rust components:
```bash
cargo build
```

## Making Changes

### Rust Changes

1. Make your changes in the `src/` directory
2. Run tests: `cargo test`
3. Run clippy for linting: `cargo clippy`
4. Format code: `cargo fmt`

### Python Changes

1. Make changes in the `copy_paste/` directory
2. Run type checking with `pyright`
3. Format with `black`
4. Run linting with `ruff`

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=copy_paste
```

### Integration Tests

```bash
# Run Rust integration tests
cargo test --test integration_tests
```

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and commit: `git commit -am 'Add feature'`
3. Push to your fork: `git push origin feature/your-feature`
4. Submit a pull request

## Code Style

- Python: Follow PEP 8, use type hints
- Rust: Follow Rust conventions, use clippy recommendations
- Documentation: Write docstrings for all public functions

## Reporting Issues

Please report issues on the [GitHub issue tracker](https://github.com/GeorgePearse/rust-copy-paste/issues).

Include:
- Python version
- Rust version (if applicable)
- Minimal reproducible example
- Error messages or stack traces

## Questions?

Feel free to open an issue or discussion on GitHub!
