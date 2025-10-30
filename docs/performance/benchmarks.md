# Benchmarks

The Copy-Paste augmentation includes comprehensive performance benchmarks to track and optimize performance across releases.

## Running Benchmarks

### Install Dependencies

```bash
uv pip install -e ".[benchmarks]"
```

### Run All Benchmarks

```bash
python benchmarks/collect_metrics.py
```

This will:
1. Generate test data
2. Run benchmarks for all scenarios
3. Collect system metrics
4. Store results in `metrics/metrics.json`
5. Display a summary report

### Run Specific Benchmarks

```bash
# Run only baseline tests
pytest benchmarks/benchmark_copy_paste.py::TestCopyPasteBenchmarks::test_transform -k "baseline_512x512_1obj" -v

# Run with specific options
pytest benchmarks/benchmark_copy_paste.py -v --benchmark-warmup=on --benchmark-disable-gc

# Save raw benchmark data
pytest benchmarks/benchmark_copy_paste.py --benchmark-json=results.json
```

## Benchmark Scenarios

The benchmarking suite tests multiple scenarios to cover different use cases:

| Scenario | Configuration | Typical Time | Typical Throughput |
|----------|---------------|--------------|-------------------|
| baseline_512x512_1obj | 512×512 image, 1 object | 20-25 ms | 40-50 img/s |
| baseline_512x512_3obj | 512×512 image, 3 objects | 25-35 ms | 30-40 img/s |
| large_1024x1024_1obj | 1024×1024 pixels | 50-65 ms | 15-20 img/s |
| large_2048x2048_1obj | 2048×2048 pixels | 80-125 ms | 8-12 img/s |
| multi_object_5obj | 512×512 image, 5 objects | 40-60 ms | 15-25 img/s |
| multi_object_10obj | 512×512 image, 10 objects | 125-200 ms | 5-8 img/s |
| with_rotation | With random rotation | +10-15% overhead | |
| with_scaling | With random scaling | +5-10% overhead | |
| random_background | Background generation | +20-30% overhead | |
| xray_blending | X-ray blend mode | +10-20% overhead | |

## Performance Tips

### Optimize for Production

1. **Use Release Build**: Always use `cargo build --release`
2. **Batch Processing**: Process multiple images in batches for better throughput
3. **Image Size**: Smaller images are faster; consider downsampling if possible
4. **Object Count**: Fewer objects paste faster; adjust `max_paste_objects`

### Performance Comparison

Compare performance across different configurations:

```bash
# Baseline
pytest benchmarks/benchmark_copy_paste.py -v

# After optimization
pytest benchmarks/benchmark_copy_paste.py -v --benchmark-json=optimized.json

# Compare results
pytest benchmarks/benchmark_copy_paste.py --benchmark-compare=optimized.json
```

## Understanding Results

### Key Metrics

- **mean**: Average processing time
- **stddev**: Standard deviation (consistency)
- **min/max**: Minimum and maximum times observed
- **throughput**: Images processed per second

### Performance Expectations

Typical performance on modern hardware:

```
512×512, 1 object:    20-25 ms (~45 img/s)
1024×1024, 1 object:  50-65 ms (~15 img/s)
2048×2048, 1 object:  80-125 ms (~10 img/s)
```

Performance scales linearly with:
- Image dimensions (O(W×H))
- Number of objects (O(n))

## Continuous Benchmarking

Benchmarks are automatically run on every commit to `master`:

1. GitHub Actions workflow executes benchmarks
2. Results are stored in `metrics/metrics.json`
3. Dashboard visualizes trends over time
4. Regressions are detected automatically

See [Metrics Dashboard](metrics.md) for live performance data.

## Troubleshooting

### Inconsistent Results
- Close other applications
- Disable dynamic CPU scaling
- Run benchmarks multiple times
- Check system load: `top` or `htop`

### Slow Performance
- Verify release build: `cargo build --release`
- Check CPU usage during benchmarks
- Profile with: `perf stat python benchmarks/collect_metrics.py`
- Compare against baseline metrics

### Missing Data
- Ensure pytest-benchmark is installed
- Check for errors in benchmark runs
- Verify write access to `metrics/` directory
