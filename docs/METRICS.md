# Performance Metrics System

This document describes the automated performance benchmarking system for rust-copy-paste.

## Overview

Performance metrics are automatically collected on every push to `master` branch using GitHub Actions. Results are stored in JSONL format and visualized on the [metrics dashboard](https://georgepearse.github.io/rust-copy-paste/).

## How It Works

### 1. Automated Benchmarks

Every commit to master triggers:

```
GitHub Push → GitHub Actions → pytest-benchmark → collect_metrics.py → metrics/metrics.json → GitHub Pages
```

### 2. Data Format (JSONL)

Each line in `metrics/metrics.json` is a complete JSON object:

```json
{
  "host": {
    "os": "Linux-5.10.0-36-cloud-amd64-x86_64",
    "cpu": "AMD EPYC 7763 64-Core Processor",
    "mem": "16.0 GB",
    "python_version": "3.10.12"
  },
  "timestamp": 1730304000,
  "revision": "8d5ffdc",
  "metrics": {
    "baseline_512x512_1obj/time_ms": [22.5, "ms"],
    "baseline_512x512_1obj/throughput": [44.4, "img/s"],
    "baseline_512x512_1obj/max_ms": [28.3, "ms"],
    "large_2048x2048_1obj/time_ms": [89.2, "ms"],
    "large_2048x2048_1obj/throughput": [11.2, "img/s"],
    "large_2048x2048_1obj/max_ms": [120.5, "ms"]
  }
}
```

## Benchmark Scenarios

The following scenarios are tested on each run:

| Scenario | Description | Key Metric |
|----------|-------------|-----------|
| `baseline_512x512_1obj` | Standard case: 512×512 image, 1 object | Baseline throughput |
| `baseline_512x512_3obj` | Standard case: 512×512 image, 3 objects | Multi-object overhead |
| `large_1024x1024_1obj` | Medium image: 1024×1024 pixels | Scaling behavior |
| `large_2048x2048_1obj` | Large image: 2048×2048 pixels | Maximum size handling |
| `multi_object_5obj` | Heavy: 512×512 image, 5 objects | Augmentation overhead |
| `multi_object_10obj` | Very heavy: 512×512 image, 10 objects | Worst-case scenario |
| `with_rotation` | With rotation: 0-360° random | Rotation cost |
| `with_scaling` | With scaling: 0.8-1.2× random | Scaling cost |
| `random_background` | Random background generation | Background cost |
| `xray_blending` | X-ray blend mode | Alternative blending |

## Metrics Collected

For each scenario:

- **time_ms**: Median processing time in milliseconds (lower is better)
- **throughput**: Images processed per second (higher is better)
- **max_ms**: Maximum processing time (P100 latency)

## Running Benchmarks Locally

### Install benchmark dependencies:

```bash
uv pip install -e ".[benchmarks]"
```

### Run benchmarks manually:

```bash
python benchmarks/collect_metrics.py
```

This will:
1. Generate/use the dummy dataset
2. Run pytest-benchmark on all scenarios
3. Collect system metrics (OS, CPU, memory)
4. Append results to `metrics/metrics.json`
5. Display a summary

### Run specific benchmarks:

```bash
# Run only baseline tests
pytest benchmarks/benchmark_copy_paste.py::TestCopyPasteBenchmarks::test_transform[baseline_512x512_1obj] -v

# Run with warmup and disable garbage collection
pytest benchmarks/benchmark_copy_paste.py -v --benchmark-warmup=on --benchmark-disable-gc

# Save raw benchmark data
pytest benchmarks/benchmark_copy_paste.py --benchmark-json=results.json
```

## Dashboard Features

The metrics dashboard at https://georgepearse.github.io/rust-copy-paste/ provides:

### Charts

1. **Processing Time** - Trend of median processing time over commits
2. **Throughput** - Images per second over time
3. **Performance Comparison** - Latest results by scenario (bar chart)
4. **Maximum Time** - P100 latency trends

### Controls

- **Normalize to first value**: View relative performance changes
- **Download Metrics**: Export raw JSONL data
- **Interactive tooltips**: Hover for exact values

## GitHub Actions Workflow

Location: `.github/workflows/benchmarks.yml`

**Triggers:**
- Push to `master` or `main` branch
- Weekly schedule (Sundays at midnight UTC)

**Steps:**
1. Checkout code with full git history
2. Setup Python 3.10 + uv
3. Install dependencies including `[benchmarks]` extras
4. Generate dummy dataset
5. Run `collect_metrics.py`
6. Commit and push metrics to repository
7. Upload metrics as GitHub artifact
8. Generate summary in workflow run

## Performance Expectations

Based on typical hardware:

| Scenario | Expected Throughput | Expected Time |
|----------|-------------------|--------------|
| Baseline (512×512, 1 obj) | 40-50 img/s | 20-25 ms |
| Large (2048×2048, 1 obj) | 8-12 img/s | 80-125 ms |
| Heavy (512×512, 10 obj) | 5-8 img/s | 125-200 ms |

**Note:** Actual performance depends on:
- Hardware (CPU cores, RAM, disk speed)
- System load during benchmark run
- Python/library versions
- Caching effectiveness

## Regression Detection

Compare against baseline to identify performance regressions:

```javascript
// Example: Calculate % change
latest_throughput = 45.2  // img/s
baseline_throughput = 48.1  // img/s
regression = ((baseline - latest) / baseline) * 100
// Result: 5.9% regression
```

## Customizing Benchmarks

### Add a new scenario:

1. Edit `benchmarks/scenarios.py`:
   ```python
   SCENARIOS = {
       "my_scenario": BenchmarkConfig(
           name="My Custom Scenario",
           image_width=512,
           image_height=512,
           max_paste_objects=2,
           use_rotation=True,
       )
   }
   ```

2. Benchmarks will automatically run for the new scenario

### Modify collection script:

Edit `benchmarks/collect_metrics.py` to:
- Change benchmark parameters
- Add custom metric collection
- Modify system info captured

## Integration with CI/CD

The benchmark workflow:
- ✅ Runs automatically on every master commit
- ✅ Does NOT block merges (continues on error)
- ✅ Commits metrics back to repository
- ✅ Enables GitHub Pages visualization
- ⏱️ Takes ~5-10 minutes to complete

## Troubleshooting

### Metrics not updating

1. Check workflow run: GitHub → Actions → Performance Benchmarks
2. Review error logs in workflow step
3. Verify `metrics/metrics.json` file exists
4. Ensure GitHub Pages is enabled (Settings → Pages)

### Missing data points

- New commits may not have metrics (workflow in progress)
- Failed benchmarks are skipped
- Check workflow artifacts for details

### Dashboard not loading

1. Clear browser cache
2. Check browser console for errors
3. Verify `metrics/metrics.json` file is readable
4. Ensure GitHub Pages is configured

## Best Practices

1. **Regular monitoring**: Check dashboard weekly for trends
2. **Investigate regressions**: >5-10% drops warrant investigation
3. **Note infrastructure changes**: Document system/library updates that affect metrics
4. **Comparison context**: Compare within similar hardware/conditions
5. **Long-term trends**: Focus on trends rather than individual data points

## See Also

- [Benchmark Code](../benchmarks/benchmark_copy_paste.py)
- [Metrics Collection](../benchmarks/collect_metrics.py)
- [Raw Metrics Data](../metrics/metrics.json)
- [Dashboard](https://georgepearse.github.io/rust-copy-paste/)
