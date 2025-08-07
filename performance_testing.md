# Performance Testing for EVoC

This directory contains performance benchmark tests for the EVoC library, specifically targeting the `knn_graph` module which is performance-critical for the overall library functionality.

## Overview

The performance test suite is designed to:

1. **Detect Performance Regressions**: Catch significant slowdowns in code changes
2. **Monitor Scaling Behavior**: Ensure algorithms scale appropriately with data size
3. **Cross-Platform Compatibility**: Work across different hardware configurations
4. **CI/CD Integration**: Provide automated performance monitoring

## Running Performance Tests

### Basic Usage

```bash
# Run all performance tests
python -m pytest evoc/tests/test_knn_graph_performance.py -m performance -v

# Run specific performance test
python -m pytest evoc/tests/test_knn_graph_performance.py::TestKNNGraphPerformance::test_baseline_performance_check -v

# Run performance tests without capturing output (see print statements)
python -m pytest evoc/tests/test_knn_graph_performance.py -m performance -v -s
```

### Using the Performance Runner Script

The included script provides more advanced functionality:

```bash
# Basic performance test run
python scripts/run_performance_tests.py

# Save results to JSON file
python scripts/run_performance_tests.py --output results_20241207.json

# Compare against baseline performance
python scripts/run_performance_tests.py --baseline baseline_results.json --threshold 1.3

# Generate human-readable report
python scripts/run_performance_tests.py --report performance_report.json
```

### Regular vs Performance Tests

To run regular unit tests without performance tests:

```bash
# Run all tests except performance tests
python -m pytest -m "not performance"

# Run only regular knn_graph tests
python -m pytest evoc/tests/test_knn_graph.py
```

## Test Categories

### 1. Scaling Performance Tests
- **Purpose**: Verify performance scales appropriately with dataset size
- **Datasets**: Various combinations of (samples, features)
- **Metrics**: Samples processed per second, absolute duration
- **Tolerance**: Hardware-adaptive expectations

### 2. Parameter Performance Tests
- **Purpose**: Compare performance across different algorithm configurations
- **Parameters**: n_neighbors, n_trees, algorithm settings
- **Metrics**: Relative performance between configurations
- **Expectations**: All configurations should complete in reasonable time

### 3. Data Type Performance Tests
- **Purpose**: Compare performance across different input data types
- **Types**: float32, uint8, int8
- **Metrics**: Processing time for each data type
- **Expectations**: All types should have acceptable performance

### 4. Threading Performance Tests
- **Purpose**: Verify multi-threading provides expected speedup
- **Configurations**: 1 thread vs multiple threads
- **Metrics**: Speedup ratio, absolute performance
- **Hardware-Aware**: Adapts to available CPU cores

### 5. Memory Usage Tests
- **Purpose**: Monitor memory scaling behavior
- **Metrics**: Memory usage growth with dataset size
- **Expectations**: Memory should not grow faster than O(n^1.5)
- **Requirements**: Requires `psutil` library (optional)

### 6. Reproducibility Tests
- **Purpose**: Ensure consistent performance across runs
- **Metrics**: Coefficient of variation across multiple runs
- **Expectations**: Performance should be stable (CV < 20%)

### 7. Baseline Performance Tests
- **Purpose**: Establish and monitor performance baselines
- **Output**: Standardized performance report
- **Usage**: Reference for regression detection

## Hardware Adaptivity

The tests are designed to work across different hardware configurations:

- **CPU Detection**: Automatically detects available CPU cores
- **Memory Monitoring**: Optional memory usage tracking with `psutil`
- **Adaptive Thresholds**: Performance expectations scale with hardware capabilities
- **Fallback Modes**: Graceful degradation when optional dependencies unavailable

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Tests
on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest psutil
    
    - name: Run performance tests
      run: |
        python scripts/run_performance_tests.py --output performance_results.json
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: performance_results.json
```

### Baseline Comparison

To implement regression detection:

1. **Establish Baseline**: Run tests on a reference configuration
2. **Store Results**: Save baseline results as JSON
3. **Compare in CI**: Use baseline comparison in subsequent runs
4. **Alert on Regression**: Fail CI if performance degrades beyond threshold

```bash
# Create baseline (run once on reference system)
python scripts/run_performance_tests.py --output baseline_performance.json

# Compare in CI
python scripts/run_performance_tests.py --baseline baseline_performance.json --threshold 1.5
```

## Performance Expectations

### Typical Performance Ranges

These are rough guidelines and will vary significantly by hardware:

| Dataset Size | Expected Throughput | Use Case |
|--------------|-------------------|----------|
| 1K samples   | > 100 samples/sec | Small embeddings |
| 5K samples   | > 50 samples/sec  | Medium datasets |
| 10K samples  | > 20 samples/sec  | Large datasets |

### Regression Thresholds

- **Default Threshold**: 1.5x (50% slower triggers alert)
- **Conservative**: 1.3x (30% slower)
- **Aggressive**: 1.2x (20% slower)

## Troubleshooting

### Common Issues

1. **Tests Too Slow**: Reduce dataset sizes in parameterized tests
2. **Hardware Variance**: Adjust performance expectations in test assertions
3. **Memory Issues**: Install `psutil` for better memory monitoring
4. **CI Failures**: Use more conservative thresholds for CI environments

### Performance Debugging

```python
# Enable verbose output to see timing details
pytest evoc/tests/test_knn_graph_performance.py -v -s

# Profile specific functions
import cProfile
cProfile.run('knn_graph(data, n_neighbors=30)')
```

## Dependencies

### Required
- `numpy`
- `scikit-learn`
- `pytest`

### Optional
- `psutil` - For memory usage monitoring and CPU detection
- `pytest-json-report` - For JSON test result output
- `pytest-benchmark` - Alternative benchmarking framework

## Future Enhancements

- **Historical Tracking**: Database storage of performance trends
- **Visualization**: Plots of performance over time
- **Alerting**: Integration with monitoring systems
- **Profiling**: Automatic profiling of slow tests
- **Hardware Benchmarking**: Standardized hardware performance scoring
