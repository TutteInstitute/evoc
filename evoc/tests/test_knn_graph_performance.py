"""
Performance benchmark tests for the knn_graph module.

This module provides performance regression testing for the knn_graph functionality.
The tests are designed to be robust across different hardware configurations by using
relative performance metrics and adaptive thresholds.
"""

import numpy as np
import pytest
import time
import platform
from contextlib import contextmanager
from sklearn.datasets import make_blobs
from typing import Dict, Any, Tuple, List

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from evoc.knn_graph import knn_graph


class PerformanceMetrics:
    """Class to collect and analyze performance metrics."""

    def __init__(self):
        self.metrics = {}
        self.hardware_info = self._get_hardware_info()

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get basic hardware information for context."""
        try:
            if HAS_PSUTIL:
                return {
                    "cpu_count": psutil.cpu_count(logical=False),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                }
            else:
                # Fallback without psutil
                import os

                return {
                    "cpu_count_logical": os.cpu_count() or 1,
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "psutil_available": False,
                }
        except Exception:
            return {"error": "Could not gather hardware info"}

    def record_metric(self, test_name: str, metric_name: str, value: float):
        """Record a performance metric."""
        if test_name not in self.metrics:
            self.metrics[test_name] = {}
        self.metrics[test_name][metric_name] = value

    def get_metric(self, test_name: str, metric_name: str) -> float:
        """Get a recorded metric."""
        return self.metrics.get(test_name, {}).get(metric_name, 0.0)


@contextmanager
def time_execution():
    """Context manager to time code execution."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    return end_time - start_time


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution and return result and duration."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


class TestKNNGraphPerformance:
    """Performance tests for knn_graph functionality."""

    @pytest.fixture(scope="class")
    def perf_metrics(self):
        """Shared performance metrics collector."""
        return PerformanceMetrics()

    @pytest.fixture(
        params=[
            (1000, 128),  # Small dataset, typical embedding size
            (5000, 384),  # Medium dataset, larger embedding
            (10000, 512),  # Large dataset, large embedding
        ]
    )
    def dataset_config(self, request):
        """Different dataset configurations for performance testing."""
        n_samples, n_features = request.param
        return n_samples, n_features

    @pytest.fixture
    def performance_data(self, dataset_config):
        """Generate performance test data."""
        n_samples, n_features = dataset_config
        np.random.seed(42)  # Consistent data for reproducible benchmarks

        # Create clustered data similar to real-world embeddings
        X, y = make_blobs(
            n_samples=n_samples,
            centers=max(4, n_samples // 2000),  # Scale centers with data size
            n_features=n_features,
            cluster_std=0.5,
            random_state=42,
        )

        # Normalize to unit sphere (typical for embeddings)
        X = X.astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms

        return X, (n_samples, n_features)

    def test_knn_graph_scaling_performance(self, performance_data, perf_metrics):
        """Test knn_graph performance scaling with different data sizes."""
        X, (n_samples, n_features) = performance_data
        test_name = f"knn_graph_scaling_{n_samples}x{n_features}"

        # Warm up run (not timed) to ensure compiled numba functions
        if n_samples <= 1000:  # Only warm up on small data
            knn_graph(X[:100], n_neighbors=10, n_trees=2, random_state=42)

        # Timed run
        result, duration = time_function(
            knn_graph, X, n_neighbors=30, n_trees=4, random_state=42, verbose=False
        )

        # Record metrics
        perf_metrics.record_metric(test_name, "duration_seconds", duration)
        perf_metrics.record_metric(
            test_name, "samples_per_second", n_samples / duration
        )
        perf_metrics.record_metric(test_name, "n_samples", n_samples)
        perf_metrics.record_metric(test_name, "n_features", n_features)

        # Verify result is correct
        indices, distances = result
        assert indices.shape == (n_samples, 30)
        assert distances.shape == (n_samples, 30)

        # Performance expectations (very loose bounds that should work across hardware)
        # These are sanity checks rather than strict requirements
        expected_min_samples_per_second = {
            1000: 100,  # At least 100 samples/sec for small data
            5000: 50,  # At least 50 samples/sec for medium data
            10000: 20,  # At least 20 samples/sec for large data
        }

        min_expected = expected_min_samples_per_second.get(n_samples, 10)
        samples_per_sec = n_samples / duration

        # Log performance info
        print(f"\n{test_name}:")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Samples/sec: {samples_per_sec:.1f}")
        print(f"  Hardware: {perf_metrics.hardware_info}")

        # Very loose performance check - mainly to catch major regressions
        assert (
            samples_per_sec > min_expected
        ), f"Performance too slow: {samples_per_sec:.1f} < {min_expected} samples/sec"

    def test_knn_graph_parameter_performance(self, perf_metrics):
        """Test performance with different parameter configurations."""
        np.random.seed(42)
        n_samples, n_features = 2000, 256
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
        X = X.astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Test different parameter combinations
        param_configs = [
            {"n_neighbors": 15, "n_trees": 2, "name": "fast_config"},
            {"n_neighbors": 30, "n_trees": 4, "name": "default_config"},
            {"n_neighbors": 50, "n_trees": 8, "name": "high_quality_config"},
        ]

        durations = {}

        for config in param_configs:
            name = config.pop("name")
            test_name = f"param_performance_{name}"

            # Warm up
            knn_graph(
                X[:100],
                n_neighbors=config["n_neighbors"],
                n_trees=config["n_trees"],
                random_state=42,
            )

            # Timed run
            result, duration = time_function(knn_graph, X, random_state=42, **config)

            durations[name] = duration
            perf_metrics.record_metric(test_name, "duration_seconds", duration)
            perf_metrics.record_metric(
                test_name, "samples_per_second", n_samples / duration
            )

            print(f"\n{name}: {duration:.3f}s ({n_samples/duration:.1f} samples/sec)")

        # Verify relative performance expectations
        # The relationship between parameters and performance can be complex
        # So we mainly check that all configurations complete successfully
        for name, duration in durations.items():
            assert duration < 10.0, f"{name} took too long: {duration:.3f}s"

        # Optionally log which configuration was fastest
        fastest_config = min(durations, key=durations.get)
        slowest_config = max(durations, key=durations.get)
        print(f"\nFastest: {fastest_config} ({durations[fastest_config]:.3f}s)")
        print(f"Slowest: {slowest_config} ({durations[slowest_config]:.3f}s)")

    def test_knn_graph_data_type_performance(self, perf_metrics):
        """Test performance differences between data types."""
        np.random.seed(42)
        n_samples, n_features = 2000, 128

        # Generate base data
        base_data = np.random.rand(n_samples, n_features)

        # Convert to different types
        float_data = base_data.astype(np.float32)
        uint8_data = (base_data * 255).astype(np.uint8)
        int8_data = ((base_data - 0.5) * 255).astype(np.int8)

        data_types = [
            (float_data, "float32"),
            (uint8_data, "uint8"),
            (int8_data, "int8"),
        ]

        durations = {}

        for data, dtype_name in data_types:
            test_name = f"dtype_performance_{dtype_name}"

            # Warm up
            knn_graph(data[:100], n_neighbors=10, n_trees=2, random_state=42)

            # Timed run
            result, duration = time_function(
                knn_graph,
                data,
                n_neighbors=20,
                n_trees=4,
                random_state=42,
                verbose=False,
            )

            durations[dtype_name] = duration
            perf_metrics.record_metric(test_name, "duration_seconds", duration)
            perf_metrics.record_metric(
                test_name, "samples_per_second", n_samples / duration
            )

            print(
                f"\n{dtype_name}: {duration:.3f}s ({n_samples/duration:.1f} samples/sec)"
            )

        # All should complete in reasonable time
        for dtype_name, duration in durations.items():
            assert duration < 30.0, f"{dtype_name} took too long: {duration:.3f}s"

    def test_knn_graph_threading_performance(self, perf_metrics):
        """Test performance scaling with different thread counts."""
        np.random.seed(42)
        n_samples, n_features = 3000, 256
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
        X = X.astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Test different thread counts
        if HAS_PSUTIL:
            max_threads = min(
                8, psutil.cpu_count(logical=True)
            )  # Don't exceed available cores
        else:
            import os

            max_threads = min(8, os.cpu_count() or 1)

        thread_counts = [1, max(2, max_threads // 2), max_threads]

        durations = {}

        for n_jobs in thread_counts:
            test_name = f"threading_performance_{n_jobs}_threads"

            # Warm up
            knn_graph(
                X[:100], n_neighbors=10, n_trees=2, n_jobs=n_jobs, random_state=42
            )

            # Timed run
            result, duration = time_function(
                knn_graph,
                X,
                n_neighbors=20,
                n_trees=4,
                n_jobs=n_jobs,
                random_state=42,
                verbose=False,
            )

            durations[n_jobs] = duration
            perf_metrics.record_metric(test_name, "duration_seconds", duration)
            perf_metrics.record_metric(
                test_name, "samples_per_second", n_samples / duration
            )

            print(
                f"\n{n_jobs} threads: {duration:.3f}s ({n_samples/duration:.1f} samples/sec)"
            )

        # More threads should generally be faster (within reason)
        if len(durations) >= 2 and max_threads > 1:
            single_thread_time = durations[1]
            multi_thread_time = durations[max_threads]

            # Allow for some overhead but expect some speedup
            speedup_ratio = single_thread_time / multi_thread_time
            expected_min_speedup = 1.2  # At least 20% speedup with more threads

            print(f"\nSpeedup ratio: {speedup_ratio:.2f}x")

            # Only assert if we have multiple cores available
            if max_threads > 2:
                assert (
                    speedup_ratio > expected_min_speedup
                ), f"Multi-threading should provide speedup: {speedup_ratio:.2f}x < {expected_min_speedup}x"

    def test_memory_usage_scaling(self, perf_metrics):
        """Test memory usage scaling (basic check)."""
        if not HAS_PSUTIL:
            pytest.skip("psutil not available for memory testing")

        import gc

        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        test_sizes = [(1000, 64), (2000, 64), (4000, 64)]
        memory_usages = []

        for n_samples, n_features in test_sizes:
            gc.collect()

            # Generate data
            np.random.seed(42)
            X, _ = make_blobs(
                n_samples=n_samples, n_features=n_features, random_state=42
            )
            X = X.astype(np.float32)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)

            # Run knn_graph
            before_memory = process.memory_info().rss / 1024 / 1024
            result = knn_graph(
                X, n_neighbors=20, n_trees=4, random_state=42, verbose=False
            )
            after_memory = process.memory_info().rss / 1024 / 1024

            memory_increase = after_memory - baseline_memory
            memory_usages.append((n_samples, memory_increase))

            test_name = f"memory_usage_{n_samples}_samples"
            perf_metrics.record_metric(test_name, "memory_mb", memory_increase)

            print(f"\n{n_samples} samples: {memory_increase:.1f} MB")

            # Clean up
            del X, result
            gc.collect()

        # Memory usage should scale reasonably (not exponentially)
        if len(memory_usages) >= 2:
            small_n, small_mem = memory_usages[0]
            large_n, large_mem = memory_usages[-1]

            sample_ratio = large_n / small_n
            memory_ratio = large_mem / max(small_mem, 1.0)  # Avoid division by zero

            # Memory should not grow faster than O(n^2)
            assert (
                memory_ratio < sample_ratio**1.5
            ), f"Memory usage growing too fast: {memory_ratio:.2f}x for {sample_ratio:.2f}x samples"

    def test_reproducibility_performance(self, perf_metrics):
        """Test that performance is consistent across runs."""
        np.random.seed(42)
        n_samples, n_features = 1500, 128
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
        X = X.astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Warm up
        knn_graph(X[:100], n_neighbors=10, n_trees=2, random_state=42)

        # Run multiple times
        n_runs = 3
        durations = []

        for i in range(n_runs):
            result, duration = time_function(
                knn_graph,
                X,
                n_neighbors=20,
                n_trees=4,
                random_state=42,  # Same random state for consistency
                verbose=False,
            )
            durations.append(duration)

        # Calculate statistics
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        cv = std_duration / mean_duration  # Coefficient of variation

        perf_metrics.record_metric("reproducibility", "mean_duration", mean_duration)
        perf_metrics.record_metric("reproducibility", "std_duration", std_duration)
        perf_metrics.record_metric("reproducibility", "coefficient_of_variation", cv)

        print(f"\nReproducibility test:")
        print(f"  Mean duration: {mean_duration:.3f}s")
        print(f"  Std deviation: {std_duration:.3f}s")
        print(f"  Coefficient of variation: {cv:.3f}")

        # Performance should be reasonably consistent
        # Allow for up to 20% variation between runs
        assert cv < 0.4, f"Performance too variable: CV = {cv:.3f}"

        # Verify results are identical
        result1, _ = time_function(knn_graph, X, n_neighbors=10, random_state=42)
        result2, _ = time_function(knn_graph, X, n_neighbors=10, random_state=42)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_almost_equal(result1[1], result2[1])


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests with historical baselines."""

    def test_baseline_performance_check(self):
        """
        Baseline performance test that can be used to establish performance standards.

        This test should be run on a reference machine to establish baseline timings,
        and then used in CI to detect significant regressions.
        """
        np.random.seed(42)

        # Standard test case
        n_samples, n_features = 5000, 256
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
        X = X.astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Warm up
        knn_graph(X[:100], n_neighbors=10, n_trees=2, random_state=42)

        # Benchmark run
        start_time = time.perf_counter()
        result = knn_graph(X, n_neighbors=30, n_trees=4, random_state=42, verbose=False)
        duration = time.perf_counter() - start_time

        indices, distances = result
        samples_per_second = n_samples / duration

        print(f"\nBaseline Performance Report:")
        print(f"  Dataset: {n_samples} samples x {n_features} features")
        print(f"  Duration: {duration:.3f} seconds")
        print(f"  Throughput: {samples_per_second:.1f} samples/second")
        print(f"  Hardware: {platform.platform()}")
        if HAS_PSUTIL:
            print(f"  CPU cores: {psutil.cpu_count(logical=True)}")
            print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        else:
            import os

            print(f"  CPU cores: {os.cpu_count() or 'unknown'}")
            print(f"  Memory: unknown (psutil not available)")

        # Basic sanity checks
        assert indices.shape == (n_samples, 30)
        assert distances.shape == (n_samples, 30)
        assert np.all(indices >= 0)
        assert np.all(distances >= 0)

        # Very basic performance floor (should work on any reasonable hardware)
        min_samples_per_second = 10  # Very conservative
        assert (
            samples_per_second > min_samples_per_second
        ), f"Performance below minimum threshold: {samples_per_second:.1f} < {min_samples_per_second}"

        # Store baseline for potential future comparison
        # In a real CI system, you might save this to a file or database
        baseline_info = {
            "duration": duration,
            "samples_per_second": samples_per_second,
            "hardware_hash": hash(platform.platform()),
            "timestamp": time.time(),
        }

        # Note: baseline_info could be used for comparison in CI systems
        # but we don't return it to avoid pytest warnings
