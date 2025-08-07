#!/usr/bin/env python3
"""
Performance benchmark runner for evoc knn_graph module.

This script runs performance tests and generates a report that can be used
for performance regression monitoring in CI/CD pipelines.
"""

import argparse
import json
import sys
import time
import platform
import subprocess
from pathlib import Path


def run_performance_tests(output_file=None, verbose=False):
    """Run performance tests and collect results."""
    
    print("Running EVoC knn_graph performance benchmarks...")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print("-" * 60)
    
    # Run pytest with performance markers
    cmd = [
        sys.executable, "-m", "pytest", 
        "evoc/tests/test_knn_graph_performance.py",
        "-m", "performance",
        "-v"
    ]
    
    if verbose:
        cmd.append("-s")
    
    # Add JSON report plugin if available
    try:
        import pytest_json_report
        if output_file:
            cmd.extend(["--json-report", f"--json-report-file={output_file}"])
    except ImportError:
        print("Note: pytest-json-report not installed, basic output only")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\nAll performance tests passed in {duration:.1f} seconds!")
    else:
        print(f"\nSome performance tests failed or had issues.")
        if not verbose and result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    
    return result.returncode == 0


def generate_performance_report(test_results_file, output_file):
    """Generate a human-readable performance report."""
    
    try:
        with open(test_results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Test results file {test_results_file} not found")
        return False
    except json.JSONDecodeError:
        print(f"Could not parse JSON from {test_results_file}")
        return False
    
    # Extract performance metrics
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'total_tests': data.get('summary', {}).get('total', 0),
        'passed_tests': data.get('summary', {}).get('passed', 0),
        'failed_tests': data.get('summary', {}).get('failed', 0),
        'duration': data.get('duration', 0),
        'tests': []
    }
    
    # Process individual test results
    for test in data.get('tests', []):
        if 'performance' in test.get('keywords', []):
            test_info = {
                'name': test.get('nodeid', '').split('::')[-1],
                'duration': test.get('duration', 0),
                'outcome': test.get('outcome', 'unknown'),
                'stdout': test.get('call', {}).get('stdout', '')
            }
            report['tests'].append(test_info)
    
    # Write report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report written to {output_file}")
    return True


def check_performance_regression(current_file, baseline_file, threshold=1.5):
    """
    Check for performance regressions by comparing current results to baseline.
    
    Args:
        current_file: JSON file with current test results
        baseline_file: JSON file with baseline results
        threshold: Maximum allowed slowdown ratio (e.g., 1.5 = 50% slower)
    
    Returns:
        bool: True if no significant regressions detected
    """
    
    try:
        with open(current_file, 'r') as f:
            current = json.load(f)
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading performance data: {e}")
        return False
    
    print(f"Comparing performance to baseline from {baseline.get('timestamp', 'unknown')}")
    
    regressions = []
    improvements = []
    
    # Compare test durations
    current_tests = {t['name']: t for t in current.get('tests', [])}
    baseline_tests = {t['name']: t for t in baseline.get('tests', [])}
    
    for test_name in current_tests:
        if test_name in baseline_tests:
            current_duration = current_tests[test_name]['duration']
            baseline_duration = baseline_tests[test_name]['duration']
            
            if baseline_duration > 0:
                ratio = current_duration / baseline_duration
                
                if ratio > threshold:
                    regressions.append({
                        'test': test_name,
                        'current': current_duration,
                        'baseline': baseline_duration,
                        'ratio': ratio
                    })
                elif ratio < 0.8:  # 20% improvement
                    improvements.append({
                        'test': test_name,
                        'current': current_duration,
                        'baseline': baseline_duration,
                        'ratio': ratio
                    })
    
    # Report results
    if regressions:
        print(f"\n⚠️  Performance regressions detected:")
        for reg in regressions:
            print(f"  {reg['test']}: {reg['ratio']:.2f}x slower "
                  f"({reg['current']:.3f}s vs {reg['baseline']:.3f}s)")
    
    if improvements:
        print(f"\n✅ Performance improvements:")
        for imp in improvements:
            print(f"  {imp['test']}: {imp['ratio']:.2f}x faster "
                  f"({imp['current']:.3f}s vs {imp['baseline']:.3f}s)")
    
    if not regressions and not improvements:
        print("\n✅ No significant performance changes detected")
    
    return len(regressions) == 0


def main():
    parser = argparse.ArgumentParser(description="Run EVoC performance benchmarks")
    parser.add_argument("--output", "-o", help="Output file for test results (JSON)")
    parser.add_argument("--report", "-r", help="Generate human-readable report file")
    parser.add_argument("--baseline", "-b", help="Compare against baseline performance file")
    parser.add_argument("--threshold", "-t", type=float, default=1.5,
                       help="Regression threshold (default: 1.5x slower)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Default output file
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"performance_results_{timestamp}.json"
    
    # Run performance tests
    success = run_performance_tests(args.output, args.verbose)
    
    if not success:
        print("Performance tests failed")
        return 1
    
    # Generate report if requested
    if args.report:
        generate_performance_report(args.output, args.report)
    
    # Check for regressions if baseline provided
    if args.baseline:
        if not check_performance_regression(args.output, args.baseline, args.threshold):
            print("Performance regression detected!")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
