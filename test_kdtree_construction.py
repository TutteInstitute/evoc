#!/usr/bin/env python3
"""
Test script to compare numba KDTree construction with sklearn KDTree.
Focuses on data structure comparison and construction time performance.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree as SklearnKDTree
from evoc.numba_kdtree import build_kdtree, NumbaKDTree, NodeData

def generate_test_data(n_samples, n_features, random_state=42):
    """Generate random test data."""
    np.random.seed(random_state)
    return np.random.rand(n_samples, n_features).astype(np.float32)

def compare_tree_structures(sklearn_tree, numba_tree, data):
    """Compare the data structures between sklearn and numba trees."""
    print("\n=== Tree Structure Comparison ===")
    
    # Get sklearn internal arrays
    sk_data, sk_idx_array, sk_node_data, sk_node_bounds = sklearn_tree.get_arrays()
    
    print(f"Data shape: {data.shape}")
    print(f"Sklearn data shape: {sk_data.shape}")
    print(f"Numba data shape: {numba_tree.data.shape}")
    
    # Compare data arrays
    data_match = np.allclose(sk_data, numba_tree.data, rtol=1e-5)
    print(f"Data arrays match: {data_match}")
    
    # Compare index arrays (initially should be identical)
    initial_idx_match = np.array_equal(sk_idx_array, numba_tree.idx_array)
    print(f"Index arrays initially match: {initial_idx_match}")
    
    # Compare node bounds shapes
    print(f"Sklearn node_bounds shape: {sk_node_bounds.shape}")
    print(f"Numba node_bounds shape: {numba_tree.node_bounds.shape}")
    bounds_shape_match = sk_node_bounds.shape == numba_tree.node_bounds.shape
    print(f"Node bounds shapes match: {bounds_shape_match}")
    
    # Compare node data structures
    print(f"Sklearn node_data shape: {sk_node_data.shape}")
    print(f"Numba node_data components:")
    print(f"  idx_start: {numba_tree.idx_start.shape}")
    print(f"  idx_end: {numba_tree.idx_end.shape}")
    print(f"  radius: {numba_tree.radius.shape}")
    print(f"  is_leaf: {numba_tree.is_leaf.shape}")
    
    # Check if we have the same number of nodes
    n_nodes_sklearn = sk_node_data.shape[0]
    n_nodes_numba = numba_tree.idx_start.shape[0]
    print(f"Number of nodes - Sklearn: {n_nodes_sklearn}, Numba: {n_nodes_numba}")
    nodes_count_match = n_nodes_sklearn == n_nodes_numba
    print(f"Node counts match: {nodes_count_match}")
    
    return {
        'data_match': data_match,
        'initial_idx_match': initial_idx_match,
        'bounds_shape_match': bounds_shape_match,
        'nodes_count_match': nodes_count_match
    }

def time_construction(data, leaf_size=40, n_runs=5):
    """Time the construction of both tree types."""
    sklearn_times = []
    numba_times = []
    
    print(f"\nTiming construction for {data.shape[0]} samples, {data.shape[1]} dimensions...")
    
    for run in range(n_runs):
        # Time sklearn construction
        start_time = time.time()
        sklearn_tree = SklearnKDTree(data, leaf_size=leaf_size)
        sklearn_time = time.time() - start_time
        sklearn_times.append(sklearn_time)
        
        # Time numba construction
        start_time = time.time()
        numba_tree = build_kdtree(data, leaf_size=leaf_size)
        numba_time = time.time() - start_time
        numba_times.append(numba_time)
        
        print(f"  Run {run+1}: Sklearn={sklearn_time:.4f}s, Numba={numba_time:.4f}s")
    
    sklearn_avg = np.mean(sklearn_times)
    numba_avg = np.mean(numba_times)
    speedup = sklearn_avg / numba_avg
    
    print(f"Average times: Sklearn={sklearn_avg:.4f}s, Numba={numba_avg:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return sklearn_avg, numba_avg, speedup

def run_dimension_scaling_test():
    """Test how construction time scales with dimensionality."""
    print("\n" + "="*60)
    print("DIMENSION SCALING TEST")
    print("="*60)
    
    n_samples = 5000
    dimensions = [2, 3, 5, 8, 10, 12, 15]
    results = []
    
    for dim in dimensions:
        print(f"\n--- Testing {dim}D data ---")
        data = generate_test_data(n_samples, dim)
        
        sklearn_time, numba_time, speedup = time_construction(data, n_runs=3)
        
        # Test structure comparison for first few dimensions
        if dim <= 5:
            sklearn_tree = SklearnKDTree(data, leaf_size=40)
            numba_tree = build_kdtree(data, leaf_size=40)
            structure_comparison = compare_tree_structures(sklearn_tree, numba_tree, data)
        else:
            structure_comparison = {}
        
        results.append({
            'dimensions': dim,
            'n_samples': n_samples,
            'sklearn_time': sklearn_time,
            'numba_time': numba_time,
            'speedup': speedup,
            **structure_comparison
        })
    
    return results

def run_sample_scaling_test():
    """Test how construction time scales with number of samples."""
    print("\n" + "="*60)
    print("SAMPLE SIZE SCALING TEST")
    print("="*60)
    
    dimensions = 8
    sample_sizes = [1000, 2500, 5000, 10000, 20000, 50000]
    results = []
    
    for n_samples in sample_sizes:
        print(f"\n--- Testing {n_samples} samples ---")
        data = generate_test_data(n_samples, dimensions)
        
        sklearn_time, numba_time, speedup = time_construction(data, n_runs=3)
        
        # Test structure comparison for smaller datasets
        if n_samples <= 5000:
            sklearn_tree = SklearnKDTree(data, leaf_size=40)
            numba_tree = build_kdtree(data, leaf_size=40)
            structure_comparison = compare_tree_structures(sklearn_tree, numba_tree, data)
        else:
            structure_comparison = {}
        
        results.append({
            'n_samples': n_samples,
            'dimensions': dimensions,
            'sklearn_time': sklearn_time,
            'numba_time': numba_time,
            'speedup': speedup,
            **structure_comparison
        })
    
    return results

def plot_results(dim_results, sample_results):
    """Create plots showing the performance comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dimension scaling - absolute times
    dims = [r['dimensions'] for r in dim_results]
    sklearn_times_dim = [r['sklearn_time'] for r in dim_results]
    numba_times_dim = [r['numba_time'] for r in dim_results]
    
    ax1.plot(dims, sklearn_times_dim, 'o-', label='Sklearn', color='blue')
    ax1.plot(dims, numba_times_dim, 'o-', label='Numba', color='red')
    ax1.set_xlabel('Dimensions')
    ax1.set_ylabel('Construction Time (s)')
    ax1.set_title('Construction Time vs Dimensions\n(5000 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dimension scaling - speedup
    speedups_dim = [r['speedup'] for r in dim_results]
    ax2.plot(dims, speedups_dim, 'o-', color='green')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Speedup (Sklearn/Numba)')
    ax2.set_title('Speedup vs Dimensions')
    ax2.grid(True, alpha=0.3)
    
    # Sample scaling - absolute times
    samples = [r['n_samples'] for r in sample_results]
    sklearn_times_sample = [r['sklearn_time'] for r in sample_results]
    numba_times_sample = [r['numba_time'] for r in sample_results]
    
    ax3.loglog(samples, sklearn_times_sample, 'o-', label='Sklearn', color='blue')
    ax3.loglog(samples, numba_times_sample, 'o-', label='Numba', color='red')
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Construction Time (s)')
    ax3.set_title('Construction Time vs Sample Size\n(8 dimensions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sample scaling - speedup
    speedups_sample = [r['speedup'] for r in sample_results]
    ax4.semilogx(samples, speedups_sample, 'o-', color='green')
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Speedup (Sklearn/Numba)')
    ax4.set_title('Speedup vs Sample Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kdtree_construction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("KDTree Construction Performance Comparison")
    print("=========================================")
    
    # Quick validation test
    print("\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)
    
    data = generate_test_data(1000, 3)
    sklearn_tree = SklearnKDTree(data, leaf_size=40)
    numba_tree = build_kdtree(data, leaf_size=40)
    
    structure_comparison = compare_tree_structures(sklearn_tree, numba_tree, data)
    
    if not all(structure_comparison.values()):
        print("\nWARNING: Structure comparison failed!")
        print("Issues found:")
        for key, value in structure_comparison.items():
            if not value:
                print(f"  - {key}: {value}")
        print("\nProceeding with performance tests anyway...")
    else:
        print("\n✓ All structure comparisons passed!")
    
    # Run scaling tests
    dim_results = run_dimension_scaling_test()
    sample_results = run_sample_scaling_test()
    
    # Create summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nDimension Scaling Results:")
    df_dim = pd.DataFrame(dim_results)
    print(df_dim[['dimensions', 'sklearn_time', 'numba_time', 'speedup']].round(4))
    
    print("\nSample Size Scaling Results:")
    df_sample = pd.DataFrame(sample_results)
    print(df_sample[['n_samples', 'sklearn_time', 'numba_time', 'speedup']].round(4))
    
    avg_speedup = np.mean([r['speedup'] for r in dim_results + sample_results])
    print(f"\nOverall average speedup: {avg_speedup:.2f}x")
    
    # Create plots
    try:
        plot_results(dim_results, sample_results)
        print("\nPlots saved as 'kdtree_construction_comparison.png'")
    except Exception as e:
        print(f"\nCould not create plots: {e}")
    
    return dim_results, sample_results

if __name__ == "__main__":
    dim_results, sample_results = main()
