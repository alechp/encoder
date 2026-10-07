"""
Test suite for the visualization layer.

Tests all major visualization functions without displaying plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from visualizer import EmbeddingVisualizer
import os


def test_2d_visualization():
    """Test 2D embedding visualization."""
    print("Testing 2D visualization...", end=" ")

    viz = EmbeddingVisualizer()

    # Generate sample embeddings
    embeddings = np.random.randn(50, 128)
    colors = np.random.randint(0, 3, 50)

    # Test without saving
    coords = viz.visualize_embeddings_2d(
        embeddings,
        method='pca',
        colors=colors,
        title='Test 2D'
    )

    assert coords.shape == (50, 2), f"Expected shape (50, 2), got {coords.shape}"
    plt.close('all')

    print("✓ PASSED")


def test_3d_visualization():
    """Test 3D embedding visualization."""
    print("Testing 3D visualization...", end=" ")

    viz = EmbeddingVisualizer()

    # Generate sample embeddings
    embeddings = np.random.randn(30, 64)

    # Test 3D projection
    coords = viz.visualize_embeddings_3d(
        embeddings,
        method='pca',
        title='Test 3D'
    )

    assert coords.shape == (30, 3), f"Expected shape (30, 3), got {coords.shape}"
    plt.close('all')

    print("✓ PASSED")


def test_attention_visualization():
    """Test attention pattern visualization."""
    print("Testing attention visualization...", end=" ")

    viz = EmbeddingVisualizer()

    # Generate sample attention weights
    seq_len = 10
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=-1, keepdims=True)

    tokens = [f"T{i}" for i in range(seq_len)]

    # Test visualization
    viz.visualize_attention_patterns(
        attention,
        tokens=tokens,
        title='Test Attention'
    )

    plt.close('all')

    print("✓ PASSED")


def test_positional_encoding():
    """Test positional encoding visualization."""
    print("Testing positional encoding...", end=" ")

    viz = EmbeddingVisualizer()

    # Test with small configuration
    viz.visualize_positional_encoding(
        seq_len=32,
        embed_dim=64
    )

    plt.close('all')

    print("✓ PASSED")


def test_embedding_evolution():
    """Test embedding evolution visualization."""
    print("Testing embedding evolution...", end=" ")

    viz = EmbeddingVisualizer()

    # Generate sequence of embeddings
    n_layers = 4
    batch_size = 10
    seq_len = 8
    embed_dim = 32

    embeddings_sequence = [
        np.random.randn(batch_size, seq_len, embed_dim) for _ in range(n_layers)
    ]

    layer_names = ['Input'] + [f'Layer {i}' for i in range(1, n_layers)]

    # Test visualization
    viz.visualize_embedding_evolution(
        embeddings_sequence,
        layer_names=layer_names,
        sample_idx=0
    )

    plt.close('all')

    print("✓ PASSED")


def test_comparison():
    """Test embedding comparison."""
    print("Testing embedding comparison...", end=" ")

    viz = EmbeddingVisualizer()

    # Generate multiple embedding sets
    embeddings_dict = {
        'Set A': np.random.randn(30, 64),
        'Set B': np.random.randn(30, 64) + 1.0,
        'Set C': np.random.randn(30, 64) - 1.0,
    }

    # Test comparison
    viz.compare_embeddings(
        embeddings_dict,
        method='pca'
    )

    plt.close('all')

    print("✓ PASSED")


def test_realtime_setup():
    """Test real-time visualization setup."""
    print("Testing real-time setup...", end=" ")

    viz = EmbeddingVisualizer()

    # Test setup (without actual updates)
    rt_viz = viz.create_realtime_plot(update_interval=100, max_points=50)

    assert rt_viz is not None, "Real-time setup failed"
    assert hasattr(rt_viz, 'streaming_data'), "Missing streaming_data"

    plt.close('all')

    print("✓ PASSED")


def test_save_functionality():
    """Test saving visualizations to files."""
    print("Testing save functionality...", end=" ")

    # Create output directory
    os.makedirs('test_outputs', exist_ok=True)

    viz = EmbeddingVisualizer()
    embeddings = np.random.randn(20, 32)

    # Test saving
    viz.visualize_embeddings_2d(
        embeddings,
        method='pca',
        save_path='test_outputs/test_save.png'
    )

    # Check file exists
    assert os.path.exists('test_outputs/test_save.png'), "File was not saved"

    # Cleanup
    os.remove('test_outputs/test_save.png')
    os.rmdir('test_outputs')

    plt.close('all')

    print("✓ PASSED")


def test_with_jax_arrays():
    """Test compatibility with JAX arrays."""
    print("Testing JAX array compatibility...", end=" ")

    try:
        import jax.numpy as jnp

        viz = EmbeddingVisualizer()

        # Create JAX arrays
        embeddings_jax = jnp.array(np.random.randn(20, 32))

        # Test visualization
        viz.visualize_embeddings_2d(
            embeddings_jax,
            method='pca',
            title='JAX Test'
        )

        plt.close('all')

        print("✓ PASSED")
    except ImportError:
        print("⚠ SKIPPED (JAX not available)")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Visualization Layer Tests")
    print("=" * 60 + "\n")

    tests = [
        test_2d_visualization,
        test_3d_visualization,
        test_attention_visualization,
        test_positional_encoding,
        test_embedding_evolution,
        test_comparison,
        test_realtime_setup,
        test_save_functionality,
        test_with_jax_arrays,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed == 0:
        print("🎉 All visualization tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
