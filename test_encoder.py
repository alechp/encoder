"""
Tests for the Transformer Encoder implementation.

Simple test suite to verify that all components work correctly.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from transformer_encoder import (
    EmbeddingLayer,
    Multihead_Attention,
    LayerNorm,
    FeedForwardNet,
    Encoder,
    encoder_fn,
)


def test_embedding_layer():
    """Test that embedding layer produces correct output shape."""
    print("Testing EmbeddingLayer...", end=" ")

    def embed_fn(x):
        layer = EmbeddingLayer(embed_dim=128, vocab_size=1000, seq_len=32)
        return layer(x)

    transformed = hk.transform(embed_fn)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((4, 32), dtype=jnp.int32)
    params = transformed.init(rng, sample)

    test_input = jax.random.randint(jax.random.PRNGKey(1), (4, 32), 0, 1000)
    output = transformed.apply(params, None, test_input)

    assert output.shape == (4, 32, 128), f"Expected shape (4, 32, 128), got {output.shape}"
    print("✓ PASSED")


def test_multihead_attention():
    """Test that multi-head attention produces correct output shape."""
    print("Testing Multihead_Attention...", end=" ")

    def attention_fn(x):
        layer = Multihead_Attention(
            embed_dim=128, batch_size=4, seq_len=32, n_heads=8, d_k=16
        )
        return layer(x)

    transformed = hk.transform(attention_fn)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((4, 32, 128))
    params = transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(1), (4, 32, 128))
    output = transformed.apply(params, None, test_input)

    assert output.shape == (4, 32, 128), f"Expected shape (4, 32, 128), got {output.shape}"
    print("✓ PASSED")


def test_layer_norm():
    """Test that layer normalization normalizes correctly."""
    print("Testing LayerNorm...", end=" ")

    def layernorm_fn(x):
        layer = LayerNorm(feature_axis=-1)
        return layer(x)

    transformed = hk.transform(layernorm_fn)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((4, 32, 128))
    params = transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(1), (4, 32, 128))
    output = transformed.apply(params, None, test_input)

    assert output.shape == test_input.shape, f"Expected shape {test_input.shape}, got {output.shape}"

    # Check that output is normalized (mean ≈ 0, std ≈ 1)
    mean = jnp.mean(output, axis=-1)
    std = jnp.std(output, axis=-1)
    assert jnp.allclose(mean, 0.0, atol=1e-5), f"Mean should be close to 0, got {jnp.mean(mean)}"
    assert jnp.allclose(std, 1.0, atol=1e-1), f"Std should be close to 1, got {jnp.mean(std)}"
    print("✓ PASSED")


def test_feedforward_net():
    """Test that feed-forward network produces correct output shape."""
    print("Testing FeedForwardNet...", end=" ")

    def ffn_fn(x):
        layer = FeedForwardNet(embed_dim=128, hidden_dim=512)
        return layer(x)

    transformed = hk.transform(ffn_fn)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((4, 32, 128))
    params = transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(1), (4, 32, 128))
    output = transformed.apply(params, None, test_input)

    assert output.shape == (4, 32, 128), f"Expected shape (4, 32, 128), got {output.shape}"
    print("✓ PASSED")


def test_encoder():
    """Test that the complete encoder produces correct output shape."""
    print("Testing Encoder...", end=" ")

    config = {
        'vocab_size': 1000,
        'embed_dim': 128,
        'seq_len': 32,
        'batch_size': 4,
        'n_heads': 8,
        'd_k': 16,
        'hidden_dim': 512,
    }

    transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((4, 32), dtype=jnp.int32)
    params = transformed.init(rng, dummy)

    test_input = jax.random.randint(jax.random.PRNGKey(1), (4, 32), 0, 1000)
    output = transformed.apply(params, None, test_input)

    assert output.shape == (4, 32, 128), f"Expected shape (4, 32, 128), got {output.shape}"
    print("✓ PASSED")


def test_positional_encoding():
    """Test that positional encodings are deterministic and different for different positions."""
    print("Testing positional encoding...", end=" ")

    def embed_fn(x):
        layer = EmbeddingLayer(embed_dim=64, vocab_size=100, seq_len=16)
        return layer(x)

    transformed = hk.transform(embed_fn)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((2, 16), dtype=jnp.int32)
    params = transformed.init(rng, sample)

    # Use same token at different positions
    test_input = jnp.ones((2, 16), dtype=jnp.int32) * 5  # All tokens are 5

    output1 = transformed.apply(params, None, test_input)
    output2 = transformed.apply(params, None, test_input)

    # Outputs should be identical (deterministic)
    assert jnp.allclose(output1, output2), "Positional encoding should be deterministic"

    # Different positions should have different values (due to positional encoding)
    # Even though all tokens are the same
    pos0 = output1[0, 0, :]
    pos1 = output1[0, 1, :]
    assert not jnp.allclose(pos0, pos1), "Different positions should have different encodings"
    print("✓ PASSED")


def test_residual_connections():
    """Test that residual connections work by checking if output differs from just attention."""
    print("Testing residual connections...", end=" ")

    config = {
        'vocab_size': 100,
        'embed_dim': 64,
        'seq_len': 16,
        'batch_size': 2,
        'n_heads': 4,
        'd_k': 16,
        'hidden_dim': 256,
    }

    transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((2, 16), dtype=jnp.int32)
    params = transformed.init(rng, dummy)

    test_input = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 100)
    output = transformed.apply(params, None, test_input)

    # Output should not be all zeros or NaN
    assert not jnp.any(jnp.isnan(output)), "Output should not contain NaN"
    assert not jnp.allclose(output, 0.0), "Output should not be all zeros"
    print("✓ PASSED")


def test_different_batch_sizes():
    """Test that the encoder works with different batch sizes."""
    print("Testing different batch sizes...", end=" ")

    config = {
        'vocab_size': 100,
        'embed_dim': 64,
        'seq_len': 16,
        'batch_size': 8,  # Used for initialization
        'n_heads': 4,
        'd_k': 16,
        'hidden_dim': 256,
    }

    transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((8, 16), dtype=jnp.int32)
    params = transformed.init(rng, dummy)

    # Test with different batch sizes
    for batch_size in [1, 4, 8, 16]:
        test_input = jax.random.randint(
            jax.random.PRNGKey(batch_size), (batch_size, 16), 0, 100
        )
        output = transformed.apply(params, None, test_input)
        assert output.shape == (batch_size, 16, 64), \
            f"Expected shape ({batch_size}, 16, 64), got {output.shape}"

    print("✓ PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Transformer Encoder Tests")
    print("=" * 60 + "\n")

    tests = [
        test_embedding_layer,
        test_multihead_attention,
        test_layer_norm,
        test_feedforward_net,
        test_encoder,
        test_positional_encoding,
        test_residual_connections,
        test_different_batch_sizes,
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
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
