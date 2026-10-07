"""
Example usage of the Transformer Encoder implementation.

This script demonstrates:
1. Basic encoder initialization and forward pass
2. Individual component usage
3. Different model configurations
4. Parameter inspection
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


def example_1_basic_usage():
    """Example 1: Basic encoder initialization and forward pass."""
    print("=" * 60)
    print("Example 1: Basic Encoder Usage")
    print("=" * 60)

    # Define model configuration
    config = {
        'vocab_size': 1000,
        'embed_dim': 128,
        'seq_len': 32,
        'batch_size': 4,
        'n_heads': 4,
        'd_k': 32,  # embed_dim / n_heads
        'hidden_dim': 512,
    }

    # Create the encoder
    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    params = encoder_transformed.init(rng, dummy_input)

    print("\n✓ Encoder initialized successfully!")

    # Create sample input (batch of token sequences)
    sample_input = jax.random.randint(
        jax.random.PRNGKey(0),
        (config['batch_size'], config['seq_len']),
        0,
        config['vocab_size']
    )

    # Forward pass
    output = encoder_transformed.apply(params, None, sample_input)

    print(f"\nInput shape:  {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Forward pass completed successfully!")
    print()


def example_2_component_inspection():
    """Example 2: Inspect individual components."""
    print("=" * 60)
    print("Example 2: Individual Component Inspection")
    print("=" * 60)

    # Test Embedding Layer
    print("\n1. Embedding Layer")
    print("-" * 40)

    def embedding_fn(x):
        layer = EmbeddingLayer(embed_dim=64, vocab_size=500, seq_len=16)
        return layer(x)

    embedding_transformed = hk.transform(embedding_fn)
    rng = jax.random.PRNGKey(1)
    sample = jnp.zeros((2, 16), dtype=jnp.int32)
    embedding_params = embedding_transformed.init(rng, sample)

    test_input = jax.random.randint(jax.random.PRNGKey(2), (2, 16), 0, 500)
    embeddings = embedding_transformed.apply(embedding_params, None, test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {embeddings.shape}")
    print("✓ Embeddings include positional encodings")

    # Test Multi-Head Attention
    print("\n2. Multi-Head Attention")
    print("-" * 40)

    def attention_fn(x):
        layer = Multihead_Attention(
            embed_dim=64, batch_size=2, seq_len=16, n_heads=4, d_k=16
        )
        return layer(x)

    attention_transformed = hk.transform(attention_fn)
    rng = jax.random.PRNGKey(3)
    sample = jnp.zeros((2, 16, 64))
    attention_params = attention_transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(4), (2, 16, 64))
    attention_output = attention_transformed.apply(attention_params, None, test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {attention_output.shape}")
    print("✓ Multi-head attention preserves shape")

    # Test Layer Normalization
    print("\n3. Layer Normalization")
    print("-" * 40)

    def layernorm_fn(x):
        layer = LayerNorm(feature_axis=-1)
        return layer(x)

    layernorm_transformed = hk.transform(layernorm_fn)
    rng = jax.random.PRNGKey(5)
    sample = jnp.zeros((2, 16, 64))
    layernorm_params = layernorm_transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(6), (2, 16, 64))
    normalized_output = layernorm_transformed.apply(layernorm_params, None, test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {normalized_output.shape}")
    print(f"Input mean:   {jnp.mean(test_input, axis=-1)[0, 0]:.4f}")
    print(f"Output mean:  {jnp.mean(normalized_output, axis=-1)[0, 0]:.4f}")
    print(f"Output std:   {jnp.std(normalized_output, axis=-1)[0, 0]:.4f}")
    print("✓ Layer normalization normalizes features")

    # Test Feed-Forward Network
    print("\n4. Feed-Forward Network")
    print("-" * 40)

    def ffn_fn(x):
        layer = FeedForwardNet(embed_dim=64, hidden_dim=256)
        return layer(x)

    ffn_transformed = hk.transform(ffn_fn)
    rng = jax.random.PRNGKey(7)
    sample = jnp.zeros((2, 16, 64))
    ffn_params = ffn_transformed.init(rng, sample)

    test_input = jax.random.normal(jax.random.PRNGKey(8), (2, 16, 64))
    ffn_output = ffn_transformed.apply(ffn_params, None, test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Hidden dim:   256")
    print(f"Output shape: {ffn_output.shape}")
    print("✓ FFN: 64 -> 256 -> 64 with GELU activation")
    print()


def example_3_different_configs():
    """Example 3: Different model configurations."""
    print("=" * 60)
    print("Example 3: Different Model Configurations")
    print("=" * 60)

    configs = {
        "Tiny": {
            'vocab_size': 1000,
            'embed_dim': 64,
            'seq_len': 32,
            'batch_size': 8,
            'n_heads': 2,
            'd_k': 32,
            'hidden_dim': 256,
        },
        "Small": {
            'vocab_size': 5000,
            'embed_dim': 256,
            'seq_len': 64,
            'batch_size': 16,
            'n_heads': 4,
            'd_k': 64,
            'hidden_dim': 1024,
        },
        "Base": {
            'vocab_size': 30000,
            'embed_dim': 512,
            'seq_len': 128,
            'batch_size': 32,
            'n_heads': 8,
            'd_k': 64,
            'hidden_dim': 2048,
        },
    }

    for name, config in configs.items():
        print(f"\n{name} Model Configuration:")
        print("-" * 40)
        for key, value in config.items():
            print(f"  {key:12s}: {value:>6}")

        # Initialize
        encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
        params = encoder_transformed.init(rng, dummy_input)

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"  Parameters  : {param_count:,}")
        print(f"  ✓ Initialized successfully")

    print()


def example_4_parameter_structure():
    """Example 4: Inspect parameter structure."""
    print("=" * 60)
    print("Example 4: Parameter Structure")
    print("=" * 60)

    config = {
        'vocab_size': 1000,
        'embed_dim': 128,
        'seq_len': 32,
        'batch_size': 4,
        'n_heads': 4,
        'd_k': 32,
        'hidden_dim': 512,
    }

    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    params = encoder_transformed.init(rng, dummy_input)

    print("\nParameter tree structure:")
    print("-" * 40)

    def print_tree(tree, indent=0):
        for key, value in tree.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_tree(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value.shape}")

    print_tree(params)

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\nTotal parameters: {total_params:,}")
    print()


def example_5_batch_processing():
    """Example 5: Process batches of different sequences."""
    print("=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)

    config = {
        'vocab_size': 100,
        'embed_dim': 64,
        'seq_len': 16,
        'batch_size': 4,
        'n_heads': 4,
        'd_k': 16,
        'hidden_dim': 256,
    }

    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    params = encoder_transformed.init(rng, dummy_input)

    print("\nProcessing different sequences:")
    print("-" * 40)

    # Create example sequences (representing tokens)
    sequences = [
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Short sequence (padded)
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0],  # Medium
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],  # Full
        [40, 41, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Very short
    ]

    batch = jnp.array(sequences, dtype=jnp.int32)
    outputs = encoder_transformed.apply(params, None, batch)

    for i, (seq, out) in enumerate(zip(sequences, outputs)):
        non_padding = sum(1 for x in seq if x != 0)
        print(f"Sequence {i+1}: {non_padding:2d} tokens -> output shape: {out.shape}")

    print("\n✓ All sequences processed successfully")
    print("✓ Padding tokens (0) are processed but typically masked in attention")
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Transformer Encoder Examples".center(58) + "║")
    print("║" + "  JAX + Haiku Implementation".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run all examples
    example_1_basic_usage()
    example_2_component_inspection()
    example_3_different_configs()
    example_4_parameter_structure()
    example_5_batch_processing()

    print("=" * 60)
    print("All examples completed successfully! 🎉")
    print("=" * 60)
    print()
