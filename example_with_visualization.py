"""
Complete Example: Transformer Encoder + Visualization

This example demonstrates the full integration of the encoder
with the visualization layer to provide insights into the model.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

from transformer_encoder import encoder_fn, EmbeddingLayer
from visualizer import EmbeddingVisualizer


def main():
    """Complete example with visualization."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Transformer Encoder + Visualization Integration".center(68) + "║")
    print("║" + "  Complete Example".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # ============================================================
    # 1. Setup Encoder
    # ============================================================
    print("=" * 70)
    print("Step 1: Setting up Transformer Encoder")
    print("=" * 70)

    config = {
        'vocab_size': 500,
        'embed_dim': 128,
        'seq_len': 24,
        'batch_size': 60,  # 20 samples per category
        'n_heads': 4,
        'd_k': 32,
        'hidden_dim': 512,
    }

    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Embedding dim:   {config['embed_dim']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Batch size:      {config['batch_size']}")
    print()

    # Initialize encoder
    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    params = encoder_transformed.init(rng, dummy_input)

    print("✓ Encoder initialized successfully!")
    print()

    # ============================================================
    # 2. Generate Sample Data
    # ============================================================
    print("=" * 70)
    print("Step 2: Generating Sample Data (3 Categories)")
    print("=" * 70)

    # Create three categories with different token distributions
    n_per_category = config['batch_size'] // 3

    # Category A: Technical text (low token IDs)
    category_a = jax.random.randint(
        jax.random.PRNGKey(0),
        (n_per_category, config['seq_len']),
        0, 150
    )

    # Category B: General text (mid token IDs)
    category_b = jax.random.randint(
        jax.random.PRNGKey(1),
        (n_per_category, config['seq_len']),
        150, 350
    )

    # Category C: Specialized text (high token IDs)
    category_c = jax.random.randint(
        jax.random.PRNGKey(2),
        (n_per_category, config['seq_len']),
        350, 500
    )

    all_inputs = jnp.concatenate([category_a, category_b, category_c], axis=0)
    labels = np.array(['Technical'] * n_per_category +
                     ['General'] * n_per_category +
                     ['Specialized'] * n_per_category)
    colors = np.array([0] * n_per_category +
                     [1] * n_per_category +
                     [2] * n_per_category)

    print(f"  Category A (Technical):   {n_per_category} samples, tokens 0-150")
    print(f"  Category B (General):     {n_per_category} samples, tokens 150-350")
    print(f"  Category C (Specialized): {n_per_category} samples, tokens 350-500")
    print(f"  Total samples: {len(all_inputs)}")
    print()

    # ============================================================
    # 3. Process Through Encoder
    # ============================================================
    print("=" * 70)
    print("Step 3: Processing Through Encoder")
    print("=" * 70)

    outputs = encoder_transformed.apply(params, None, all_inputs)
    print(f"  Encoder output shape: {outputs.shape}")

    # Get sentence-level embeddings (average pooling)
    sentence_embeddings = jnp.mean(outputs, axis=1)
    print(f"  Sentence embeddings shape: {sentence_embeddings.shape}")
    print()

    # ============================================================
    # 4. Initialize Visualizer
    # ============================================================
    print("=" * 70)
    print("Step 4: Creating Visualizations")
    print("=" * 70)
    print()

    viz = EmbeddingVisualizer()

    # Visualization 1: 2D PCA
    print("  [1/6] Creating 2D PCA visualization...")
    viz.visualize_embeddings_2d(
        np.array(sentence_embeddings),
        method='pca',
        colors=colors,
        title='Encoder Outputs: 2D PCA Projection\n(3 Text Categories)',
        save_path='outputs/encoder_2d_pca.png'
    )

    # Visualization 2: 2D t-SNE
    print("  [2/6] Creating 2D t-SNE visualization...")
    viz.visualize_embeddings_2d(
        np.array(sentence_embeddings),
        method='tsne',
        colors=colors,
        title='Encoder Outputs: 2D t-SNE Projection\n(3 Text Categories)',
        save_path='outputs/encoder_2d_tsne.png'
    )

    # Visualization 3: 3D PCA
    print("  [3/6] Creating 3D PCA visualization...")
    viz.visualize_embeddings_3d(
        np.array(sentence_embeddings),
        method='pca',
        colors=colors,
        title='Encoder Outputs: 3D PCA Projection\n(3 Text Categories)',
        save_path='outputs/encoder_3d_pca.png'
    )

    # Visualization 4: Positional Encoding
    print("  [4/6] Creating positional encoding visualization...")
    viz.visualize_positional_encoding(
        seq_len=config['seq_len'],
        embed_dim=config['embed_dim'],
        save_path='outputs/positional_encoding.png'
    )

    # Visualization 5: Simulated Attention Pattern
    print("  [5/6] Creating attention pattern visualization...")
    # Note: This is simulated. To get real attention, modify the encoder to return attention weights
    seq_len_viz = 12
    attention_sim = np.random.rand(seq_len_viz, seq_len_viz)
    attention_sim = attention_sim / attention_sim.sum(axis=-1, keepdims=True)
    tokens = ["The", "transformer", "encoder", "processes", "input", "tokens",
             "using", "multi", "head", "attention", "mechanisms", "."]

    viz.visualize_attention_patterns(
        attention_sim,
        tokens=tokens,
        title='Simulated Attention Pattern\n(To get real attention, modify encoder)',
        save_path='outputs/attention_pattern.png'
    )

    # Visualization 6: Compare with raw embeddings
    print("  [6/6] Creating embedding comparison...")

    # Get raw embeddings (before encoder)
    def embedding_only_fn(x):
        layer = EmbeddingLayer(
            embed_dim=config['embed_dim'],
            vocab_size=config['vocab_size'],
            seq_len=config['seq_len']
        )
        return layer(x)

    embedding_transformed = hk.transform(embedding_only_fn)
    embedding_params = embedding_transformed.init(rng, dummy_input)
    raw_embeddings = embedding_transformed.apply(embedding_params, None, all_inputs)
    raw_sentence_emb = jnp.mean(raw_embeddings, axis=1)

    viz.compare_embeddings(
        {
            'Raw Embeddings': np.array(raw_sentence_emb),
            'After Encoder': np.array(sentence_embeddings),
        },
        method='pca',
        save_path='outputs/embedding_comparison.png'
    )

    print()
    print("✓ All visualizations created successfully!")
    print()

    # ============================================================
    # 5. Analysis
    # ============================================================
    print("=" * 70)
    print("Step 5: Statistical Analysis")
    print("=" * 70)
    print()

    # Compute statistics per category
    for i, category in enumerate(['Technical', 'General', 'Specialized']):
        mask = colors == i
        category_embeddings = sentence_embeddings[mask]

        mean_norm = float(jnp.mean(jnp.linalg.norm(category_embeddings, axis=1)))
        std_norm = float(jnp.std(jnp.linalg.norm(category_embeddings, axis=1)))
        mean_val = float(jnp.mean(category_embeddings))

        print(f"  {category}:")
        print(f"    Mean L2 norm: {mean_norm:.4f} ± {std_norm:.4f}")
        print(f"    Mean value:   {mean_val:.4f}")
        print()

    # ============================================================
    # 6. Summary
    # ============================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ Processed 60 samples through transformer encoder")
    print("✓ Generated 6 visualizations:")
    print("    - 2D PCA projection")
    print("    - 2D t-SNE projection")
    print("    - 3D PCA projection")
    print("    - Positional encoding patterns")
    print("    - Attention pattern (simulated)")
    print("    - Raw vs Encoded comparison")
    print()
    print("📁 All outputs saved to: outputs/")
    print()

    # List output files
    output_files = [f for f in os.listdir('outputs') if f.endswith('.png')]
    print(f"Generated {len(output_files)} visualization files:")
    for f in sorted(output_files):
        print(f"  - outputs/{f}")
    print()

    print("=" * 70)
    print("✨ Complete! Encoder + Visualization integration successful!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
