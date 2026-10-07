"""
Real-time Visualization Layer for Transformer Encoder

This module provides comprehensive visualization tools for:
- Vector embeddings in 2D/3D space
- Attention patterns and heatmaps
- Positional encodings
- Token representations through encoder layers
- Real-time updates during training/inference

Can run independently of the encoder implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import jax.numpy as jnp
from typing import Optional, List, Tuple, Dict, Union
import time
from datetime import datetime


class EmbeddingVisualizer:
    """
    Real-time visualization for transformer embeddings and attention patterns.
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize the visualizer.

        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.animation_data = []

    def visualize_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = 'pca',
        title: str = 'Embedding Space Visualization',
        colors: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize embeddings in 2D space using dimensionality reduction.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Optional labels for each point
            method: Reduction method ('pca', 'tsne')
            title: Plot title
            colors: Optional color values for each point
            save_path: If provided, save the plot to this path
        """
        # Convert JAX arrays to numpy if needed
        if isinstance(embeddings, jnp.ndarray):
            embeddings = np.array(embeddings)

        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            subtitle = f'PCA (explains {sum(explained_var)*100:.1f}% variance)'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords = reducer.fit_transform(embeddings)
            subtitle = 't-SNE'
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        if colors is not None:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                               cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)

        # Add labels if provided
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")

        plt.show()

        return coords

    def visualize_embeddings_3d(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = 'pca',
        title: str = 'Embedding Space Visualization (3D)',
        colors: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize embeddings in 3D space using dimensionality reduction.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Optional labels for each point
            method: Reduction method ('pca', 'tsne')
            title: Plot title
            colors: Optional color values for each point
            save_path: If provided, save the plot to this path
        """
        # Convert JAX arrays to numpy if needed
        if isinstance(embeddings, jnp.ndarray):
            embeddings = np.array(embeddings)

        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=3)
            coords = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            subtitle = f'PCA (explains {sum(explained_var)*100:.1f}% variance)'
        elif method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords = reducer.fit_transform(embeddings)
            subtitle = 't-SNE'
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=colors, cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      s=100, alpha=0.6)

        # Add labels if provided
        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label,
                       fontsize=8, alpha=0.7)

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_zlabel('Component 3', fontsize=12)
        ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D visualization to {save_path}")

        plt.show()

        return coords

    def visualize_attention_patterns(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
        head_idx: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize attention weights as a heatmap.

        Args:
            attention_weights: Attention matrix of shape (n_heads, seq_len, seq_len)
                             or (seq_len, seq_len) for single head
            tokens: Optional list of token strings for labels
            head_idx: Which attention head to visualize (if multi-head)
            title: Plot title
            save_path: If provided, save the plot to this path
        """
        # Convert JAX arrays to numpy if needed
        if isinstance(attention_weights, jnp.ndarray):
            attention_weights = np.array(attention_weights)

        # Handle multi-head attention
        if len(attention_weights.shape) == 3:
            attention_matrix = attention_weights[head_idx]
        else:
            attention_matrix = attention_weights

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12)

        # Set ticks and labels
        if tokens is not None:
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)

        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)

        if title is None:
            title = f'Attention Pattern (Head {head_idx})'
        ax.set_title(title, fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved attention visualization to {save_path}")

        plt.show()

    def visualize_positional_encoding(
        self,
        seq_len: int = 128,
        embed_dim: int = 512,
        save_path: Optional[str] = None,
    ):
        """
        Visualize sinusoidal positional encodings.

        Args:
            seq_len: Sequence length
            embed_dim: Embedding dimension
            save_path: If provided, save the plot to this path
        """
        # Generate positional encodings
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        pos_encoding = np.zeros((seq_len, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        # Create visualization
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Full heatmap
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.imshow(pos_encoding.T, cmap='RdBu', aspect='auto')
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Dimension', fontsize=12)
        ax1.set_title('Positional Encoding Heatmap', fontsize=14, pad=10)
        plt.colorbar(im1, ax=ax1)

        # 2. Selected dimensions over positions
        ax2 = fig.add_subplot(gs[1, 0])
        dims_to_plot = [0, 1, embed_dim//4, embed_dim//2, embed_dim-1]
        for dim in dims_to_plot:
            ax2.plot(position, pos_encoding[:, dim],
                    label=f'Dim {dim}', alpha=0.7)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Encoding Value', fontsize=12)
        ax2.set_title('Encoding vs Position (Selected Dims)', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Selected positions over dimensions
        ax3 = fig.add_subplot(gs[1, 1])
        positions_to_plot = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
        for pos in positions_to_plot:
            ax3.plot(range(embed_dim), pos_encoding[pos, :],
                    label=f'Pos {pos}', alpha=0.7)
        ax3.set_xlabel('Dimension', fontsize=12)
        ax3.set_ylabel('Encoding Value', fontsize=12)
        ax3.set_title('Encoding vs Dimension (Selected Pos)', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'Sinusoidal Positional Encoding\n(seq_len={seq_len}, embed_dim={embed_dim})',
                    fontsize=16, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved positional encoding visualization to {save_path}")

        plt.show()

    def visualize_embedding_evolution(
        self,
        embeddings_sequence: List[np.ndarray],
        layer_names: Optional[List[str]] = None,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Visualize how embeddings change through encoder layers.

        Args:
            embeddings_sequence: List of embedding arrays from different layers
            layer_names: Names for each layer
            sample_idx: Which sample to visualize
            save_path: If provided, save the plot to this path
        """
        n_layers = len(embeddings_sequence)

        if layer_names is None:
            layer_names = [f'Layer {i}' for i in range(n_layers)]

        # Extract the specific sample
        sample_embeddings = [emb[sample_idx] for emb in embeddings_sequence]

        # Convert to numpy
        sample_embeddings = [np.array(emb) if isinstance(emb, jnp.ndarray)
                           else emb for emb in sample_embeddings]

        # Create figure
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(2, n_layers, figure=fig, hspace=0.3, wspace=0.4)

        # Plot each layer's embedding as heatmap
        for i, (emb, name) in enumerate(zip(sample_embeddings, layer_names)):
            ax = fig.add_subplot(gs[0, i])

            if len(emb.shape) == 1:
                emb_to_plot = emb.reshape(-1, 1)
            else:
                emb_to_plot = emb.T

            im = ax.imshow(emb_to_plot, cmap='RdBu', aspect='auto')
            ax.set_title(name, fontsize=11)
            ax.set_xlabel('Token' if len(emb.shape) > 1 else '', fontsize=9)
            ax.set_ylabel('Dimension', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot statistics evolution
        ax_stats = fig.add_subplot(gs[1, :])

        means = [np.mean(emb) for emb in sample_embeddings]
        stds = [np.std(emb) for emb in sample_embeddings]
        norms = [np.linalg.norm(emb) for emb in sample_embeddings]

        x = np.arange(n_layers)
        ax_stats.plot(x, means, 'o-', label='Mean', linewidth=2, markersize=8)
        ax_stats.plot(x, stds, 's-', label='Std Dev', linewidth=2, markersize=8)

        ax2 = ax_stats.twinx()
        ax2.plot(x, norms, '^-', color='red', label='L2 Norm',
                linewidth=2, markersize=8)

        ax_stats.set_xlabel('Layer', fontsize=12)
        ax_stats.set_ylabel('Mean / Std Dev', fontsize=12)
        ax2.set_ylabel('L2 Norm', fontsize=12, color='red')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(layer_names, rotation=45, ha='right')
        ax_stats.grid(True, alpha=0.3)
        ax_stats.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax_stats.set_title('Embedding Statistics Through Layers', fontsize=12)

        plt.suptitle(f'Embedding Evolution (Sample {sample_idx})',
                    fontsize=14, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved evolution visualization to {save_path}")

        plt.show()

    def create_realtime_plot(
        self,
        update_interval: int = 100,
        max_points: int = 1000,
    ):
        """
        Create a real-time updating plot for streaming embeddings.

        Args:
            update_interval: Milliseconds between updates
            max_points: Maximum number of points to display

        Returns:
            Animation object and update function
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.suptitle('Real-time Embedding Visualization', fontsize=16)

        # Initialize empty data structures
        self.streaming_data = {
            'embeddings': [],
            'reduced_coords': [],
            'statistics': {'mean': [], 'std': [], 'norm': []},
            'timestamps': []
        }
        self.max_points = max_points

        # Set up subplots
        ax_2d = self.axes[0, 0]
        ax_2d.set_title('2D Projection (PCA)')
        ax_2d.set_xlabel('PC1')
        ax_2d.set_ylabel('PC2')
        self.scatter_2d = ax_2d.scatter([], [], alpha=0.6, s=50)

        ax_stats = self.axes[0, 1]
        ax_stats.set_title('Statistics Over Time')
        ax_stats.set_xlabel('Time Steps')
        ax_stats.set_ylabel('Value')
        self.line_mean, = ax_stats.plot([], [], 'b-', label='Mean')
        self.line_std, = ax_stats.plot([], [], 'r-', label='Std')
        self.line_norm, = ax_stats.plot([], [], 'g-', label='Norm')
        ax_stats.legend()
        ax_stats.grid(True, alpha=0.3)

        ax_dist = self.axes[1, 0]
        ax_dist.set_title('Value Distribution')
        ax_dist.set_xlabel('Value')
        ax_dist.set_ylabel('Frequency')

        ax_heatmap = self.axes[1, 1]
        ax_heatmap.set_title('Recent Embeddings')
        ax_heatmap.set_xlabel('Dimension')
        ax_heatmap.set_ylabel('Sample')

        plt.tight_layout()

        return self

    def update_realtime(self, new_embedding: np.ndarray):
        """
        Update the real-time visualization with new embedding.

        Args:
            new_embedding: New embedding vector to add
        """
        if isinstance(new_embedding, jnp.ndarray):
            new_embedding = np.array(new_embedding)

        # Add to data
        self.streaming_data['embeddings'].append(new_embedding)
        self.streaming_data['timestamps'].append(time.time())

        # Keep only recent data
        if len(self.streaming_data['embeddings']) > self.max_points:
            self.streaming_data['embeddings'].pop(0)
            self.streaming_data['timestamps'].pop(0)

        # Calculate statistics
        self.streaming_data['statistics']['mean'].append(np.mean(new_embedding))
        self.streaming_data['statistics']['std'].append(np.std(new_embedding))
        self.streaming_data['statistics']['norm'].append(np.linalg.norm(new_embedding))

        # Update plots
        self._update_plots()

    def _update_plots(self):
        """Internal method to update all subplots."""
        if len(self.streaming_data['embeddings']) < 2:
            return

        embeddings = np.array(self.streaming_data['embeddings'])

        # Update 2D projection
        if embeddings.shape[1] >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            self.scatter_2d.set_offsets(coords)
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()

        # Update statistics
        steps = range(len(self.streaming_data['statistics']['mean']))
        self.line_mean.set_data(steps, self.streaming_data['statistics']['mean'])
        self.line_std.set_data(steps, self.streaming_data['statistics']['std'])
        self.line_norm.set_data(steps, self.streaming_data['statistics']['norm'])
        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()

        # Update distribution
        self.axes[1, 0].clear()
        self.axes[1, 0].hist(embeddings[-1], bins=30, alpha=0.7, color='blue')
        self.axes[1, 0].set_title('Value Distribution (Latest)')
        self.axes[1, 0].set_xlabel('Value')
        self.axes[1, 0].set_ylabel('Frequency')

        # Update heatmap
        recent_embeddings = embeddings[-min(50, len(embeddings)):]
        self.axes[1, 1].clear()
        im = self.axes[1, 1].imshow(recent_embeddings, cmap='viridis', aspect='auto')
        self.axes[1, 1].set_title('Recent Embeddings')
        self.axes[1, 1].set_xlabel('Dimension')
        self.axes[1, 1].set_ylabel('Sample')

        plt.draw()
        plt.pause(0.001)

    def compare_embeddings(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        method: str = 'pca',
        save_path: Optional[str] = None,
    ):
        """
        Compare multiple sets of embeddings in the same space.

        Args:
            embeddings_dict: Dictionary mapping names to embedding arrays
            method: Reduction method ('pca', 'tsne')
            save_path: If provided, save the plot to this path
        """
        # Combine all embeddings
        all_embeddings = []
        all_labels = []
        all_groups = []

        for group_name, embeddings in embeddings_dict.items():
            if isinstance(embeddings, jnp.ndarray):
                embeddings = np.array(embeddings)

            all_embeddings.append(embeddings)
            all_labels.extend([f"{group_name}_{i}" for i in range(len(embeddings))])
            all_groups.extend([group_name] * len(embeddings))

        all_embeddings = np.vstack(all_embeddings)

        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(all_embeddings)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(all_embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each group with different color
        unique_groups = list(embeddings_dict.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

        for i, group_name in enumerate(unique_groups):
            mask = np.array(all_groups) == group_name
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      label=group_name, color=colors[i], s=100, alpha=0.6)

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'Embedding Comparison ({method.upper()})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison visualization to {save_path}")

        plt.show()


def demo_visualizer():
    """Demonstration of the visualization capabilities."""
    print("=" * 60)
    print("Embedding Visualizer Demo")
    print("=" * 60)
    print()

    viz = EmbeddingVisualizer()

    # 1. Generate sample embeddings
    print("1. Generating sample embeddings...")
    np.random.seed(42)
    n_samples = 100
    embed_dim = 128

    # Create three clusters
    cluster1 = np.random.randn(n_samples//3, embed_dim) + np.array([3, 0] + [0]*(embed_dim-2))
    cluster2 = np.random.randn(n_samples//3, embed_dim) + np.array([-3, 0] + [0]*(embed_dim-2))
    cluster3 = np.random.randn(n_samples//3, embed_dim) + np.array([0, 3] + [0]*(embed_dim-2))

    embeddings = np.vstack([cluster1, cluster2, cluster3])
    labels = ['A']*len(cluster1) + ['B']*len(cluster2) + ['C']*len(cluster3)
    colors = np.array([0]*len(cluster1) + [1]*len(cluster2) + [2]*len(cluster3))

    print(f"   Created {n_samples} embeddings with dimension {embed_dim}")
    print()

    # 2. 2D visualization
    print("2. Visualizing embeddings in 2D (PCA)...")
    viz.visualize_embeddings_2d(
        embeddings,
        method='pca',
        colors=colors,
        title='Sample Embeddings (3 Clusters)'
    )

    # 3. Positional encoding
    print("3. Visualizing positional encodings...")
    viz.visualize_positional_encoding(seq_len=64, embed_dim=128)

    # 4. Attention pattern (simulated)
    print("4. Visualizing attention patterns...")
    seq_len = 10
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=-1, keepdims=True)  # Normalize
    tokens = [f"T{i}" for i in range(seq_len)]

    viz.visualize_attention_patterns(
        attention,
        tokens=tokens,
        title='Sample Attention Pattern'
    )

    # 5. Embedding evolution (simulated)
    print("5. Visualizing embedding evolution through layers...")
    n_layers = 4
    sample_sequence = [embeddings + np.random.randn(*embeddings.shape) * 0.1 * i
                       for i in range(n_layers)]
    layer_names = ['Input'] + [f'Layer {i}' for i in range(1, n_layers)]

    viz.visualize_embedding_evolution(
        sample_sequence,
        layer_names=layer_names,
        sample_idx=0
    )

    print()
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    demo_visualizer()
