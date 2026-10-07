"""
Transformer Encoder Implementation from Scratch with JAX and Haiku

Based on the tutorial: "Implementing a Transformer Encoder from Scratch with JAX and Haiku"
by Ryan Pégoud

This module implements all the core components of a Transformer Encoder:
- Embedding Layer with Positional Encoding
- Multi-Head Attention
- Layer Normalization
- Position-wise Feed-Forward Network
- Complete Encoder Block
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
import haiku as hk
from functools import partial
import math
from typing import Optional


class EmbeddingLayer(hk.Module):
    """
    Embedding layer that combines token embeddings with sinusoidal positional encodings.

    The positional encoding uses sine and cosine functions of different frequencies:
    - PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        seq_len: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_layer = hk.Embed(self.vocab_size, self.embed_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Token indices of shape (batch_size, seq_len)

        Returns:
            Embeddings with positional encodings of shape (batch_size, seq_len, embed_dim)
        """
        embeddings = self.embedding_layer(x)
        positional_encodings = self._batched_positional_encoding(
            jnp.arange(self.seq_len),
            jnp.arange(self.embed_dim),
        )
        return embeddings + positional_encodings

    @partial(jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, None, 0), out_axes=1)
    def _batched_positional_encoding(self, pos: jnp.ndarray, dim: jnp.ndarray) -> jnp.ndarray:
        """
        Compute positional encodings using sine for even dimensions and cosine for odd dimensions.

        Args:
            pos: Position indices
            dim: Dimension indices

        Returns:
            Positional encoding matrix
        """
        def _even_encoding():
            return jnp.sin(pos / (jnp.power(10_000, 2 * dim / self.embed_dim)))

        def _odd_encoding():
            return jnp.cos(pos / (jnp.power(10_000, 2 * dim / self.embed_dim)))

        is_even = dim % 2 == 0
        return lax.cond(is_even, _even_encoding, _odd_encoding)


class Multihead_Attention(hk.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need".

    The multi-head attention splits the embedding dimension into multiple heads,
    computes scaled dot-product attention for each head in parallel, then
    concatenates and projects the results.
    """

    def __init__(
        self,
        embed_dim: int,
        batch_size: int,
        seq_len: int,
        n_heads: int,
        d_k: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_k = d_k  # Dimension per head, typically embed_dim / n_heads

        # Initialize weight matrices for Q, K, V projections
        self.w_q = hk.Linear(self.n_heads * self.d_k, name="W_Q")
        self.w_k = hk.Linear(self.n_heads * self.d_k, name="W_K")
        self.w_v = hk.Linear(self.n_heads * self.d_k, name="W_V")
        self.w_o = hk.Linear(self.embed_dim, name="W_O")

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply multi-head attention.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size = x.shape[0]

        # Project inputs to Q, K, V
        q = self.w_q(x)  # (batch_size, seq_len, n_heads * d_k)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape and transpose for multi-head attention
        # (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Compute scaled dot-product attention for all heads in parallel
        attention_output = self._scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and apply output projection
        attention_output = self._combine_heads(attention_output, batch_size)
        output = self.w_o(attention_output)

        return output

    def _split_heads(self, x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """
        Split the last dimension into (n_heads, d_k) and transpose for multi-head attention.
        """
        x = x.reshape(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_k)

    def _combine_heads(self, x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """
        Inverse of _split_heads: concatenate the attention heads.
        """
        x = x.transpose(0, 2, 1, 3)  # (batch_size, seq_len, n_heads, d_k)
        return x.reshape(batch_size, -1, self.n_heads * self.d_k)

    def _scaled_dot_product_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            q: Queries of shape (batch_size, n_heads, seq_len, d_k)
            k: Keys of shape (batch_size, n_heads, seq_len, d_k)
            v: Values of shape (batch_size, n_heads, seq_len, d_k)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, n_heads, seq_len, d_k)
        """
        # Compute attention scores
        matmul_qk = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # (batch_size, n_heads, seq_len, seq_len)

        # Scale by sqrt(d_k)
        scaled_attention_logits = matmul_qk / jnp.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits = jnp.where(mask == 0, -1e9, scaled_attention_logits)

        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention weights to values
        output = jnp.matmul(attention_weights, v)

        return output


class LayerNorm(hk.Module):
    """
    Layer Normalization as described in "Layer Normalization" (Ba et al., 2016).

    Normalizes the features for each sample in a batch independently.
    Unlike batch normalization, layer normalization operates across the features,
    which is more suitable for sequences where each sample may have a unique distribution.
    """

    def __init__(
        self,
        feature_axis: int = -1,
        epsilon: float = 1e-6,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.feature_axis = feature_axis
        self.epsilon = epsilon

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply layer normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor with learnable scale (alpha) and shift (beta)
        """
        # Get the feature dimension
        feature_dim = x.shape[self.feature_axis]

        # Initialize learnable parameters
        alpha = hk.get_parameter(
            "alpha",
            shape=[feature_dim],
            init=hk.initializers.Constant(1.0),
        )
        beta = hk.get_parameter(
            "beta",
            shape=[feature_dim],
            init=hk.initializers.Constant(0.0),
        )

        # Compute mean and variance
        mean = jnp.mean(x, axis=self.feature_axis, keepdims=True)
        variance = jnp.var(x, axis=self.feature_axis, keepdims=True)

        # Normalize
        normalized = (x - mean) / jnp.sqrt(variance + self.epsilon)

        # Scale and shift
        return alpha * normalized + beta


class FeedForwardNet(hk.Module):
    """
    Position-wise Feed-Forward Network.

    A fully connected network applied to each position separately and identically.
    Consists of two linear transformations with a GELU activation in between:
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dense1 = hk.Linear(hidden_dim, name="dense1")
        self.dense2 = hk.Linear(embed_dim, name="dense2")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.dense1(x)
        x = jax.nn.gelu(x)
        x = self.dense2(x)
        return x


class Encoder(hk.Module):
    """
    Complete Transformer Encoder block.

    Combines all components:
    1. Embedding Layer (token embeddings + positional encodings)
    2. Multi-Head Attention with residual connection and layer normalization
    3. Feed-Forward Network with residual connection and layer normalization

    Architecture:
    Input -> Embeddings -> [Multi-Head Attention -> Add & Norm] -> [FFN -> Add & Norm] -> Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seq_len: int,
        batch_size: int,
        n_heads: int,
        d_k: int,
        hidden_dim: int,
        name: str = "Encoder",
    ) -> None:
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_k = d_k  # Common value: int(self.embed_dim / self.n_heads)
        self.hidden_dim = hidden_dim

        # Initialize all sub-modules
        self.embedding_layer = EmbeddingLayer(
            embed_dim, vocab_size, seq_len, name="Embedding_layer"
        )
        self.multihead_attention = Multihead_Attention(
            embed_dim, batch_size, seq_len, n_heads, d_k, name="Multihead_Attention"
        )
        self.post_attention_layer_norm = LayerNorm(
            feature_axis=-1, name="Post_Attention_Layer_Norm"
        )
        self.post_feedforward_layer_norm = LayerNorm(
            feature_axis=-1, name="Post_FeedForward_Layer_Norm"
        )
        self.feedforward_net = FeedForwardNet(
            embed_dim, hidden_dim, name="FeedForward_Net"
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the encoder.

        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            Encoded representations of shape (batch_size, seq_len, embed_dim)
        """
        # Step 1: Get embeddings with positional encodings
        embeddings = self.embedding_layer(x)

        # Step 2: Multi-head attention with residual connection and layer norm
        attention_matrices = self.multihead_attention(embeddings)
        normalized_residual_attention = self.post_attention_layer_norm(
            attention_matrices + embeddings
        )

        # Step 3: Feed-forward network with residual connection and layer norm
        feedforward_outputs = self.feedforward_net(normalized_residual_attention)
        output = self.post_feedforward_layer_norm(
            feedforward_outputs + normalized_residual_attention
        )

        return output


# Example usage and initialization
def encoder_fn(x: jnp.ndarray, config: dict) -> jnp.ndarray:
    """
    Wrapper function for the Encoder to be used with hk.transform.

    Args:
        x: Input token indices
        config: Configuration dictionary with model hyperparameters

    Returns:
        Encoder output
    """
    encoder = Encoder(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        n_heads=config['n_heads'],
        d_k=config['d_k'],
        hidden_dim=config['hidden_dim'],
    )
    return encoder(x)


def create_encoder(config: dict):
    """
    Create and initialize a Transformer Encoder.

    Args:
        config: Configuration dictionary with the following keys:
            - vocab_size: Size of the vocabulary
            - embed_dim: Embedding dimension
            - seq_len: Maximum sequence length
            - batch_size: Batch size
            - n_heads: Number of attention heads
            - d_k: Dimension per attention head (typically embed_dim / n_heads)
            - hidden_dim: Hidden dimension of feed-forward network (typically 4 * embed_dim)

    Returns:
        Tuple of (init_fn, apply_fn) from hk.transform

    Example:
        >>> config = {
        ...     'vocab_size': 10000,
        ...     'embed_dim': 512,
        ...     'seq_len': 128,
        ...     'batch_size': 32,
        ...     'n_heads': 8,
        ...     'd_k': 64,  # 512 / 8
        ...     'hidden_dim': 2048,  # 4 * 512
        ... }
        >>> encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))
        >>>
        >>> # Initialize parameters
        >>> rng = jax.random.PRNGKey(42)
        >>> dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
        >>> params = encoder_transformed.init(rng, dummy_input)
        >>>
        >>> # Apply to real data
        >>> real_input = jnp.array([[1, 2, 3, 4, 5]])  # Example token indices
        >>> output = encoder_transformed.apply(params, None, real_input)
    """
    # Create the transformed encoder
    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))

    return encoder_transformed


if __name__ == "__main__":
    """
    Example demonstrating how to use the Transformer Encoder.
    """
    # Define model configuration
    config = {
        'vocab_size': 10000,
        'embed_dim': 512,
        'seq_len': 128,
        'batch_size': 32,
        'n_heads': 8,
        'd_k': 64,  # embed_dim / n_heads
        'hidden_dim': 2048,  # Typically 4 * embed_dim
    }

    # Create the encoder
    encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))

    # Initialize parameters with dummy data
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    params = encoder_transformed.init(rng, dummy_input)

    print("Encoder initialized successfully!")
    print(f"Parameter structure: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

    # Apply to sample data
    sample_input = jax.random.randint(
        jax.random.PRNGKey(0),
        (config['batch_size'], config['seq_len']),
        0,
        config['vocab_size']
    )
    output = encoder_transformed.apply(params, None, sample_input)

    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({config['batch_size']}, {config['seq_len']}, {config['embed_dim']})")
