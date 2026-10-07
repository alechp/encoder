# Transformer Encoder from Scratch with JAX and Haiku

This repository contains a complete implementation of a Transformer Encoder built from scratch using JAX and Haiku, based on the tutorial ["Implementing a Transformer Encoder from Scratch with JAX and Haiku"](https://towardsdatascience.com/implementing-a-transformer-encoder-from-scratch-with-jax-and-haiku-791d31b4f0dd/) by Ryan Pégoud.

## 🚀 Quick Start

**Interactive Menu (Recommended):**

```bash
./run.sh          # macOS/Linux
python run.py     # Windows or any platform
```

**Or use command-line options:**

```bash
./run.sh --full          # Run everything (tests + demos)
./run.sh --encoder       # Encoder only
./run.sh --viz           # Visualization only
./run.sh --tests         # Tests only
./run.sh --quick         # Quick verification
./run.sh --interactive   # Interactive viz menu
```

### Menu Options

1. **Full Demo** - Complete walkthrough (tests + encoder + visualization)
2. **Encoder Demo Only** - Core transformer implementation
3. **Visualization Demo Only** - Visualization layer demos
4. **Tests Only** - Run all tests
5. **Quick Test** - Verify installation
6. **Interactive Visualization** - Hands-on visualization menu

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md).

## Overview

This implementation includes all the fundamental building blocks of a Transformer Encoder:

1. **Embedding Layer** - Token embeddings combined with sinusoidal positional encodings
2. **Multi-Head Attention** - Parallel attention mechanism with multiple heads
3. **Layer Normalization** - Feature-wise normalization with learnable parameters
4. **Position-wise Feed-Forward Network** - Two-layer fully connected network with GELU activation
5. **Complete Encoder Block** - All components assembled with residual connections
6. **🎨 Visualization Layer** - Real-time embedding and attention visualization (NEW!)

## 📊 Visualization Features

The repository includes **two visualization systems**:

### 🌐 Web Visualizer (Recommended - NEW!)

Modern, interactive JavaScript web application:

```bash
./start_visualizer.sh    # Start web server
# Open http://localhost:5000 in your browser
```

**Features:**
- 🎨 Beautiful modern UI with dark theme
- 📊 Interactive Plotly.js visualizations (2D/3D)
- ⚡ Real-time WebSocket streaming
- 🎯 PCA, t-SNE, positional encoding
- 📱 Responsive design
- 🚀 No Python plotting libraries needed

See [viz_app/README.md](viz_app/README.md) for details.

### 🐍 Python Visualizer (Alternative)

Command-line visualization tools:

```bash
python realtime_demo.py  # Interactive menu
python visualizer.py     # Standalone demos
```

**Features:**
- 2D/3D matplotlib visualizations
- Batch processing
- Export to PNG
- Integration with training

See [VISUALIZATION.md](VISUALIZATION.md) for complete documentation.

## Architecture

```
Input (Token IDs)
    ↓
Embedding Layer (Token Embeddings + Positional Encodings)
    ↓
Multi-Head Attention
    ↓
Add & Normalize (Residual Connection + Layer Norm)
    ↓
Feed-Forward Network
    ↓
Add & Normalize (Residual Connection + Layer Norm)
    ↓
Output (Encoded Representations)
```

## Key Features

### Positional Encoding
Uses sinusoidal functions to encode position information:
- Even dimensions: `sin(pos / 10000^(2i/embed_dim))`
- Odd dimensions: `cos(pos / 10000^(2i/embed_dim))`

### Multi-Head Attention
Implements scaled dot-product attention across multiple heads:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Residual Connections
Skip connections around both the attention and feed-forward blocks to:
- Solve the vanishing gradient problem
- Enable gradients to flow more efficiently
- Lead to faster convergence and more stable training

### Layer Normalization
Normalizes across features rather than batch dimension, which is more suitable for sequences where each sample may have unique distributions.

## Installation

### Option 1: Automated Setup (Recommended)

Use the provided run scripts to automatically set up everything:

```bash
# On macOS/Linux
./run.sh

# On Windows or any platform
python run.py
```

The run scripts will:
- Create a virtual environment
- Install all dependencies
- Run tests to verify everything works
- Run comprehensive examples
- Display the demo

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import jax
import jax.numpy as jnp
import haiku as hk
from transformer_encoder import encoder_fn

# Define model configuration
config = {
    'vocab_size': 10000,    # Size of vocabulary
    'embed_dim': 512,       # Embedding dimension
    'seq_len': 128,         # Maximum sequence length
    'batch_size': 32,       # Batch size
    'n_heads': 8,           # Number of attention heads
    'd_k': 64,              # Dimension per head (embed_dim / n_heads)
    'hidden_dim': 2048,     # FFN hidden dimension (typically 4 * embed_dim)
}

# Create the encoder using hk.transform
encoder_transformed = hk.transform(lambda x: encoder_fn(x, config))

# Initialize parameters
rng = jax.random.PRNGKey(42)
dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
params = encoder_transformed.init(rng, dummy_input)

# Apply to real data
real_input = jax.random.randint(
    jax.random.PRNGKey(0),
    (config['batch_size'], config['seq_len']),
    0,
    config['vocab_size']
)
output = encoder_transformed.apply(params, None, real_input)

print(f"Input shape: {real_input.shape}")   # (32, 128)
print(f"Output shape: {output.shape}")      # (32, 128, 512)
```

### Running the Example

The repository includes a complete example in the main file:

```bash
python transformer_encoder.py
```

This will:
1. Initialize the encoder with default configuration
2. Print the parameter structure
3. Process sample input data
4. Display input/output shapes

## Components

### EmbeddingLayer
Combines token embeddings with positional encodings using JAX's `vmap` for efficient vectorization.

```python
from transformer_encoder import EmbeddingLayer

embedding_layer = EmbeddingLayer(
    embed_dim=512,
    vocab_size=10000,
    seq_len=128
)
```

### Multihead_Attention
Implements multi-head scaled dot-product attention with separate Q, K, V projections.

```python
from transformer_encoder import Multihead_Attention

attention = Multihead_Attention(
    embed_dim=512,
    batch_size=32,
    seq_len=128,
    n_heads=8,
    d_k=64
)
```

### LayerNorm
Layer normalization with learnable scale (alpha) and shift (beta) parameters.

```python
from transformer_encoder import LayerNorm

layer_norm = LayerNorm(feature_axis=-1)
```

### FeedForwardNet
Position-wise feed-forward network with GELU activation.

```python
from transformer_encoder import FeedForwardNet

ffn = FeedForwardNet(
    embed_dim=512,
    hidden_dim=2048
)
```

### Encoder
Complete encoder block combining all components with residual connections.

```python
from transformer_encoder import Encoder

encoder = Encoder(
    vocab_size=10000,
    embed_dim=512,
    seq_len=128,
    batch_size=32,
    n_heads=8,
    d_k=64,
    hidden_dim=2048
)
```

## Haiku and JAX Paradigm

This implementation follows JAX's functional programming paradigm using Haiku:

1. **Module Definition**: Each component is defined as an `hk.Module` with `__init__` and `__call__` methods
2. **Transformation**: Modules are wrapped in functions and transformed using `hk.transform`
3. **Initialization**: `init()` creates the parameter tree using dummy data
4. **Application**: `apply()` processes real data using the initialized parameters

Example:
```python
# Define a function containing the module
def forward_fn(x):
    encoder = Encoder(...)
    return encoder(x)

# Transform it to get init and apply functions
encoder_transformed = hk.transform(forward_fn)

# Initialize
params = encoder_transformed.init(rng, dummy_input)

# Apply
output = encoder_transformed.apply(params, None, real_input)
```

## Configuration Guidelines

### Typical Hyperparameters

- **embed_dim**: Common values are 512 (base) or 768 (BERT-base)
- **n_heads**: Typically 8 or 12
- **d_k**: Usually `embed_dim / n_heads`
- **hidden_dim**: Typically `4 * embed_dim` (2048 for embed_dim=512)
- **seq_len**: Depends on your task (128, 256, 512, etc.)

### Example Configurations

**Small Model:**
```python
config = {
    'vocab_size': 5000,
    'embed_dim': 256,
    'seq_len': 64,
    'batch_size': 16,
    'n_heads': 4,
    'd_k': 64,
    'hidden_dim': 1024,
}
```

**Base Model (BERT-like):**
```python
config = {
    'vocab_size': 30000,
    'embed_dim': 768,
    'seq_len': 512,
    'batch_size': 32,
    'n_heads': 12,
    'd_k': 64,
    'hidden_dim': 3072,
}
```

**Large Model:**
```python
config = {
    'vocab_size': 50000,
    'embed_dim': 1024,
    'seq_len': 512,
    'batch_size': 16,
    'n_heads': 16,
    'd_k': 64,
    'hidden_dim': 4096,
}
```

## Technical Details

### JAX Utilities Used

- **`vmap`**: Vectorizes operations over different attention heads and matrices
- **`jit`**: Just-in-time compilation for performance optimization
- **`tree_map`**: Maps operations over PyTree structures
- **`lax.cond`**: Conditional operations in a functionally pure way

### Design Decisions

1. **Sinusoidal Positional Encodings**: Fixed encodings rather than learned, allowing the model to extrapolate to longer sequences
2. **GELU Activation**: Used in the feed-forward network (common in modern transformers like BERT and GPT)
3. **Layer Normalization**: Applied after residual connections (Post-LN), though Pre-LN is also common
4. **Xavier/Glorot Initialization**: Used by default in Haiku's `hk.Linear`

## References

1. [Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Vaswani et al., Google
2. [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450) - Ba et al.
3. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
4. [Original Tutorial](https://towardsdatascience.com/implementing-a-transformer-encoder-from-scratch-with-jax-and-haiku-791d31b4f0dd/) - Ryan Pégoud

## Next Steps

This implementation covers the encoder half of a Transformer. To build a complete Transformer, you would also need:

1. **Decoder**: Similar architecture but includes masked self-attention and cross-attention
2. **Training Loop**: Using Optax for optimization
3. **Task-Specific Heads**: Classification, language modeling, etc.

## License

This implementation is based on the tutorial by Ryan Pégoud and is provided for educational purposes.

## Acknowledgments

- Ryan Pégoud for the original tutorial
- The JAX and Haiku teams at Google DeepMind
- The authors of "Attention is All You Need"
