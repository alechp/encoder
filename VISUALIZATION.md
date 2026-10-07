# 📊 Visualization Layer Documentation

The visualization layer provides comprehensive, real-time visualization capabilities for Transformer embeddings and attention patterns. It runs independently of the encoder and can be used with any embedding system.

## Features

### 1. **2D/3D Embedding Visualization**
- Dimensionality reduction using PCA or t-SNE
- Color-coded clusters
- Interactive labeling
- Support for JAX and NumPy arrays

### 2. **Real-time Streaming**
- Live updates as embeddings are generated
- Statistical tracking (mean, std, norm)
- Distribution histograms
- Recent embedding heatmaps

### 3. **Attention Pattern Visualization**
- Heatmap visualization of attention weights
- Multi-head attention support
- Token-level attention patterns
- Customizable color schemes

### 4. **Positional Encoding Visualization**
- Sinusoidal encoding patterns
- Dimension-wise analysis
- Position-wise analysis
- Multiple configuration support

### 5. **Embedding Evolution Tracking**
- Layer-by-layer visualization
- Statistical evolution plots
- Norm tracking through layers
- Comparative analysis

### 6. **Embedding Comparison**
- Side-by-side comparison of different embeddings
- Support for multiple embedding sets
- Shared visualization space
- Color-coded groups

## Quick Start

### Basic Usage

```python
from visualizer import EmbeddingVisualizer
import numpy as np

# Create visualizer
viz = EmbeddingVisualizer()

# Generate sample embeddings
embeddings = np.random.randn(100, 128)  # 100 samples, 128 dims

# Visualize in 2D
viz.visualize_embeddings_2d(
    embeddings,
    method='pca',
    title='My Embeddings'
)
```

### With Transformer Encoder

```python
import jax
import jax.numpy as jnp
import haiku as hk
from transformer_encoder import encoder_fn
from visualizer import EmbeddingVisualizer

# Setup encoder
config = {
    'vocab_size': 1000,
    'embed_dim': 128,
    'seq_len': 32,
    'batch_size': 50,
    'n_heads': 4,
    'd_k': 32,
    'hidden_dim': 512,
}

encoder = hk.transform(lambda x: encoder_fn(x, config))
rng = jax.random.PRNGKey(42)
params = encoder.init(rng, jnp.zeros((50, 32), dtype=jnp.int32))

# Generate embeddings
inputs = jax.random.randint(jax.random.PRNGKey(0), (50, 32), 0, 1000)
outputs = encoder.apply(params, None, inputs)

# Average pool over sequence
sentence_embeddings = jnp.mean(outputs, axis=1)

# Visualize
viz = EmbeddingVisualizer()
viz.visualize_embeddings_2d(
    sentence_embeddings,
    method='pca',
    title='Encoder Output Embeddings'
)
```

## API Reference

### EmbeddingVisualizer

Main class for all visualization operations.

#### Methods

##### `visualize_embeddings_2d(embeddings, labels=None, method='pca', title='...', colors=None, save_path=None)`

Visualize embeddings in 2D space.

**Parameters:**
- `embeddings` (np.ndarray): Shape (n_samples, embed_dim)
- `labels` (List[str], optional): Text labels for each point
- `method` (str): 'pca' or 'tsne'
- `title` (str): Plot title
- `colors` (np.ndarray, optional): Color values for each point
- `save_path` (str, optional): Path to save the plot

**Returns:**
- `np.ndarray`: 2D coordinates after reduction

**Example:**
```python
coords = viz.visualize_embeddings_2d(
    embeddings,
    labels=['token1', 'token2', ...],
    method='pca',
    colors=cluster_ids,
    save_path='embeddings_2d.png'
)
```

##### `visualize_embeddings_3d(embeddings, labels=None, method='pca', title='...', colors=None, save_path=None)`

Visualize embeddings in 3D space.

**Parameters:** Same as `visualize_embeddings_2d`

**Returns:**
- `np.ndarray`: 3D coordinates after reduction

##### `visualize_attention_patterns(attention_weights, tokens=None, head_idx=0, title=None, save_path=None)`

Visualize attention weights as a heatmap.

**Parameters:**
- `attention_weights` (np.ndarray): Shape (n_heads, seq_len, seq_len) or (seq_len, seq_len)
- `tokens` (List[str], optional): Token labels
- `head_idx` (int): Which attention head to visualize
- `title` (str, optional): Plot title
- `save_path` (str, optional): Path to save

**Example:**
```python
viz.visualize_attention_patterns(
    attention_weights,
    tokens=['The', 'cat', 'sat'],
    head_idx=0,
    title='Attention Head 0'
)
```

##### `visualize_positional_encoding(seq_len=128, embed_dim=512, save_path=None)`

Visualize sinusoidal positional encodings.

**Parameters:**
- `seq_len` (int): Sequence length
- `embed_dim` (int): Embedding dimension
- `save_path` (str, optional): Path to save

**Example:**
```python
viz.visualize_positional_encoding(
    seq_len=128,
    embed_dim=512,
    save_path='pos_encoding.png'
)
```

##### `visualize_embedding_evolution(embeddings_sequence, layer_names=None, sample_idx=0, save_path=None)`

Visualize how embeddings change through encoder layers.

**Parameters:**
- `embeddings_sequence` (List[np.ndarray]): List of embeddings from each layer
- `layer_names` (List[str], optional): Names for each layer
- `sample_idx` (int): Which sample to visualize
- `save_path` (str, optional): Path to save

**Example:**
```python
# Collect embeddings from each layer
layer_embeddings = [input_emb, layer1_out, layer2_out, final_out]

viz.visualize_embedding_evolution(
    layer_embeddings,
    layer_names=['Input', 'Layer 1', 'Layer 2', 'Output'],
    sample_idx=0
)
```

##### `create_realtime_plot(update_interval=100, max_points=1000)`

Create a real-time updating plot for streaming embeddings.

**Parameters:**
- `update_interval` (int): Milliseconds between updates
- `max_points` (int): Maximum number of points to display

**Returns:**
- `self`: For method chaining

**Example:**
```python
viz = EmbeddingVisualizer()
viz.create_realtime_plot(update_interval=100, max_points=200)

# Generate embeddings and update in real-time
for embedding in embedding_stream:
    viz.update_realtime(embedding)
    time.sleep(0.05)
```

##### `update_realtime(new_embedding)`

Update the real-time visualization with a new embedding.

**Parameters:**
- `new_embedding` (np.ndarray): New embedding vector to add

##### `compare_embeddings(embeddings_dict, method='pca', save_path=None)`

Compare multiple sets of embeddings in the same space.

**Parameters:**
- `embeddings_dict` (Dict[str, np.ndarray]): Dictionary mapping names to embedding arrays
- `method` (str): 'pca' or 'tsne'
- `save_path` (str, optional): Path to save

**Example:**
```python
viz.compare_embeddings(
    {
        'Raw Embeddings': raw_embeddings,
        'After Layer 1': layer1_embeddings,
        'Final Output': final_embeddings,
    },
    method='pca',
    save_path='comparison.png'
)
```

## Demo Scripts

### 1. Standalone Visualizer Demo

```bash
python visualizer.py
```

Demonstrates all visualization capabilities with synthetic data.

### 2. Real-time Demo

```bash
python realtime_demo.py
```

Interactive menu with multiple demos:
- Encoder embeddings (2D/3D)
- Positional encoding visualization
- Attention patterns
- Embedding comparison
- Real-time streaming

## Examples

### Example 1: Visualize Training Progress

```python
from visualizer import EmbeddingVisualizer
import numpy as np

viz = EmbeddingVisualizer()

# During training, collect embeddings
embeddings_by_epoch = {
    'Epoch 1': epoch1_embeddings,
    'Epoch 10': epoch10_embeddings,
    'Epoch 50': epoch50_embeddings,
    'Epoch 100': epoch100_embeddings,
}

# Compare how embeddings change
viz.compare_embeddings(
    embeddings_by_epoch,
    method='tsne',
    save_path='training_progress.png'
)
```

### Example 2: Analyze Attention Patterns

```python
# Extract attention weights from encoder
# (You would need to modify the encoder to return attention weights)

tokens = ['The', 'quick', 'brown', 'fox', 'jumps']
attention = extract_attention_weights(encoder, tokens)  # Your function

# Visualize all heads
for head_idx in range(n_heads):
    viz.visualize_attention_patterns(
        attention,
        tokens=tokens,
        head_idx=head_idx,
        save_path=f'attention_head_{head_idx}.png'
    )
```

### Example 3: Real-time Monitoring

```python
import time
from visualizer import EmbeddingVisualizer

viz = EmbeddingVisualizer()
viz.create_realtime_plot(update_interval=50, max_points=500)

# Simulate real-time embedding generation
for i in range(1000):
    # Generate or compute embedding
    embedding = generate_embedding(i)  # Your function

    # Update visualization
    viz.update_realtime(embedding)

    # Small delay
    time.sleep(0.01)
```

### Example 4: Cluster Analysis

```python
from sklearn.cluster import KMeans
from visualizer import EmbeddingVisualizer

# Get embeddings
embeddings = get_encoder_embeddings(texts)

# Cluster
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)

# Visualize with cluster colors
viz = EmbeddingVisualizer()
viz.visualize_embeddings_2d(
    embeddings,
    colors=clusters,
    title='Embedding Clusters (K-Means, k=5)',
    save_path='clusters.png'
)
```

## Configuration Options

### Dimensionality Reduction Methods

**PCA (Principal Component Analysis)**
- Faster computation
- Linear dimensionality reduction
- Good for visualization
- Shows variance explained

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Better for non-linear structures
- Preserves local structure
- Slower for large datasets
- Good for cluster visualization

### Color Schemes

All matplotlib colormaps are supported:
- `viridis` (default)
- `plasma`
- `RdBu` (red-blue)
- `coolwarm`
- `tab10` (categorical)

## Performance Tips

1. **Large Datasets**: Use PCA instead of t-SNE for datasets > 1000 samples
2. **Real-time**: Limit `max_points` to keep visualization responsive
3. **High-dimensional**: Use PCA to reduce to ~50 dims before t-SNE
4. **Memory**: Clear old embeddings periodically in streaming mode

## Output Directory

All visualizations with `save_path` are saved to:
```
outputs/
├── encoder_embeddings_2d.png
├── encoder_embeddings_3d.png
├── pos_encoding_small.png
├── attention_focused.png
└── ...
```

Create the directory first:
```bash
mkdir -p outputs
```

## Integration with Training

```python
class TrainingVisualizer:
    def __init__(self):
        self.viz = EmbeddingVisualizer()
        self.embeddings_history = []

    def on_epoch_end(self, epoch, embeddings):
        """Call this at the end of each epoch"""
        self.embeddings_history.append(embeddings)

        # Visualize current embeddings
        self.viz.visualize_embeddings_2d(
            embeddings,
            title=f'Epoch {epoch}',
            save_path=f'outputs/epoch_{epoch}.png'
        )

    def create_evolution_video(self):
        """Create animation showing embedding evolution"""
        # Use saved images to create video
        pass
```

## Troubleshooting

### Issue: Plots not displaying
**Solution**: Make sure you're not running in a headless environment. For headless:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Issue: t-SNE is slow
**Solution**: Reduce perplexity or use PCA for large datasets
```python
viz.visualize_embeddings_2d(embeddings, method='pca')  # Faster
```

### Issue: Out of memory in real-time mode
**Solution**: Reduce `max_points`
```python
viz.create_realtime_plot(max_points=100)  # Lower limit
```

### Issue: JAX arrays not working
**Solution**: Arrays are automatically converted. If issues persist:
```python
import numpy as np
embeddings_np = np.array(embeddings)
viz.visualize_embeddings_2d(embeddings_np)
```

## Requirements

```bash
pip install numpy matplotlib scikit-learn jax jaxlib
```

## License

Part of the Transformer Encoder implementation. See main README for details.
