"""
Flask Backend for Transformer Encoder Visualization

Serves the web-based visualization interface and provides API endpoints
for encoder data, embeddings, and real-time streaming.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_encoder import encoder_fn, EmbeddingLayer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import time
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transformer-viz-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
encoder_instance = None
encoder_params = None
encoder_config = None


def initialize_encoder(config=None):
    """Initialize the transformer encoder with given config."""
    global encoder_instance, encoder_params, encoder_config

    if config is None:
        config = {
            'vocab_size': 1000,
            'embed_dim': 128,
            'seq_len': 32,
            'batch_size': 50,
            'n_heads': 4,
            'd_k': 32,
            'hidden_dim': 512,
        }

    encoder_config = config
    encoder_instance = hk.transform(lambda x: encoder_fn(x, config))

    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((config['batch_size'], config['seq_len']), dtype=jnp.int32)
    encoder_params = encoder_instance.init(rng, dummy_input)

    return True


@app.route('/')
def index():
    """Serve the main visualization interface."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current encoder configuration."""
    if encoder_config is None:
        initialize_encoder()
    return jsonify(encoder_config)


@app.route('/api/config', methods=['POST'])
def set_config():
    """Update encoder configuration."""
    config = request.json
    initialize_encoder(config)
    return jsonify({'status': 'success', 'config': encoder_config})


@app.route('/api/encode', methods=['POST'])
def encode_batch():
    """Encode a batch of sequences."""
    global encoder_instance, encoder_params, encoder_config

    if encoder_instance is None:
        initialize_encoder()

    data = request.json
    input_ids = np.array(data.get('input_ids', []))

    if len(input_ids.shape) == 1:
        input_ids = input_ids.reshape(1, -1)

    # Pad or truncate to seq_len
    seq_len = encoder_config['seq_len']
    if input_ids.shape[1] < seq_len:
        padding = np.zeros((input_ids.shape[0], seq_len - input_ids.shape[1]), dtype=np.int32)
        input_ids = np.concatenate([input_ids, padding], axis=1)
    else:
        input_ids = input_ids[:, :seq_len]

    # Encode
    outputs = encoder_instance.apply(encoder_params, None, jnp.array(input_ids, dtype=jnp.int32))

    # Average pool for sentence embeddings
    sentence_embeddings = jnp.mean(outputs, axis=1)

    return jsonify({
        'embeddings': sentence_embeddings.tolist(),
        'outputs': outputs.tolist(),
        'shape': list(outputs.shape)
    })


@app.route('/api/visualize/pca', methods=['POST'])
def visualize_pca():
    """Compute PCA projection of embeddings."""
    data = request.json
    embeddings = np.array(data.get('embeddings', []))
    n_components = data.get('n_components', 2)

    if len(embeddings) < n_components:
        return jsonify({'error': 'Not enough samples for PCA'}), 400

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    return jsonify({
        'coordinates': coords.tolist(),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'total_variance': float(sum(pca.explained_variance_ratio_))
    })


@app.route('/api/visualize/tsne', methods=['POST'])
def visualize_tsne():
    """Compute t-SNE projection of embeddings."""
    data = request.json
    embeddings = np.array(data.get('embeddings', []))
    n_components = data.get('n_components', 2)
    perplexity = data.get('perplexity', min(30, len(embeddings) - 1))

    if len(embeddings) < 2:
        return jsonify({'error': 'Not enough samples for t-SNE'}), 400

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    return jsonify({
        'coordinates': coords.tolist()
    })


@app.route('/api/positional_encoding', methods=['GET'])
def get_positional_encoding():
    """Generate positional encoding matrix."""
    seq_len = int(request.args.get('seq_len', 128))
    embed_dim = int(request.args.get('embed_dim', 512))

    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

    pos_encoding = np.zeros((seq_len, embed_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return jsonify({
        'encoding': pos_encoding.tolist(),
        'shape': [seq_len, embed_dim]
    })


@app.route('/api/generate_samples', methods=['POST'])
def generate_samples():
    """Generate sample embeddings for visualization."""
    data = request.json
    n_samples = data.get('n_samples', 60)
    n_categories = data.get('n_categories', 3)

    if encoder_instance is None:
        initialize_encoder()

    samples_per_category = n_samples // n_categories
    all_inputs = []
    labels = []

    for cat in range(n_categories):
        start_token = cat * (encoder_config['vocab_size'] // n_categories)
        end_token = (cat + 1) * (encoder_config['vocab_size'] // n_categories)

        category_inputs = jax.random.randint(
            jax.random.PRNGKey(cat),
            (samples_per_category, encoder_config['seq_len']),
            start_token, end_token
        )

        all_inputs.append(category_inputs)
        labels.extend([f'Category {cat}'] * samples_per_category)

    all_inputs = jnp.concatenate(all_inputs, axis=0)

    # Encode
    outputs = encoder_instance.apply(encoder_params, None, all_inputs)
    sentence_embeddings = jnp.mean(outputs, axis=1)

    return jsonify({
        'embeddings': sentence_embeddings.tolist(),
        'labels': labels,
        'input_ids': all_inputs.tolist()
    })


# WebSocket events for real-time streaming
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('status', {'message': 'Connected to encoder server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('start_streaming')
def handle_start_streaming(data):
    """Start streaming embeddings."""
    n_steps = data.get('n_steps', 100)
    interval = data.get('interval', 0.1)

    def stream_embeddings():
        embed_dim = encoder_config['embed_dim'] if encoder_config else 128

        for step in range(n_steps):
            # Generate synthetic embedding
            t = step / 50.0
            embedding = np.random.randn(embed_dim) * 0.5
            embedding[0] = np.sin(t * 2 * np.pi) * 2
            embedding[1] = np.cos(t * 2 * np.pi) * 2
            embedding[2] = np.sin(t * 4 * np.pi) * 1.5

            socketio.emit('embedding_update', {
                'step': step,
                'embedding': embedding.tolist(),
                'timestamp': time.time()
            })

            time.sleep(interval)

        socketio.emit('streaming_complete', {'total_steps': n_steps})

    thread = Thread(target=stream_embeddings)
    thread.daemon = True
    thread.start()

    emit('streaming_started', {'n_steps': n_steps})


@socketio.on('stop_streaming')
def handle_stop_streaming():
    """Stop streaming embeddings."""
    emit('streaming_stopped', {'message': 'Streaming stopped'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Transformer Encoder Visualization Server")
    print("="*60)
    print("\nInitializing encoder...")
    initialize_encoder()
    print("✓ Encoder initialized")
    print(f"\n📊 Server running at: http://localhost:5000")
    print("   Open this URL in your browser to access the visualization\n")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
