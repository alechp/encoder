# 🎨 Transformer Encoder Web Visualizer

Modern, interactive web-based visualization for transformer embeddings.

## Features

### 🚀 Real-time Visualization
- **2D/3D PCA** - Principal component analysis with variance explained
- **2D/3D t-SNE** - t-Distributed stochastic neighbor embedding
- **Positional Encoding** - Interactive heatmap of sinusoidal encodings
- **Real-time Streaming** - Live embedding updates via WebSocket

### 💡 Interactive Controls
- Adjust number of samples and categories
- Switch between visualization modes
- Real-time configuration display
- Live statistics and variance tracking

### 🎯 Professional UI
- Modern dark theme with gradient backgrounds
- Smooth animations and transitions
- Responsive design (works on mobile)
- Interactive Plotly.js visualizations

## Quick Start

### 1. Install Dependencies

```bash
# From the main encoder directory
pip install flask flask-cors flask-socketio python-socketio simple-websocket
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
cd viz_app
python app.py
```

Or from the main directory:

```bash
python viz_app/app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

You'll see a modern web interface with:
- Left sidebar with controls
- Large visualization area
- Real-time statistics
- Configuration display

## Usage

### Generate & Visualize

1. **Set Parameters**
   - Adjust "Number of Samples" slider (10-200)
   - Adjust "Categories" slider (2-5)

2. **Generate Samples**
   - Click "Generate Samples"
   - Wait for encoder to process

3. **Choose Visualization**
   - Select mode from dropdown:
     - 2D PCA (fast, shows variance)
     - 3D PCA (interactive 3D plot)
     - 2D t-SNE (better clustering)
     - 3D t-SNE (3D clusters)
     - Positional Encoding (heatmap)
     - Real-time Stream (live updates)

4. **Visualize**
   - Click "Visualize" to render
   - Interact with plot (zoom, pan, rotate)

### Real-time Streaming

1. Select "Real-time Stream" mode
2. Click "Start Stream"
3. Watch embeddings update live
4. Click "Stop Stream" to pause

## Architecture

### Backend (Flask)

```
viz_app/
├── app.py              # Flask server with API endpoints
├── templates/
│   └── index.html      # Main HTML template
└── static/
    ├── css/
    │   └── style.css   # Modern CSS styling
    └── js/
        └── app.js      # JavaScript application
```

### API Endpoints

#### GET `/`
Main visualization interface

#### GET `/api/config`
Get encoder configuration

#### POST `/api/config`
Update encoder configuration

#### POST `/api/encode`
Encode batch of sequences
```json
{
  "input_ids": [[1, 2, 3, ...]]
}
```

#### POST `/api/visualize/pca`
Compute PCA projection
```json
{
  "embeddings": [[...], [...]],
  "n_components": 2
}
```

#### POST `/api/visualize/tsne`
Compute t-SNE projection
```json
{
  "embeddings": [[...], [...]],
  "n_components": 2,
  "perplexity": 30
}
```

#### GET `/api/positional_encoding`
Generate positional encoding matrix
```
?seq_len=128&embed_dim=512
```

#### POST `/api/generate_samples`
Generate sample embeddings
```json
{
  "n_samples": 60,
  "n_categories": 3
}
```

### WebSocket Events

#### Client → Server
- `connect` - Client connected
- `start_streaming` - Start embedding stream
- `stop_streaming` - Stop streaming

#### Server → Client
- `status` - Connection status
- `embedding_update` - New embedding data
- `streaming_started` - Stream initiated
- `streaming_complete` - Stream finished

## Frontend Stack

### Libraries
- **Plotly.js** - Interactive plots (2D/3D scatter, heatmaps)
- **D3.js** - Data visualization utilities
- **Socket.IO** - Real-time WebSocket communication

### Features
- Responsive grid layout
- Dark theme with CSS variables
- Smooth animations
- Loading states
- Error handling

## Development

### Run in Debug Mode

```python
# In app.py, already enabled:
socketio.run(app, debug=True)
```

### Customize Configuration

Edit default config in `app.py`:

```python
config = {
    'vocab_size': 1000,
    'embed_dim': 128,
    'seq_len': 32,
    'batch_size': 50,
    'n_heads': 4,
    'd_k': 32,
    'hidden_dim': 512,
}
```

### Add New Visualization

1. Add API endpoint in `app.py`
2. Add mode to dropdown in `index.html`
3. Implement visualization in `app.js`
4. Add styling in `style.css`

## Performance

### Recommendations

- **Samples**: 10-100 for t-SNE, up to 200 for PCA
- **Streaming**: 0.05s interval works well for visualization
- **Browser**: Chrome/Firefox recommended for best performance

### Optimization

The app automatically:
- Limits streaming buffer to 200 points
- Uses efficient numpy operations
- Caches encoder parameters
- Streams data via WebSocket

## Screenshots

### Main Interface
- Modern dark theme
- Interactive controls
- Real-time statistics

### 2D PCA Visualization
- Colored by category
- Variance explained shown
- Interactive zoom/pan

### 3D Visualizations
- Rotatable 3D plots
- Multi-category clustering
- Beautiful gradients

### Positional Encoding
- Heatmap visualization
- Shows sinusoidal patterns
- Configurable dimensions

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
socketio.run(app, port=5001)
```

### WebSocket Connection Failed

Check CORS settings and firewall. The app uses:
```python
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
```

### Visualization Not Rendering

- Check browser console for errors
- Ensure Plotly.js loaded: check network tab
- Try clearing cache and reloading

### Slow t-SNE

t-SNE is computationally intensive:
- Reduce number of samples
- Use PCA instead for quick preview
- Wait patiently (shows loading spinner)

## Integration

### Use with Your Data

```python
import requests

# Encode your data
response = requests.post('http://localhost:5000/api/encode', json={
    'input_ids': your_token_ids
})

embeddings = response.json()['embeddings']

# Visualize
response = requests.post('http://localhost:5000/api/visualize/pca', json={
    'embeddings': embeddings,
    'n_components': 2
})

coords = response.json()['coordinates']
```

### Custom Encoder

Modify `initialize_encoder()` in `app.py` to use your custom encoder.

## Future Enhancements

Potential additions:
- [ ] Attention heatmaps
- [ ] Layer-wise visualization
- [ ] Export plots as images
- [ ] Save/load sessions
- [ ] Custom color schemes
- [ ] Animation recording
- [ ] Batch upload interface
- [ ] Multi-model comparison

## License

Part of the Transformer Encoder project.

---

**Built with:** Flask, Plotly.js, D3.js, Socket.IO, JAX & Haiku
