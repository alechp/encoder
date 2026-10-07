// Transformer Encoder Visualizer - Main Application
// Handles all UI interactions, API calls, and visualizations

class EncoderVisualizer {
    constructor() {
        this.socket = null;
        this.currentData = null;
        this.config = null;
        this.streamingActive = false;
        this.streamData = [];

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadConfig();
    }

    setupEventListeners() {
        // Mode change
        document.getElementById('viz-mode').addEventListener('change', (e) => {
            this.handleModeChange(e.target.value);
        });

        // Range sliders
        document.getElementById('n-samples').addEventListener('input', (e) => {
            document.getElementById('n-samples-value').textContent = e.target.value;
        });

        document.getElementById('n-categories').addEventListener('input', (e) => {
            document.getElementById('n-categories-value').textContent = e.target.value;
        });

        // Buttons
        document.getElementById('generate-btn').addEventListener('click', () => this.generateSamples());
        document.getElementById('visualize-btn').addEventListener('click', () => this.visualize());
        document.getElementById('clear-btn').addEventListener('click', () => this.clear());

        // Real-time controls
        document.getElementById('start-stream-btn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stop-stream-btn').addEventListener('click', () => this.stopStreaming());
    }

    setupWebSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            this.updateStatus('Connected to server');
        });

        this.socket.on('disconnect', () => {
            this.updateStatus('Disconnected from server');
        });

        this.socket.on('embedding_update', (data) => {
            this.handleEmbeddingUpdate(data);
        });

        this.socket.on('streaming_complete', () => {
            this.streamingActive = false;
            document.getElementById('start-stream-btn').disabled = false;
            document.getElementById('stop-stream-btn').disabled = true;
            document.getElementById('stream-status').textContent = 'Stream Complete';
            this.updateStatus('Streaming complete');
        });
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            this.displayConfig(this.config);
        } catch (error) {
            console.error('Error loading config:', error);
            this.updateStatus('Error loading configuration');
        }
    }

    displayConfig(config) {
        document.getElementById('config-vocab').textContent = config.vocab_size.toLocaleString();
        document.getElementById('config-embed').textContent = config.embed_dim;
        document.getElementById('config-heads').textContent = config.n_heads;
        document.getElementById('config-seqlen').textContent = config.seq_len;
    }

    handleModeChange(mode) {
        const realtimeControls = document.getElementById('realtime-controls');

        if (mode === 'realtime') {
            realtimeControls.style.display = 'block';
        } else {
            realtimeControls.style.display = 'none';
            if (this.streamingActive) {
                this.stopStreaming();
            }
        }

        if (mode === 'positional') {
            this.visualizePositionalEncoding();
        }
    }

    async generateSamples() {
        this.showLoading(true);
        this.updateStatus('Generating samples...');

        const nSamples = parseInt(document.getElementById('n-samples').value);
        const nCategories = parseInt(document.getElementById('n-categories').value);

        try {
            const response = await fetch('/api/generate_samples', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n_samples: nSamples, n_categories: nCategories })
            });

            this.currentData = await response.json();
            this.updateStatistics(this.currentData);
            this.updateStatus('Samples generated successfully');

            // Auto-visualize
            await this.visualize();
        } catch (error) {
            console.error('Error generating samples:', error);
            this.updateStatus('Error generating samples');
        } finally {
            this.showLoading(false);
        }
    }

    async visualize() {
        if (!this.currentData) {
            this.updateStatus('No data to visualize. Generate samples first.');
            return;
        }

        const mode = document.getElementById('viz-mode').value;
        this.showLoading(true);

        try {
            switch (mode) {
                case '2d-pca':
                    await this.visualize2DPCA();
                    break;
                case '3d-pca':
                    await this.visualize3DPCA();
                    break;
                case '2d-tsne':
                    await this.visualize2DTSNE();
                    break;
                case '3d-tsne':
                    await this.visualize3DTSNE();
                    break;
                case 'positional':
                    await this.visualizePositionalEncoding();
                    break;
                default:
                    this.updateStatus('Unknown visualization mode');
            }
        } catch (error) {
            console.error('Visualization error:', error);
            this.updateStatus('Error creating visualization');
        } finally {
            this.showLoading(false);
        }
    }

    async visualize2DPCA() {
        this.updateStatus('Computing PCA projection...');

        const response = await fetch('/api/visualize/pca', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings: this.currentData.embeddings,
                n_components: 2
            })
        });

        const result = await response.json();
        const coords = result.coordinates;

        // Create color array for categories
        const labels = this.currentData.labels;
        const uniqueLabels = [...new Set(labels)];
        const colors = labels.map(label => uniqueLabels.indexOf(label));

        const trace = {
            x: coords.map(c => c[0]),
            y: coords.map(c => c[1]),
            mode: 'markers',
            type: 'scatter',
            text: labels,
            marker: {
                size: 10,
                color: colors,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Category'
                }
            }
        };

        const layout = {
            title: `2D PCA Projection<br><sub>Variance Explained: ${(result.total_variance * 100).toFixed(1)}%</sub>`,
            xaxis: { title: 'PC1', gridcolor: '#2c3e50' },
            yaxis: { title: 'PC2', gridcolor: '#2c3e50' },
            hovermode: 'closest',
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.newPlot('plot', [trace], layout, { responsive: true });

        document.getElementById('stat-variance').textContent = `${(result.total_variance * 100).toFixed(1)}%`;
        this.updateStatus('2D PCA visualization complete');
        this.updateLastUpdate();
    }

    async visualize3DPCA() {
        this.updateStatus('Computing 3D PCA projection...');

        const response = await fetch('/api/visualize/pca', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings: this.currentData.embeddings,
                n_components: 3
            })
        });

        const result = await response.json();
        const coords = result.coordinates;

        const labels = this.currentData.labels;
        const uniqueLabels = [...new Set(labels)];
        const colors = labels.map(label => uniqueLabels.indexOf(label));

        const trace = {
            x: coords.map(c => c[0]),
            y: coords.map(c => c[1]),
            z: coords.map(c => c[2]),
            mode: 'markers',
            type: 'scatter3d',
            text: labels,
            marker: {
                size: 5,
                color: colors,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Category'
                }
            }
        };

        const layout = {
            title: `3D PCA Projection<br><sub>Variance Explained: ${(result.total_variance * 100).toFixed(1)}%</sub>`,
            scene: {
                xaxis: { title: 'PC1', gridcolor: '#2c3e50' },
                yaxis: { title: 'PC2', gridcolor: '#2c3e50' },
                zaxis: { title: 'PC3', gridcolor: '#2c3e50' },
                bgcolor: '#0f3460'
            },
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.newPlot('plot', [trace], layout, { responsive: true });

        document.getElementById('stat-variance').textContent = `${(result.total_variance * 100).toFixed(1)}%`;
        this.updateStatus('3D PCA visualization complete');
        this.updateLastUpdate();
    }

    async visualize2DTSNE() {
        this.updateStatus('Computing t-SNE projection (this may take a moment)...');

        const response = await fetch('/api/visualize/tsne', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings: this.currentData.embeddings,
                n_components: 2
            })
        });

        const result = await response.json();
        const coords = result.coordinates;

        const labels = this.currentData.labels;
        const uniqueLabels = [...new Set(labels)];
        const colors = labels.map(label => uniqueLabels.indexOf(label));

        const trace = {
            x: coords.map(c => c[0]),
            y: coords.map(c => c[1]),
            mode: 'markers',
            type: 'scatter',
            text: labels,
            marker: {
                size: 10,
                color: colors,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Category'
                }
            }
        };

        const layout = {
            title: '2D t-SNE Projection',
            xaxis: { title: 't-SNE 1', gridcolor: '#2c3e50' },
            yaxis: { title: 't-SNE 2', gridcolor: '#2c3e50' },
            hovermode: 'closest',
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.newPlot('plot', [trace], layout, { responsive: true });

        this.updateStatus('2D t-SNE visualization complete');
        this.updateLastUpdate();
    }

    async visualize3DTSNE() {
        this.updateStatus('Computing 3D t-SNE projection (this may take a moment)...');

        const response = await fetch('/api/visualize/tsne', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embeddings: this.currentData.embeddings,
                n_components: 3
            })
        });

        const result = await response.json();
        const coords = result.coordinates;

        const labels = this.currentData.labels;
        const uniqueLabels = [...new Set(labels)];
        const colors = labels.map(label => uniqueLabels.indexOf(label));

        const trace = {
            x: coords.map(c => c[0]),
            y: coords.map(c => c[1]),
            z: coords.map(c => c[2]),
            mode: 'markers',
            type: 'scatter3d',
            text: labels,
            marker: {
                size: 5,
                color: colors,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Category'
                }
            }
        };

        const layout = {
            title: '3D t-SNE Projection',
            scene: {
                xaxis: { title: 't-SNE 1', gridcolor: '#2c3e50' },
                yaxis: { title: 't-SNE 2', gridcolor: '#2c3e50' },
                zaxis: { title: 't-SNE 3', gridcolor: '#2c3e50' },
                bgcolor: '#0f3460'
            },
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.newPlot('plot', [trace], layout, { responsive: true });

        this.updateStatus('3D t-SNE visualization complete');
        this.updateLastUpdate();
    }

    async visualizePositionalEncoding() {
        this.updateStatus('Generating positional encoding...');

        const seqLen = this.config?.seq_len || 128;
        const embedDim = this.config?.embed_dim || 512;

        const response = await fetch(`/api/positional_encoding?seq_len=${seqLen}&embed_dim=${embedDim}`);
        const result = await response.json();
        const encoding = result.encoding;

        const trace = {
            z: encoding,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true
        };

        const layout = {
            title: `Positional Encoding<br><sub>Sequence Length: ${seqLen}, Embedding Dim: ${embedDim}</sub>`,
            xaxis: { title: 'Embedding Dimension', gridcolor: '#2c3e50' },
            yaxis: { title: 'Position', gridcolor: '#2c3e50' },
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.newPlot('plot', [trace], layout, { responsive: true });

        this.updateStatus('Positional encoding visualization complete');
        this.updateLastUpdate();
    }

    startStreaming() {
        this.streamData = [];
        this.streamingActive = true;

        document.getElementById('start-stream-btn').disabled = true;
        document.getElementById('stop-stream-btn').disabled = false;
        document.getElementById('stream-status').textContent = 'Streaming...';

        this.socket.emit('start_streaming', {
            n_steps: 200,
            interval: 0.05
        });

        this.updateStatus('Real-time streaming started');
    }

    stopStreaming() {
        this.streamingActive = false;

        document.getElementById('start-stream-btn').disabled = false;
        document.getElementById('stop-stream-btn').disabled = true;
        document.getElementById('stream-status').textContent = 'Stopped';

        this.socket.emit('stop_streaming');
        this.updateStatus('Streaming stopped');
    }

    handleEmbeddingUpdate(data) {
        this.streamData.push({
            step: data.step,
            embedding: data.embedding,
            timestamp: data.timestamp
        });

        // Keep only recent data
        if (this.streamData.length > 200) {
            this.streamData.shift();
        }

        // Update visualization
        this.updateStreamVisualization();
    }

    updateStreamVisualization() {
        if (this.streamData.length < 2) return;

        // Extract first 2 dimensions for 2D plot
        const trace = {
            x: this.streamData.map(d => d.embedding[0]),
            y: this.streamData.map(d => d.embedding[1]),
            mode: 'markers+lines',
            type: 'scatter',
            marker: {
                size: 6,
                color: this.streamData.map((_, i) => i),
                colorscale: 'Viridis',
                showscale: true
            },
            line: {
                color: 'rgba(74, 144, 226, 0.3)',
                width: 1
            }
        };

        const layout = {
            title: `Real-time Embedding Stream<br><sub>Points: ${this.streamData.length}</sub>`,
            xaxis: { title: 'Dimension 1', gridcolor: '#2c3e50' },
            yaxis: { title: 'Dimension 2', gridcolor: '#2c3e50' },
            plot_bgcolor: '#0f3460',
            paper_bgcolor: '#0f3460',
            font: { color: '#eaeaea' }
        };

        Plotly.react('plot', [trace], layout, { responsive: true });
    }

    clear() {
        Plotly.purge('plot');
        this.currentData = null;
        this.streamData = [];

        document.getElementById('stat-samples').textContent = '0';
        document.getElementById('stat-dims').textContent = '0';
        document.getElementById('stat-variance').textContent = '-';

        this.updateStatus('Cleared');
    }

    updateStatistics(data) {
        if (data.embeddings) {
            document.getElementById('stat-samples').textContent = data.embeddings.length;
            document.getElementById('stat-dims').textContent = data.embeddings[0]?.length || 0;
        }
    }

    updateStatus(message) {
        document.getElementById('status-text').textContent = message;
    }

    updateLastUpdate() {
        const now = new Date().toLocaleTimeString();
        document.getElementById('last-update').textContent = now;
    }

    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'flex' : 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.visualizer = new EncoderVisualizer();
});
