# 📋 Menu Reference Card

Quick reference for all run script options and menu choices.

---

## 🎯 Quick Commands

### Interactive Menu
```bash
./run.sh          # macOS/Linux - Show menu
python run.py     # Cross-platform - Show menu
```

### Direct Commands
```bash
./run.sh --full          # or: python run.py --full
./run.sh --encoder       # or: python run.py --encoder
./run.sh --viz           # or: python run.py --viz
./run.sh --tests         # or: python run.py --tests
./run.sh --quick         # or: python run.py --quick
./run.sh --interactive   # or: python run.py --interactive
./run.sh --help          # or: python run.py --help
```

---

## 📊 Menu Options

### [1] Run Full Demo
**What it does:**
- ✅ Runs encoder tests (8 tests)
- ✅ Runs visualization tests (9 tests)
- ✅ Runs encoder examples (5 examples)
- ✅ Runs main encoder demo
- ✅ Generates visualizations (6 images)

**Time:** ~2-3 minutes
**Output:** Complete walkthrough + outputs/ directory

**Use when:**
- First time setup
- Want to see everything
- Verifying full functionality

---

### [2] Run Encoder Demo Only
**What it does:**
- ✅ Runs encoder tests (8 tests)
- ✅ Runs encoder examples (5 examples)
- ✅ Runs main encoder demo

**Time:** ~1-2 minutes
**Output:** Test results + example outputs

**Use when:**
- Focused on transformer architecture
- Don't need visualizations
- Quick encoder verification

---

### [3] Run Visualization Demo Only
**What it does:**
- ✅ Runs visualization tests (9 tests)
- ✅ Generates 6 visualizations
  - 2D PCA projection
  - 2D t-SNE projection
  - 3D PCA projection
  - Positional encoding
  - Attention patterns
  - Embedding comparison

**Time:** ~1-2 minutes
**Output:** outputs/ directory with 6 PNG files

**Use when:**
- Testing visualization layer
- Generating example plots
- Focused on embeddings analysis

---

### [4] Run Tests Only
**What it does:**
- ✅ Runs encoder tests (8 tests)
- ✅ Runs visualization tests (9 tests)
- ❌ Skips demos and examples

**Time:** ~30 seconds
**Output:** Test results only

**Use when:**
- CI/CD pipelines
- Quick verification
- Development workflow
- After code changes

---

### [5] Quick Test
**What it does:**
- ✅ Verifies Python installation
- ✅ Checks all imports (JAX, Haiku, matplotlib, etc.)
- ❌ Doesn't run tests or demos

**Time:** ~5 seconds
**Output:** Installation status

**Use when:**
- First time setup check
- Debugging installation issues
- Verifying environment
- Fastest verification

---

### [6] Interactive Visualization Menu (Python)
**What it does:**
- 🎨 Launches `realtime_demo.py`
- Shows submenu with options:
  1. Encoder Embeddings (2D/3D)
  2. Positional Encoding
  3. Attention Patterns
  4. Embedding Comparison
  5. Real-time Streaming
  6. Run All Demos

**Time:** Varies by selection
**Output:** Interactive plots and saved images

**Use when:**
- Prefer Python matplotlib
- Batch processing
- Offline visualization
- Exporting to PNG

---

### [7] 🌐 Start Web Visualizer (Recommended!)
**What it does:**
- 🚀 Starts Flask web server
- 🌐 Opens interactive web interface at `http://localhost:5000`
- 📊 Modern UI with Plotly.js visualizations
- ⚡ Real-time WebSocket streaming
- 📱 Responsive design (works on mobile)

**Time:** Runs until stopped (Ctrl+C)
**Output:** Web application with live visualizations

**Use when:**
- Want best visualization experience
- Need interactive 3D plots
- Presenting to others
- Real-time monitoring
- **This is the recommended way to visualize!**

---

## 🔧 Command-Line Flags

### Bash Script (`./run.sh`)
```bash
--full, -f          # Run full demo
--encoder, -e       # Run encoder demo only
--viz, -v           # Run visualization demo only
--tests, -t         # Run tests only
--quick, -q         # Quick installation test
--interactive, -i   # Interactive visualization menu (Python)
--web, -w           # Start web visualizer (Recommended!)
--help, -h          # Show help
```

### Python Script (`python run.py`)
```bash
--full, -f          # Run full demo
--encoder, -e       # Run encoder demo only
--viz, -v           # Run visualization demo only
--tests, -t         # Run tests only
--quick, -q         # Quick installation test
--interactive, -i   # Interactive visualization menu (Python)
--web, -w           # Start web visualizer (Recommended!)
--help, -h          # Show help
```

---

## 🎓 Recommended Workflow

### First Time Setup
```bash
./run.sh              # Show menu, select [5] Quick Test
# If successful:
./run.sh --full       # Run complete demo
```

### Daily Development
```bash
./run.sh --tests      # Run tests after changes
./run.sh --encoder    # Verify encoder works
./run.sh --viz        # Generate visualizations
```

### Demonstration / Presentation
```bash
./run.sh --viz        # Generate all plots
# Or:
./run.sh --interactive # Interactive exploration
```

### CI/CD Pipeline
```bash
./run.sh --tests      # Just run tests
# Or:
./run.sh --quick      # Fastest verification
```

---

## 📁 Output Locations

### Test Results
- Console output only
- ✅ or ✗ for each test

### Encoder Examples
- Console output with shapes and statistics

### Visualization Outputs
```
outputs/
├── encoder_2d_pca.png           (261 KB)
├── encoder_2d_tsne.png          (240 KB)
├── encoder_3d_pca.png           (768 KB)
├── positional_encoding.png      (869 KB)
├── attention_pattern.png        (211 KB)
└── embedding_comparison.png     (328 KB)
```

---

## 💡 Tips

### Speed Up Runs
- Use `--quick` for fastest check
- Use `--tests` to skip demos
- Run specific components only

### Skip Setup
If environment is already set up:
```bash
source venv/bin/activate         # Activate env
python test_encoder.py           # Run directly
python example_with_visualization.py  # Run directly
```

### Multiple Runs
The run script is idempotent:
- Safe to run multiple times
- Reuses existing venv
- Updates dependencies if needed

### Debugging
```bash
./run.sh --quick      # Check installation
./run.sh --tests      # Verify functionality
./run.sh --encoder    # Test encoder only
./run.sh --viz        # Test visualization only
```

---

## 🚀 Quick Reference Table

| Option | Time | Tests | Examples | Visualizations | Output Files |
|--------|------|-------|----------|----------------|--------------|
| Full Demo | 2-3 min | ✅ Both | ✅ Yes | ✅ Yes | 6 images |
| Encoder Only | 1-2 min | ✅ Encoder | ✅ Yes | ❌ No | None |
| Viz Only | 1-2 min | ✅ Viz | ❌ No | ✅ Yes | 6 images |
| Tests Only | 30 sec | ✅ Both | ❌ No | ❌ No | None |
| Quick Test | 5 sec | ❌ No | ❌ No | ❌ No | None |
| Interactive (Python) | Varies | ❌ No | ❌ No | 🎨 Custom | Varies |
| **Web Visualizer** ⭐ | **Runs until stopped** | **❌ No** | **❌ No** | **🌐 Live Web** | **None** |

---

## 🎯 Common Use Cases

**"I just cloned the repo"**
```bash
./run.sh              # Menu → [5] Quick Test → [1] Full Demo
```

**"I made code changes"**
```bash
./run.sh --tests      # Quick verification
```

**"I need visualizations for a presentation"**
```bash
./run.sh --viz        # Generates all plots
```

**"I want to explore interactively"**
```bash
./run.sh --interactive # Launch viz menu
```

**"CI/CD check"**
```bash
./run.sh --tests      # Or --quick for faster
```

**"Show me just the encoder"**
```bash
./run.sh --encoder    # Skip visualization
```

---

## ℹ️ Help

For more information:
- `./run.sh --help` - Command-line help
- `README.md` - Complete documentation
- `QUICKSTART.md` - Setup guide
- `VISUALIZATION.md` - Visualization API
- `PROJECT_SUMMARY.md` - Full project overview

---

*Last Updated: 2026-02-24*
