# 🚀 Quick Start Guide

Get up and running with the Transformer Encoder in 2 minutes!

## Option 1: Automated Setup (Recommended)

### Interactive Menu

Simply run:

```bash
./run.sh          # macOS/Linux
python run.py     # Windows or any platform
```

You'll see an interactive menu:

```
╔════════════════════════════════════════════════════════════╗
║          Transformer Encoder from Scratch                 ║
║          JAX + Haiku + Visualization Layer                ║
╚════════════════════════════════════════════════════════════╝

Demo Menu
════════════════════════════════════════════════════════════

  [1] Run Full Demo (Tests + Encoder + Visualization)
  [2] Run Encoder Demo Only (Tests + Examples)
  [3] Run Visualization Demo Only
  [4] Run Tests Only (Encoder + Visualization)
  [5] Quick Test (Skip demos, just verify installation)
  [6] Interactive Visualization Menu

  [q] Quit
```

### Command-Line Options

Skip the menu with direct commands:

```bash
# Full demo (everything)
./run.sh --full

# Just encoder
./run.sh --encoder

# Just visualization
./run.sh --viz

# Tests only
./run.sh --tests

# Quick install check
./run.sh --quick

# Interactive viz menu
./run.sh --interactive
```

The run script automatically:
- ✓ Checks Python installation
- ✓ Creates virtual environment
- ✓ Installs all dependencies
- ✓ Runs selected demos

## Option 2: Manual Setup

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Examples
```bash
# Run tests
python test_encoder.py

# Run examples
python example.py

# Run main demo
python transformer_encoder.py
```

## What Gets Installed?

- **JAX** - High-performance numerical computing
- **JAXlib** - JAX backend library
- **dm-haiku** - Neural network library for JAX
- **optax** - Gradient processing and optimization (optional)
- **numpy** - Numerical arrays
- **matplotlib** - Plotting (optional)

## Quick Test

After setup, verify everything works:

```python
import jax
import jax.numpy as jnp
import haiku as hk
from transformer_encoder import encoder_fn

# Quick configuration
config = {
    'vocab_size': 1000,
    'embed_dim': 128,
    'seq_len': 32,
    'batch_size': 4,
    'n_heads': 4,
    'd_k': 32,
    'hidden_dim': 512,
}

# Create encoder
encoder = hk.transform(lambda x: encoder_fn(x, config))

# Initialize
rng = jax.random.PRNGKey(42)
params = encoder.init(rng, jnp.zeros((4, 32), dtype=jnp.int32))

# Test forward pass
sample = jax.random.randint(jax.random.PRNGKey(0), (4, 32), 0, 1000)
output = encoder.apply(params, None, sample)

print(f"✓ Success! Output shape: {output.shape}")
# Expected: ✓ Success! Output shape: (4, 32, 128)
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'jax'`
**Solution:** Make sure you activated the virtual environment
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Issue: `ImportError: cannot import name 'tree_map'`
**Solution:** Update the code (this is fixed in the current version)

### Issue: Permission denied when running `./run.sh`
**Solution:** Make the script executable
```bash
chmod +x run.sh
```

### Issue: Python version too old
**Solution:** Install Python 3.8 or higher
```bash
python3 --version  # Check your version
```

## Next Steps

After successful setup:

1. **Explore the Examples** - See `example.py` for 5 comprehensive examples
2. **Read the Docs** - Check `README.md` for detailed documentation
3. **Modify the Code** - Experiment with different configurations
4. **Train a Model** - Extend with a training loop using Optax

## File Overview

```
encoder/
├── run.sh               # Bash run script (macOS/Linux)
├── run.py               # Python run script (cross-platform)
├── transformer_encoder.py  # Main implementation
├── example.py           # 5 comprehensive examples
├── test_encoder.py      # Test suite (8 tests)
├── requirements.txt     # Python dependencies
├── README.md            # Full documentation
└── QUICKSTART.md        # This file
```

## Help

If you encounter any issues:

1. Check this guide for common issues
2. Read the full `README.md`
3. Make sure all dependencies are installed
4. Verify Python version is 3.8+

Happy learning! 🎓
