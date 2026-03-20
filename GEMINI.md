# minGPT Project Context

minGPT is a minimal, clean, and educational PyTorch re-implementation of the GPT (Generative Pre-trained Transformer) architecture. It is designed to be small and interpretable while supporting training, inference, and loading of pre-trained GPT-2 weights.

## Project Overview

- **Core Model:** A decoder-only Transformer following the GPT-2/GPT-3 design.
- **Key Files:**
  - `mingpt/model.py`: Defines the `GPT` model, `Block`, and `CausalSelfAttention` layers.
  - `mingpt/trainer.py`: A generic training loop with support for callbacks and multiple devices (CUDA, MPS, CPU).
  - `mingpt/bpe.py`: Tokenizer implementation using Byte Pair Encoding.
  - `mingpt/utils.py`: Configuration management (`CfgNode`) and utility functions.
- **Projects:** Examples located in `projects/` (e.g., `chargpt`, `adder`) demonstrate how to use the library for different tasks.

## Building and Running

### Installation
The project can be installed in editable mode:
```bash
pip install -e .
```

### Training
To run an example project (e.g., character-level GPT):
```bash
# Ensure input.txt exists in the project directory
python projects/chargpt/chargpt.py
```

### Testing
Unit tests are located in the `tests/` directory:
```bash
python -m unittest discover tests
```

### Notebooks
- `demo.ipynb`: Basic usage of `GPT` and `Trainer` for a sorting task.
- `generate.ipynb`: Loading pre-trained GPT-2 weights and generating text.

## Development Conventions

### Configuration Management
The project uses a custom `CfgNode` class (defined in `mingpt/utils.py`) for hierarchical configuration.
- Classes like `GPT` and `Trainer` provide a `get_default_config()` static method.
- Configurations can be overridden via command-line arguments (e.g., `--model.n_layer=6`).

### Weight Initialization
- Follows GPT-2/GPT-3 standards: `Normal(0.0, 0.02)` for weights, zeros for biases.
- Residual projections (`c_proj`) are scaled by `1/sqrt(2 * n_layer)` to account for accumulation on the residual path.

### Optimizer Setup
- Uses `AdamW` with weight decay separation.
- **Decay:** Applied only to 2D weights (e.g., `nn.Linear`).
- **No Decay:** Applied to 1D parameters (biases, `nn.LayerNorm`, `nn.Embedding`).

### Device Support
The `Trainer` automatically selects the best available device:
1. `cuda` (NVIDIA GPUs)
2. `mps` (Apple Silicon)
3. `cpu` (Fallback)

## Architectural Insights

- **Self-Attention:** Implements `CausalSelfAttention` with a triangular mask to ensure causality.
- **Modular Blocks:** The `Block` class encapsulates LayerNorm, Self-Attention, and MLP sub-blocks.
- **Hugging Face Integration:** The `GPT.from_pretrained(model_type)` method allows loading weights directly from Hugging Face's `transformers` library.
