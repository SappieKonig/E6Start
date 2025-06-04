# ML Competition Framework

A minimal framework for ML competitions with Hydra configs, WandB sweeps, and modular components.

## Structure

- `src/models/` - Model implementations inheriting from BaseModel
- `src/augmentations/` - Data augmentation classes inheriting from BaseAugmentation  
- `src/postprocessing/` - Prediction postprocessing classes inheriting from BasePostprocessing
- `configs/` - Hydra configuration files
- `sweeps/` - WandB sweep configurations
- `data/` - Dataset directory

## Usage

```bash
# Install dependencies
uv sync

# Run training
uv run python train.py

# Run WandB sweep
wandb sweep sweeps/example_sweep.yaml
wandb agent <sweep_id>
```

## Adding Components

1. Create new classes inheriting from base classes
2. Add corresponding config files in `configs/`
3. Update `defaults` in main config to use new components