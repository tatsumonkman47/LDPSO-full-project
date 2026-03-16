# LDPSO — Learned Diffusion Prior for SWOT Observations

Portable project for running SWOT-SST inpainting experiments across HPC systems.

## Repository Structure

```
├── SWOT-LDPSO/          # Main project: priors package + experiment scripts
│   ├── priors/          # Core diffusion prior library
│   └── experiments/     # Experiment configs and training scripts
│       └── swot-sst-inpainting_V2/  # SWOT-SST inpainting experiment
├── dawgz_with_python_singularity/   # Custom dawgz scheduler (HPC job orchestration)
├── inox_local/          # Custom inox neural network library (JAX-based)
├── jax_environment.yml  # Conda environment specification
├── sub_JAX_job_base.sh  # Base HPC job submission script
└── submission_sh_scripts/  # Additional submission scripts
```

## Setup

1. **Create the conda environment:**
   ```bash
   conda env create -f jax_environment.yml
   ```

2. **Install local packages (in editable mode):**
   ```bash
   pip install -e dawgz_with_python_singularity/
   pip install -e inox_local/
   pip install -e SWOT-LDPSO/
   ```

## Notes

- Run outputs (`runs/`, `outputs/`, `wandb/`) are excluded from version control via `.gitignore`.
- Model checkpoints and data files (`*.pt`, `*.npy`, `*.pkl`, etc.) are also excluded.
- Each HPC system may need adjustments to the submission scripts.
