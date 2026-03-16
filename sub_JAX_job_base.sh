#!/bin/bash
#PBS -l select=1:ncpus=64:ngpus=4:model=mil_a100:mem=500gb
#PBS -l walltime=24:00:00
#PBS -l place=free:excl
#PBS -q gpu_normal@pbspl4
#PBS -j oe

# Exit on error
set -e
cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

# Initialize modules system FIRST
source /usr/share/modules/init/bash
# OR try: source /usr/local/lib/global.profile

# Load modules and activate env
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
export CONDA_ENVS_PATH=/home3/tmonkman/swbuild3/.conda/envs/
export CONDA_PKGS_DIRS=/home3/tmonkman/swbuild3/.conda/pkgs/
source activate jax_base || { echo "Failed to activate conda env"; exit 1; }

# Set lap number (default to 0, can be overridden)
export LAP_NUMBER=${LAP_NUMBER:-0}
export OMP_NUM_THREADS=64
# Force unbuffered Python output
export PYTHONUNBUFFERED=1  

# Set wandb API key (get it from: wandb login --relogin on login node)
export WANDB_API_KEY="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"
# Force wandb offline mode on compute nodes
export WANDB_MODE=offline

# Run single Python process - JAX will find all 4 GPUs on this node
TARGET_DIR=$PBS_O_WORKDIR/SWOT-LDPSO/experiments/swot-sst-inpainting_V2
cd $TARGET_DIR || { echo "Failed to cd to $TARGET_DIR"; exit 1; }
python -u train_with_hydra_pleiades.py --config-name=0061_cg_1-iter_ddim_sst_5tstep_crho03_warmup_only.yaml
