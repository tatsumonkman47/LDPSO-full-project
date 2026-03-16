#!/bin/bash
#PBS -l select=2:ncpus=64:ngpus=4:model=mil_a100:mem=500gb
#PBS -l walltime=24:00:00
#PBS -l place=free:excl
#PBS -q gpu_normal@pbspl4
#PBS -j oe

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

# Load modules and activate env
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
export CONDA_ENVS_PATH=/home3/tmonkman/swbuild3/.conda/envs/
export CONDA_PKGS_DIRS=/home3/tmonkman/swbuild3/.conda/pkgs/
source activate jax_base

# Set lap number (default to 0, can be overridden)
export LAP_NUMBER=${LAP_NUMBER:-0}

# Gather node info
NODES=($(cat $PBS_NODEFILE | uniq))
NUM_NODES=${#NODES[@]}
NUM_GPUS_PER_NODE=4
WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_NODE))
MASTER_ADDR=${NODES[0]}
JOBNUM=$(echo $PBS_JOBID | sed 's/[^0-9].*//')
MASTER_PORT=$((12000 + ($JOBNUM % 10000)))

echo "Nodes: ${NODES[@]}"
echo "Master Addr: $MASTER_ADDR"
echo "World Size: $WORLD_SIZE"
echo "Lap Number: $LAP_NUMBER"

C=0
for node in "${NODES[@]}"; do
    if [ "$node" == "$(hostname)" ]; then
        echo "Running locally on $node (rank $C)"
        cd $PBS_O_WORKDIR
        source /usr/local/lib/global.profile
        module purge
        module use -a /swbuild/analytix/tools/modulefiles
        module load miniconda3/v4
        export CONDA_ENVS_PATH=/home3/tmonkman/swbuild3/.conda/envs/
        export CONDA_PKGS_DIRS=/home3/tmonkman/swbuild3/.conda/pkgs/
        eval "$(/swbuild/analytix/tools/miniconda3_220407/bin/conda shell.bash hook)"
        conda activate jax_base
        which python
        export OMP_NUM_THREADS=64
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export WORLD_SIZE=$WORLD_SIZE
        export RANK=$C
        export LAP_NUMBER=$LAP_NUMBER
        cd /nobackup/tmonkman/LDPSO/SWOT-LDPSO/experiments/swot-sst-inpainting_V2
        python train_single_lap.py \
            --config-name=001_cg_1-iter_test.yaml &
    else
        echo "SSHing into $node (rank $C)"
        ssh $node "
            cd $PBS_O_WORKDIR
            source /usr/local/lib/global.profile
            module purge
            module use -a /swbuild/analytix/tools/modulefiles
            module load miniconda3/v4
            export CONDA_ENVS_PATH=/home3/tmonkman/swbuild3/.conda/envs/
            export CONDA_PKGS_DIRS=/home3/tmonkman/swbuild3/.conda/pkgs/
            eval "$(/swbuild/analytix/tools/miniconda3_220407/bin/conda shell.bash hook)"
            conda activate jax_base
            which python
            export OMP_NUM_THREADS=64
            export MASTER_ADDR=$MASTER_ADDR
            export MASTER_PORT=$MASTER_PORT
            export WORLD_SIZE=$WORLD_SIZE
            export RANK=$C
            export LAP_NUMBER=$LAP_NUMBER
            cd /nobackup/tmonkman/LDPSO/SWOT-LDPSO/experiments/swot-sst-inpainting_V2
            python train_single_lap.py \
                --config-name=001_cg_1-iter_test.yaml
        " &
    fi
    C=$((C + 1))
    sleep 2
done

wait