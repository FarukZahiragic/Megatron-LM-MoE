#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:15:00
#SBATCH --job-name=test-moe-head-ep2
#SBATCH --output=/iopsstor/scratch/cscs/%u/test-moe-head-ep2-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/test-moe-head-ep2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --reservation=SD-69241-apertus-1-5-0

CONTAINER_ENV=/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/apertus/Megatron-LM-MoE

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export WORLD_SIZE=$SLURM_NPROCS   # = 4

echo "START TIME: $(date)"
echo "MASTER_ADDR=$MASTER_ADDR  WORLD_SIZE=$WORLD_SIZE"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID WORLD_SIZE=$WORLD_SIZE \
    MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    PYTHONPATH=$MEGATRON_LM_DIR \
    python -m pytest --override-ini='addopts=-v -s' $MEGATRON_LM_DIR/multimodal_moe_head/tests/test_ep2.py
    "

echo "END TIME: $(date)"
