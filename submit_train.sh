#!/bin/bash

#=========================================================================================
# SLURM SBATCH SCRIPT FOR PRE-COMPUTING LATENTS WITH ACCELERATE
#=========================================================================================
# This script launches a multi-CPU job using PyTorch Accelerate to pre-compute
# latents and text embeddings for the ImageNet dataset.

#--------------------
# JOB CONFIGURATION
#--------------------

#SBATCH --job-name=imagenet_sit    # A descriptive name for your job
#SBATCH --partition=gpu-single   # IMPORTANT: Replace with the actual GPU partition name
#SBATCH --gres=gpu:H200:4             # Request 4 H200 GPUs
#SBATCH --ntasks=1                    # We launch all processes from a single task
#SBATCH --cpus-per-task=24           # Request 24 CPU cores for the task (6 per GPU process)
#SBATCH --mem=250G                    # Request 250GB of system RAM
#SBATCH --time=60:00:00               # Maximum walltime for the job (e.g., 8 hours)

#--------------------
# LOGGING
#--------------------

#SBATCH --output=output/train_%j.log   # Standard output log file (%j is the job ID)
#SBATCH --error=output/train_%j.err    # Standard error log file

#--------------------
# JOB EXECUTION
#--------------------

echo "========================================================="
echo "Starting job execution"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================="

# --- 1. SETUP THE ENVIRONMENT ---
# Load the required miniconda module.
echo "Loading miniconda module..."
module load devel/miniconda/3

# Activate the target conda environment.
# sbatch runs in a non-interactive shell, so you must initialize conda first.
echo "Activating conda environment: flux"
# source $(conda info --base)/etc/profile.d/conda.sh
source activate flux

# --- 2. RUN THE PRE-COMPUTATION SCRIPT ---
# We use `srun` to ensure SLURM's process and resource tracking works correctly.
# `accelerate` will handle launching the 4 processes across the 4 allocated GPUs.
echo "Launching accelerate script..."
# srun bash micro_diffusion/datasets/scripts/get_diffdb_dataset.sh ./datadir/diffdb small 4
# srun bash micro_diffusion/datasets/scripts/get_cc12m_dataset.sh ./datadir/cc12m small 4
# srun bash micro_diffusion/datasets/scripts/get_textcaps_dataset.sh ./datadir/textcaps 4
# srun bash micro_diffusion/datasets/scripts/get_coco_dataset.sh ./datadir/coco2014 4
# srun bash micro_diffusion/datasets/scripts/get_sa1b_dataset.sh ./datadir/sa1b small 4

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs allocated: $NUM_GPUS"

# NUM_WORKERS=$(($NUM_GPUS * 6))
# echo "Number of CPUs allocated: $NUM_WORKERS"

# Set the effective batch size and calculate per-GPU batch size
BATCH_SIZE_PER_GPU=128 
BATCH_SIZE=$(($BATCH_SIZE_PER_GPU * $NUM_GPUS))

srun accelerate launch --num_processes $NUM_GPUS --config_file accelerate_config.yaml train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --epochs=200 \
  --batch-size=$BATCH_SIZE \
  --learning-rate=1e-4 \
  --mixed-precision="bf16" \
  --max-train-steps=400000 \
  --seed=25 \
  --path-type="linear" \
  --prediction="v" \
  --num-workers=6 \
  --weighting="uniform" \
  --model="SiT-L/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="/export/data/jmakadiy/meanflow/" \
  --exp-name="sit-B-Meanflow" \
  --data-dir=/export/data/jmakadiy/datasets/imageNet/ \

echo "========================================================="
echo "Job finished with exit code $?"
echo "========================================================="