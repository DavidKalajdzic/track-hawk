#!/bin/bash
#SBATCH --job-name=gr00t_finetune
#SBATCH --ntasks=1                     
#SBATCH --nodes=1                      
#SBATCH --gres=gpu:2                   
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --account=master
#SBATCH --partition=test       # if you want to run on the A100s, but there are only 2GPUs available


# Load the necessary modules and activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

module load gcc cuda

export CUDA_HOME=$HOME/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH



python "Isaac-GR00T/scripts/gr00t_finetune.py" \
  --dataset-path "data_track_hawk/dataset_drone_control/" \
  --output-dir ./checkpoints \
  --data-config track_hawk \
  --batch-size 8 \
  --max-steps 5000 \
  --num-gpus 2 \
  --save-steps 1000 \
  --base-model-path nvidia/GR00T-N1-2B \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --lora-rank 32 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --dataloader-num-workers 16 \
  --report-to tensorboard \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --no-tf32 \
  --attn_implementation eager # or run with flash_attention_2 which is faster
