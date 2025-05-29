source /mloscratch/users/kalajdzi/.bashrc
conda env list > envsyolo.txt
conda activate gr00t

sudo apt-get update
sudo apt-get install --fix-missing -y libgl1

python /mloscratch/users/kalajdzi/track-hawk/Isaac-GR00T/scripts/gr00t_finetune.py \
  --dataset-path /mloscratch/users/kalajdzi/track-hawk/data_track_hawk/dataset_drone_control/ \
  --output-dir /mloscratch/users/kalajdzi/track-hawk/checkpoints \
  --data-config track_hawk \
  --batch-size 48 \
  --max-steps 5000 \
  --num-gpus 4 \
  --save-steps 1000 \
  --base-model-path nvidia/GR00T-N1-2B \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --lora-rank 0 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --dataloader-num-workers 32 \
  --report-to wandb \
  --embodiment-tag new_embodiment \
  --video-backend decord

mkdir "camarche$(date +%Y-%m-%d_%H-%M-%S)"
