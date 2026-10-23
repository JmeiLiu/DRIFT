#!/usr/bin/env bash
set -euo pipefail

conda_executable="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_executable" ]]; then
    echo "Conda not found. Initialize Conda before running this script." >&2
    exit 1
fi
source "$("$conda_executable" info --base)/etc/profile.d/conda.sh"
conda activate drift

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export DRIFT_FLOW_MODEL="${DRIFT_FLOW_MODEL:-$HOME/models/stable-diffusion-3.5-medium}"
export DRIFT_SD_MODEL="${DRIFT_SD_MODEL:-$HOME/models/stable-diffusion-v1-5}"
export DRIFT_CLIP_MODEL="${DRIFT_CLIP_MODEL:-$HOME/models/clip-vit-large-patch14}"
export DRIFT_PICK_MODEL="${DRIFT_PICK_MODEL:-$HOME/models/PickScore_v1}"
export DRIFT_DREAMSIM_MODEL="${DRIFT_DREAMSIM_MODEL:-$HOME/models/dreamsim}"

case "${1:-}" in
  sd35_shaping_hps)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29501 flow/diversity_reward_shaping.py \
            --config "config/flow.py:hps_v2_sd3" \
            --config.train.alpha "0.6" \
            --config.sample.batch_size "6" \
            --config.sample.num_generations "24" \
            --config.sample.mini_batch "12" \
            --config.train.batch_size "2" \
            --config.train.gradient_accumulation_steps "36" \
            --config.train.d_clip_max "1" \
            --config.mixed_precision "fp16" \
            --config.sample.num_steps "16" \
            --config.train.timestep_fraction "0.89" \
            --config.num_epochs "400" \
            --config.run_name "hpsv2_drift_reward_sd3.5"
    ;;
  sd35_shaping_pickscore)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29501 flow/diversity_reward_shaping.py \
            --config "config/flow.py:pickscore_v2_sd3" \
            --config.train.alpha "0.6" \
            --config.sample.batch_size "6" \
            --config.sample.num_generations "24" \
            --config.sample.mini_batch "12" \
            --config.train.batch_size "2" \
            --config.train.gradient_accumulation_steps "36" \
            --config.train.d_clip_max "1" \
            --config.mixed_precision "fp16" \
            --config.sample.num_steps "16" \
            --config.train.timestep_fraction "0.89" \
            --config.num_epochs "400" \
            --config.run_name "pickscore_drift_reward_sd3.5"
    ;;
  sd15_shaping_hps)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29504  diffusion/diversity_reward_shaping.py \
            --config config/diffusion.py:hps_v2 \
            --config.train.alpha "0.5" \
            --config.sample.batch_size "16"\
            --config.train.batch_size "2" \
            --config.sample.num_generations "8" \
            --config.train.d_clip_max "1"\
            --config.mixed_precision "fp16" \
            --config.num_epochs "200" \
            --config.run_name "hpsv2_drift_reward_sd1.5"
    ;;
  sd15_sampling_pickscore)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29501 diffusion/reward_concentrated_sampling.py \
            --config config/diffusion.py:pickscore \
            --config.sample.batch_size "16"\
            --config.train.batch_size "2" \
            --config.sample.num_generations "8" \
            --config.mixed_precision "fp16" \
            --config.run_name "pickscore_drift_sampling_sd1.5"
    ;;
  sd15_shaping_pickscore)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29502 diffusion/diversity_reward_shaping.py \
            --config config/diffusion.py:pickscore \
            --config.train.alpha "0.5" \
            --config.sample.batch_size "16"\
            --config.train.batch_size "2" \
            --config.sample.num_generations "8" \
            --config.train.d_clip_max "1"\
            --config.mixed_precision "fp16" \
            --config.num_epochs "200" \
            --config.run_name "pickscore_drift_reward_sd1.5"
    ;;
  sd15_prompt_pickscore)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29502 diffusion/noise_conditioned_prompting.py \
            --config config/diffusion.py:pickscore \
            --config.sample.batch_size "16"\
            --config.train.batch_size "2" \
            --config.sample.num_generations "8" \
            --config.mixed_precision "fp16" \
            --config.noise  "0.05" \
            --config.num_epochs "200" \
            --config.run_name "pickscore_drift_noise_sd1.5"
    ;;
  *)
    echo "Usage: bash train.sh <run>"
    printf '  %s\n' sd35_shaping_hps sd35_shaping_pickscore sd15_shaping_hps sd15_sampling_pickscore sd15_shaping_pickscore sd15_prompt_pickscore
    exit 1
    ;;
esac
