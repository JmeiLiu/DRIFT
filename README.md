# DRIFT

Implementation of **Beyond the Dirac Delta: Mitigating Diversity Collapse in Reinforcement Fine-Tuning for Image Generation**.

## Installation

Linux and an NVIDIA GPU with a CUDA 12.8-compatible driver. Use the same environment for SDv1.5 and SD3.5-M.

```bash
conda env create -f environment.yaml
conda activate drift
accelerate config
```

Both SDv1.5 and SD3.5-M support PickScore and HPSv2 rewards.

## Running

Set the model paths and GPU IDs in `train.sh` for your machine.

**Sampling — Reward-Concentrated Sampling (SDv1.5, PickScore)**

```bash
bash train.sh sd15_sampling_pickscore
```

**Noise — Noise-Conditioned Prompting (SDv1.5, PickScore)**

```bash
bash train.sh sd15_prompt_pickscore
```

**Reward Shaping**

```bash
# SDv1.5
bash train.sh sd15_shaping_pickscore
bash train.sh sd15_shaping_hps

# SD3.5-M
bash train.sh sd35_shaping_pickscore
bash train.sh sd35_shaping_hps
```

Run `bash train.sh` to list all available commands. SD3.5-M requires access to the base model; use `huggingface-cli login` if needed.

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party terms and attribution.
