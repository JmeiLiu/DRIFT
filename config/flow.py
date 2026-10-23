import ml_collections
import importlib.util
import os

spec = importlib.util.spec_from_file_location("flow_base", os.path.join(os.path.dirname(__file__), "flow_base.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def _training_base():
    config = base.get_config()

    config.pretrained.model = os.environ.get("DRIFT_FLOW_MODEL", "stabilityai/stable-diffusion-3.5-medium")
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 16

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.per_prompt_stat_tracking = True
    return config

def hps_v2_sd3():
    config = _training_base()
    config.run_name = "hpsv2_drift_reward_sd3.5"
    # sd3.5 medium
    config.pretrained.model =os.environ.get("DRIFT_FLOW_MODEL", "stabilityai/stable-diffusion-3.5-medium")
    config.prompt_fn = "training_prompts"
    config.reward_fn = "hps_v2"
    config.sample.num_steps = 16
    config.sample.guidance_scale = 4.5

    config.train.cfg = True
    config.resolution = 512

    config.sample.batch_size = 6
    config.sample.num_generations = 24
    config.sample.mini_batch = 12
    config.sample.num_batches_per_epoch = 16
    config.train.batch_size = 2  # Global batch size: 4 on two GPUs.
    config.train.gradient_accumulation_steps = 36
    config.train.alpha = 0.6

    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.89
    # kl loss
    config.train.beta = 0
    # Whether to use the std of all samples or the current group's.
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 10 # epoch

    config.per_prompt_stat_tracking = True
    config.sample.sde_type = "sde"
    return config

def pickscore_v2_sd3():
    config = _training_base()
    config.run_name = "pickscore_drift_reward_sd3.5"
    # sd3.5 medium
    config.pretrained.model = os.environ.get("DRIFT_FLOW_MODEL", "stabilityai/stable-diffusion-3.5-medium")
    config.prompt_fn = "training_prompts"
    config.reward_fn = "PickScore"
    config.sample.num_steps = 16
    config.sample.guidance_scale = 4.5

    config.train.cfg = True
    config.resolution = 512

    config.sample.batch_size = 6
    config.sample.num_generations = 24
    config.sample.mini_batch = 12
    config.sample.num_batches_per_epoch = 16
    config.train.batch_size = 2  # Global batch size: 4 on two GPUs.
    config.train.gradient_accumulation_steps = 36
    config.train.alpha = 0.6

    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.89
    # kl loss
    config.train.beta = 0

    # Whether to use the std of all samples or the current group's.
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 10 # epoch
    config.per_prompt_stat_tracking = True
    config.sample.sde_type = "sde"
    return config

def get_config(name):
    presets = {'hps_v2_sd3': hps_v2_sd3, 'pickscore_v2_sd3': pickscore_v2_sd3}
    return presets[name]()
