# Modified for DRIFT. See LICENSE and NOTICE for license and attribution details.
import importlib.util
import sys
import os

module_name = "base"
module_path = os.path.join(os.path.dirname(__file__), "base.py")

spec = importlib.util.spec_from_file_location(module_name, module_path)
base = importlib.util.module_from_spec(spec)
sys.modules[module_name] = base
spec.loader.exec_module(base)

def _training_base():
    config = base.get_config()

    config.pretrained.model = os.environ.get("DRIFT_SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 10
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 16

    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 16

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config

def hps_v2():
    config = _training_base()
    config.run_name = "hpsv2_drift_reward_sd1.5"
    config.prompt_fn = "simple_animals"
    config.reward_fn = "hps_v2"

    config.num_epochs = 100
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 16
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 16

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config

def pickscore():
    config = _training_base()
    config.run_name = "pickscore_drift_reward_sd1.5"
    config.reward_fn = "PickScore"
    config.prompt_fn = "simple_animals"
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 16
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 16
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config

def get_config(name):
    presets = {'hps_v2': hps_v2, 'pickscore': pickscore}
    return presets[name]()
