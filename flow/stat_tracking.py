import numpy as np

class PerPromptStatTracker_GRPO:
    def __init__(self, global_std = False):
        self.global_std = global_std

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if self.global_std :
                mean = np.mean(prompt_rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(prompt_rewards)
                std = np.std(prompt_rewards) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages
