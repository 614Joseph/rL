# random_renderer.py

import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def render_random_actions(
    env_id: str,
    n_steps: int = 1000,
    n_envs: int = 1,
    normalize: bool = False,
    vecnormalize_path: str = "vec_normalize.pkl"
):
    """
    Render a Gym environment using random actions with a DummyVecEnv that
    uses a lambda to create an environment with render_mode='human'.
    
    Parameters
    ----------
    env_id : str
        The Gym environment ID (e.g., 'LunarLander-v2').
    n_steps : int
        Number of steps to render.
    n_envs : int
        Number of parallel environments.
    normalize : bool
        Whether to apply VecNormalize to the environment.
    vecnormalize_path : str
        Path to a saved VecNormalize stats file (if it exists).
    """
    print("[random_renderer] Rendering with random actions...")

    # Create a DummyVecEnv that uses gym.make with render_mode='human'
    env = DummyVecEnv([lambda: gym.make(env_id, render_mode='human') for _ in range(n_envs)])

    # If normalization is requested, load existing VecNormalize stats if available;
    # otherwise, wrap the environment with a new VecNormalize.
    if normalize:
        if os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)
            print(f"[random_renderer] Loaded VecNormalize stats from {vecnormalize_path}")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            print("[random_renderer] Created new VecNormalize wrapper for random actions.")

    # Reset the environment
    obs = env.reset()
    for step in range(n_steps):
        # Always sample a list of random actions, one per environment.
        action = [env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, info = env.step(action)
        env.render("human")

        # Reset environment if any instance is done.
        if isinstance(dones, np.ndarray):
            if dones.any():
                print(f"[random_renderer] Done in at least one env at step {step+1}. Resetting env(s).")
                obs = env.reset()
        else:
            if dones:
                print(f"[random_renderer] Episode done at step {step+1}. Resetting env.")
                obs = env.reset()

    env.close()
    print("[random_renderer] Finished rendering random actions.")
