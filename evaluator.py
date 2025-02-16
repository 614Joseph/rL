#evaluator.py
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import os


class RLEvaluator:
    """
    A class to evaluate and render a trained RL model.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : stable_baselines3 model
            A trained Stable Baselines model (A2C, PPO, etc.).
        """
        self.model = model
        if self.model is None:
            raise ValueError("[Evaluator] No model provided.")

    def evaluate_model(self, n_eval_episodes: int = 5) -> tuple:
        """
        Evaluate the model over n_eval_episodes.

        Parameters
        ----------
        n_eval_episodes : int
            Number of episodes to evaluate.

        Returns
        -------
        (mean_reward, std_reward) : tuple
            The mean and standard deviation of rewards.
        """
        env = self.model.get_env()
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=n_eval_episodes)
        print(f"[Evaluator] Evaluated over {n_eval_episodes} episodes. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def render_model(self, 
                    n_steps: int = 1000, 
                    env_id: str = "LunarLander-v2", 
                    model_type: str = "PPO"):
        """
        Render the model for a specified number of steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to render.
        env_id : str
            Gymnasium environment ID to render.
        model_type : str
            The RL algorithm used (e.g., 'PPO', 'A2C', etc.).
        """

        # Construct the VecNormalize stats filename from model_type and env_id
        vec_filename = f"{model_type}_{env_id}_vecnormalize.pkl"
        models_dir = "models"
        full_vec_path = os.path.join(models_dir, vec_filename)
        print(f"[Evaluator] Attempting to load VecNormalize stats from: {full_vec_path}")

        # Create the environment with explicit render_mode
        env = DummyVecEnv([lambda: gym.make(env_id, render_mode='human')])

        # If the file exists, load VecNormalize; otherwise, render unnormalized env
        if os.path.exists(full_vec_path):
            env = VecNormalize.load(full_vec_path, env)
            # Disable training updates to keep normalization stats fixed
            env.training = False
            env.norm_reward = False
            print("[Evaluator] VecNormalize successfully loaded and disabled for training.")
        else:
            print(f"[Evaluator] VecNormalize file not found at {full_vec_path}. Proceeding without normalization.")

        obs = env.reset()
        for step in range(n_steps):
            if self.model is None:
                # Use random actions if no model was provided
                action = env.action_space.sample()
            else:
                # Use the trained model to predict actions
                action, _ = self.model.predict(obs)

            obs, rewards, dones, info = env.step(action)
            env.render()

            # If running multiple parallel envs, check if any done
            if isinstance(dones, np.ndarray):
                if dones.any():
                    print(f"[Evaluator] Episode finished at step {step+1}. Resetting environment.")
                    obs = env.reset()
            else:
                if dones:
                    print(f"[Evaluator] Episode finished at step {step+1}. Resetting environment.")
                    obs = env.reset()

        print("[Evaluator] Render complete.")
