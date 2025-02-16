import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.env_util import make_vec_env


class RLTrainer:
    """
    A class for training or loading a reinforcement learning model using Stable Baselines3.
    """

    def __init__(self,
                 env_id: str = "LunarLander-v2",
                 model_type: str = "A2C",
                 n_envs: int = 4,
                 total_timesteps: int = 200_000,
                 normalize_env: bool = True):
        """
        Parameters
        ----------
        env_id : str
            The ID of the gym environment (e.g., 'LunarLander-v2').
        model_type : str
            The type of RL model to train or load ('A2C' or 'PPO').
        n_envs : int
            Number of parallel environments.
        total_timesteps : int
            The total training timesteps.
        normalize_env : bool
            Whether to apply VecNormalize to the environment.
        """
        self.env_id = env_id
        self.model_type = model_type.upper()
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.normalize_env = normalize_env

        # Prepare the environment
        self.env = make_vec_env(self.env_id, n_envs=self.n_envs)

        if self.normalize_env:
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)

        self.model = None

    def train(self):
        """
        Train the RL model on the environment.
        """
        
        if self.model_type == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=1)
        elif self.model_type == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        print(f"[Trainer] Starting training of {self.model_type} on {self.env_id}")
        self.model.learn(total_timesteps=self.total_timesteps)
        print("[Trainer] Training finished.")

    def save(self, model_filename: str = None, vecnormalize_filename: str = None):
        """
        Save the trained model and VecNormalize stats (if any) into the 'models' folder.

        Parameters
        ----------
        model_filename : str
            Filename to save the model. If None, auto-generate using model_type + env_id.
        vecnormalize_filename : str
            Filename to save VecNormalize stats. If None, auto-generate similarly.
        """
        if self.model is None:
            raise RuntimeError("[Trainer] Cannot save - no model found. Please train first.")

        # Ensure the 'models' directory exists.
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Auto-generate model filename if not provided
        if model_filename is None:
            model_filename = f"{self.model_type}_{self.env_id}.zip"
        full_model_path = os.path.join(models_dir, model_filename)

        self.model.save(full_model_path)
        print(f"[Trainer] Model saved to {full_model_path}")

        # Auto-generate VecNormalize filename if needed
        if self.normalize_env and isinstance(self.env, VecNormalize):
            if vecnormalize_filename is None:
                vecnormalize_filename = f"{self.model_type}_{self.env_id}_vecnormalize.pkl"
            full_vecnormalize_path = os.path.join(models_dir, vecnormalize_filename)
            self.env.save(full_vecnormalize_path)
            print(f"[Trainer] VecNormalize stats saved to {full_vecnormalize_path}")

    def load(self, model_filename: str = None, vecnormalize_filename: str = None):
        """
        Load a pre-trained model from file. If filenames are None, we'll auto-generate
        from the stored model_type and env_id. If a file does not exist, model stays None.

        Parameters
        ----------
        model_filename : str
            Filename of the saved model. If None, auto-generate from model_type + env_id.
        vecnormalize_filename : str
            Filename of the saved VecNormalize stats (if used). If None, auto-generate.
        """
        models_dir = "models"

        # If user did not provide a model filename, auto-generate
        if model_filename is None:
            model_filename = f"{self.model_type}_{self.env_id}.zip"
        full_model_path = os.path.join(models_dir, model_filename)

        # Re-initialize the environment
        self.env = make_vec_env(self.env_id, n_envs=self.n_envs)

        # If using VecNormalize, try loading stats
        if self.normalize_env:
            if vecnormalize_filename is None:
                vecnormalize_filename = f"{self.model_type}_{self.env_id}_vecnormalize.pkl"
            full_vecnormalize_path = os.path.join(models_dir, vecnormalize_filename)

            # Only load if file actually exists
            if os.path.exists(full_vecnormalize_path):
                self.env = VecNormalize.load(full_vecnormalize_path, self.env)
                print(f"[Trainer] Loaded VecNormalize stats from {full_vecnormalize_path}")
            else:
                print(f"[Trainer] No VecNormalize file found at {full_vecnormalize_path}. Proceeding without it.")

        # Attempt to load the model itself
        if os.path.exists(full_model_path):
            if self.model_type == "A2C":
                self.model = A2C.load(full_model_path, env=self.env)
            elif self.model_type == "PPO":
                self.model = PPO.load(full_model_path, env=self.env)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            print(f"[Trainer] Model loaded from {full_model_path}")
        else:
            print(f"[Trainer] No model found at {full_model_path}.")
            self.model = None

    def get_env(self) -> VecEnv:
        """
        Returns the underlying environment (VecEnv or VecNormalize).
        """
        return self.env
