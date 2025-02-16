import csv
import os
from datetime import datetime
import gymnasium as gym

class EnvDataCollector:
    """
    A class for collecting environment data and writing it to a CSV file.
    """

    def __init__(self, env_id: str, csv_filename: str = "env_data"):
        """
        Parameters
        ----------
        env_id : str
            The gym environment ID to collect data from (e.g. 'LunarLander-v2').
        csv_filename : str
            The base CSV filename to store the environment data. If the filename
            ends with ".csv", the extension will be removed automatically.
        """
        self.env_id = env_id

        # Ensure the 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")

        # Remove '.csv' from csv_filename if it's present
        if csv_filename.lower().endswith('.csv'):
            csv_filename = csv_filename[:-4]

        # Create a unique CSV filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.csv_filename = os.path.join("data", f"{csv_filename}_{timestamp}.csv")
        self.env = gym.make(env_id)

    def collect_data(self, num_steps: int = 1000):
        """
        Collect environment data for a specified number of steps using random actions.

        Parameters
        ----------
        num_steps : int
            Number of steps to run the environment.
        """
        # Reset environment
        obs, _ = self.env.reset()

        # Write the CSV header
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Step',
                'x', 'y', 'x_velocity', 'y_velocity', 'angle',
                'angular_velocity', 'left_leg_contact', 'right_leg_contact',
                'Action', 'Reward', 'Done', 'Truncated', 'Info'
            ])

        # Run the environment and collect data
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            for step in range(num_steps):
                action = self.env.action_space.sample()  # random action
                obs, reward, done, truncated, info = self.env.step(action)

                # Each environment can have different obs shapes; adjust as needed
                writer.writerow([
                    step,
                    obs[0],  # x
                    obs[1],  # y
                    obs[2],  # x_velocity
                    obs[3],  # y_velocity
                    obs[4],  # angle
                    obs[5],  # angular_velocity
                    obs[6],  # left_leg_contact
                    obs[7],  # right_leg_contact
                    action,
                    reward,
                    done,
                    truncated,
                    info
                ])

                if done:
                    print(f"[DataCollector] Episode finished after {step+1} timesteps")
                    obs, _ = self.env.reset()  # reset environment if done

        self.env.close()
        print(f"[DataCollector] Data collection complete. CSV saved to {self.csv_filename}")
