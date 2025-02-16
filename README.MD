# Reinforcement Learning CLI Project

This project provides a command-line interface (CLI) for training, testing, collecting data, and rendering reinforcement learning (RL) models using [Stable Baselines3](https://stable-baselines3.readthedocs.io/) and [Gymnasium](https://gymnasium.farama.org/).

> **Note:**  
> This project supports RL algorithms **A2C** and **PPO** and automatically handles saving and loading of both the trained model and environment normalization statistics (using `VecNormalize`) if you choose to use the `--normalize` flag.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
    - [Collect Data](#collect-data)
    - [Train a Model](#train-a-model)
    - [Test a Model](#test-a-model)
    - [Render a Model](#render-a-model)
5. [Examples](#examples)
6. [Notes](#notes)

## Requirements

This project requires the following packages:

- **gymnasium**
- **stable-baselines3**

These dependencies are listed in the `requirements.txt` file:

```txt
gymnasium
stable-baselines3 
```

## Installation
Clone the repository:


```bash
git clone https://github.com/matt/rl_cli_project.git

cd rl_cli_project
```
## Install the dependencies:

```bash
pip install -r requirements.txt
Tip: It is recommended to use a virtual environment to manage dependencies.
``` 


## Project Structure
```
.
├── main.py               # Main CLI entry point
├── env_data_collector.py # Module for collecting environment data to CSV
├── trainer.py            # Module defining RLTrainer for training, saving, and loading models
├── evaluator.py          # Module for evaluating and rendering RL models
├── random_renderer.py    # Module for rendering environments with random actions
├── requirements.txt      # Required Python packages
└── README.md             # This documentation file
main.py
Contains the CLI interface with four subcommands: collect, train, test, and render.

trainer.py
Defines the RLTrainer class which creates a vectorized environment, applies normalization (if specified), trains models, and saves/loads both the model and VecNormalize statistics.

evaluator.py
Contains the logic to evaluate a loaded model through a series of episodes.

random_renderer.py
Provides functionality to render the environment using random actions (useful for debugging or exploration).
``` 
## Usage
The CLI supports four main commands:

- **Collect Data**:  
  Collect environment data and store it in a CSV file.

- **Train a Model**:  
  Train an RL model on a specified Gymnasium environment.

- **Test a Model**:  
  Evaluate a previously trained model.

- **Render a Model**:  
  Render a trained model or, alternatively, use random actions if no model is provided.

### Collect Data
**Command:**

```bash
python main.py collect --env_id "LunarLander-v2" --steps 1000 --csv_filename "env_data.csv" 
```
#### Data Collection Parameters

- `--env_id`: Gymnasium environment ID.
- `--steps`: Number of steps to collect.
- `--csv_filename`: CSV file to store collected data.


### Train a Model
**Command:**

```bash
python main.py train --env_id "LunarLander-v2" --model_type PPO --timesteps 20000 --normalize
```
#### Training Parameters

- `--env_id`: Gymnasium environment ID.
- `--model_type`: RL algorithm to use (`A2C` or `PPO`).
- `--timesteps`: Number of training timesteps.
- `--normalize`: Flag to apply `VecNormalize` to standardize observations and rewards.

--model_path and --vecnormalize_path: (Optional) If not specified, filenames are auto-generated as:
Model: PPO_LunarLander-v2.zip
VecNormalize stats: PPO_LunarLander-v2_vecnormalize.pkl
Test a Model
Command:

```bash
python main.py test --env_id "LunarLander-v2" --model_type PPO --normalize --eval_episodes 5
```
#### Evaluation Parameters

- `--env_id`: Gymnasium environment ID.
- `--model_type`: RL algorithm that was used (`A2C` or `PPO`).
- `--normalize`: Must be set if the model was trained with normalization.
- `--eval_episodes`: Number of episodes to run during evaluation.

**Note:** Optional model and normalization paths will be auto-generated if not provided.

### Render a Model
**Command:**

```bash
python main.py render --env_id "LunarLander-v2" --model_type PPO --normalize --steps 2000
```
#### Rendering Parameters

- `--env_id`: Gymnasium environment ID.
- `--model_type`: RL algorithm (`A2C` or `PPO`).
- `--normalize`: Must be set if the model was trained with normalization.
- `--steps`: Number of steps to render.

**Note:** The `model` and `VecNormalize` stats are loaded from auto-generated file names if not provided.

#### Render with Random Actions
**Command:**

```bash
python main.py render --env_id "LunarLander-v2" --model_type null --steps 2000
```

Use the keyword ```null for --model_type``` to render the environment using random actions (without loading a model).

## Examples
### Training a PPO with VecNormalize:

```bash
python main.py train --env_id "LunarLander-v2" --model_type PPO --timesteps 20000 --normalize
```


This process will train a `PPO` model, wrap the environment with `VecNormalize`, and automatically save:

- The model as `models/PPO_LunarLander-v2.zip`
- The normalization stats as `models/PPO_LunarLander-v2_vecnormalize.pkl`

### Testing the Trained Model:

```bash
python main.py test --env_id "LunarLander-v2" --model_type PPO --normalize --eval_episodes 5
```

This loads the auto-generated model and normalization stats and evaluates the model for 5 episodes.

### Rendering the Trained Model:

```bash
python main.py render --env_id "LunarLander-v2" --model_type PPO --normalize --steps 2000
```
Loads and renders the model for 2000 steps. After rendering, the updated normalization statistics are re-saved.

### Rendering with Random Actions:

```bash

python main.py render --env_id "LunarLander-v2" --model_type null --steps 2000
```
This renders the environment using random actions (without loading any model).

### **Notes**

### **Normalization (VecNormalize)**
Using the `--normalize` flag is recommended when your training environment benefits from standardized observations and rewards. When this flag is set:

- The environment is wrapped with `VecNormalize` during training, testing, and rendering.
- Normalization statistics are automatically saved and loaded using auto-generated filenames unless specified otherwise.

### **File Locations**
- All models and normalization files are saved inside a directory called `models` (which is automatically created if it doesn't exist).

### **Customization**
- You can override the default model and normalization file names using the `--model_path` and `--vecnormalize_path` parameters.

### **Python Version**  
The project is tested with **Python 3.7 and above**.







Python Version:
The project is tested with Python 3.7 and above.
