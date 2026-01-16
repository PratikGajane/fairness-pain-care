# Fairness-Aware Pain Care Recommendation Meta-Algorithm

This repository contains the implementation of a meta-algorithm for fairness-aware pain care recommendations using reinforcement learning. The algorithm implements a two-level nested decision-making strategy for feature selection and treatment recommendation.

## Overview

The meta-algorithm makes decisions at two levels:
- **Level 1**: Selects optimal feature sets from patient information
- **Level 2**: Selects optimal pain care recommendations using selected features

The algorithm optimizes for both utility (treatment effectiveness) and fairness (minimal disparity across demographic groups).

## Project Structure

```
fairness-pain-care/
│
├── policy_base.py           # Abstract base classes for Policy1 and Policy2
├── thompson_sampling.py     # Thompson Sampling implementation (Policy1)
├── linucb_policy.py        # LinUCB implementation (Policy2 example)
├── mock_environment.py     # Mock environment for testing
├── meta_algorithm.py       # Main meta-algorithm implementation
├── example_usage.py        # Example usage and demonstrations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fairness-pain-care.git
cd fairness-pain-care
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from thompson_sampling import ThompsonSampling
from linucb_policy import LinUCB
from mock_environment import MockEnvironment
from meta_algorithm import MetaAlgorithm

# Create environment
env = MockEnvironment(n_timesteps=50000, n_features=8, n_actions=3)

# Create Policy1 (feature set selection)
policy1 = ThompsonSampling(n_actions=255, alpha_prior=1, beta_prior=5)

# Create Policy2 (recommendation selection)
policy2 = LinUCB(n_actions=3, n_features=8, alpha=1.0)

# Create and run meta-algorithm
meta_algo = MetaAlgorithm(
    policy1=policy1,
    policy2=policy2,
    environment=env,
    fairness_weight=0.5,
    demographic_attribute='gender',
    save_metrics=True,        
    n_metric_points=100    
)

results = meta_algo.run(n_timesteps=50000)
```

### Output Files

The algorithm automatically saves detailed fairness metrics to `fairness_metrics_timesteps.csv` with columns:
- timestep
- avg_reward_male, avg_reward_female, fairness_gender
- avg_reward_white, avg_reward_nonwhite, fairness_race
- avg_reward_under65, avg_reward_65plus, fairness_age

### Running Examples

Run the example script to see demonstrations:

```bash
python example_usage.py
```

This will run three examples:
1. Basic usage with Thompson Sampling + LinUCB
2. Custom Policy2 implementation
3. Different fairness-utility weight combinations

## Implementing Custom Policies

### Custom Policy1 (Level 1)

Inherit from `Policy1Base` and implement:
- `select_action()`: Select a feature set
- `update(action, reward)`: Update based on observed reward

```python
from policy_base import Policy1Base

class CustomPolicy1(Policy1Base):
    def __init__(self, n_actions):
        super().__init__(n_actions)
        # Your initialization
    
    def select_action(self):
        # Your action selection logic
        pass
    
    def update(self, action, reward):
        # Your update logic
        pass
```

### Custom Policy2 (Level 2)

Inherit from `Policy2Base` and implement:
- `select_action(context)`: Select a recommendation given context
- `update(context, action, reward)`: Update based on observed reward
- `reset()`: Reset when feature set changes

```python
from policy_base import Policy2Base

class CustomPolicy2(Policy2Base):
    def __init__(self, n_actions, n_features):
        super().__init__(n_actions, n_features)
        # Your initialization
    
    def select_action(self, context):
        # Your action selection logic
        pass
    
    def update(self, context, action, reward):
        # Your update logic
        pass
    
    def reset(self):
        # Reset parameters
        pass
```

Users can implement advanced algorithms like:
- NeuralUCB
- NN-UCB
- Neural Thompson Sampling
- Any other contextual bandit algorithm

## Using Real Clinical Data

To use this code with real clinical data from Dataset #2 for Piette et al, Data in Brief: Data for a Reinforcement Learning Intervention to Treat Chronic Pain (https://data.mendeley.com/datasets/33mkbm32dz/1):

1. Download the dataset
2. Replace `MockEnvironment` with your data loader

Example:
```python
class RealDataEnvironment:
    def __init__(self, data_path):
        # Load your data
        self.data = load_data(data_path)
    
    def get_context(self, timestep, feature_indices):
        # Return patient context with selected features
        pass
    
    def get_reward(self, timestep, feature_indices, action):
        # Return observed reward
        pass
```

## Parameters

### ThompsonSampling
- `n_actions` (int): Number of feature sets (default: 255)
- `alpha_prior` (float): Alpha parameter for Beta prior (default: 1)
- `beta_prior` (float): Beta parameter for Beta prior (default: 5)

### LinUCB
- `n_actions` (int): Number of recommendations (default: 3)
- `n_features` (int): Number of features (default: 8)
- `alpha` (float): Exploration parameter (default: 1.0)

### MetaAlgorithm
- `fairness_weight` (float): Weight for fairness (0 to 1, default: 0.5)
- `demographic_attribute` (str): 'gender', 'race', or 'age' (default: 'gender')

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gajane2024investigatinggenderfairnessmachine,
      title={Investigating Fairness in Machine Learning-driven Personalized Care for Chronic Pain}, 
      author={Pratik Gajane and Sean Newman and John D. Piette},
      year={2024},
      eprint={2402.19226},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.19226}, 
}
```

## License

GNU General Public License v3.0

## Acknowledgments

This work uses data from the clinical trial: Piette et al. (2022) Dataset available at: https://data.mendeley.com/datasets/33mkbm32dz/1
