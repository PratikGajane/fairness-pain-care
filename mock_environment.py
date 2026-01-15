"""
Mock environment for testing the meta-algorithm.
Generates synthetic patient data with demographics and rewards.
"""

import numpy as np


class MockEnvironment:
    """
    Mock environment for pain care recommendation system.
    
    Simulates patient contexts, demographics, and rewards for testing purposes.
    In real deployment, replace this with actual clinical data.
    """
    
    def __init__(self, n_timesteps=50000, n_features=8, n_actions=3, random_seed=42):
        """
        Initialize mock environment.
        
        Args:
            n_timesteps (int): Total number of time steps. Default is 50000.
            n_features (int): Number of base features. Default is 8.
            n_actions (int): Number of recommendations. Default is 3.
            random_seed (int): Random seed for reproducibility. Default is 42.
        """
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_actions = n_actions
        
        np.random.seed(random_seed)
        
        # Generate synthetic data for all time steps
        self._generate_data()
        
        # Track history for fairness computation
        self.history = {
            'male': {'rewards': [], 'counts': 0},
            'female': {'rewards': [], 'counts': 0},
            'white': {'rewards': [], 'counts': 0},
            'non_white': {'rewards': [], 'counts': 0},
            'age_65_plus': {'rewards': [], 'counts': 0},
            'age_under_65': {'rewards': [], 'counts': 0}
        }
    
    def _generate_data(self):
        """
        Generate synthetic patient contexts and demographics for all time steps.
        """
        # Generate random contexts (8 features) for all time steps
        # Features are normalized to [0, 1] range
        self.contexts = np.random.rand(self.n_timesteps, self.n_features)
        
        # Generate demographics (binary for simplicity)
        # Gender: 0 = male, 1 = female
        self.gender = np.random.binomial(1, 0.5, self.n_timesteps)
        
        # Race: 0 = white, 1 = non-white
        self.race = np.random.binomial(1, 0.3, self.n_timesteps)
        
        # Age: 0 = under 65, 1 = 65+
        self.age = np.random.binomial(1, 0.4, self.n_timesteps)
        
        # Generate true reward coefficients for each action
        # In real setting, these would be learned from data
        self.true_theta = np.random.randn(self.n_actions, self.n_features) * 0.5
    
    def get_context(self, timestep, feature_set_indices):
        """
        Get patient context for given time step with selected features.
        
        Args:
            timestep (int): Current time step
            feature_set_indices (list): Indices of selected features (subset of 0-7)
        
        Returns:
            np.ndarray: Context vector with only selected features
        """
        # Extract selected features from full context
        full_context = self.contexts[timestep]
        selected_context = full_context[feature_set_indices]
        
        return selected_context
    
    def get_reward(self, timestep, feature_set_indices, action):
        """
        Get reward for taking an action with selected features.
        
        Args:
            timestep (int): Current time step
            feature_set_indices (list): Indices of selected features
            action (int): Selected recommendation (0 to n_actions-1)
        
        Returns:
            float: Reward (continuous, in [0, 1])
        """
        # Get context with selected features
        context = self.get_context(timestep, feature_set_indices)
        
        # Pad context to match full feature dimension if needed
        padded_context = np.zeros(self.n_features)
        padded_context[feature_set_indices] = context
        
        # Compute base reward as linear function of context
        base_reward = np.dot(self.true_theta[action], padded_context)
        
        # Add noise and clip to [0, 1]
        noisy_reward = base_reward + np.random.randn() * 0.1
        reward = np.clip(noisy_reward, 0, 1)
        
        # Update history for fairness computation
        self._update_history(timestep, reward)
        
        return reward
    
    def _update_history(self, timestep, reward):
        """
        Update reward history for fairness computation.
        
        Args:
            timestep (int): Current time step
            reward (float): Observed reward
        """
        # Update gender groups
        if self.gender[timestep] == 0:
            self.history['male']['rewards'].append(reward)
            self.history['male']['counts'] += 1
        else:
            self.history['female']['rewards'].append(reward)
            self.history['female']['counts'] += 1
        
        # Update race groups
        if self.race[timestep] == 0:
            self.history['white']['rewards'].append(reward)
            self.history['white']['counts'] += 1
        else:
            self.history['non_white']['rewards'].append(reward)
            self.history['non_white']['counts'] += 1
        
        # Update age groups
        if self.age[timestep] == 0:
            self.history['age_under_65']['rewards'].append(reward)
            self.history['age_under_65']['counts'] += 1
        else:
            self.history['age_65_plus']['rewards'].append(reward)
            self.history['age_65_plus']['counts'] += 1
    
    def compute_fairness(self, demographic_attribute):
        """
        Compute fairness score for a demographic attribute.
        
        Fairness = 1 - |avg_reward_group1 - avg_reward_group2|
        
        Args:
            demographic_attribute (str): One of 'gender', 'race', 'age'
        
        Returns:
            float: Fairness score (higher is better, max is 1.0)
        """
        if demographic_attribute == 'gender':
            group1_key, group2_key = 'male', 'female'
        elif demographic_attribute == 'race':
            group1_key, group2_key = 'white', 'non_white'
        elif demographic_attribute == 'age':
            group1_key, group2_key = 'age_under_65', 'age_65_plus'
        else:
            raise ValueError(f"Unknown demographic attribute: {demographic_attribute}")
        
        # Compute average rewards for each group
        if self.history[group1_key]['counts'] > 0:
            avg_reward_group1 = np.mean(self.history[group1_key]['rewards'])
        else:
            avg_reward_group1 = 0.0
        
        if self.history[group2_key]['counts'] > 0:
            avg_reward_group2 = np.mean(self.history[group2_key]['rewards'])
        else:
            avg_reward_group2 = 0.0
        
        # Fairness = 1 - disparity
        fairness = 1.0 - abs(avg_reward_group1 - avg_reward_group2)
        
        return fairness
    
    def compute_utility(self):
        """
        Compute overall utility (average reward across all time steps so far).
        
        Returns:
            float: Utility score
        """
        all_rewards = []
        for group_data in self.history.values():
            all_rewards.extend(group_data['rewards'])
        
        if len(all_rewards) > 0:
            utility = np.mean(all_rewards)
        else:
            utility = 0.0
        
        return utility
    
    def feature_set_id_to_indices(self, feature_set_id):
        """
        Convert feature set ID (0-254) to actual feature indices.
        
        Feature set ID represents binary encoding of which features are selected.
        For example, ID=5 (binary: 00000101) means features 0 and 2 are selected.
        
        Args:
            feature_set_id (int): Feature set ID (0 to 254, excluding empty set)
        
        Returns:
            list: Indices of selected features
        """
        # Add 1 to avoid empty set (ID=0 represents feature set {0}, not {})
        binary_representation = feature_set_id + 1
        
        # Convert to binary and get indices of 1s
        feature_indices = []
        for i in range(self.n_features):
            if binary_representation & (1 << i):
                feature_indices.append(i)
        
        return feature_indices