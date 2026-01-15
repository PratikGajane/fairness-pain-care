"""
Thompson Sampling implementation for Level 1 (feature set selection).
Uses Beta distributions with Bernoulli sampling for continuous rewards.
"""

import numpy as np
from policy_base import Policy1Base


class ThompsonSampling(Policy1Base):
    """
    Thompson Sampling policy for feature set selection at Level 1.
    
    Uses Beta distributions for each action (feature set) and updates
    via Bernoulli sampling from continuous rewards.
    """
    
    def __init__(self, n_actions=255, alpha_prior=1, beta_prior=5, random_seed=None):
        """
        Initialize Thompson Sampling.
        
        Args:
            n_actions (int): Number of actions (feature sets). Default is 255.
            alpha_prior (float): Alpha parameter for Beta prior. Default is 1.
            beta_prior (float): Beta parameter for Beta prior. Default is 5.
            random_seed (int): Random seed for reproducibility. Default is None.
        """
        super().__init__(n_actions)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Initialize Beta distributions for all actions
        self.alpha = np.ones(n_actions) * alpha_prior
        self.beta = np.ones(n_actions) * beta_prior
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def select_action(self):
        """
        Select a feature set using Thompson Sampling.
        
        Samples from Beta distribution for each action and selects
        the action with highest sampled value.
        
        Returns:
            int: Selected feature set ID (0 to n_actions-1)
        """
        # Sample from Beta distribution for each action
        theta_samples = np.random.beta(self.alpha, self.beta)
        
        # Select action with highest sampled value
        action = np.argmax(theta_samples)
        
        return action
    
    def update(self, action, reward):
        """
        Update Beta distribution for selected action.
        
        Uses Bernoulli sampling: draws binary outcome from Bernoulli(reward),
        then updates Beta distribution based on that outcome.
        
        Args:
            action (int): Feature set ID that was selected
            reward (float): Observed reward (continuous, in [0,1])
        """
        # Bernoulli sampling from continuous reward
        # If reward = 0.6, draw from {0,1} with P(1) = 0.6
        binary_reward = np.random.binomial(1, reward)
        
        # Standard Beta updates based on binary outcome
        if binary_reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
    
    def get_posterior_means(self):
        """
        Get posterior mean for each action.
        
        Returns:
            np.ndarray: Posterior means (expected rewards) for all actions
        """
        return self.alpha / (self.alpha + self.beta)
    
    def reset(self):
        """
        Reset Beta distributions to prior.
        """
        self.alpha = np.ones(self.n_actions) * self.alpha_prior
        self.beta = np.ones(self.n_actions) * self.beta_prior