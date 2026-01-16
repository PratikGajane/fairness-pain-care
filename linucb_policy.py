"""
LinUCB implementation for Level 2 (recommendation selection).
This is a simple example implementation that users can use as reference.
"""

import numpy as np
from policy_base import Policy2Base


class LinUCB(Policy2Base):
    """
    LinUCB policy for recommendation selection at Level 2.
    
    This is a simple implementation provided as an example.
    Users can implement other contextual bandit algorithms
    (NeuralUCB, NN-UCB, Neural Thompson Sampling, etc.) by
    inheriting from Policy2Base.
    """
    
    def __init__(self, n_actions=3, n_features=8, alpha=0.3, random_seed=None):
        """
        Initialize LinUCB.
        
        Args:
            n_actions (int): Number of actions (recommendations). Default is 3.
            n_features (int): Maximum number of features. Default is 8.
            alpha (float): Exploration parameter. Default is 0.3
            random_seed (int): Random seed for reproducibility. Default is None.
        """
        super().__init__(n_actions, n_features)
        self.alpha = alpha
        self.current_feature_dim = n_features  # Track current feature dimension
        
        # Initialize parameters for each action
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_actions)]
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def select_action(self, context):
        """
        Select a recommendation using LinUCB.
        
        Args:
            context (np.ndarray): Patient context vector (selected features)
        
        Returns:
            int: Selected recommendation ID (0 to n_actions-1)
        """
        # Get the dimension of the current context
        context_dim = len(context) if context.ndim == 1 else context.shape[0]
        
        # Pad context to full feature dimension with zeros
        padded_context = np.zeros(self.n_features)
        padded_context[:context_dim] = context if context.ndim == 1 else context.flatten()
        
        # Ensure context is column vector
        padded_context = padded_context.reshape(-1, 1)
        
        # Calculate UCB for each action
        p_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Compute A_a^{-1}
            A_inv = np.linalg.inv(self.A[a])
            
            # Compute theta_hat = A_a^{-1} * b_a
            theta_hat = A_inv @ self.b[a]
            
            # Compute UCB: theta_hat^T * x + alpha * sqrt(x^T * A_a^{-1} * x)
            mean_reward = (theta_hat.T @ padded_context)[0, 0]
            uncertainty = self.alpha * np.sqrt((padded_context.T @ A_inv @ padded_context)[0, 0])
            
            p_values[a] = mean_reward + uncertainty
        
        # Select action with highest UCB
        action = np.argmax(p_values)
        
        return action
    
    def update(self, context, action, reward):
        """
        Update LinUCB parameters for selected action.
        
        Args:
            context (np.ndarray): Patient context vector that was used
            action (int): Recommendation ID that was selected
            reward (float): Observed reward
        """
        # Get the dimension of the current context
        context_dim = len(context) if context.ndim == 1 else context.shape[0]
        
        # Pad context to full feature dimension with zeros
        padded_context = np.zeros(self.n_features)
        padded_context[:context_dim] = context if context.ndim == 1 else context.flatten()
        
        # Ensure context is column vector
        padded_context = padded_context.reshape(-1, 1)
        
        # Update A_a = A_a + x * x^T
        self.A[action] += padded_context @ padded_context.T
        
        # Update b_a = b_a + r * x
        self.b[action] += reward * padded_context
    
    def reset(self):
        """
        Reset LinUCB parameters. Called when feature set changes at Level 1.
        """
        self.A = [np.identity(self.n_features) for _ in range(self.n_actions)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_actions)]
