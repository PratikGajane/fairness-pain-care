"""
Abstract base classes for Policy1 (Level 1) and Policy2 (Level 2).
Users can implement custom policies by inheriting from these classes.
"""

from abc import ABC, abstractmethod
import numpy as np

class Policy1Base(ABC):
    """
    Abstract base class for Level 1 policy (feature set selection).
    
    Level 1 selects which feature set to use from 255 possible feature sets
    (all non-empty subsets of 8 base features).
    """
    
    def __init__(self, n_actions):
        """
        Initialize Policy1.
        
        Args:
            n_actions (int): Number of actions (feature sets). Default is 255.
        """
        self.n_actions = n_actions
    
    @abstractmethod
    def select_action(self):
        """
        Select a feature set (action).
        
        Returns:
            int: Selected feature set ID (0 to n_actions-1)
        """
        pass
    
    @abstractmethod
    def update(self, action, reward):
        """
        Update policy parameters based on observed reward.
        
        Args:
            action (int): Feature set ID that was selected
            reward (float): Observed reward (continuous, in [0,1])
        """
        pass


class Policy2Base(ABC):
    """
    Abstract base class for Level 2 policy (recommendation selection).
    
    Level 2 selects which pain care recommendation to make given patient context
    and selected features from Level 1.
    """
    
    def __init__(self, n_actions, n_features):
        """
        Initialize Policy2.
        
        Args:
            n_actions (int): Number of actions (recommendations). Default is 3.
            n_features (int): Maximum number of features (8 base features)
        """
        self.n_actions = n_actions
        self.n_features = n_features
    
    @abstractmethod
    def select_action(self, context):
        """
        Select a recommendation given patient context.
        
        Args:
            context (np.ndarray): Patient context vector (selected features only)
        
        Returns:
            int: Selected recommendation ID (0 to n_actions-1)
        """
        pass
    
    @abstractmethod
    def update(self, context, action, reward):
        """
        Update policy parameters based on observed reward.
        
        Args:
            context (np.ndarray): Patient context vector that was used
            action (int): Recommendation ID that was selected
            reward (float): Observed reward
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset policy parameters. Called when feature set changes at Level 1.
        """
        pass
