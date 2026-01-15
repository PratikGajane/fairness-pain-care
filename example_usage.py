"""
Example usage of the meta-algorithm with Thompson Sampling and LinUCB.

This script demonstrates:
1. How to run the meta-algorithm with default policies
2. How to implement a custom Policy2 (LinUCB example provided)
3. How to analyze results
"""

import numpy as np
import matplotlib.pyplot as plt

# Import components
from policy_base import Policy1Base, Policy2Base
from thompson_sampling import ThompsonSampling
from linucb_policy import LinUCB
from mock_environment import MockEnvironment
from meta_algorithm import MetaAlgorithm


def example_1_basic_usage():
    """
    Example 1: Basic usage with Thompson Sampling + LinUCB
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create environment
    env = MockEnvironment(n_timesteps=50000, n_features=8, n_actions=3, random_seed=42)
    
    # Create Policy1 (Thompson Sampling for feature set selection)
    policy1 = ThompsonSampling(
        n_actions=255,        # 255 non-empty feature sets
        alpha_prior=1,        # Can be changed
        beta_prior=5,         # Can be changed
        random_seed=42
    )
    
    # Create Policy2 (LinUCB for recommendation selection)
    policy2 = LinUCB(
        n_actions=3,          # 3 pain care recommendations
        n_features=8,         # 8 base features
        alpha=1.0,            # Exploration parameter
        random_seed=42
    )
    
    # Create meta-algorithm
    meta_algo = MetaAlgorithm(
        policy1=policy1,
        policy2=policy2,
        environment=env,
        fairness_weight=0.5,           # Equal weight for fairness and utility
        demographic_attribute='gender', # Can be 'gender', 'race', or 'age'
        save_metrics=True,             # Save detailed fairness metrics
        n_metric_points=100            # Save metrics at 100 equally spaced timesteps
    )
    
    # Run algorithm
    results = meta_algo.run(n_timesteps=50000)
    
    # Analyze feature set selection
    # In real experiments, you would know the Pareto-optimal feature set ID
    # Here we just analyze the most selected one
    meta_algo.analyze_feature_set_selection(pareto_optimal_id=None)
    
    # Plot results
    #plot_results(results)
    
    return results


def example_2_custom_policy2():
    """
    Example 2: Implementing a custom Policy2
    
    This shows how users can implement their own contextual bandit algorithms
    (NeuralUCB, NN-UCB, Neural Thompson Sampling, etc.) by inheriting from Policy2Base.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Policy2 Implementation")
    print("=" * 60)
    
    # Custom Policy2 implementation (simple epsilon-greedy for demonstration)
    class CustomEpsilonGreedy(Policy2Base):
        """
        Example custom Policy2: Epsilon-Greedy
        
        This is just for demonstration. Users can implement more sophisticated
        algorithms like NeuralUCB, NN-UCB, Neural Thompson Sampling, etc.
        """
        
        def __init__(self, n_actions=3, n_features=8, epsilon=0.1, random_seed=None):
            super().__init__(n_actions, n_features)
            self.epsilon = epsilon
            
            # Simple Q-value estimates
            self.Q = np.zeros(n_actions)
            self.counts = np.zeros(n_actions)
            
            if random_seed is not None:
                np.random.seed(random_seed)
        
        def select_action(self, context):
            """Select action using epsilon-greedy"""
            if np.random.rand() < self.epsilon:
                # Explore: random action
                return np.random.randint(self.n_actions)
            else:
                # Exploit: best action
                return np.argmax(self.Q)
        
        def update(self, context, action, reward):
            """Update Q-values"""
            self.counts[action] += 1
            # Running average
            self.Q[action] += (reward - self.Q[action]) / self.counts[action]
        
        def reset(self):
            """Reset Q-values"""
            self.Q = np.zeros(self.n_actions)
            self.counts = np.zeros(self.n_actions)
    
    # Create environment
    env = MockEnvironment(n_timesteps=10000, random_seed=42)
    
    # Create Policy1 (Thompson Sampling)
    policy1 = ThompsonSampling(n_actions=255, random_seed=42)
    
    # Create custom Policy2 (Epsilon-Greedy)
    policy2 = CustomEpsilonGreedy(n_actions=3, n_features=8, epsilon=0.1, random_seed=42)
    
    # Create and run meta-algorithm
    meta_algo = MetaAlgorithm(
        policy1=policy1,
        policy2=policy2,
        environment=env,
        fairness_weight=0.5,
        demographic_attribute='gender'
    )
    
    results = meta_algo.run(n_timesteps=10000)
    
    print("\nThis demonstrates how to implement custom Policy2 algorithms.")
    print("Users can similarly implement NeuralUCB, NN-UCB, Neural TS, etc.")
    
    return results


def plot_results(results):
    """
    Plot utility and fairness over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot utility
    axes[0].plot(results['utility'], label='Utility', color='blue', alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Utility')
    axes[0].set_title('Utility over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot fairness
    axes[1].plot(results['fairness'], label='Fairness', color='green', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Fairness')
    axes[1].set_title('Fairness over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nResults plot saved as 'results.png'")


def example_3_different_fairness_weights():
    """
    Example 3: Running with different fairness-utility weight combinations
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Fairness-Utility Weights")
    print("=" * 60)
    
    # Test different weight combinations
    weight_combinations = [
        (0.3, 0.7),  # 30% fairness, 70% utility
        (0.5, 0.5),  # 50% fairness, 50% utility
        (0.7, 0.3),  # 70% fairness, 30% utility
    ]
    
    results_dict = {}
    
    for fairness_w, utility_w in weight_combinations:
        print(f"\n--- Running with weights: Fairness={fairness_w}, Utility={utility_w} ---")
        
        # Create fresh environment and policies for each run
        env = MockEnvironment(n_timesteps=10000, random_seed=42)
        policy1 = ThompsonSampling(n_actions=255, random_seed=42)
        policy2 = LinUCB(n_actions=3, n_features=8, random_seed=42)
        
        meta_algo = MetaAlgorithm(
            policy1=policy1,
            policy2=policy2,
            environment=env,
            fairness_weight=fairness_w,
            demographic_attribute='gender',
            save_metrics=False  # Disable for multiple runs to avoid overwriting
        )
        
        results = meta_algo.run(n_timesteps=10000)
        results_dict[f"{fairness_w}:{utility_w}"] = results
        
        print(f"Final utility: {results['utility'][-1]:.4f}")
        print(f"Final fairness: {results['fairness'][-1]:.4f}")
    
    return results_dict


if __name__ == "__main__":
    # Run examples
    
    # Example 1: Basic usage
    results1 = example_1_basic_usage()
    
    # Example 2: Custom Policy2
    results2 = example_2_custom_policy2()
    
    # Example 3: Different fairness weights
    results3 = example_3_different_fairness_weights()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nTo use this code with real clinical data:")
    print("1. Replace MockEnvironment with your data loader")
    print("2. Implement advanced Policy2 algorithms (NeuralUCB, etc.)")
    print("3. Run experiments with different fairness weights")
    print("4. Analyze Pareto-optimal feature set selection")
    print("=" * 60)