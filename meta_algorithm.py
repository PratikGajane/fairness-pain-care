"""
Meta-algorithm for fairness-aware pain care recommendations.
Implements two-level nested decision-making strategy.
"""

import numpy as np
import csv


class MetaAlgorithm:
    """
    Meta-algorithm for selecting feature sets and pain care recommendations.
    
    Implements a two-level nested strategy:
    - Level 1: Selects feature sets using Policy1
    - Level 2: Selects recommendations using Policy2 with selected features
    
    This algorithm can use any contextual bandit algorithm as black-box
    for Policy1 and Policy2 by passing implementations that inherit from
    Policy1Base and Policy2Base.
    """
    
    def __init__(self, policy1, policy2, environment, 
                 fairness_weight=0.5, demographic_attribute='gender',
                 save_metrics=True, n_metric_points=100):
        """
        Initialize meta-algorithm.
        
        Args:
            policy1: Policy for Level 1 (feature set selection).
                     Must inherit from Policy1Base.
            policy2: Policy for Level 2 (recommendation selection).
                     Must inherit from Policy2Base.
            environment: Environment that provides contexts and rewards.
            fairness_weight (float): Weight for fairness in reward computation.
                                    reward = (1-w)*utility + w*fairness
                                    Default is 0.5 (equal weights).
            demographic_attribute (str): Demographic attribute for fairness.
                                        One of 'gender', 'race', 'age'.
                                        Default is 'gender'.
            save_metrics (bool): Whether to save detailed fairness metrics.
                                Default is True.
            n_metric_points (int): Number of equally spaced timesteps to save metrics.
                                  Default is 100.
        """
        self.policy1 = policy1
        self.policy2 = policy2
        self.environment = environment
        self.fairness_weight = fairness_weight
        self.utility_weight = 1.0 - fairness_weight
        self.demographic_attribute = demographic_attribute
        self.save_metrics = save_metrics
        self.n_metric_points = n_metric_points
        
        # Track history
        self.feature_set_history = []
        self.recommendation_history = []
        self.reward_history = []
        self.utility_history = []
        self.fairness_history = []
        
        # Detailed metrics for saving
        self.detailed_metrics = []
    
    def run(self, n_timesteps):
        """
        Run meta-algorithm for specified number of time steps.
        
        Args:
            n_timesteps (int): Number of time steps to run
        
        Returns:
            dict: Results containing histories of selections and rewards
        """
        print(f"Running meta-algorithm for {n_timesteps} time steps...")
        print(f"Fairness weight: {self.fairness_weight}, Utility weight: {self.utility_weight}")
        print(f"Demographic attribute: {self.demographic_attribute}")
        print("-" * 60)
        
        # Determine which timesteps to save detailed metrics
        if self.save_metrics:
            metric_timesteps = set(np.linspace(0, n_timesteps - 1, self.n_metric_points, dtype=int))
        else:
            metric_timesteps = set()
        
        for t in range(n_timesteps):
            # Progress indicator
            if (t + 1) % 5000 == 0:
                print(f"Time step {t + 1}/{n_timesteps}")
            
            # ===== LEVEL 1: Selection of Feature Sets =====
            # Use Policy1 to select a feature set
            feature_set_id = self.policy1.select_action()
            
            # Convert feature set ID to actual feature indices
            feature_indices = self.environment.feature_set_id_to_indices(feature_set_id)
            
            # ===== LEVEL 2: Selection of Pain Care Recommendations =====
            # Get patient context with selected features
            context = self.environment.get_context(t, feature_indices)
            
            # Use Policy2 to select a recommendation
            recommendation = self.policy2.select_action(context)
            
            # Apply recommendation and observe reward from environment
            instantaneous_reward = self.environment.get_reward(t, feature_indices, recommendation)
            
            # Compute utility (average reward so far)
            utility = self.environment.compute_utility()
            
            # Compute fairness (1 - disparity in average rewards)
            fairness = self.environment.compute_fairness(self.demographic_attribute)
            
            # Compute combined reward for Policy1
            # reward = (1 - fairness_weight) * utility + fairness_weight * fairness
            combined_reward = self.utility_weight * utility + self.fairness_weight * fairness
            
            # ===== UPDATE POLICIES =====
            # Update Policy2 with instantaneous reward
            self.policy2.update(context, recommendation, instantaneous_reward)
            
            # Update Policy1 with combined reward (utility + fairness)
            self.policy1.update(feature_set_id, combined_reward)
            
            # Record history
            self.feature_set_history.append(feature_set_id)
            self.recommendation_history.append(recommendation)
            self.reward_history.append(instantaneous_reward)
            self.utility_history.append(utility)
            self.fairness_history.append(fairness)
            
            # Save detailed metrics at specified timesteps
            if t in metric_timesteps:
                self._save_detailed_metrics(t)
        
        print("-" * 60)
        print("Meta-algorithm completed!")
        print(f"Final utility: {self.utility_history[-1]:.4f}")
        print(f"Final fairness: {self.fairness_history[-1]:.4f}")
        
        # Save detailed metrics to CSV
        if self.save_metrics and len(self.detailed_metrics) > 0:
            self._write_metrics_to_csv()
        
        # Return results
        results = {
            'feature_sets': self.feature_set_history,
            'recommendations': self.recommendation_history,
            'rewards': self.reward_history,
            'utility': self.utility_history,
            'fairness': self.fairness_history
        }
        
        return results
    
    def analyze_feature_set_selection(self, pareto_optimal_id=None):
        """
        Analyze how often different feature sets were selected.
        
        Args:
            pareto_optimal_id (int): ID of Pareto-optimal feature set (if known)
        
        Returns:
            dict: Analysis results
        """
        feature_set_counts = {}
        for fs_id in self.feature_set_history:
            feature_set_counts[fs_id] = feature_set_counts.get(fs_id, 0) + 1
        
        total_selections = len(self.feature_set_history)
        
        print("\n" + "=" * 60)
        print("FEATURE SET SELECTION ANALYSIS")
        print("=" * 60)
        print(f"Total time steps: {total_selections}")
        print(f"Unique feature sets selected: {len(feature_set_counts)}")
        
        # Most frequently selected feature sets
        sorted_fs = sorted(feature_set_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 most selected feature sets:")
        for i, (fs_id, count) in enumerate(sorted_fs[:5]):
            percentage = (count / total_selections) * 100
            print(f"  {i+1}. Feature Set {fs_id}: {count} times ({percentage:.2f}%)")
        
        # Pareto-optimal selection rate
        if pareto_optimal_id is not None:
            pareto_count = feature_set_counts.get(pareto_optimal_id, 0)
            pareto_percentage = (pareto_count / total_selections) * 100
            
            # Expected rate under uniform random selection
            expected_rate = (1 / self.policy1.n_actions) * 100
            
            print(f"\nPareto-optimal feature set (ID {pareto_optimal_id}):")
            print(f"  Selected: {pareto_count} times ({pareto_percentage:.2f}%)")
            print(f"  Expected (uniform random): {expected_rate:.4f}%")
            
            if pareto_percentage > expected_rate:
                print(f"  ✓ Selected {pareto_percentage/expected_rate:.1f}x more than random")
            
        print("=" * 60)
        
        return {
            'counts': feature_set_counts,
            'total_selections': total_selections,
            'unique_feature_sets': len(feature_set_counts)
        }
    
    def _save_detailed_metrics(self, timestep):
        """
        Save detailed fairness metrics for current timestep.
        
        Args:
            timestep (int): Current timestep
        """
        # Get average rewards for all demographic groups
        history = self.environment.history
        
        # Gender
        avg_male = np.mean(history['male']['rewards']) if history['male']['counts'] > 0 else 0.0
        avg_female = np.mean(history['female']['rewards']) if history['female']['counts'] > 0 else 0.0
        fairness_gender = 1.0 - abs(avg_male - avg_female)
        
        # Race
        avg_white = np.mean(history['white']['rewards']) if history['white']['counts'] > 0 else 0.0
        avg_nonwhite = np.mean(history['non_white']['rewards']) if history['non_white']['counts'] > 0 else 0.0
        fairness_race = 1.0 - abs(avg_white - avg_nonwhite)
        
        # Age
        avg_under65 = np.mean(history['age_under_65']['rewards']) if history['age_under_65']['counts'] > 0 else 0.0
        avg_65plus = np.mean(history['age_65_plus']['rewards']) if history['age_65_plus']['counts'] > 0 else 0.0
        fairness_age = 1.0 - abs(avg_under65 - avg_65plus)
        
        # Store metrics
        self.detailed_metrics.append({
            'timestep': timestep + 1,  # 1-indexed for readability
            'avg_reward_male': avg_male,
            'avg_reward_female': avg_female,
            'fairness_gender': fairness_gender,
            'avg_reward_white': avg_white,
            'avg_reward_nonwhite': avg_nonwhite,
            'fairness_race': fairness_race,
            'avg_reward_under65': avg_under65,
            'avg_reward_65plus': avg_65plus,
            'fairness_age': fairness_age
        })
    
    def _write_metrics_to_csv(self, filename='fairness_metrics_timesteps.csv'):
        """
        Write detailed fairness metrics to CSV file.
        
        Args:
            filename (str): Output filename. Default is 'fairness_metrics_timesteps.csv'
        """
        if len(self.detailed_metrics) == 0:
            return
        
        # Define CSV headers
        headers = [
            'timestep',
            'avg_reward_male', 'avg_reward_female', 'fairness_gender',
            'avg_reward_white', 'avg_reward_nonwhite', 'fairness_race',
            'avg_reward_under65', 'avg_reward_65plus', 'fairness_age'
        ]
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.detailed_metrics)
        
        print(f"\nDetailed fairness metrics saved to '{filename}'")
        print(f"Total timesteps recorded: {len(self.detailed_metrics)}")