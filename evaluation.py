"""
Clean and organized policy evaluation script for reinforcement learning models.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common import type_aliases
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, VecNormalize, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


class PolicyEvaluator:
    """Handles policy evaluation with support for zero-shot testing and rule-based policies."""
    
    def __init__(self):
        self.rule_policies = {
            'cartpole_ppo': self._cartpole_ppo_rule,
            'cartpole_a2c': self._cartpole_a2c_rule,
            'cartpole_dqn': self._cartpole_dqn_rule,
            'mountaincar_ppo': self._mountaincar_ppo_rule,
            'mountaincar_a2c': self._mountaincar_a2c_rule,
            'mountaincar_dqn': self._mountaincar_dqn_rule,
            'acrobot_ppo': self._acrobot_ppo_rule,
            'acrobot_a2c': self._acrobot_a2c_rule,
            'acrobot_dqn': self._acrobot_dqn_rule,
        }
    
    def evaluate_policy(
        self,
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        zero_shot: Optional[int] = None,
        zero_shot_set: Optional[List[int]] = None,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """Evaluate a trained policy."""
        return self._evaluate_policy_core(
            model, env, n_eval_episodes, zero_shot, zero_shot_set,
            deterministic, render, callback, reward_threshold,
            return_episode_rewards, warn, use_rule_policy=False
        )
    
    def evaluate_rule_policy(
        self,
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        rule_name: str,
        n_eval_episodes: int = 10,
        zero_shot: Optional[int] = None,
        zero_shot_set: Optional[List[int]] = None,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """Evaluate using a rule-based policy."""
        if rule_name not in self.rule_policies:
            raise ValueError(f"Rule policy '{rule_name}' not found. Available: {list(self.rule_policies.keys())}")
        
        return self._evaluate_policy_core(
            model, env, n_eval_episodes, zero_shot, zero_shot_set,
            deterministic, render, callback, reward_threshold,
            return_episode_rewards, warn, use_rule_policy=True, rule_name=rule_name
        )
    
    def _evaluate_policy_core(
        self,
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int,
        zero_shot: Optional[int],
        zero_shot_set: Optional[List[int]],
        deterministic: bool,
        render: bool,
        callback: Optional[Callable],
        reward_threshold: Optional[float],
        return_episode_rewards: bool,
        warn: bool,
        use_rule_policy: bool = False,
        rule_name: str = None,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """Core evaluation logic."""
        # Setup environment
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        
        # Check for Monitor wrapper
        is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
        if not is_monitor_wrapped and warn:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards.",
                UserWarning,
            )
        
        # Initialize tracking variables
        n_envs = env.num_envs
        episode_rewards = []
        episode_lengths = []
        episode_counts = np.zeros(n_envs, dtype="int")
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        
        # Reset environment and apply zero-shot modifications
        observations = env.reset()
        observations = self._apply_zero_shot(observations, zero_shot, zero_shot_set)
        
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        # Main evaluation loop
        while (episode_counts < episode_count_targets).any():
            if use_rule_policy:
                actions = self.rule_policies[rule_name](observations)
            else:
                actions, states = model.predict(
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
            
            new_observations, rewards, dones, infos = env.step(actions)
            new_observations = self._apply_zero_shot(new_observations, zero_shot, zero_shot_set)
            
            current_rewards += rewards
            current_lengths += 1
            
            # Process episode completions
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done
                    
                    if callback is not None:
                        callback(locals(), globals())
                    
                    if dones[i]:
                        if is_monitor_wrapped and "episode" in info.keys():
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                        
                        episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0
            
            observations = new_observations
            
            if render:
                env.render()
        
        # Calculate and return results
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
        
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward
    
    def _apply_zero_shot(self, observations, zero_shot: Optional[int], zero_shot_set: Optional[List[int]]):
        """Apply zero-shot modifications to observations."""
        if zero_shot is not None:
            observations[0][zero_shot] = 0
        
        if zero_shot_set is not None:
            for i in zero_shot_set:
                observations[0][i] = 0
        
        return observations
    
    # Rule-based policies
    def _cartpole_ppo_rule(self, observations):
        """CartPole PPO rule-based policy."""
        obs = observations[0]
        # Polynomial policy
        decision = ((-0.193*observations[0][0]) + 
                    (-0.523*observations[0][1]) + 
                    (-1*observations[0][2]) + 
                    (-1*observations[0][3]) +0.0014)
        return np.array([0 if decision > 0 else 1])
    
    def _cartpole_a2c_rule(self, observations):
        """CartPole A2C rule-based policy."""
        obs = observations[0]
        decision = ((-0.4875 * obs[0]) + (-0.9811 * obs[1]) + 
                   (-1 * obs[2]) + (-1 * obs[3]))
        return np.array([0 if decision > 0 else 1])
    
    def _cartpole_dqn_rule(self, observations):
        """CartPole DQN rule-based policy."""
        obs = observations[0]
        decision = ((-0.5 * obs[0]) + (-0.687 * obs[1]) + 
                   (-1.09 * obs[2]) + (-1 * obs[3]) - 0.018)
        return np.array([0 if decision > 0 else 1])
    
    def _mountaincar_ppo_rule(self, observations):
        """MountainCar PPO rule-based policy."""
        obs = observations[0]
        decision = (0.35 * obs[0]) - obs[1] - 0.3
        return np.array([0 if decision > 0 else 2])
    
    def _mountaincar_a2c_rule(self, observations):
        """MountainCar A2C rule-based policy."""
        obs = observations[0]
        decision = (0.003 * obs[0]) - obs[1] - 0.12
        return np.array([0 if decision > 0 else 2])
    
    def _mountaincar_dqn_rule(self, observations):
        """MountainCar DQN rule-based policy."""
        obs = observations[0]
        decision = (0.013 * obs[0]) - obs[1] + 0.0033
        return np.array([0 if decision > 0 else 2])
    
    def _acrobot_ppo_rule(self, observations):
        """Acrobot PPO rule-based policy."""
        obs = observations[0]
        decision = ((0.0642 * obs[0]) + (-0.4282 * obs[1]) + 
                   (-0.0432 * obs[2]) + (-0.0003 * obs[3]) + 
                   (0.3616 * obs[4]) + (-1 * obs[5]) - 0.0048)
        return np.array([0 if decision > 0 else 2])
    
    def _acrobot_a2c_rule(self, observations):
        """Acrobot A2C rule-based policy."""
        obs = observations[0]
        decision = ((0.0262 * obs[0]) + (0.1928 * obs[1]) + 
                   (0.0131 * obs[2]) + (0.5353 * obs[3]) + 
                   (1.1534 * obs[4]) + (-1 * obs[5]) + 0.0008)
        return np.array([0 if decision > 0 else 2])
    
    def _acrobot_dqn_rule(self, observations):
        """Acrobot DQN rule-based policy."""
        obs = observations[0]
        decision = ((1.7925 * obs[0]) + (-1.8633 * obs[1]) + 
                   (-1.4589 * obs[2]) + (0.2775 * obs[3]) + 
                   (-0.6890 * obs[4]) + (-1 * obs[5]) - 0.4376)
        return np.array([0 if decision > 0 else 2])


class ExperimentRunner:
    """Handles loading models and running experiments."""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.model_classes = {
            'ppo': PPO,
            'a2c': A2C,
            'dqn': DQN
        }
    
    def load_model(self, model_type: str, model_path: str):
        """Load a trained model."""
        if model_type.lower() not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_classes[model_type.lower()]
        return model_class.load(model_path)
    
    def create_environment(self, env_name: str, n_envs: int = 1, normalize_path: Optional[str] = None):
        """Create and configure environment."""
        env = make_vec_env(lambda: gym.make(env_name), n_envs=n_envs)
        
        if normalize_path:
            env = VecNormalize.load(normalize_path, env)
        
        return env
    
    def run_comparison_experiment(self, model, env, rule_name: str, n_episodes: int = 10):
        """Run comparison between rule-based and trained policy."""
        print('-----------------------------------------------------------------------')
        
        # Evaluate rule-based policy
        print('\nRule-based policy:')
        rule_mean, rule_std = self.evaluator.evaluate_rule_policy(
            model, env, rule_name, n_eval_episodes=n_episodes
        )
        print(f"Mean Reward: {rule_mean:.2f}")
        print(f"Standard Deviation: {rule_std:.2f}")
        
        # Evaluate trained policy
        print('\nTrained policy:')
        trained_mean, trained_std = self.evaluator.evaluate_policy(
            model, env, n_eval_episodes=n_episodes
        )
        print(f"Mean Reward: {trained_mean:.2f}")
        print(f"Standard Deviation: {trained_std:.2f}")
        
        return {
            'rule': {'mean': rule_mean, 'std': rule_std},
            'trained': {'mean': trained_mean, 'std': trained_std}
        }
    
    def run_zero_shot_experiment(self, model, env, n_features: int, n_episodes: int = 100):
        """Run zero-shot experiments by masking individual features."""
        print(f"\nZero-shot analysis for {n_features} features:")
        results = {}
        
        for i in range(n_features):
            mean_reward, std_reward = self.evaluator.evaluate_policy(
                model, env, n_eval_episodes=n_episodes, zero_shot=i
            )
            results[f'feature_{i}'] = {'mean': mean_reward, 'std': std_reward}
            print(f'Feature {i} masked - Mean: {mean_reward:.2f}, Std: {std_reward:.2f}')
        
        return results


# Predefined model paths
MODEL_PATHS = {
    # CartPole models
    'cartpole_ppo': './model/cartpole_ppo',
    'cartpole_a2c': './model/cartpole_a2c',
    'cartpole_dqn': './model/cartpole_dqn',
    
    # MountainCar models
    'mountaincar_ppo': './model/mountaincar_ppo',
    'mountaincar_a2c': './model/mountaincar_a2c',
    'mountaincar_dqn': './model/mountaincar_dqn',
    
    # Acrobot models
    'acrobot_ppo': './model/acrobot_ppo',
    'acrobot_a2c': './model/acrobot_a2c',
    'acrobot_dqn': './model/acrobot_dqn',
}

# Predefined normalization paths
NORMALIZE_PATHS = {
    'mountaincar_ppo': './vec_env/vec_normalize_ppo_mountaincar.pkl',
    'mountaincar_a2c': './vec_env/vec_normalize_a2c_mountaincar.pkl',
    'acrobot_ppo': './vec_env/vec_normalize_ppo_acrobot.pkl',
    'acrobot_a2c': './vec_env/vec_normalize_a2c_acrobot.pkl',
}

# Environment configurations
ENV_CONFIGS = {
    'cartpole': {
        'env_name': 'CartPole-v1',
        'n_features': 4,
        'rule_names': ['cartpole_ppo', 'cartpole_a2c', 'cartpole_dqn']
    },
    'mountaincar': {
        'env_name': 'MountainCar-v0',
        'n_features': 2,
        'rule_names': ['mountaincar_ppo', 'mountaincar_a2c', 'mountaincar_dqn']
    },
    'acrobot': {
        'env_name': 'Acrobot-v1',
        'n_features': 6,
        'rule_names': ['acrobot_ppo', 'acrobot_a2c', 'acrobot_dqn']
    }
}


def main(
    model_key: str = 'cartpole_ppo',
    n_episodes: int = 10,
    run_zero_shot: bool = False,
    use_custom_path: bool = False,
    custom_model_path: Optional[str] = None,
    custom_normalize_path: Optional[str] = None
):
    """
    Main execution function.
    
    Args:
        model_key: Key for predefined model (e.g., 'cartpole_ppo', 'mountaincar_a2c')
        n_episodes: Number of episodes for evaluation
        run_zero_shot: Whether to run zero-shot experiments
        use_custom_path: If True, use custom_model_path instead of predefined
        custom_model_path: Custom path to model (only used if use_custom_path=True)
        custom_normalize_path: Custom normalization path (overrides predefined)
    """
    runner = ExperimentRunner()
    
    try:
        # Determine model path
        if use_custom_path:
            if custom_model_path is None:
                raise ValueError("custom_model_path must be specified when use_custom_path=True")
            model_path = custom_model_path
        else:
            if model_key not in MODEL_PATHS:
                raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_PATHS.keys())}")
            model_path = MODEL_PATHS[model_key]
        
        # Extract model type and environment from key
        env_type = model_key.split('_')[0]  # e.g., 'cartpole' from 'cartpole_ppo'
        model_type = model_key.split('_')[1]  # e.g., 'ppo' from 'cartpole_ppo'
        
        if env_type not in ENV_CONFIGS:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        env_config = ENV_CONFIGS[env_type]
        env_name = env_config['env_name']
        n_features = env_config['n_features']
        
        # Determine normalization path
        normalize_path = custom_normalize_path or NORMALIZE_PATHS.get(model_key)
        
        # Load model and create environment
        print(f"Loading {model_type.upper()} model for {env_type}: {model_key}")
        print(f"Model path: {model_path}")
        model = runner.load_model(model_type, model_path)
        
        print(f"Creating environment: {env_name}")
        if normalize_path:
            print(f"Using normalization: {normalize_path}")
        env = runner.create_environment(env_name, normalize_path=normalize_path)
        
        # Run comparison experiment
        print(f"Running comparison experiment with rule: {model_key}")
        results = runner.run_comparison_experiment(
            model, env, model_key, n_episodes
        )
        
        # Optionally run zero-shot experiment
        if run_zero_shot:
            print(f"Running zero-shot experiment with {n_features} features")
            zero_shot_results = runner.run_zero_shot_experiment(model, env, n_features)
        
        return results
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    """
    model_key examples:
    - 'cartpole_ppo/a2c/dqn'
    - 'mountaincar_ppo/a2c/dqn'
    - 'acrobot_ppo/a2c/dqn'
    """
    main(
        model_key='cartpole_ppo',
        n_episodes=10,
        run_zero_shot=False
    )
    