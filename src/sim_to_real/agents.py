"""Reinforcement learning agents for sim-to-real transfer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
from tqdm import tqdm

from .utils import set_seed, get_device, SafetyLimits
from .domain_randomization import DomainRandomization


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    algorithm: str = "PPO"  # PPO, SAC, TD3
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    target_update_interval: int = 1
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: float = 0.0
    clip_range: float = 0.2
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False


class ResidualPolicy(nn.Module):
    """Residual policy that adds corrections to a base controller."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """Initialize residual policy.
        
        Args:
            observation_dim: Dimension of observation space.
            action_dim: Dimension of action space.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function.
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            observations: Input observations.
            
        Returns:
            Residual actions.
        """
        return self.network(observations)


class ResidualController:
    """Residual controller combining classical control with learned corrections."""
    
    def __init__(
        self,
        policy: ResidualPolicy,
        base_controller: Optional[Any] = None,
        residual_scale: float = 0.1
    ):
        """Initialize residual controller.
        
        Args:
            policy: Learned residual policy.
            base_controller: Classical base controller.
            residual_scale: Scaling factor for residual actions.
        """
        self.policy = policy
        self.base_controller = base_controller
        self.residual_scale = residual_scale
        
    def get_action(
        self, 
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """Get action combining base controller and residual policy.
        
        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic action.
            
        Returns:
            Combined action.
        """
        # Get base controller action
        if self.base_controller is not None:
            base_action = self.base_controller.get_action(observation)
        else:
            base_action = np.zeros(self.policy.action_dim)
            
        # Get residual action
        obs_tensor = torch.FloatTensor(self._flatten_observation(observation)).unsqueeze(0)
        
        with torch.no_grad():
            residual_action = self.policy(obs_tensor).squeeze(0).numpy()
            
        # Combine actions
        total_action = base_action + self.residual_scale * residual_action
        
        return total_action
        
    def _flatten_observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten observation dictionary to vector.
        
        Args:
            observation: Observation dictionary.
            
        Returns:
            Flattened observation vector.
        """
        return np.concatenate([
            observation["position"],
            observation["velocity"],
            observation["goal_position"],
            observation["distance_to_goal"]
        ])


class PPOAgent:
    """PPO agent for sim-to-real transfer."""
    
    def __init__(
        self,
        env: gym.Env,
        config: AgentConfig,
        safety_limits: Optional[SafetyLimits] = None,
        use_wandb: bool = False
    ):
        """Initialize PPO agent.
        
        Args:
            env: Training environment.
            config: Agent configuration.
            safety_limits: Safety limits for control.
            use_wandb: Whether to use Weights & Biases logging.
        """
        self.env = env
        self.config = config
        self.safety_limits = safety_limits or SafetyLimits()
        self.use_wandb = use_wandb
        
        # Initialize PPO
        self.agent = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=2048,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=config.use_sde,
            sde_sample_freq=config.sde_sample_freq,
            use_sde_at_warmup=config.use_sde_at_warmup,
            normalize_advantage=config.normalize_advantage,
            tensorboard_log="./logs/",
            device=get_device(),
            verbose=1
        )
        
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """Train the agent.
        
        Args:
            total_timesteps: Total training timesteps.
            callback: Training callback.
        """
        if self.use_wandb:
            wandb.init(project="sim-to-real-robotics", config=self.config.__dict__)
            
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        if self.use_wandb:
            wandb.finish()
            
    def predict(
        self, 
        observation: Dict[str, np.ndarray], 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Predict action for given observation.
        
        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic action.
            
        Returns:
            Action and optional state value.
        """
        return self.agent.predict(observation, deterministic=deterministic)
        
    def save(self, path: str) -> None:
        """Save the trained agent.
        
        Args:
            path: Path to save the model.
        """
        self.agent.save(path)
        
    def load(self, path: str) -> None:
        """Load a trained agent.
        
        Args:
            path: Path to load the model from.
        """
        self.agent = PPO.load(path, env=self.env)


class SACAgent:
    """SAC agent for sim-to-real transfer."""
    
    def __init__(
        self,
        env: gym.Env,
        config: AgentConfig,
        safety_limits: Optional[SafetyLimits] = None,
        use_wandb: bool = False
    ):
        """Initialize SAC agent.
        
        Args:
            env: Training environment.
            config: Agent configuration.
            safety_limits: Safety limits for control.
            use_wandb: Whether to use Weights & Biases logging.
        """
        self.env = env
        self.config = config
        self.safety_limits = safety_limits or SafetyLimits()
        self.use_wandb = use_wandb
        
        # Initialize SAC
        self.agent = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=1000,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            ent_coef=config.ent_coef,
            target_update_interval=config.target_update_interval,
            target_entropy="auto",
            use_sde=config.use_sde,
            sde_sample_freq=config.sde_sample_freq,
            use_sde_at_warmup=config.use_sde_at_warmup,
            tensorboard_log="./logs/",
            device=get_device(),
            verbose=1
        )
        
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """Train the agent.
        
        Args:
            total_timesteps: Total training timesteps.
            callback: Training callback.
        """
        if self.use_wandb:
            wandb.init(project="sim-to-real-robotics", config=self.config.__dict__)
            
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        if self.use_wandb:
            wandb.finish()
            
    def predict(
        self, 
        observation: Dict[str, np.ndarray], 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Predict action for given observation.
        
        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic action.
            
        Returns:
            Action and optional state value.
        """
        return self.agent.predict(observation, deterministic=deterministic)
        
    def save(self, path: str) -> None:
        """Save the trained agent.
        
        Args:
            path: Path to save the model.
        """
        self.agent.save(path)
        
    def load(self, path: str) -> None:
        """Load a trained agent.
        
        Args:
            path: Path to load the model from.
        """
        self.agent = SAC.load(path, env=self.env)


class SimToRealCallback(BaseCallback):
    """Callback for sim-to-real transfer monitoring."""
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        verbose: int = 1
    ):
        """Initialize callback.
        
        Args:
            eval_env: Environment for evaluation.
            eval_freq: Evaluation frequency.
            n_eval_episodes: Number of episodes for evaluation.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_returns = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True
        
    def _evaluate(self) -> None:
        """Evaluate the current policy."""
        episode_returns = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_return += reward
                done = terminated or truncated
                
            episode_returns.append(episode_return)
            
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        self.eval_returns.append(mean_return)
        
        if self.verbose > 0:
            print(f"Eval return: {mean_return:.2f} ± {std_return:.2f}")
            
        # Log to tensorboard
        self.logger.record("eval/mean_return", mean_return)
        self.logger.record("eval/std_return", std_return)
        
        # Log to wandb if available
        if hasattr(self, "wandb_run"):
            self.wandb_run.log({
                "eval/mean_return": mean_return,
                "eval/std_return": std_return,
                "timestep": self.num_timesteps
            })
