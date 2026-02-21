"""Domain randomization for sim-to-real transfer."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class RandomizationRange:
    """Range for parameter randomization."""
    min_val: float
    max_val: float
    
    def sample(self) -> float:
        """Sample a random value from the range."""
        return random.uniform(self.min_val, self.max_val)


class DomainRandomization:
    """Domain randomization for improving sim-to-real transfer."""
    
    def __init__(
        self,
        friction_range: Tuple[float, float] = (0.1, 0.8),
        mass_range: Tuple[float, float] = (0.8, 1.2),
        noise_level: float = 0.1,
        delay_range: Tuple[float, float] = (0.0, 0.1),
        actuator_noise_range: Tuple[float, float] = (0.0, 0.05),
        sensor_noise_range: Tuple[float, float] = (0.0, 0.02),
        enable_visual_randomization: bool = True,
        enable_dynamics_randomization: bool = True,
        enable_sensor_randomization: bool = True
    ):
        """Initialize domain randomization.
        
        Args:
            friction_range: Range for friction coefficient randomization.
            mass_range: Range for mass randomization.
            noise_level: General noise level for actions.
            delay_range: Range for action delay randomization.
            actuator_noise_range: Range for actuator noise.
            sensor_noise_range: Range for sensor noise.
            enable_visual_randomization: Enable visual domain randomization.
            enable_dynamics_randomization: Enable dynamics randomization.
            enable_sensor_randomization: Enable sensor randomization.
        """
        self.friction_range = RandomizationRange(*friction_range)
        self.mass_range = RandomizationRange(*mass_range)
        self.noise_level = noise_level
        self.delay_range = RandomizationRange(*delay_range)
        self.actuator_noise_range = RandomizationRange(*actuator_noise_range)
        self.sensor_noise_range = RandomizationRange(*sensor_noise_range)
        
        self.enable_visual_randomization = enable_visual_randomization
        self.enable_dynamics_randomization = enable_dynamics_randomization
        self.enable_sensor_randomization = enable_sensor_randomization
        
        # Visual randomization parameters
        self.lighting_range = RandomizationRange(0.5, 2.0)
        self.texture_range = RandomizationRange(0.1, 2.0)
        self.color_range = RandomizationRange(0.5, 1.5)
        
    def randomize_dynamics(self) -> Dict[str, float]:
        """Randomize dynamics parameters.
        
        Returns:
            Dictionary of randomized dynamics parameters.
        """
        if not self.enable_dynamics_randomization:
            return {}
            
        return {
            "friction": self.friction_range.sample(),
            "mass": self.mass_range.sample(),
            "action_delay": self.delay_range.sample(),
            "actuator_noise": self.actuator_noise_range.sample(),
        }
    
    def randomize_sensors(self) -> Dict[str, float]:
        """Randomize sensor parameters.
        
        Returns:
            Dictionary of randomized sensor parameters.
        """
        if not self.enable_sensor_randomization:
            return {}
            
        return {
            "sensor_noise": self.sensor_noise_range.sample(),
            "sensor_delay": self.delay_range.sample(),
        }
    
    def randomize_visual(self) -> Dict[str, float]:
        """Randomize visual parameters.
        
        Returns:
            Dictionary of randomized visual parameters.
        """
        if not self.enable_visual_randomization:
            return {}
            
        return {
            "lighting": self.lighting_range.sample(),
            "texture": self.texture_range.sample(),
            "color": self.color_range.sample(),
        }
    
    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Add noise to action.
        
        Args:
            action: Original action.
            
        Returns:
            Action with added noise.
        """
        noise = np.random.normal(0, self.noise_level, action.shape)
        return action + noise
    
    def add_sensor_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add noise to sensor observations.
        
        Args:
            observation: Original observation.
            
        Returns:
            Observation with added noise.
        """
        if not self.enable_sensor_randomization:
            return observation
            
        noise_level = self.sensor_noise_range.sample()
        noise = np.random.normal(0, noise_level, observation.shape)
        return observation + noise


class ProgressiveDomainRandomization:
    """Progressive domain randomization that increases difficulty over time."""
    
    def __init__(
        self,
        base_randomization: DomainRandomization,
        max_episodes: int = 1000,
        difficulty_schedule: str = "linear"
    ):
        """Initialize progressive randomization.
        
        Args:
            base_randomization: Base domain randomization.
            max_episodes: Maximum number of episodes for full randomization.
            difficulty_schedule: Schedule type ('linear', 'exponential', 'step').
        """
        self.base_randomization = base_randomization
        self.max_episodes = max_episodes
        self.difficulty_schedule = difficulty_schedule
        
    def get_randomization_factor(self, episode: int) -> float:
        """Get randomization factor for current episode.
        
        Args:
            episode: Current episode number.
            
        Returns:
            Randomization factor between 0 and 1.
        """
        progress = min(episode / self.max_episodes, 1.0)
        
        if self.difficulty_schedule == "linear":
            return progress
        elif self.difficulty_schedule == "exponential":
            return progress ** 2
        elif self.difficulty_schedule == "step":
            return 1.0 if progress > 0.5 else 0.0
        else:
            raise ValueError(f"Unknown schedule: {self.difficulty_schedule}")
    
    def get_current_randomization(self, episode: int) -> DomainRandomization:
        """Get domain randomization for current episode.
        
        Args:
            episode: Current episode number.
            
        Returns:
            Scaled domain randomization.
        """
        factor = self.get_randomization_factor(episode)
        
        # Create a copy and scale the ranges
        current_randomization = DomainRandomization(
            friction_range=(
                self.base_randomization.friction_range.min_val,
                self.base_randomization.friction_range.min_val + 
                (self.base_randomization.friction_range.max_val - 
                 self.base_randomization.friction_range.min_val) * factor
            ),
            mass_range=(
                self.base_randomization.mass_range.min_val,
                self.base_randomization.mass_range.min_val + 
                (self.base_randomization.mass_range.max_val - 
                 self.base_randomization.mass_range.min_val) * factor
            ),
            noise_level=self.base_randomization.noise_level * factor,
            delay_range=(
                self.base_randomization.delay_range.min_val,
                self.base_randomization.delay_range.min_val + 
                (self.base_randomization.delay_range.max_val - 
                 self.base_randomization.delay_range.min_val) * factor
            ),
            actuator_noise_range=(
                self.base_randomization.actuator_noise_range.min_val,
                self.base_randomization.actuator_noise_range.min_val + 
                (self.base_randomization.actuator_noise_range.max_val - 
                 self.base_randomization.actuator_noise_range.min_val) * factor
            ),
            sensor_noise_range=(
                self.base_randomization.sensor_noise_range.min_val,
                self.base_randomization.sensor_noise_range.min_val + 
                (self.base_randomization.sensor_noise_range.max_val - 
                 self.base_randomization.sensor_noise_range.min_val) * factor
            ),
            enable_visual_randomization=self.base_randomization.enable_visual_randomization,
            enable_dynamics_randomization=self.base_randomization.enable_dynamics_randomization,
            enable_sensor_randomization=self.base_randomization.enable_sensor_randomization
        )
        
        return current_randomization
