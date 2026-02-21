"""Core utilities for sim-to-real transfer."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple, Union
import warnings


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def safe_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalize a vector to unit length.
    
    Args:
        vector: Input vector.
        eps: Small value to avoid division by zero.
        
    Returns:
        Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def clip_action(action: np.ndarray, action_space: Tuple[float, float]) -> np.ndarray:
    """Clip action to valid range.
    
    Args:
        action: Action to clip.
        action_space: (low, high) bounds for action space.
        
    Returns:
        Clipped action.
    """
    low, high = action_space
    return np.clip(action, low, high)


def compute_reality_gap(
    sim_performance: float, 
    real_performance: float
) -> Dict[str, float]:
    """Compute reality gap metrics.
    
    Args:
        sim_performance: Performance in simulation.
        real_performance: Performance in real world.
        
    Returns:
        Dictionary containing gap metrics.
    """
    if sim_performance == 0:
        warnings.warn("Simulation performance is zero, cannot compute relative gap")
        return {"absolute_gap": real_performance - sim_performance, "relative_gap": float('inf')}
    
    absolute_gap = real_performance - sim_performance
    relative_gap = absolute_gap / abs(sim_performance)
    
    return {
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "transfer_efficiency": real_performance / sim_performance if sim_performance > 0 else 0.0
    }


class SafetyLimits:
    """Safety limits for robotic control."""
    
    def __init__(
        self,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0,
        max_jerk: float = 5.0,
        max_force: float = 100.0,
        emergency_stop_threshold: float = 0.8
    ):
        """Initialize safety limits.
        
        Args:
            max_velocity: Maximum allowed velocity.
            max_acceleration: Maximum allowed acceleration.
            max_jerk: Maximum allowed jerk.
            max_force: Maximum allowed force.
            emergency_stop_threshold: Threshold for emergency stop.
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.max_force = max_force
        self.emergency_stop_threshold = emergency_stop_threshold
        
    def check_velocity_limit(self, velocity: np.ndarray) -> bool:
        """Check if velocity is within limits."""
        return np.all(np.abs(velocity) <= self.max_velocity)
    
    def check_force_limit(self, force: np.ndarray) -> bool:
        """Check if force is within limits."""
        return np.all(np.abs(force) <= self.max_force)
    
    def should_emergency_stop(self, error: float) -> bool:
        """Check if emergency stop should be triggered."""
        return error > self.emergency_stop_threshold
