"""Tests for sim-to-real transfer package."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from sim_to_real.utils import set_seed, get_device, safe_normalize, clip_action, compute_reality_gap, SafetyLimits
from sim_to_real.domain_randomization import DomainRandomization, ProgressiveDomainRandomization
from sim_to_real.agents import ResidualPolicy, ResidualController
from sim_to_real.evaluation import EvaluationMetrics, SimToRealEvaluator, Leaderboard


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        assert True  # If no exception is raised, test passes
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_safe_normalize(self):
        """Test safe normalization."""
        # Test normal vector
        vector = np.array([3.0, 4.0])
        normalized = safe_normalize(vector)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # Test zero vector
        zero_vector = np.array([0.0, 0.0])
        normalized_zero = safe_normalize(zero_vector)
        np.testing.assert_array_equal(normalized_zero, zero_vector)
    
    def test_clip_action(self):
        """Test action clipping."""
        action = np.array([2.0, -3.0])
        clipped = clip_action(action, (-1.0, 1.0))
        expected = np.array([1.0, -1.0])
        np.testing.assert_array_equal(clipped, expected)
    
    def test_compute_reality_gap(self):
        """Test reality gap computation."""
        sim_perf = 100.0
        real_perf = 80.0
        
        gap = compute_reality_gap(sim_perf, real_perf)
        
        assert gap["absolute_gap"] == -20.0
        assert gap["relative_gap"] == -0.2
        assert gap["transfer_efficiency"] == 0.8
    
    def test_safety_limits(self):
        """Test safety limits."""
        limits = SafetyLimits(max_velocity=1.0, max_force=100.0)
        
        # Test velocity limit
        assert limits.check_velocity_limit(np.array([0.5, 0.8]))
        assert not limits.check_velocity_limit(np.array([1.5, 0.8]))
        
        # Test force limit
        assert limits.check_force_limit(np.array([50.0, 80.0]))
        assert not limits.check_force_limit(np.array([150.0, 80.0]))
        
        # Test emergency stop
        assert limits.should_emergency_stop(0.9)
        assert not limits.should_emergency_stop(0.5)


class TestDomainRandomization:
    """Test domain randomization."""
    
    def test_domain_randomization_init(self):
        """Test domain randomization initialization."""
        dr = DomainRandomization()
        assert dr.friction_range.min_val == 0.1
        assert dr.friction_range.max_val == 0.8
        assert dr.noise_level == 0.1
    
    def test_randomize_dynamics(self):
        """Test dynamics randomization."""
        dr = DomainRandomization()
        params = dr.randomize_dynamics()
        
        assert "friction" in params
        assert "mass" in params
        assert dr.friction_range.min_val <= params["friction"] <= dr.friction_range.max_val
        assert dr.mass_range.min_val <= params["mass"] <= dr.mass_range.max_val
    
    def test_add_action_noise(self):
        """Test action noise addition."""
        dr = DomainRandomization(noise_level=0.1)
        action = np.array([1.0, 0.0])
        noisy_action = dr.add_action_noise(action)
        
        assert noisy_action.shape == action.shape
        assert not np.array_equal(noisy_action, action)
    
    def test_progressive_randomization(self):
        """Test progressive domain randomization."""
        base_dr = DomainRandomization()
        pdr = ProgressiveDomainRandomization(base_dr, max_episodes=1000)
        
        # Test linear schedule
        factor_0 = pdr.get_randomization_factor(0)
        factor_500 = pdr.get_randomization_factor(500)
        factor_1000 = pdr.get_randomization_factor(1000)
        
        assert factor_0 == 0.0
        assert factor_500 == 0.5
        assert factor_1000 == 1.0


class TestAgents:
    """Test agent components."""
    
    def test_residual_policy(self):
        """Test residual policy."""
        policy = ResidualPolicy(observation_dim=4, action_dim=2)
        
        obs = torch.randn(1, 4)
        action = policy(obs)
        
        assert action.shape == (1, 2)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    
    def test_residual_controller(self):
        """Test residual controller."""
        policy = ResidualPolicy(observation_dim=4, action_dim=2)
        controller = ResidualController(policy)
        
        observation = {
            "position": np.array([1.0, 2.0]),
            "velocity": np.array([0.5, -0.3]),
            "goal_position": np.array([5.0, 5.0]),
            "distance_to_goal": np.array([3.0])
        }
        
        action = controller.get_action(observation)
        
        assert action.shape == (2,)
        assert isinstance(action, np.ndarray)


class TestEvaluation:
    """Test evaluation components."""
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        metrics = EvaluationMetrics(
            success_rate=0.8,
            mean_return=100.0,
            std_return=10.0,
            mean_episode_length=50.0,
            std_episode_length=5.0,
            mean_distance_to_goal=0.5,
            std_distance_to_goal=0.1,
            control_effort=20.0,
            smoothness=5.0,
            safety_violations=2
        )
        
        assert metrics.success_rate == 0.8
        assert metrics.mean_return == 100.0
        assert metrics.safety_violations == 2
    
    def test_leaderboard(self):
        """Test leaderboard functionality."""
        leaderboard = Leaderboard("test_leaderboard.json")
        
        # Create mock metrics
        sim_metrics = EvaluationMetrics(
            success_rate=0.8,
            mean_return=100.0,
            std_return=10.0,
            mean_episode_length=50.0,
            std_episode_length=5.0,
            mean_distance_to_goal=0.5,
            std_distance_to_goal=0.1,
            control_effort=20.0,
            smoothness=5.0,
            safety_violations=2
        )
        
        # Add entry
        leaderboard.add_entry("test_method", sim_metrics)
        
        # Get top methods
        top_methods = leaderboard.get_top_methods("mean_return", n=1)
        assert len(top_methods) == 1
        assert top_methods[0]["method_name"] == "test_method"
        
        # Convert to DataFrame
        df = leaderboard.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["method_name"] == "test_method"


class TestIntegration:
    """Integration tests."""
    
    @patch('pybullet.connect')
    @patch('pybullet.setAdditionalSearchPath')
    @patch('pybullet.setGravity')
    @patch('pybullet.setTimeStep')
    @patch('pybullet.loadURDF')
    def test_environment_initialization(self, mock_load_urdf, mock_set_timestep, 
                                     mock_set_gravity, mock_set_path, mock_connect):
        """Test environment initialization."""
        # Mock PyBullet functions
        mock_connect.return_value = 0
        mock_load_urdf.return_value = 1
        
        from sim_to_real.environments import SimToRealEnvironment
        
        env = SimToRealEnvironment(
            robot_type="mobile_robot",
            render_mode=None
        )
        
        assert env.robot_type == "mobile_robot"
        assert env.action_space is not None
        assert env.observation_space is not None


if __name__ == "__main__":
    pytest.main([__file__])
