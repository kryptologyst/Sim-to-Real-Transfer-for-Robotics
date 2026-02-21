"""Simulation environments for sim-to-real transfer."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import pybullet as p
import pybullet_data
import threading
import time
from dataclasses import dataclass

from .domain_randomization import DomainRandomization
from .utils import SafetyLimits, clip_action, safe_normalize


@dataclass
class RobotConfig:
    """Configuration for robot parameters."""
    urdf_path: str
    base_position: Tuple[float, float, float] = (0, 0, 0)
    base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)
    max_velocity: float = 1.0
    max_force: float = 100.0
    control_frequency: int = 100  # Hz


class MobileRobotEnv(gym.Env):
    """Mobile robot environment for sim-to-real transfer."""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        domain_randomization: Optional[DomainRandomization] = None,
        safety_limits: Optional[SafetyLimits] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000
    ):
        """Initialize mobile robot environment.
        
        Args:
            robot_config: Robot configuration.
            domain_randomization: Domain randomization settings.
            safety_limits: Safety limits for control.
            render_mode: Rendering mode ('human', 'rgb_array', None).
            max_episode_steps: Maximum steps per episode.
        """
        super().__init__()
        
        self.robot_config = robot_config
        self.domain_randomization = domain_randomization or DomainRandomization()
        self.safety_limits = safety_limits or SafetyLimits()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # [linear_vel, angular_vel]
        
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "goal_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "distance_to_goal": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })
        
        # Environment state
        self.robot_id: Optional[int] = None
        self.goal_position: np.ndarray = np.array([5.0, 5.0])
        self.current_step = 0
        self.episode_reward = 0.0
        
        # PyBullet setup
        self.physics_client = None
        self._setup_physics()
        
    def _setup_physics(self) -> None:
        """Setup PyBullet physics simulation."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.robot_config.control_frequency)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Initial observation and info.
        """
        super().reset(seed=seed)
        
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
            
        # Load robot
        self.robot_id = p.loadURDF(
            self.robot_config.urdf_path,
            self.robot_config.base_position,
            self.robot_config.base_orientation
        )
        
        # Set random goal position
        self.goal_position = np.random.uniform(-5, 5, 2)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Apply domain randomization
        if self.domain_randomization:
            self._apply_domain_randomization()
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to execute.
            
        Returns:
            Observation, reward, terminated, truncated, info.
        """
        # Clip action to valid range
        action = clip_action(action, (-1.0, 1.0))
        
        # Apply domain randomization to action
        if self.domain_randomization:
            action = self.domain_randomization.add_action_noise(action)
            
        # Convert action to robot commands
        linear_vel = action[0] * self.robot_config.max_velocity
        angular_vel = action[1] * self.robot_config.max_velocity
        
        # Apply safety limits
        if not self.safety_limits.check_velocity_limit(np.array([linear_vel, angular_vel])):
            linear_vel = np.clip(linear_vel, -self.safety_limits.max_velocity, self.safety_limits.max_velocity)
            angular_vel = np.clip(angular_vel, -self.safety_limits.max_velocity, self.safety_limits.max_velocity)
        
        # Apply control commands
        self._apply_control(linear_vel, angular_vel)
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(observation)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        # Update step counter
        self.current_step += 1
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _apply_control(self, linear_vel: float, angular_vel: float) -> None:
        """Apply control commands to robot.
        
        Args:
            linear_vel: Linear velocity command.
            angular_vel: Angular velocity command.
        """
        # Simple differential drive model
        wheel_base = 0.3  # Distance between wheels
        wheel_radius = 0.05
        
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2) / wheel_radius
        
        # Apply wheel velocities (assuming wheel joints are indices 0 and 1)
        p.setJointMotorControl2(
            self.robot_id, 0, p.VELOCITY_CONTROL, 
            targetVelocity=left_wheel_vel, force=self.robot_config.max_force
        )
        p.setJointMotorControl2(
            self.robot_id, 1, p.VELOCITY_CONTROL, 
            targetVelocity=right_wheel_vel, force=self.robot_config.max_force
        )
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation.
        
        Returns:
            Current observation dictionary.
        """
        # Get robot position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        position = np.array([pos[0], pos[1]], dtype=np.float32)
        
        # Get robot velocity
        vel, _ = p.getBaseVelocity(self.robot_id)
        velocity = np.array([vel[0], vel[1]], dtype=np.float32)
        
        # Compute distance to goal
        distance_to_goal = np.linalg.norm(position - self.goal_position)
        
        observation = {
            "position": position,
            "velocity": velocity,
            "goal_position": self.goal_position.astype(np.float32),
            "distance_to_goal": np.array([distance_to_goal], dtype=np.float32),
        }
        
        # Apply sensor noise
        if self.domain_randomization:
            observation["position"] = self.domain_randomization.add_sensor_noise(observation["position"])
            observation["velocity"] = self.domain_randomization.add_sensor_noise(observation["velocity"])
            
        return observation
        
    def _compute_reward(self, observation: Dict[str, np.ndarray]) -> float:
        """Compute reward for current state.
        
        Args:
            observation: Current observation.
            
        Returns:
            Reward value.
        """
        position = observation["position"]
        distance_to_goal = observation["distance_to_goal"][0]
        
        # Distance-based reward
        reward = -distance_to_goal
        
        # Bonus for reaching goal
        if distance_to_goal < 0.5:
            reward += 100.0
            
        # Penalty for excessive velocity
        velocity_magnitude = np.linalg.norm(observation["velocity"])
        if velocity_magnitude > self.safety_limits.max_velocity:
            reward -= 10.0
            
        return reward
        
    def _is_terminated(self, observation: Dict[str, np.ndarray]) -> bool:
        """Check if episode is terminated.
        
        Args:
            observation: Current observation.
            
        Returns:
            True if episode is terminated.
        """
        distance_to_goal = observation["distance_to_goal"][0]
        return distance_to_goal < 0.3
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info.
        
        Returns:
            Info dictionary.
        """
        return {
            "episode_reward": self.episode_reward,
            "current_step": self.current_step,
            "goal_position": self.goal_position.tolist(),
        }
        
    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to the environment."""
        dynamics_params = self.domain_randomization.randomize_dynamics()
        
        # Apply friction randomization
        if "friction" in dynamics_params:
            p.changeDynamics(
                self.robot_id, -1, 
                lateralFriction=dynamics_params["friction"]
            )
            
        # Apply mass randomization
        if "mass" in dynamics_params:
            p.changeDynamics(
                self.robot_id, -1,
                massMultiplier=dynamics_params["mass"]
            )
            
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            Rendered image if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "rgb_array":
            # Get camera image
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=5,
                yaw=0,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
            )
            
            _, _, rgb_array, _, _ = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )
            
            return rgb_array
            
        return None
        
    def close(self) -> None:
        """Close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class SimToRealEnvironment:
    """Main sim-to-real environment wrapper."""
    
    def __init__(
        self,
        robot_type: str = "mobile_robot",
        domain_randomization: Optional[DomainRandomization] = None,
        safety_limits: Optional[SafetyLimits] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """Initialize sim-to-real environment.
        
        Args:
            robot_type: Type of robot ('mobile_robot', 'manipulator', 'quadrotor').
            domain_randomization: Domain randomization settings.
            safety_limits: Safety limits for control.
            render_mode: Rendering mode.
            **kwargs: Additional arguments.
        """
        self.robot_type = robot_type
        self.domain_randomization = domain_randomization or DomainRandomization()
        self.safety_limits = safety_limits or SafetyLimits()
        self.render_mode = render_mode
        
        # Create robot configuration
        if robot_type == "mobile_robot":
            robot_config = RobotConfig(
                urdf_path="robots/mobile_robot.urdf",
                **kwargs
            )
            self.env = MobileRobotEnv(
                robot_config=robot_config,
                domain_randomization=self.domain_randomization,
                safety_limits=self.safety_limits,
                render_mode=self.render_mode
            )
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
            
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(**kwargs)
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        return self.env.step(action)
        
    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        return self.env.render()
        
    def close(self) -> None:
        """Close environment."""
        self.env.close()
        
    @property
    def action_space(self):
        """Action space."""
        return self.env.action_space
        
    @property
    def observation_space(self):
        """Observation space."""
        return self.env.observation_space
