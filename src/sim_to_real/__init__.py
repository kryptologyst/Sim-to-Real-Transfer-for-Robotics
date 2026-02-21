"""Sim-to-Real Transfer for Robotics - Main Package."""

__version__ = "0.1.0"
__author__ = "Robotics Research Team"
__email__ = "research@example.com"

from sim_to_real.environments import SimToRealEnvironment
from sim_to_real.agents import PPOAgent, SACAgent
from sim_to_real.domain_randomization import DomainRandomization
from sim_to_real.evaluation import SimToRealEvaluator

__all__ = [
    "SimToRealEnvironment",
    "PPOAgent", 
    "SACAgent",
    "DomainRandomization",
    "SimToRealEvaluator",
]
