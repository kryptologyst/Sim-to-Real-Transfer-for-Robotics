#!/usr/bin/env python3
"""
Quick start script for sim-to-real transfer.

This script provides a minimal example to get started quickly.
"""

import numpy as np
from sim_to_real import SimToRealEnvironment, PPOAgent, AgentConfig, DomainRandomization
from sim_to_real.utils import set_seed

def main():
    """Quick start example."""
    print("🚀 Quick Start: Sim-to-Real Transfer")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment with domain randomization
    domain_randomization = DomainRandomization(
        friction_range=(0.1, 0.8),
        mass_range=(0.8, 1.2),
        noise_level=0.1
    )
    
    env = SimToRealEnvironment(
        robot_type="mobile_robot",
        domain_randomization=domain_randomization,
        render_mode=None
    )
    
    print(f"✅ Environment created: {env.robot_type}")
    
    # Create and train agent
    agent_config = AgentConfig(algorithm="PPO", learning_rate=3e-4)
    agent = PPOAgent(env, agent_config)
    
    print("🤖 Training agent...")
    agent.train(10000)  # Quick training
    print("✅ Training completed!")
    
    # Test the trained agent
    print("🧪 Testing agent...")
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(100):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"✅ Test completed! Total reward: {total_reward:.2f}")
    print("\n🎉 Quick start successful!")
    print("\nNext steps:")
    print("- Run full example: python examples/sim_to_real_example.py")
    print("- Try interactive demo: streamlit run demo/app.py")
    print("- Check documentation: README.md")

if __name__ == "__main__":
    main()
