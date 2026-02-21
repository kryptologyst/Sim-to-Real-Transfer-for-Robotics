#!/usr/bin/env python3
"""
Example script demonstrating sim-to-real transfer for robotics.

This script shows how to:
1. Set up a simulation environment with domain randomization
2. Train a reinforcement learning agent
3. Evaluate sim-to-real transfer performance
4. Compare different methods

Run with: python examples/sim_to_real_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch

from sim_to_real import (
    SimToRealEnvironment, 
    PPOAgent, 
    SACAgent, 
    AgentConfig, 
    DomainRandomization,
    SimToRealEvaluator,
    Leaderboard
)
from sim_to_real.utils import set_seed, get_device


def main():
    """Main example function."""
    print("🤖 Sim-to-Real Transfer for Robotics Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    print(f"Using device: {get_device()}")
    
    # Configuration
    config = {
        "robot_type": "mobile_robot",
        "domain_randomization": {
            "friction_range": (0.1, 0.8),
            "mass_range": (0.8, 1.2),
            "noise_level": 0.1,
            "enable_dynamics_randomization": True,
            "enable_sensor_randomization": True
        },
        "agent": {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
            "clip_range": 0.2
        },
        "training": {
            "total_timesteps": 50000,
            "eval_freq": 10000
        },
        "evaluation": {
            "n_eval_episodes": 50
        }
    }
    
    # Create environment with domain randomization
    print("\n1. Setting up environment...")
    domain_randomization = DomainRandomization(**config["domain_randomization"])
    
    env = SimToRealEnvironment(
        robot_type=config["robot_type"],
        domain_randomization=domain_randomization,
        render_mode=None  # Set to "human" for visualization
    )
    
    print(f"Environment created: {env.robot_type}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create agent
    print("\n2. Creating agent...")
    agent_config = AgentConfig(**config["agent"])
    
    if agent_config.algorithm == "PPO":
        agent = PPOAgent(env, agent_config)
    elif agent_config.algorithm == "SAC":
        agent = SACAgent(env, agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {agent_config.algorithm}")
    
    print(f"Agent created: {agent_config.algorithm}")
    
    # Train agent
    print(f"\n3. Training agent for {config['training']['total_timesteps']} timesteps...")
    agent.train(config["training"]["total_timesteps"])
    print("Training completed!")
    
    # Save trained agent
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{agent_config.algorithm}_trained.pt"
    agent.save(str(checkpoint_path))
    print(f"Agent saved to {checkpoint_path}")
    
    # Evaluate agent
    print(f"\n4. Evaluating agent...")
    evaluator = SimToRealEvaluator(env, n_eval_episodes=config["evaluation"]["n_eval_episodes"])
    
    metrics = evaluator.evaluate_agent(agent, env, deterministic=True)
    
    print("Evaluation Results:")
    print(f"  Success Rate: {metrics.success_rate:.3f}")
    print(f"  Mean Return: {metrics.mean_return:.3f} ± {metrics.std_return:.3f}")
    print(f"  Mean Episode Length: {metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}")
    print(f"  Control Effort: {metrics.control_effort:.3f}")
    print(f"  Smoothness: {metrics.smoothness:.3f}")
    print(f"  Safety Violations: {metrics.safety_violations}")
    
    # Demonstrate domain randomization effects
    print("\n5. Demonstrating domain randomization...")
    demonstrate_domain_randomization(env, agent)
    
    # Compare different noise levels
    print("\n6. Comparing different noise levels...")
    compare_noise_levels()
    
    # Create leaderboard entry
    print("\n7. Updating leaderboard...")
    leaderboard = Leaderboard("leaderboard.json")
    leaderboard.add_entry(
        method_name=f"{agent_config.algorithm}_baseline",
        sim_metrics=metrics,
        config=config
    )
    print("Leaderboard updated!")
    
    # Generate summary plot
    print("\n8. Generating summary plot...")
    generate_summary_plot(metrics)
    
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("- Try different algorithms (SAC, TD3)")
    print("- Experiment with domain randomization parameters")
    print("- Run the Streamlit demo: streamlit run demo/app.py")
    print("- Check the leaderboard: leaderboard.json")


def demonstrate_domain_randomization(env, agent):
    """Demonstrate the effects of domain randomization."""
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    results = []
    
    for noise_level in noise_levels:
        # Create environment with specific noise level
        domain_randomization = DomainRandomization(noise_level=noise_level)
        test_env = SimToRealEnvironment(
            robot_type="mobile_robot",
            domain_randomization=domain_randomization,
            render_mode=None
        )
        
        # Evaluate agent
        evaluator = SimToRealEvaluator(test_env, n_eval_episodes=20)
        metrics = evaluator.evaluate_agent(agent, test_env, deterministic=True)
        
        results.append({
            "noise_level": noise_level,
            "success_rate": metrics.success_rate,
            "mean_return": metrics.mean_return
        })
        
        test_env.close()
    
    print("Domain Randomization Effects:")
    for result in results:
        print(f"  Noise Level {result['noise_level']:.2f}: "
              f"Success Rate {result['success_rate']:.3f}, "
              f"Return {result['mean_return']:.2f}")


def compare_noise_levels():
    """Compare performance across different noise levels."""
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    algorithms = ["PPO", "SAC"]
    
    results = []
    
    for algorithm in algorithms:
        for noise_level in noise_levels:
            print(f"  Testing {algorithm} with noise level {noise_level}...")
            
            # Create environment
            domain_randomization = DomainRandomization(noise_level=noise_level)
            env = SimToRealEnvironment(
                robot_type="mobile_robot",
                domain_randomization=domain_randomization,
                render_mode=None
            )
            
            # Create and train agent
            agent_config = AgentConfig(algorithm=algorithm, learning_rate=3e-4)
            
            if algorithm == "PPO":
                agent = PPOAgent(env, agent_config)
            elif algorithm == "SAC":
                agent = SACAgent(env, agent_config)
            
            # Quick training
            agent.train(10000)
            
            # Evaluate
            evaluator = SimToRealEvaluator(env, n_eval_episodes=20)
            metrics = evaluator.evaluate_agent(agent, env, deterministic=True)
            
            results.append({
                "algorithm": algorithm,
                "noise_level": noise_level,
                "success_rate": metrics.success_rate,
                "mean_return": metrics.mean_return
            })
            
            env.close()
    
    # Print comparison
    print("\nAlgorithm Comparison:")
    print("Algorithm | Noise Level | Success Rate | Mean Return")
    print("-" * 50)
    for result in results:
        print(f"{result['algorithm']:9} | {result['noise_level']:11.2f} | "
              f"{result['success_rate']:12.3f} | {result['mean_return']:10.2f}")


def generate_summary_plot(metrics):
    """Generate a summary plot of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Sim-to-Real Transfer Evaluation Summary", fontsize=16)
    
    # Success rate
    axes[0, 0].bar(["Success Rate"], [metrics.success_rate], color='green', alpha=0.7)
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_ylim(0, 1)
    
    # Mean return with error bars
    axes[0, 1].bar(["Mean Return"], [metrics.mean_return], 
                   yerr=[metrics.std_return], color='blue', alpha=0.7, capsize=5)
    axes[0, 1].set_title("Mean Return")
    axes[0, 1].set_ylabel("Return")
    
    # Control effort
    axes[1, 0].bar(["Control Effort"], [metrics.control_effort], color='orange', alpha=0.7)
    axes[1, 0].set_title("Control Effort")
    axes[1, 0].set_ylabel("Effort")
    
    # Safety violations
    axes[1, 1].bar(["Safety Violations"], [metrics.safety_violations], color='red', alpha=0.7)
    axes[1, 1].set_title("Safety Violations")
    axes[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    
    # Save plot
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    plot_path = assets_dir / "evaluation_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Summary plot saved to {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
