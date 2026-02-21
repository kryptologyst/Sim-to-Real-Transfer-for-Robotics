"""Command-line interface for sim-to-real transfer."""

import typer
from typing import Optional, List
from pathlib import Path
import yaml
import torch
import numpy as np

from .environments import SimToRealEnvironment
from .agents import PPOAgent, SACAgent, AgentConfig
from .domain_randomization import DomainRandomization
from .evaluation import SimToRealEvaluator, Leaderboard
from .utils import set_seed, get_device

app = typer.Typer(help="Sim-to-Real Transfer for Robotics CLI")


@app.command()
def train(
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("checkpoints", help="Output directory for checkpoints"),
    seed: int = typer.Option(42, help="Random seed"),
    use_wandb: bool = typer.Option(False, help="Use Weights & Biases logging"),
    total_timesteps: int = typer.Option(100000, help="Total training timesteps")
):
    """Train a sim-to-real transfer agent."""
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    domain_randomization = DomainRandomization(**config.get("domain_randomization", {}))
    env = SimToRealEnvironment(
        robot_type=config["robot_type"],
        domain_randomization=domain_randomization,
        render_mode=config.get("render_mode")
    )
    
    # Create agent
    agent_config = AgentConfig(**config.get("agent", {}))
    
    if agent_config.algorithm == "PPO":
        agent = PPOAgent(env, agent_config, use_wandb=use_wandb)
    elif agent_config.algorithm == "SAC":
        agent = SACAgent(env, agent_config, use_wandb=use_wandb)
    else:
        raise ValueError(f"Unsupported algorithm: {agent_config.algorithm}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train agent
    print(f"Training {agent_config.algorithm} agent for {total_timesteps} timesteps...")
    agent.train(total_timesteps)
    
    # Save agent
    checkpoint_path = output_path / f"{agent_config.algorithm}_final.pt"
    agent.save(str(checkpoint_path))
    print(f"Agent saved to {checkpoint_path}")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(..., help="Path to trained agent checkpoint"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    n_episodes: int = typer.Option(100, help="Number of evaluation episodes"),
    render: bool = typer.Option(False, help="Render during evaluation"),
    output_dir: str = typer.Option("results", help="Output directory for results")
):
    """Evaluate a trained sim-to-real transfer agent."""
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    domain_randomization = DomainRandomization(**config.get("domain_randomization", {}))
    env = SimToRealEnvironment(
        robot_type=config["robot_type"],
        domain_randomization=domain_randomization,
        render_mode="human" if render else None
    )
    
    # Load agent
    agent_config = AgentConfig(**config.get("agent", {}))
    
    if agent_config.algorithm == "PPO":
        agent = PPOAgent(env, agent_config)
    elif agent_config.algorithm == "SAC":
        agent = SACAgent(env, agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {agent_config.algorithm}")
    
    agent.load(checkpoint_path)
    
    # Create evaluator
    evaluator = SimToRealEvaluator(env, n_eval_episodes=n_episodes)
    
    # Evaluate agent
    print(f"Evaluating agent for {n_episodes} episodes...")
    metrics = evaluator.evaluate_agent(agent, env, render=render)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics.success_rate:.3f}")
    print(f"Mean Return: {metrics.mean_return:.3f} ± {metrics.std_return:.3f}")
    print(f"Control Effort: {metrics.control_effort:.3f}")
    print(f"Smoothness: {metrics.smoothness:.3f}")
    print(f"Safety Violations: {metrics.safety_violations}")


@app.command()
def compare(
    checkpoint_paths: List[str] = typer.Option(..., help="Paths to trained agent checkpoints"),
    method_names: List[str] = typer.Option(..., help="Names for the methods"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    n_episodes: int = typer.Option(100, help="Number of evaluation episodes"),
    output_dir: str = typer.Option("results", help="Output directory for results")
):
    """Compare multiple sim-to-real transfer methods."""
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    domain_randomization = DomainRandomization(**config.get("domain_randomization", {}))
    env = SimToRealEnvironment(
        robot_type=config["robot_type"],
        domain_randomization=domain_randomization
    )
    
    # Create evaluator
    evaluator = SimToRealEvaluator(env, n_eval_episodes=n_episodes)
    
    # Load agents
    agents = {}
    agent_config = AgentConfig(**config.get("agent", {}))
    
    for checkpoint_path, method_name in zip(checkpoint_paths, method_names):
        if agent_config.algorithm == "PPO":
            agent = PPOAgent(env, agent_config)
        elif agent_config.algorithm == "SAC":
            agent = SACAgent(env, agent_config)
        else:
            raise ValueError(f"Unsupported algorithm: {agent_config.algorithm}")
        
        agent.load(checkpoint_path)
        agents[method_name] = agent
    
    # Compare methods
    print("Comparing methods...")
    results_df = evaluator.compare_methods(agents)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_path / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Generate plots
    plot_path = output_path / "comparison_plots.png"
    evaluator.plot_evaluation_results(results_df, str(plot_path))
    print(f"Plots saved to {plot_path}")
    
    # Generate report
    report_path = output_path / "evaluation_report.md"
    report = evaluator.generate_report(results_df, str(report_path))
    print(f"Report saved to {report_path}")
    
    # Print summary
    print("\nComparison Summary:")
    print(results_df.groupby("method")["mean_return"].mean().sort_values(ascending=False))


@app.command()
def demo(
    checkpoint_path: str = typer.Option(..., help="Path to trained agent checkpoint"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    n_episodes: int = typer.Option(5, help="Number of demo episodes")
):
    """Run interactive demo of trained agent."""
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment with rendering
    domain_randomization = DomainRandomization(**config.get("domain_randomization", {}))
    env = SimToRealEnvironment(
        robot_type=config["robot_type"],
        domain_randomization=domain_randomization,
        render_mode="human"
    )
    
    # Load agent
    agent_config = AgentConfig(**config.get("agent", {}))
    
    if agent_config.algorithm == "PPO":
        agent = PPOAgent(env, agent_config)
    elif agent_config.algorithm == "SAC":
        agent = SACAgent(env, agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {agent_config.algorithm}")
    
    agent.load(checkpoint_path)
    
    # Run demo episodes
    print(f"Running {n_episodes} demo episodes...")
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        
        print(f"Episode return: {episode_return:.2f}")
    
    env.close()


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
