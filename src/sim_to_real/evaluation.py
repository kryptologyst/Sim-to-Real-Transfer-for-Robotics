"""Evaluation metrics and tools for sim-to-real transfer."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm

from .utils import compute_reality_gap, SafetyLimits
from .environments import SimToRealEnvironment


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    success_rate: float
    mean_return: float
    std_return: float
    mean_episode_length: float
    std_episode_length: float
    mean_distance_to_goal: float
    std_distance_to_goal: float
    control_effort: float
    smoothness: float
    safety_violations: int
    reality_gap: Optional[Dict[str, float]] = None


class SimToRealEvaluator:
    """Evaluator for sim-to-real transfer performance."""
    
    def __init__(
        self,
        sim_env: SimToRealEnvironment,
        real_env: Optional[SimToRealEnvironment] = None,
        safety_limits: Optional[SafetyLimits] = None,
        n_eval_episodes: int = 100
    ):
        """Initialize evaluator.
        
        Args:
            sim_env: Simulation environment.
            real_env: Real environment (optional).
            safety_limits: Safety limits for evaluation.
            n_eval_episodes: Number of episodes for evaluation.
        """
        self.sim_env = sim_env
        self.real_env = real_env
        self.safety_limits = safety_limits or SafetyLimits()
        self.n_eval_episodes = n_eval_episodes
        
    def evaluate_agent(
        self,
        agent: Any,
        environment: SimToRealEnvironment,
        deterministic: bool = True,
        render: bool = False
    ) -> EvaluationMetrics:
        """Evaluate an agent in an environment.
        
        Args:
            agent: Trained agent.
            environment: Environment to evaluate in.
            deterministic: Whether to use deterministic actions.
            render: Whether to render during evaluation.
            
        Returns:
            Evaluation metrics.
        """
        episode_returns = []
        episode_lengths = []
        distances_to_goal = []
        control_efforts = []
        smoothness_scores = []
        safety_violations = 0
        
        for episode in tqdm(range(self.n_eval_episodes), desc="Evaluating"):
            obs, _ = environment.reset()
            episode_return = 0
            episode_length = 0
            episode_distance = 0
            episode_control_effort = 0
            episode_smoothness = 0
            prev_action = None
            
            done = False
            while not done:
                # Get action from agent
                if hasattr(agent, 'predict'):
                    action, _ = agent.predict(obs, deterministic=deterministic)
                else:
                    action = agent.get_action(obs, deterministic=deterministic)
                
                # Check safety limits
                if not self.safety_limits.check_velocity_limit(action):
                    safety_violations += 1
                
                # Compute control effort and smoothness
                episode_control_effort += np.sum(np.abs(action))
                if prev_action is not None:
                    episode_smoothness += np.sum(np.abs(action - prev_action))
                prev_action = action.copy()
                
                # Step environment
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                episode_length += 1
                episode_distance += obs["distance_to_goal"][0]
                
                done = terminated or truncated
                
                if render:
                    environment.render()
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            distances_to_goal.append(episode_distance / episode_length)
            control_efforts.append(episode_control_effort)
            smoothness_scores.append(episode_smoothness)
        
        # Compute success rate (episodes that reached goal)
        success_rate = np.mean([d < 0.5 for d in distances_to_goal])
        
        # Compute metrics
        metrics = EvaluationMetrics(
            success_rate=success_rate,
            mean_return=np.mean(episode_returns),
            std_return=np.std(episode_returns),
            mean_episode_length=np.mean(episode_lengths),
            std_episode_length=np.std(episode_lengths),
            mean_distance_to_goal=np.mean(distances_to_goal),
            std_distance_to_goal=np.std(distances_to_goal),
            control_effort=np.mean(control_efforts),
            smoothness=np.mean(smoothness_scores),
            safety_violations=safety_violations
        )
        
        return metrics
    
    def evaluate_sim_to_real_transfer(
        self,
        agent: Any,
        deterministic: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """Evaluate sim-to-real transfer performance.
        
        Args:
            agent: Trained agent.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Dictionary containing simulation and real-world metrics.
        """
        results = {}
        
        # Evaluate in simulation
        print("Evaluating in simulation...")
        sim_metrics = self.evaluate_agent(
            agent, self.sim_env, deterministic=deterministic
        )
        results["simulation"] = sim_metrics
        
        # Evaluate in real world if available
        if self.real_env is not None:
            print("Evaluating in real world...")
            real_metrics = self.evaluate_agent(
                agent, self.real_env, deterministic=deterministic
            )
            results["real_world"] = real_metrics
            
            # Compute reality gap
            reality_gap = compute_reality_gap(
                sim_metrics.mean_return,
                real_metrics.mean_return
            )
            
            # Add reality gap to both metrics
            sim_metrics.reality_gap = reality_gap
            real_metrics.reality_gap = reality_gap
            
        return results
    
    def compare_methods(
        self,
        agents: Dict[str, Any],
        deterministic: bool = True
    ) -> pd.DataFrame:
        """Compare multiple methods.
        
        Args:
            agents: Dictionary of agent names to agents.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for name, agent in agents.items():
            print(f"Evaluating {name}...")
            
            # Evaluate in simulation
            sim_metrics = self.evaluate_agent(
                agent, self.sim_env, deterministic=deterministic
            )
            
            result = {
                "method": name,
                "environment": "simulation",
                "success_rate": sim_metrics.success_rate,
                "mean_return": sim_metrics.mean_return,
                "std_return": sim_metrics.std_return,
                "mean_episode_length": sim_metrics.mean_episode_length,
                "control_effort": sim_metrics.control_effort,
                "smoothness": sim_metrics.smoothness,
                "safety_violations": sim_metrics.safety_violations,
            }
            results.append(result)
            
            # Evaluate in real world if available
            if self.real_env is not None:
                real_metrics = self.evaluate_agent(
                    agent, self.real_env, deterministic=deterministic
                )
                
                reality_gap = compute_reality_gap(
                    sim_metrics.mean_return,
                    real_metrics.mean_return
                )
                
                result = {
                    "method": name,
                    "environment": "real_world",
                    "success_rate": real_metrics.success_rate,
                    "mean_return": real_metrics.mean_return,
                    "std_return": real_metrics.std_return,
                    "mean_episode_length": real_metrics.mean_episode_length,
                    "control_effort": real_metrics.control_effort,
                    "smoothness": real_metrics.smoothness,
                    "safety_violations": real_metrics.safety_violations,
                    "reality_gap_absolute": reality_gap["absolute_gap"],
                    "reality_gap_relative": reality_gap["relative_gap"],
                    "transfer_efficiency": reality_gap["transfer_efficiency"],
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_evaluation_results(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results.
            save_path: Path to save plots.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Sim-to-Real Transfer Evaluation Results", fontsize=16)
        
        # Success rate
        sns.barplot(data=results_df, x="method", y="success_rate", hue="environment", ax=axes[0, 0])
        axes[0, 0].set_title("Success Rate")
        axes[0, 0].set_ylabel("Success Rate")
        
        # Mean return
        sns.barplot(data=results_df, x="method", y="mean_return", hue="environment", ax=axes[0, 1])
        axes[0, 1].set_title("Mean Return")
        axes[0, 1].set_ylabel("Mean Return")
        
        # Control effort
        sns.barplot(data=results_df, x="method", y="control_effort", hue="environment", ax=axes[0, 2])
        axes[0, 2].set_title("Control Effort")
        axes[0, 2].set_ylabel("Control Effort")
        
        # Smoothness
        sns.barplot(data=results_df, x="method", y="smoothness", hue="environment", ax=axes[1, 0])
        axes[1, 0].set_title("Smoothness")
        axes[1, 0].set_ylabel("Smoothness")
        
        # Safety violations
        sns.barplot(data=results_df, x="method", y="safety_violations", hue="environment", ax=axes[1, 1])
        axes[1, 1].set_title("Safety Violations")
        axes[1, 1].set_ylabel("Safety Violations")
        
        # Reality gap (if available)
        if "reality_gap_absolute" in results_df.columns:
            sim_data = results_df[results_df["environment"] == "simulation"]
            real_data = results_df[results_df["environment"] == "real_world"]
            
            if not real_data.empty:
                gap_data = pd.DataFrame({
                    "method": sim_data["method"],
                    "reality_gap": real_data["reality_gap_absolute"]
                })
                sns.barplot(data=gap_data, x="method", y="reality_gap", ax=axes[1, 2])
                axes[1, 2].set_title("Reality Gap (Absolute)")
                axes[1, 2].set_ylabel("Reality Gap")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """Generate evaluation report.
        
        Args:
            results_df: DataFrame with evaluation results.
            save_path: Path to save report.
            
        Returns:
            Report string.
        """
        report = []
        report.append("# Sim-to-Real Transfer Evaluation Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("")
        
        for method in results_df["method"].unique():
            method_data = results_df[results_df["method"] == method]
            report.append(f"### {method}")
            report.append("")
            
            for env in method_data["environment"].unique():
                env_data = method_data[method_data["environment"] == env]
                report.append(f"**{env.title()} Environment:**")
                report.append(f"- Success Rate: {env_data['success_rate'].iloc[0]:.3f}")
                report.append(f"- Mean Return: {env_data['mean_return'].iloc[0]:.3f} ± {env_data['std_return'].iloc[0]:.3f}")
                report.append(f"- Control Effort: {env_data['control_effort'].iloc[0]:.3f}")
                report.append(f"- Smoothness: {env_data['smoothness'].iloc[0]:.3f}")
                report.append(f"- Safety Violations: {env_data['safety_violations'].iloc[0]}")
                
                if "reality_gap_absolute" in env_data.columns and not pd.isna(env_data['reality_gap_absolute'].iloc[0]):
                    report.append(f"- Reality Gap (Absolute): {env_data['reality_gap_absolute'].iloc[0]:.3f}")
                    report.append(f"- Reality Gap (Relative): {env_data['reality_gap_relative'].iloc[0]:.3f}")
                    report.append(f"- Transfer Efficiency: {env_data['transfer_efficiency'].iloc[0]:.3f}")
                
                report.append("")
        
        # Best performing method
        if "real_world" in results_df["environment"].values:
            real_data = results_df[results_df["environment"] == "real_world"]
            best_method = real_data.loc[real_data["mean_return"].idxmax(), "method"]
            report.append(f"## Best Performing Method: {best_method}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text


class Leaderboard:
    """Leaderboard for tracking sim-to-real transfer performance."""
    
    def __init__(self, save_path: str = "leaderboard.json"):
        """Initialize leaderboard.
        
        Args:
            save_path: Path to save leaderboard data.
        """
        self.save_path = Path(save_path)
        self.entries = []
        
        if self.save_path.exists():
            self.load()
    
    def add_entry(
        self,
        method_name: str,
        sim_metrics: EvaluationMetrics,
        real_metrics: Optional[EvaluationMetrics] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add entry to leaderboard.
        
        Args:
            method_name: Name of the method.
            sim_metrics: Simulation metrics.
            real_metrics: Real-world metrics (optional).
            config: Method configuration (optional).
        """
        entry = {
            "method_name": method_name,
            "timestamp": time.time(),
            "sim_metrics": {
                "success_rate": sim_metrics.success_rate,
                "mean_return": sim_metrics.mean_return,
                "std_return": sim_metrics.std_return,
                "control_effort": sim_metrics.control_effort,
                "smoothness": sim_metrics.smoothness,
                "safety_violations": sim_metrics.safety_violations,
            },
            "config": config or {}
        }
        
        if real_metrics is not None:
            entry["real_metrics"] = {
                "success_rate": real_metrics.success_rate,
                "mean_return": real_metrics.mean_return,
                "std_return": real_metrics.std_return,
                "control_effort": real_metrics.control_effort,
                "smoothness": real_metrics.smoothness,
                "safety_violations": real_metrics.safety_violations,
            }
            
            if real_metrics.reality_gap is not None:
                entry["reality_gap"] = real_metrics.reality_gap
        
        self.entries.append(entry)
        self.save()
    
    def get_top_methods(self, metric: str = "mean_return", n: int = 10) -> List[Dict[str, Any]]:
        """Get top performing methods.
        
        Args:
            metric: Metric to rank by.
            n: Number of top methods to return.
            
        Returns:
            List of top methods.
        """
        # Sort by metric (use real-world metrics if available, otherwise simulation)
        sorted_entries = sorted(
            self.entries,
            key=lambda x: x.get("real_metrics", x["sim_metrics"]).get(metric, 0),
            reverse=True
        )
        
        return sorted_entries[:n]
    
    def save(self) -> None:
        """Save leaderboard to file."""
        with open(self.save_path, "w") as f:
            json.dump(self.entries, f, indent=2)
    
    def load(self) -> None:
        """Load leaderboard from file."""
        with open(self.save_path, "r") as f:
            self.entries = json.load(f)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert leaderboard to DataFrame.
        
        Returns:
            DataFrame with leaderboard data.
        """
        data = []
        for entry in self.entries:
            row = {
                "method_name": entry["method_name"],
                "timestamp": entry["timestamp"],
            }
            
            # Add simulation metrics
            for key, value in entry["sim_metrics"].items():
                row[f"sim_{key}"] = value
            
            # Add real-world metrics if available
            if "real_metrics" in entry:
                for key, value in entry["real_metrics"].items():
                    row[f"real_{key}"] = value
            
            # Add reality gap if available
            if "reality_gap" in entry:
                for key, value in entry["reality_gap"].items():
                    row[f"reality_gap_{key}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
