"""Streamlit demo for sim-to-real transfer."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import torch

from sim_to_real import SimToRealEnvironment, PPOAgent, SACAgent, AgentConfig, DomainRandomization
from sim_to_real.evaluation import SimToRealEvaluator, Leaderboard
from sim_to_real.utils import set_seed

st.set_page_config(
    page_title="Sim-to-Real Transfer Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sim-to-Real Transfer for Robotics")
st.markdown("Interactive demo for evaluating sim-to-real transfer methods")

# Sidebar configuration
st.sidebar.header("Configuration")

# Load default config
config_path = Path("configs/default.yaml")
if config_path.exists():
    with open(config_path, "r") as f:
        default_config = yaml.safe_load(f)
else:
    default_config = {
        "robot_type": "mobile_robot",
        "domain_randomization": {
            "friction_range": [0.1, 0.8],
            "mass_range": [0.8, 1.2],
            "noise_level": 0.1
        },
        "agent": {
            "algorithm": "PPO",
            "learning_rate": 3e-4
        }
    }

# Configuration options
robot_type = st.sidebar.selectbox(
    "Robot Type",
    ["mobile_robot", "manipulator", "quadrotor"],
    index=0
)

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["PPO", "SAC", "TD3"],
    index=0
)

# Domain randomization parameters
st.sidebar.subheader("Domain Randomization")
friction_min = st.sidebar.slider("Friction Min", 0.0, 1.0, 0.1)
friction_max = st.sidebar.slider("Friction Max", 0.0, 1.0, 0.8)
mass_min = st.sidebar.slider("Mass Min", 0.5, 2.0, 0.8)
mass_max = st.sidebar.slider("Mass Max", 0.5, 2.0, 1.2)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

# Safety limits
st.sidebar.subheader("Safety Limits")
max_velocity = st.sidebar.slider("Max Velocity", 0.1, 2.0, 1.0)
max_force = st.sidebar.slider("Max Force", 10.0, 200.0, 100.0)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Environment", "Training", "Evaluation", "Analysis"])

with tab1:
    st.header("Environment Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment Parameters")
        st.write(f"**Robot Type:** {robot_type}")
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Friction Range:** [{friction_min:.2f}, {friction_max:.2f}]")
        st.write(f"**Mass Range:** [{mass_min:.2f}, {mass_max:.2f}]")
        st.write(f"**Noise Level:** {noise_level:.2f}")
        st.write(f"**Max Velocity:** {max_velocity:.2f}")
        st.write(f"**Max Force:** {max_force:.2f}")
    
    with col2:
        st.subheader("Robot Description")
        st.code("""
        Mobile Robot Features:
        - Differential drive
        - 2 wheels + caster
        - Mass: 10kg
        - Wheel radius: 0.05m
        - Wheel base: 0.3m
        """)
    
    # Environment setup
    if st.button("Initialize Environment"):
        with st.spinner("Setting up environment..."):
            try:
                domain_randomization = DomainRandomization(
                    friction_range=(friction_min, friction_max),
                    mass_range=(mass_min, mass_max),
                    noise_level=noise_level
                )
                
                env = SimToRealEnvironment(
                    robot_type=robot_type,
                    domain_randomization=domain_randomization,
                    render_mode=None
                )
                
                st.session_state.env = env
                st.success("Environment initialized successfully!")
                
                # Display environment info
                st.subheader("Environment Information")
                st.write(f"**Action Space:** {env.action_space}")
                st.write(f"**Observation Space:** {env.observation_space}")
                
            except Exception as e:
                st.error(f"Failed to initialize environment: {str(e)}")

with tab2:
    st.header("Training")
    
    if "env" not in st.session_state:
        st.warning("Please initialize the environment first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Parameters")
            total_timesteps = st.number_input("Total Timesteps", 1000, 1000000, 100000)
            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
            
        with col2:
            st.subheader("Training Options")
            use_wandb = st.checkbox("Use Weights & Biases", False)
            save_checkpoints = st.checkbox("Save Checkpoints", True)
            progress_bar = st.checkbox("Show Progress Bar", True)
        
        if st.button("Start Training"):
            with st.spinner("Training agent..."):
                try:
                    # Create agent
                    agent_config = AgentConfig(
                        algorithm=algorithm,
                        learning_rate=learning_rate,
                        batch_size=batch_size
                    )
                    
                    if algorithm == "PPO":
                        agent = PPOAgent(st.session_state.env, agent_config, use_wandb=use_wandb)
                    elif algorithm == "SAC":
                        agent = SACAgent(st.session_state.env, agent_config, use_wandb=use_wandb)
                    else:
                        st.error(f"Algorithm {algorithm} not supported in demo")
                        st.stop()
                    
                    # Train agent
                    agent.train(total_timesteps)
                    
                    st.session_state.agent = agent
                    st.success("Training completed!")
                    
                    # Save agent if requested
                    if save_checkpoints:
                        checkpoint_path = f"checkpoints/{algorithm}_trained.pt"
                        Path("checkpoints").mkdir(exist_ok=True)
                        agent.save(checkpoint_path)
                        st.info(f"Agent saved to {checkpoint_path}")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

with tab3:
    st.header("Evaluation")
    
    if "agent" not in st.session_state:
        st.warning("Please train an agent first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Evaluation Parameters")
            n_episodes = st.number_input("Number of Episodes", 1, 1000, 100)
            deterministic = st.checkbox("Deterministic Actions", True)
            render = st.checkbox("Render During Evaluation", False)
            
        with col2:
            st.subheader("Evaluation Options")
            save_results = st.checkbox("Save Results", True)
            show_plots = st.checkbox("Show Plots", True)
        
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating agent..."):
                try:
                    evaluator = SimToRealEvaluator(
                        st.session_state.env,
                        n_eval_episodes=n_episodes
                    )
                    
                    metrics = evaluator.evaluate_agent(
                        st.session_state.agent,
                        st.session_state.env,
                        deterministic=deterministic,
                        render=render
                    )
                    
                    st.session_state.metrics = metrics
                    st.success("Evaluation completed!")
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Success Rate", f"{metrics.success_rate:.3f}")
                    with col2:
                        st.metric("Mean Return", f"{metrics.mean_return:.3f}")
                    with col3:
                        st.metric("Control Effort", f"{metrics.control_effort:.3f}")
                    with col4:
                        st.metric("Safety Violations", f"{metrics.safety_violations}")
                    
                    # Additional metrics
                    st.subheader("Detailed Metrics")
                    st.write(f"**Mean Episode Length:** {metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}")
                    st.write(f"**Mean Distance to Goal:** {metrics.mean_distance_to_goal:.3f} ± {metrics.std_distance_to_goal:.3f}")
                    st.write(f"**Smoothness:** {metrics.smoothness:.3f}")
                    
                    # Save results if requested
                    if save_results:
                        results_path = "results/evaluation_results.json"
                        Path("results").mkdir(exist_ok=True)
                        
                        results_dict = {
                            "success_rate": metrics.success_rate,
                            "mean_return": metrics.mean_return,
                            "std_return": metrics.std_return,
                            "control_effort": metrics.control_effort,
                            "smoothness": metrics.smoothness,
                            "safety_violations": metrics.safety_violations
                        }
                        
                        import json
                        with open(results_path, "w") as f:
                            json.dump(results_dict, f, indent=2)
                        
                        st.info(f"Results saved to {results_path}")
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")

with tab4:
    st.header("Analysis")
    
    # Load leaderboard if available
    leaderboard_path = Path("leaderboard.json")
    if leaderboard_path.exists():
        st.subheader("Leaderboard")
        
        try:
            leaderboard = Leaderboard(str(leaderboard_path))
            df = leaderboard.to_dataframe()
            
            if not df.empty:
                st.dataframe(df)
                
                # Plot performance over time
                if "timestamp" in df.columns and "sim_mean_return" in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(df["timestamp"], df["sim_mean_return"], marker="o")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Mean Return")
                    ax.set_title("Performance Over Time")
                    ax.grid(True)
                    st.pyplot(fig)
            else:
                st.info("No entries in leaderboard yet.")
                
        except Exception as e:
            st.error(f"Failed to load leaderboard: {str(e)}")
    else:
        st.info("No leaderboard found. Run evaluations to create one.")
    
    # Method comparison
    st.subheader("Method Comparison")
    
    if st.button("Compare Methods"):
        st.info("Method comparison feature coming soon!")
    
    # Reality gap analysis
    st.subheader("Reality Gap Analysis")
    
    if st.button("Analyze Reality Gap"):
        st.info("Reality gap analysis feature coming soon!")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This software is for research and educational purposes only. Do not use on real hardware without expert review.")
