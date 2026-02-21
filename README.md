# Sim-to-Real Transfer for Robotics

A comprehensive framework for transferring learned behaviors from simulation to real-world robotics environments.

## DISCLAIMER

**⚠️ WARNING: DO NOT USE ON REAL HARDWARE WITHOUT EXPERT REVIEW ⚠️**

This software is for research and educational purposes only. It has NOT been tested on real robotic hardware and may cause damage to equipment or injury to people. Always consult with robotics safety experts before deploying any control algorithms on real robots.

## Overview

This project implements state-of-the-art sim-to-real transfer techniques for robotics, including:

- **Domain Randomization**: Randomizing simulation parameters to improve robustness
- **Residual Learning**: Combining classical control with learned corrections
- **Adaptation Methods**: Online adaptation to bridge the reality gap
- **Reality Gap Analysis**: Quantifying and visualizing sim-to-real performance drops

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Sim-to-Real-Transfer-for-Robotics.git
cd Sim-to-Real-Transfer-for-Robotics

# Install in development mode
pip install -e ".[dev]"

# Optional: Install ROS 2 dependencies
pip install -e ".[ros2]"
```

### Basic Usage

```python
from sim_to_real import SimToRealEnvironment, DomainRandomization

# Create environment with domain randomization
env = SimToRealEnvironment(
    robot_type="mobile_robot",
    domain_randomization=DomainRandomization(
        friction_range=(0.1, 0.8),
        mass_range=(0.8, 1.2),
        noise_level=0.1
    )
)

# Train a policy
policy = train_policy(env)

# Evaluate sim-to-real transfer
results = evaluate_transfer(policy, real_world=True)
```

### Running Demos

```bash
# Interactive Streamlit demo
streamlit run demo/app.py

# Command-line interface
sim-to-real train --config configs/mobile_robot.yaml
sim-to-real evaluate --checkpoint checkpoints/best_model.pt
```

## Project Structure

```
src/
├── sim_to_real/
│   ├── environments/     # Simulation environments
│   ├── agents/          # RL agents and policies
│   ├── adaptation/      # Online adaptation methods
│   ├── domain_randomization/  # Domain randomization techniques
│   ├── evaluation/     # Evaluation metrics and tools
│   ├── visualization/   # Plotting and visualization
│   └── utils/          # Utility functions
robots/
├── urdf/              # Robot descriptions
├── meshes/            # 3D models
└── configs/           # Robot-specific configurations
configs/               # Hydra configuration files
data/                  # Datasets and logs
tests/                 # Unit tests
demo/                  # Interactive demos
assets/                # Generated plots and videos
```

## Supported Robot Types

- **Mobile Robots**: Differential drive, omnidirectional
- **Manipulators**: 6-DOF and 7-DOF arms
- **Quadrotors**: Aerial vehicles with dynamics
- **Legged Robots**: Quadruped locomotion

## Simulation Environments

- **PyBullet**: Fast physics simulation
- **MuJoCo**: High-fidelity dynamics
- **Custom**: Specialized environments for specific tasks

## Key Features

### Domain Randomization
- Friction and mass variations
- Sensor noise and delays
- Actuator dynamics modeling
- Environmental disturbances

### Adaptation Methods
- Online parameter estimation
- Meta-learning approaches
- Residual policy learning
- System identification

### Evaluation Metrics
- Reality gap quantification
- Transfer success rates
- Performance degradation analysis
- Robustness testing

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/environment/`: Environment-specific settings
- `configs/agent/`: Agent and training configurations
- `configs/domain_randomization/`: Randomization parameters
- `configs/evaluation/`: Evaluation metrics and protocols

## Safety Considerations

- **Velocity Limits**: All agents respect maximum velocity constraints
- **Emergency Stops**: Built-in emergency stop mechanisms
- **Dry Run Mode**: Test policies in simulation before real deployment
- **Safety Margins**: Conservative control limits and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{sim_to_real_robotics,
  title={Sim-to-Real Transfer for Robotics},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Sim-to-Real-Transfer-for-Robotics}
}
```

## Acknowledgments

- PyBullet physics engine
- Stable Baselines3 RL library
- OpenAI Gym/Gymnasium environments
- ROS 2 robotics framework
# Sim-to-Real-Transfer-for-Robotics
