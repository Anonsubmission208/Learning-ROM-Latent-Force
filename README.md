# Learning ROMs with Latent Force Models

This repository contains code for learning non-intrusive, physically consistent reduced-order models (ROMs) from video. We use a latent force variational autoencoder (VAE) to learn low-dimensional latent dynamics while enforcing a physics-inspired prior.

The goal is to model systems such as soft robots directly from video, without requiring full state measurements, high-fidelity simulations, or complete knowledge of the system dynamics.

## Overview

Soft robots are difficult to model because they often have nonlinear, high-dimensional dynamics. This project learns compact latent representations of system motion from video while encouraging the learned latent trajectory to follow interpretable physical dynamics.

The model combines:

- A VAE for encoding and reconstructing video frames
- A latent force model for structured latent dynamics
- Gaussian process priors for unknown external forces
- Learnable physical parameters such as frequency and damping

## Repository Structure

```text
Learning-ROM-Latent-Force/
├── configs/
│   ├── finger/
│   └── pendulum/
├── notebooks/
├── scripts/
│   ├── finger_training/
│   ├── pendulum_data_gen/
│   └── pendulum_training/
├── src/
│   └── learning_dynamics/
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
cd Learning-ROM-Latent-Force
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install torch numpy scipy matplotlib scikit-learn opencv-python jupyter
```

Add the source directory to your Python path:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Usage

### Generate Pendulum Data

```bash
python scripts/pendulum_data_gen/pendulum_data_generation.py
```

### Train Pendulum Models

```bash
bash scripts/pendulum_training/run_physics_recon.sh
bash scripts/pendulum_training/run_physics_extrap.sh
```

### Train Soft Robot Finger Models

```bash
bash scripts/finger_training/run_physics_recon.sh
bash scripts/finger_training/run_physics_extrap.sh
```