# From Explainability to Interpretability: Interpretable Reinforcement Learning Via Model Explanations

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=kreQkWaOK5#discussion)
[![Conference](https://img.shields.io/badge/Conference-RLC%202025-green)](https://rl-conference.cc/)

Official implementation of "[From Explainability to Interpretability: Interpretable Reinforcement Learning Via Model Explanations](https://openreview.net/forum?id=kreQkWaOK5#discussion)" by Peilang Li, Umer Siddique, and Yongcan Cao.

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PeilangLi/explain-to-interpret-rl.git
   cd explain-to-interpret-rl
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using conda
   conda create --name explain-interpret-rl python=3.10.12
   conda activate explain-interpret-rl
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

##  Overview

This repository implements a novel approach to interpretable reinforcement learning that bridges the gap between explainability and interpretability through model explanations. Our method leverages Shapley value explanations to generate interpretable policies.

### Key Features

- **Model Training**: Built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/#) with optimized hyperparameters from [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- **Shapley Value Explanations**: Generate explanations for RL agent decisions
- **Interpretable Policy Generation**: Convert explanations into interpretable policies
- **Comprehensive Evaluation**: Tools for assessing interpretable policy performance

### Repository Structure

```
explain-to-interpret-rl/
├── model/              # Pre-trained models for generating Shapley explanations
├── vec_env/           # Normalized vectorized environment configurations
├── notebook/          # Jupyter notebooks with implementation examples
├── evaluation.py      # Evaluation script for interpretable policies
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Usage

### Interactive Examples

The main implementation is provided in easy-to-understand Jupyter notebooks located in the `notebook/` folder. These notebooks demonstrate:

- Model training and explanation generation
- Shapley value computation
- Interpretable policy extraction
- Visualization of results

### Evaluation

To evaluate the performance of interpretable policies:

```bash
python evaluation.py
```

The pre-trained models and environment configurations are provided for reproducibility and ease of use.


For questions, issues, or collaboration inquiries, please contact:
- **Peilang Li**: [peilang.li@my.utsa.edu](mailto:peilang.li@my.utsa.edu)

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
li2025from,
title={From Explainability to Interpretability: Interpretable Reinforcement Learning Via Model Explanations},
author={Peilang Li and Umer Siddique and Yongcan Cao},
booktitle={Reinforcement Learning Conference},
year={2025},
url={https://openreview.net/forum?id=kreQkWaOK5}
}
```
---

*Reinforcement Learning Conference (RLC) 2025.*