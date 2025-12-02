# Emotion Agent: Unsupervised Deep Reinforcement Learning with Distribution-Prototype Reward for Continuous Emotional EEG Analysis (Neurocomputing 2025)
### Zhihao Zhou (stewenz@163.com), Li Zhang, Qile Liu, Gan Huang, Zhuliang Yu, Zhen Liang*.
This repository contains the official implementation of the paper "Emotion Agent: Unsupervised Deep Reinforcement Learning with Distribution-Prototype Reward for Continuous Emotional EEG Analysis," accepted in Neurocomputing 2025.
- [Neurocomputing 2025](https://doi.org/10.1016/j.neucom.2025.130951)


# Installation:
- Python 3.9.17
- PyTorch 2.0.1 (CUDA 11.8)
- NVIDIA CUDA 11.8
- NumPy 1.23.5
- Scikit-learn 1.3.0
- SciPy 1.11.3

# Preliminaries
- Prepare dataset: SEED, SEED-IV and DEAP
- Extract differential entropy features of EEG using a 1-second sliding window

# Training
- Emotion Agent model definition file: Agent.py
- Pipeline of the Emotion Agent: rl_utils.py
- implementation of training: train.py
