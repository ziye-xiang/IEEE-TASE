# Core Source Code for IEEE T-ASE Manuscript

Copyright (c) 2026 The Authors. All rights reserved.

This repository provides the core source code associated with the manuscript submitted to:

**IEEE Transactions on Automation Science and Engineering (IEEE T-ASE)**  
**Manuscript ID: T-ASE-2026-2601**

The released code provides the core implementation of the proposed QMIX-based resource allocation, TDM scheduling, and dual self-triggered distributed model predictive control framework.

The code is provided for academic research purposes. Redistribution or commercial use requires prior permission from the authors.

## Core Files

- `QMIX_Agent.m` - decentralized action selection, experience replay, Double-Q temporal-difference update, local Q-network training, and target-network synchronization.
- `QMIXMixer.m` - monotonic QMIX mixing network with state-conditioned hypernetworks.
- `QMIXMLP.m` - fully connected neural network and Adam optimizer used by the local Q-networks and mixer hypernetworks.
- `QMIX_train_core.m` - core QMIX training procedure for TDM resource allocation.
- `QMIX_evaluate_core.m` - greedy policy evaluation routine.
- `MAS_QMIX_DSTDMPC_environment_core.m` - core coupling interface between QMIX resource allocation and the Dual ST-DMPC/TDM control environment, including normalized team-reward computation.
- `choose_sample_interval.m` - TDM discrete-clock routine for determining the next admissible controller execution instant.
- `judgment_trigger.m` - TDM trigger-time correction routine for candidate triggering instants.
- `solveOptimalControlProblem.m` - finite-horizon DMPC optimization routine based on `fmincon`.

## Framework

The core implementation follows a centralized-training and decentralized-execution QMIX architecture. Each agent selects its action from local observations, while the monotonic mixing network combines individual action values into the global action value used during centralized training. The selected actions determine TDM resource-allocation patterns for the local Dual ST-DMPC controllers.

The normalized team reward combines resource-allocation performance and control performance as

`r_i = F_i * S_U,i + G_i * S_x,i`,

with the team reward obtained by summing the individual rewards over all agents.

## MATLAB

The source files are written in MATLAB. The MPC optimization routine uses `fmincon`.
