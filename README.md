# NVIDIA PhysicsNeMo Examples in Google Colab

This repository demonstrates how to set up and run NVIDIA's PhysicsNeMo framework within a Google Colab environment. It covers the installation of core components, `PhysicsNeMo-Sym` for symbolic computations, and the execution of example models like `darcy_fno` and `ldc_pinns`.

## Table of Contents
- [Introduction](#introduction)
- [PhysicsNeMo Architecture Overview](#physicsnemo-architecture-overview)
- [Google Colab Setup](#google-colab-setup)
  - [Mount Google Drive](#mount-google-drive)
  - [Clone PhysicsNeMo Repository](#clone-physicsnemo-repository)
  - [Install Core PhysicsNeMo](#install-core-physicsnemo)
  - [Install PhysicsNeMo-Sym](#install-physicsnemo-sym)
  - [Install NVIDIA DALI](#install-nvidia-dali)
- [Running Examples](#running-examples)
  - [Darcy FNO Example (`darcy_fno`)](#darcy-fno-example-darcy_fno)
  - [LDC PINNs Example (`ldc_pinns`)](#ldc-pinns-example-ldc_pinns)
- [Troubleshooting & Notes](#troubleshooting--notes)

## Introduction
PhysicsNeMo is NVIDIA's next-generation AI4Science framework designed to unify various AI-driven simulation techniques, including PINNs, Neural Operators, and CFD surrogates. This guide provides a step-by-step process to get PhysicsNeMo running in Google Colab, focusing on its main components and illustrative examples.

## PhysicsNeMo Architecture Overview

PhysicsNeMo is designed as a unified container to integrate multiple AI4Science approaches:

*   **PhysicsNeMo (Mainline)**: The overarching framework aiming to consolidate different modules.
*   **PhysicsNeMo-Sym (Symbolic)**: Historically known as `NVIDIA Modulus`, this is the core for PINN (Physics-Informed Neural Networks) and PDE symbolic systems, featuring PDE symbolic expression, automatic differentiation (AD), and boundary/constraint systems. It is being absorbed into the main PhysicsNeMo framework but often requires separate installation.
*   **physicsnemo-CFD**: A sub-module of PhysicsNeMo focused on CFD inference and engineering integration. It is not a training framework itself but rather an interface layer to embed trained AI models into CFD workflows (e.g., for inference, benchmarking, or hybrid CFD initialization).

## Google Colab Setup

The following steps were executed in a Google Colab environment.

### Mount Google Drive
This step is crucial for persistent storage of cloned repositories and training outputs.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Clone PhysicsNeMo Repository
Navigate to your desired workspace and clone the official NVIDIA PhysicsNeMo repository.

```bash
%cd /content/drive/MyDrive/AIforScience/PhysicsNemo/
!git clone https://github.com/NVIDIA/physicsnemo.git
%cd /content/drive/MyDrive/AIforScience/PhysicsNemo/physicsnemo
```

### Install Core PhysicsNeMo
Install the main PhysicsNeMo package from PyPI. A runtime restart might be necessary in Colab after installation for changes to take full effect.

```bash
!python -m pip install --upgrade pip setuptools wheel
!pip install nvidia-physicsnemo
```

Verify installation:

```bash
!python -c "import physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__)"
```

### Install PhysicsNeMo-Sym
Since some examples (like PINNs) rely on symbolic capabilities, `nvidia-physicsnemo.sym` needs to be installed separately. `Cython` is a prerequisite.

```bash
!pip install "Cython"
!pip install nvidia-physicsnemo.sym --no-build-isolation
```

Verify installation:

```bash
!python -c "import physicsnemo.sym; print('PhysicsNeMo Symbolic version:', physicsnemo.sym.__version__)"
```

### Install NVIDIA DALI
Some examples, particularly those involving data pipelines or specific hardware acceleration, might require NVIDIA DALI. If you encounter `ModuleNotFoundError: No module named 'nvidia.dali'`, install it based on your CUDA version (e.g., `cuda120`).

```bash
!pip install nvidia-dali-cuda120
```

## Running Examples

### Darcy FNO Example (`darcy_fno`)
This example demonstrates training a Fourier Neural Operator (FNO) model for Darcy flow. `wandb` (Weights & Biases) is used for experiment tracking.

Navigate to the example directory, install its specific requirements, and run the training script.

```bash
%cd /content/drive/MyDrive/AIforScience/PhysicsNemo/physicsnemo/examples/cfd/darcy_fno
!pip install -r requirements.txt
!pip install wandb
!wandb login # Follow prompts to log in
```

Run training:

```bash
!python train_fno_darcy_wandb_v3.py
```

Or override configuration parameters directly:

```bash
!python train_fno_darcy_wandb_v3.py training.batch_size=16
```

*Note: During execution, a `Warp DeprecationWarning: The symbol `warp.context.Device` will soon be removed` might appear. This is a warning related to the `warp-lang` library and might not prevent the training from proceeding, but it indicates an upcoming API change.*

### LDC PINNs Example (`ldc_pinns`)
This example showcases a Physics-Informed Neural Network (PINN) for a Lid-Driven Cavity (LDC) problem, utilizing `physicsnemo.sym`.

Navigate to the example directory and run the training script:

```bash
%cd /content/drive/MyDrive/AIforScience/PhysicsNemo/physicsnemo/examples/cfd/ldc_pinns/
!python train.py
```

*Note: Initial attempts resulted in a `ModuleNotFoundError: No module named 'nvidia.dali'`, which was resolved by installing `nvidia-dali-cuda120`. After fixing this, multiprocessing errors related to DALI (`multiprocessing/synchronize.py`) were observed, suggesting potential compatibility issues or resource contention in the Colab environment for this specific example.*

## Troubleshooting & Notes

*   **`ModuleNotFoundError: physicsnemo.sym`**: Ensure `nvidia-physicsnemo.sym` is installed with `!pip install nvidia-physicsnemo.sym --no-build-isolation`.
*   **`ModuleNotFoundError: No module named 'nvidia.dali'`**: Install `nvidia-dali-cuda120` (or the appropriate CUDA version) if this error occurs in examples like `ldc_pinns`.
*   **`Warp DeprecationWarning`**: These warnings indicate future API changes in the `warp-lang` library and usually do not halt execution but are worth noting for future compatibility.
*   **Multiprocessing Errors in `ldc_pinns`**: Even after installing DALI, some multiprocessing-related errors were observed during the `ldc_pinns` execution in Colab. This might require further investigation into DALI's multiprocessing configuration or Colab's environment limitations for such setups.
