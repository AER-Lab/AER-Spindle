# Introduction

The AER Lab introduces a customized implementation of the SPINDLE method (Sleep Phase Identification with Neural Networks for Domain-invariant LEarning), originally developed by Miladinović et al. (2019), which achieved remarkable accuracy rates of up to 99% in rodent sleep scoring. Our adaptation maintains the core methodological principles of the original SPINDLE framework while introducing a streamlined graphical user interface for enhanced accessibility and workflow automation with an accuracy rate of up to 97.25%. The system processes paired EEG/EMG recordings stored in EDF format alongside their corresponding time-labeled CSV files, automatically converting them into structured datasets suitable for model training.

# Architecture

- Building upon SPINDLE's CNN architecture
  - Our implementation:
    - Remains agnostic to changes in sleep patterns across time and frequency dimensions
    - Preserves original model parameters
    - Adds convenient features for model weight management
    - Enables automated prediction generation
    - Signal Processing
    - The preprocessing pipeline incorporates sophisticated techniques:
    - Time-frequency domain operations
    - Multi-channel analysis inspired by ASR systems
    - Advanced artifact detection and handling

# Installation - Environment Setup

### Download:

- The latest stable python version [3.1x+, at least 3.10]
- Git [https://git-scm.com/downloads]
- OR
- Conda/mini conda

## Setup

## Clone repository

```
- git clone https://github.com/AER-Lab/AER-Spindle.git
- cd AER-Spindle
```

## Create an Environment

#### Option 1: Conda Environment

```
- conda create -n SPINDLE python=3.10
- conda activate SPINDLE
```

#### Option 2: Virtual Environment

```
python -m venv SPINDLE
source SPINDLE/bin/activate  # Linux/Mac
.\SPINDLE\Scripts\activate   # Windows
```

## Install dependencies

```
pip install -r requirements.txt
```

## Usage - GUI Interface

```
python Spindle_gui.py
```

# Core Features

- Data Visualization: Plot and analyze raw and Filtered EEG/EMG data.
- Model Training: Train custom sleep prediction models using labeled datasets
- Sleep Scoring: Automated prediction using our pre-trained model or your custom model
- Data Analysis [Sleep Architecture]
- Visualize spectrograms
- Review signal quality
- Model Training
- Configurable model parameters
- Train custom models
- Compare & Evaluate Performances
- Achieves up to 97.25% agreement with expert scoring
- Efficient processing of large datasets

# Acknowledgments - Special thanks to:

- [George Saad](https://github.com/gsaaad) and the [AER Lab](https://github.com/AER-Lab/AER-Spindle)
- Original SPINDLE method developers (Miladinović et al., 2019)
- amibeuret/spindle repository contributors
