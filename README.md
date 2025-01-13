# Introduction
The AER Lab introduces a customized implementation of the SPINDLE method (Sleep Phase Identification with Neural Networks for Domain-invariant LEarning), originally developed by Miladinović et al. (2019), which achieved remarkable accuracy rates of up to 99% in rodent sleep scoring. Our adaptation maintains the core methodological principles of the original SPINDLE framework while introducing a streamlined graphical user interface for enhanced accessibility and workflow automation with an accuracy rate of up to 97.25%. The system processes paired EEG/EMG recordings stored in EDF format alongside their corresponding time-labeled CSV files, automatically converting them into structured datasets suitable for model training.

# Architecture
- Building upon SPINDLE's CNN-HMM architecture 
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
- Git [https://git-scm.com/downloads] - If cloning the repo, otherwise you can download the zip folder directly and extract.
- Conda/mini conda - If usins this method.
## Option 1: Conda Environment
```
- conda create -n SPINDLE python=3.10
- conda activate SPINDLE
```

##  Option 2: Virtual Environment
```
- python -m venv SPINDLE
- source SPINDLE/bin/activate  # Linux/Mac
- .\SPINDLE\Scripts\activate   # Windows
```

## Repository Setup
1. ###  Clone repository
```
- git clone https://github.com/AER-Lab/AER-Spindle.git
- cd SPINDLE_Sleep_Prediction
```

# Install dependencies
```
- pip install -r requirements.txt
```

# Usage - GUI Interface
- Launch the graphical interface:
```
    - python Spindle_gui.py
```

# Core Features
- Data Visualization: Plot and analyze raw EEG/EMG data
- Model Training: Train custom models using labeled datasets
- Sleep Scoring: Automated prediction using pre-trained or custom models
- Data Analysis
- Read and plot raw data
- Visualize spectrograms
- Review signal quality
- Model Training
- Select training data folder
- Configure model parameters
- Train custom models
- Sleep Scoring
- Load model weights
- Process new recordings
- Generate predictions
- Compare Performance
- Achieves up to 97.25% agreement with expert scoring
- Efficient processing of large datasets


# Acknowledgments - Special thanks to:
- [George Saad](https://github.com/gsaaad) and the AER Lab 
- Original SPINDLE method developers (Miladinović et al., 2019)
- amibeuret/spindle repository contributors