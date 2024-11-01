# AER Lab Documentation - Spindle Method for Scoring Sleep States in Mice

## Introduction
The AER Lab introduces a customized implementation of the SPINDLE method (Sleep Phase Identification with Neural Networks for Domain-invariant LEarning), originally developed by Miladinović et al. (2019), which achieved remarkable accuracy rates of up to 99% in rodent sleep scoring1. Our adaptation maintains the core methodological principles of the original SPINDLE framework while introducing a streamlined graphical user interface for enhanced accessibility and workflow automation. The system processes paired EEG/EMG recordings stored in EDF format alongside their corresponding time-labeled CSV files, automatically converting them into structured datasets suitable for model training and prediction. Building upon SPINDLE's CNN-HMM architecture, which was specifically designed to remain agnostic to changes in sleep patterns across both time and frequency dimensions, our implementation preserves the original model parameters while adding convenient features for model weight management and automated prediction generation. The tool's preprocessing pipeline incorporates the same sophisticated signal processing techniques used in the original framework, including time-frequency domain operations and multi-channel analysis inspired by automatic speech recognition systems. Following the successful validation approach of the original SPINDLE method, our system includes a comprehensive evaluation suite that enables detailed performance analysis and comparison between model predictions and expert scoring. Drawing inspiration from the amibeuret/spindle repository, we have enhanced the framework with additional user-centric features while maintaining the high accuracy standards and physiological plausibility that made the original method so effective.

## Installation
Create a Conda or Virtual Environment for this project, clone the codebase, and install the requirements:
    1. conda create -n SPINDLE python=3.9
        OR
        python -m venv SPINDLE
    2.  clone the codebase
    3.  pip install -r requirements.txt

## Quick start - GUI
A GUI-based implementation of the SPINDLE method for plotting raw data, training models, and automated sleep scoring in mice.






