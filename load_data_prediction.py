# Function to load all data
import os
import numpy as np
import pandas as pd
import torch
from preprocess_plot_spectrograms import preprocess_and_plot_edf
# Function to map labels


def load_data_prediction(file, params):
    all_spectrograms = []
    eeg_spectrograms = []
    emg_spectrogram = []
    
    eeg_file = file
            
    # Preprocess the EEG data
    data = preprocess_and_plot_edf(eeg_file, params)
    print("Data Shape:", data.shape)
    eeg1_spectrogram = data[0]
    emg_spectrogram = data[1]

    spectrograms = np.stack([eeg1_spectrogram, emg_spectrogram])
    
    # Append to list
    all_spectrograms.append(spectrograms)

    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Concatenate all spectrograms from all files
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    
    # Convert to PyTorch tensor
    all_spectrograms_tensor = torch.tensor(all_spectrograms, dtype=torch.float32)
    
    print("All spectrograms shape:", all_spectrograms_tensor.shape)
    print("Data Tensor Shape: ", data_tensor.shape)

    return data_tensor