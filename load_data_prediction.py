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

    # TODO [COMPLETE]: Try to run a loop over shape of data to create the np.stack [to prevent 2 eeg channels vs. 1 eeg channels.]
    # lOOP over the shape of data dynamically such that the last channel is always the EMG channel, and the rest are EEG channels
    # for i in range(data.shape[0]):  # Assuming data has shape (channels, time, frequency)
    #     if i == data.shape[0] - 1:  # If it's the last channel, it's EMG
    #         emg_spectrogram = data[i]
    #     else:  # All other channels are EEG
    #         eeg_spectrograms.append(data[i])

    # # Convert the list of EEG spectrograms into a numpy array using np.stack
    # eeg_spectrograms = np.stack(eeg_spectrograms)
    # print("EEG Spectrograms Shape:", eeg_spectrograms.shape)
    # print("EMG Spectrogram Shape:", emg_spectrogram.shape)
    # spectrograms = np.stack([eeg_spectrograms, emg_spectrogram])
    # ? previous implementation of hard coding the expected number of channels as per the spindle data set
    # eeg1_spectrogram = data[0]
    # eeg2_spectrogram = data[1]
    # emg_spectrogram = data[2]
    
    # Stack spectrograms along the channel axis
    # spectrograms = np.stack([eeg1_spectrogram, eeg2_spectrogram, emg_spectrogram], axis=1)
    
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