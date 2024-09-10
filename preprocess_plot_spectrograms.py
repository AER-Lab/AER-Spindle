import numpy as np
from spindle_preproc import SpindlePreproc
from utils import setup_logger_to_std,plot_spectrogram
import pyedflib


def preprocess_and_plot_edf(eeg_file, params):
    # Set the logger to print to stdout. You could skip this and ignore logs.
    setup_logger_to_std()
    
    # Load the EEG/EMG data from the EDF file
    # print("Loading EEG/EMG...", )
    all_signals, signal_header, header = pyedflib.highlevel.read_edf(eeg_file)
    # print("All signals:", all_signals)
    # print("Signal header:", signal_header)
    # print("Header:", header)
    # Preprocess the data
    # print("Preprocessing data...")
    preprocessing = SpindlePreproc(params)
    # data = preprocessing(all_signals, signal_header, np.array([0, 1]), np.array([]), np.array([2])
    data = preprocessing(all_signals, signal_header, np.array([0]), np.array([]), np.array([1]))

    # Extract spectrograms
    eeg1_spectrogram = data[0]
    # eeg2_spectrogram = data[1]
    emg_spectrogram = data[1]

    # Display the shape of the spectrograms
    print("Data shape:", data[0].shape, data[1].shape, data[2].shape)
    print("EEG1 Spectrogram Shape:", eeg1_spectrogram.shape)
    # print("EEG2 Spectrogram Shape:", eeg2_spectrogram.shape)
    print("EMG Spectrogram Shape:", emg_spectrogram.shape)

    # # Plot the spectrograms
    # plot_spectrogram(eeg1_spectrogram[0], "EEG1 Spectrogram")
    # plot_spectrogram(eeg2_spectrogram[0], "EEG2 Spectrogram")
    # plot_spectrogram(emg_spectrogram[0], "EMG Spectrogram")


    return data

