import numpy as np
from scipy import signal
import mne
import pyedflib
from os import environ as os_environ
from pathlib import Path
import logging
import sys
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from torch.optim.optimizer import Optimizer
import torch
import math
logger = logging.getLogger('Spindle-AER' + '.' + __name__)


def classification_accuracy(predictions, labels):
    accuracy = (predictions.argmax(1) == labels).float().mean()
    return accuracy


def setup_logger_to_std():
    """Logger for debugging the code during development"""
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger('Spindle-AER')
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(f_format)
    root_logger.addHandler(handler)


def resample(eegs, from_srate, to_srate):
    n_eeg, eeg_length = np.shape(eegs)
    resampled_eegs = \
        np.zeros((n_eeg, int(eeg_length * (to_srate / from_srate))))
    for i, eeg in enumerate(eegs):
        resampled_eegs[i, :] = signal.resample(
            eeg, int(len(eeg)/from_srate*to_srate))

    return resampled_eegs



def get_spectrograms(eegs, srate, window, stride, mode='magnitude'):
    # print("Initiate creating spectrograms..")
    # Calculate padding to make the signal symmetric
    padding = window // 2 - stride // 2
    print("Padding is: ", padding, "which is window size: ()", window, "divided by 2) minus stride size: (", stride, "divided by 2)")
    # initialize the spectrogram array
    n_eeg, eeg_length = np.shape(eegs)
    print("Number of EEG signals: ", n_eeg, "EEG signal length: ", eeg_length)
    # Window is the size of the window for short-time Fourier transform
    # Fourier transform is process of converting a signal from time domain to frequency domain
    # which means unwinding the signal into its constituent frequencies [the make up of frequencies]
    spectrograms = np.zeros((n_eeg, window // 2 + 1, eeg_length // stride))
    # print("Spectrogram shape: ", spectrograms.shape)
    # Loop over the EEG signals and calculate the spectrogram
    for i, eeg in enumerate(eegs):
        padded_eeg = np.pad(eeg, pad_width=padding, mode='edge')
        f, t, eeg_spectrogram = signal.spectrogram(padded_eeg, fs=srate,
                                                   nperseg=window,
                                                   noverlap=window - stride,
                                                   scaling='density',
                                                   mode=mode)
        # normalize the spectrogram to scales of the original signal
        eeg_spectrogram = (eeg_spectrogram / 2) ** 2 / window
        # print("Spectrogram is divided by 2 and squared and divided by window size")
        spectrograms[i, ...] = np.log(eeg_spectrogram + 0.00000001)
        # print("Then the log of the spectrogram is taken and added a small value to avoid log(0), which avoids division by zero/Null/Undefined")
    return spectrograms


def compress_spectrograms(eeg_specs, srate, window,
                          lowcutoff=0.5, highcutoff=12):
    nu = np.fft.rfftfreq(window, 1.0 / srate)

    new_specs = []
    for spec in eeg_specs:
        new_specs.append(spec[np.logical_and(nu >= lowcutoff,
                                             nu <= highcutoff), :])

    new_specs = np.array(new_specs)
    if len(new_specs.shape) < 2:
        new_specs = new_specs[np.newaxis, ...]
    return np.array(new_specs)


def compress_and_replicate_emg(emg_specs, srate, window,
                               lowcutoff=2, highcutoff=30,
                               replicate=1):
    nu = np.fft.rfftfreq(window, 1.0 / srate)

    new_emg_specs = np.zeros(
        (emg_specs.shape[0], replicate, emg_specs.shape[2]))
    for idx, emg_spec in enumerate(emg_specs):
        new_emg_specs[idx, ...] = np.asarray(
            [np.mean(emg_spec[np.logical_and(
                nu >= lowcutoff, nu <= highcutoff), :], 0) for _ in
             range(replicate)])

    return new_emg_specs


def normalise_spectrograms(eeg_specs):
    # print("Normalising spectrograms..")
    new_eeg_specs = np.zeros(np.shape(eeg_specs))
    for j, spec in enumerate(eeg_specs):
        for i in range(np.shape(spec)[0]):
            new_eeg_specs[j, i, :] = \
                (spec[i] - np.mean(spec[i])) / np.std(spec[i])
    # print("Normalizing allows spectrograms to be compared on the same scale, by subtracting the mean and dividing by the standard deviation")
    return new_eeg_specs


def add_neighbors(spectrograms, num_neighbors):
    length = np.shape(spectrograms)[3]
    aug_data = np.zeros((np.shape(spectrograms)[0],
                         np.shape(spectrograms)[1],
                         np.shape(spectrograms)[2],
                         length * (num_neighbors + 1)))
    for i in range(np.shape(aug_data)[0]):
        for j in range(num_neighbors + 1):
            aug_data[i, :, :, j * length:(j + 1) * length] = \
                spectrograms[int((i + j - int(num_neighbors / 2)) %
                                 len(spectrograms)), :, :, :]

    return aug_data


def make_epochs(spectrograms, num_epochs, epoch_size):
    # print("Making epochs..")
    spectrum_size = np.shape(spectrograms)[1]
    data = np.ndarray(shape=(num_epochs,
                             len(spectrograms),
                             spectrum_size,
                             epoch_size))
    for i in range(num_epochs):
        for j in range(len(spectrograms)):
            data[i, j, :, :] = \
                spectrograms[j][:, i * epoch_size:(i + 1) * epoch_size]
    
    return data
def plot_spectrogram(spectrogram, title):
    plt.figure()
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()