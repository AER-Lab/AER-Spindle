import numpy as np
from scipy import signal
import mne
import pyedflib
import constants as cnst
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



def load_edf(eeg_file):
    """This function uses mne library.
    WARNING: The function has strange behaviour, please don't use"""
    data = mne.io.read_raw_edf(eeg_file)

    sfreq = data.info['sfreq']
    signals = data.get_data()
    return signals, int(sfreq)


def plot_spectrogram(spectrogram, title):
    plt.figure()
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    
def classification_accuracy(predictions, labels):
    accuracy = (predictions.argmax(1) == labels).float().mean()
    return accuracy

def set_up_paths(root_path, data_path=None, weights_path=None, tmp=False):

    cnst.ROOT_PATH = root_path
    cnst.EXPERIMENTS_PATH = root_path / cnst.EXPERIMENTS_FOLDER
    cnst.DATA_PATH = root_path / cnst.DATA_FOLDER if not data_path else data_path
    cnst.WEIGHTS_PATH = (root_path / cnst.WEIGHTS_FOLDER if not weights_path
                         else weights_path)
    cnst.TMP_DIR = Path(os_environ['TMPDIR'] if tmp else Path(''))

    exp_id = str(time.time())
    cnst.EXPERIMENT_PATH = cnst.EXPERIMENTS_PATH / exp_id

    cnst.EXPERIMENTS_PATH.mkdir(parents=True, exist_ok=True)
    cnst.WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger_to_std():
    """Logger for debugging the code during development"""
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger(cnst.ROOT_LOGGER_STR)
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(f_format)
    root_logger.addHandler(handler)



def take_majority(labels, axis):
    u, indices = np.unique(labels, return_inverse=True)

    labels = np.apply_along_axis(np.bincount,
                                 axis,
                                 indices.reshape(labels.shape),
                                 None, np.max(indices) + 1)
    labels = u[np.argmax(labels, axis=axis)]
    return labels


def load_pyedf(eeg_file):
    f = pyedflib.EdfReader(eeg_file)
    n = f.signals_in_file
    signals = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        signals[i, :] = f.readSignal(i)
    srate = f.getSignalHeaders()[0]['sample_rate']
    return signals, srate


def slice_resample(eegs, from_srate, to_srate):
    new_len = int(len(eegs[0]) - (len(eegs[0]) % (from_srate * 4)))
    n_channels = 3
    lslice = 2 ** 15 * 100
    num_wholes = new_len // lslice
    remaining = new_len % lslice
    channels_new = [None] * n_channels
    for c in range(n_channels):
        channels_new[c] = []
        for i in range(num_wholes):
            channels_new[c].extend(
                list(signal.resample(eegs[c][i * lslice:(i + 1) * lslice],
                                     lslice // from_srate * to_srate)))
        # resample remaining part
        channels_new[c].extend(list(signal.resample(
            eegs[c][num_wholes * lslice:],
            remaining // from_srate * to_srate)))
    eegs = np.array(channels_new)
    return eegs


def resample(eegs, from_srate, to_srate):
    n_eeg, eeg_length = np.shape(eegs)
    resampled_eegs = \
        np.zeros((n_eeg, int(eeg_length * (to_srate / from_srate))))
    for i, eeg in enumerate(eegs):
        resampled_eegs[i, :] = signal.resample(
            eeg, int(len(eeg)/from_srate*to_srate))

    return resampled_eegs


def pop_eeg_filtnew(signal, srate, l_freq, h_freq):

    return mne.filter.filter_data(
        signal, srate, l_freq, h_freq,
        picks=None, filter_length='auto', l_trans_bandwidth='auto',
        h_trans_bandwidth='auto', n_jobs='cuda', method='fir',
        iir_params=None, copy=True,phase='zero', fir_window='hamming',
        fir_design='firwin', pad='reflect_limited', verbose=False)


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


def get_spectrograms_vec(eegs, srate, stride, spec_mode='magnitude'):
    """Initially I thought this would be faster than the previous function,
    but not necessarily"""
    window = srate * 2
    padding = window // 2 - stride // 2

    padded_eeg = np.pad(eegs,
                        pad_width=((0, 0), (padding, padding)),
                        mode='edge')
    f, t, eeg_spectrogram = signal.spectrogram(padded_eeg, fs=srate,
                                               nperseg=window,
                                               noverlap=window - stride,
                                               scaling='density',
                                               mode=spec_mode)
    eeg_spectrogram = (eeg_spectrogram / 2) ** 2 / window
    spectrograms = np.log(eeg_spectrogram + 0.00000001)
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


def replicate_emg(emg_specs, replicate=1):
    new_emg_specs = np.zeros((emg_specs.shape[0],
                              replicate,
                              emg_specs.shape[2]))
    for idx, emg_spec in enumerate(emg_specs):
        new_emg_specs[idx, ...] = np.asarray([np.mean(emg_spec, 0)
                                              for _ in range(replicate)])

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


def normalise_spectrogram_epochs(eeg_specs):
    """Exactly as the previous normalisation, only can be applied after
    epochs are separated"""
    mean = np.mean(eeg_specs, axis=(0, 3), keepdims=True)
    std = np.std(eeg_specs, axis=(0, 3), keepdims=True)
    norm = (eeg_specs - mean) / (std + 1e-10)
    return norm


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

def plot_confusion_mat(y_true, y_pred, classes,
                       normalize=False,
                       title=None,
                       cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.debug("Normalized confusion matrix")
    else:
        logger.debug('Confusion matrix, without normalization')

    logger.debug(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.ylim(cm.shape[0] + 0.5, 0 - 0.5)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

class RAdam(Optimizer):
    """Copy pasted from https://github.com/LiyuanLucasLiu/RAdam
    based on https://arxiv.org/abs/1908.03265 """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
