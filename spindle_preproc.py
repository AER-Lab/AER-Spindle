import numpy as np
import logging
import utils
from utils import get_spectrograms as get_specs
from utils import compress_spectrograms as compress
from utils import compress_and_replicate_emg as comp_emg


logger = logging.getLogger('Spindle-AER' + '.' + __name__)


class SpindlePreproc:
    # Initialize the preprocessing class with the parameters required
    def __init__(self, params):
        self.name = params['name']
        self.target_srate = params['target_srate']
        self.stride = params['spectrogram-stride']
        self.time_interval = params['time_interval']
        self.num_neighbors = params['num_neighbors']
        self.eeg_filtering = params['EEG-filtering']
        self.emg_filtering = params['EMG-filtering']

        logger.debug('preprocessing routine created: \n {0}'.format(self))
    # Calling the class will run this body
    def __call__(self, all_signals, signal_header,
                 eeg_idx=np.array([]),
                 eog_idx=np.array([]),
                 emg_idx=np.array([])):
        
        # Initial setup for the preprocessing and logging
        logger.debug('preprocessing data ...')
        logger.debug(f'eeg indices: {eeg_idx} and emg indices: {emg_idx} '
                     f'vs signal shape: {len(signal_header)}')
        print("Spindle Preproc - ALL SIGNALS SHAPE--RAW",all_signals.shape)
        # Downsample the signals to the target sample rate
        downsampled_signals = []
        for i, sig in enumerate(all_signals):
            srate = signal_header[i]['sample_rate']
            if srate != self.target_srate:
                downsampled_signals.append(utils.resample(
                    sig[np.newaxis], srate, self.target_srate))
            else:
                if isinstance(sig, list):
                    downsampled_signals.append(sig)
                else:
                    downsampled_signals.append(sig[np.newaxis, ...])
        # Concatenate the downsampled signals
        all_signals = np.concatenate(downsampled_signals)
        print("Spindle Preproc - ALL SIGNALS SHAPE--Re-sampled",all_signals.shape)
        srate = self.target_srate
        stride = self.stride
        win = srate * 2
        print("Sample Rate: ", srate, "Stride-Steps in # Samples: ", stride, "Window Size: ", win, f"Which is sample rate: {srate} * 2")
        # Get the spectrograms for the EEG and EMG signals
        specs = get_specs(all_signals, srate, win, stride, mode='magnitude')
        print("GET-SPEC-Spectrogram shapes: ", specs.shape)

        # comparess the spectrograms by filtering the EEG and EMG signals through the low and high cutoff frequencies
        eeg_spectrograms = compress(specs[eeg_idx, ...], srate, win,
                                    lowcutoff=self.eeg_filtering['lfreq'],
                                    highcutoff=self.eeg_filtering['hfreq'])
        emg_spectrogram = comp_emg(specs[emg_idx, ...], srate, win,
                                   lowcutoff=self.emg_filtering['lfreq'],
                                   highcutoff=self.emg_filtering['hfreq'],
                                   replicate=np.shape(eeg_spectrograms)[1])
        
        print("COMPRESS-SPEC-Spectrogram EEG Shape: ", eeg_spectrograms.shape)
        print("COMPRESS-SPEC-Spectrogram EMG Shape: ", emg_spectrogram.shape)
        
        # Concatenate the EEG and EMG spectrograms
        specs = [x for x in [eeg_spectrograms, emg_spectrogram] if x.size > 0]
        specs = np.concatenate(specs, axis=0)
        # Normalize the spectrograms 
        specs = utils.normalise_spectrograms(specs)
        print("NORMALIZE-SPEC-Spectrogram shapes: ", specs.shape)

        samples_per_epoch = int(self.time_interval * srate)
        print("Samples per epoch which is time interval * sample rate: ", self.time_interval, "*", srate, "=", samples_per_epoch)
        # This is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // self.stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        print("Samples per Epoch is: ", samples_per_epoch)
        print("Epoch Size is calculated by dividing samples per epoch by stride: ", samples_per_epoch, "/", self.stride, "=", epoch_size)
        print("Number of Epochs is calculated by dividing the length of the signal by samples per epoch: ", len(all_signals[0]), "/", samples_per_epoch, "=", num_epochs)
        data = utils.make_epochs(specs, num_epochs, epoch_size)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data
    # Str is used to print the parameters of the class
    def __str__(self):
        return f"""Preprocessing class SpindlePreprocessing replicates the 
        exact preprocessing introduced in the paper.
        parameters:
        target_srate: {self.target_srate}
        stride: {self.stride}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        eeg_filtering: lfreq={self.eeg_filtering['lfreq']}, 
                       hfreq={self.eeg_filtering['hfreq']}
        emg_filtering: lfreq={self.emg_filtering['lfreq']}, 
                       hfreq={self.emg_filtering['hfreq']}
        """
