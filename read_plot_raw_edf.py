import pandas as pd
import matplotlib.pyplot as plt
from pyedflib import highlevel
from scipy.signal import butter, sosfiltfilt, buttord
import numpy as np

# Define function to read and plot EDF file data V1
def read_plot_raw_edf(edf_file, time_unit='minutes'):
    # Multiplier based on the chosen time unit
    time_multiplier = {
        'seconds': 1,
        'minutes': 1 / 60,
        'hours': 1 / 3600
    }.get(time_unit, 1)  # Default to seconds if an invalid unit is passed

    # Read meta data - channel names
    metaData = highlevel.read_edf_header(edf_file)
    print("Meta Data:", metaData)
    channels = metaData['SignalHeaders']
    channels = [channel['label'] for channel in channels]

    try:
        # Read EDF file
        signals, meta_data, meta_data2 = highlevel.read_edf(edf_file)
    except Exception as e:
        print(f"An error occurred while reading the EDF file: {e}")
        return None
    # resample data to 256Hz
    resample_hz = 256
    signals = highlevel.resample_signal(signals, meta_data, resample_hz)
    print("Resampled data to 256Hz")
    # Convert signals to a DataFrame and transpose
    df = pd.DataFrame(signals).T
    print("Dataframe shape: ", df.shape)

    # Check the number of channel pairs
    channel_pairs = round(df.shape[1] / 2, 0)
    if channel_pairs == 1:
        print("There is ", channel_pairs, "pair of channels in the dataframe")
    else:
        print("There are ", channel_pairs, "pairs of channels in the dataframe")

    # Get the sample frequency from metadata
    sample_frequency = meta_data[0]['sample_rate']
    # if no sample_rate, try sample_frequency
    if sample_frequency is None:
        sample_frequency = meta_data[0]['sample_frequency']
    print("Sample frequency: ", sample_frequency, "Hz")

    # Create a time axis in the specified unit
    total_samples = df.shape[0]
    time_axis = [i / sample_frequency * time_multiplier for i in range(total_samples)]

    # Plot each channel pair over the time axis
    for i in range(0, df.shape[1], 2):
        print("Plotting Pair: ", i, f"Channel {channels[0]}: ", i, f"Channel {channels[1]}: ", i + 1)

        channel1 = df.iloc[:, i]
        channel2 = df.iloc[:, i + 1]

        # Plot channels with time axis
        plt.figure(figsize=(12, 8))
        
        # Plot Channel 1
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, channel1, linewidth=0.3)
        plt.title(f"EEG - (Ch. {channels[0]})")
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel("Amplitude")

        # Plot Channel 2
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, channel2, linewidth=0.1)  # Thinner line for Channel 2
        plt.title(f"EMG - (Ch. {channels[1]})")
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

# bandpass filter
def bandpass_filter_channel(data, fs, lowcut, highcut, order=4, 
                            automatic_order=True, transition_ratio=0.2, debug=False):
    """
    Simplified dynamic bandpass filter for biosignals with optional automatic order selection,
    similar in functionality to MATLAB's bandpass function.
    
    Parameters:
        data (np.ndarray): 1D signal array.
        fs (float): Sampling frequency in Hz.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        order (int): Default filter order used when automatic_order is False (default is 4).
        automatic_order (bool): If True, calculates the filter order automatically using a
                                transition band determined by transition_ratio.
        transition_ratio (float): Ratio to calculate the transition band width (only used if
                                  automatic_order is True).
        debug (bool): If True, prints filter parameters for debugging.
    
    Returns:
        np.ndarray: Filtered signal with zero-phase distortion.
        
    Raises:
        ValueError: If the provided cutoff frequencies are invalid or if the final filter
                    order is outside the acceptable range [2, 30].
    """
    # Compute Nyquist frequency
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Cutoff frequencies must satisfy: 0 < {lowcut} < {highcut} < {nyq}")

    # Normalize passband frequencies
    Wn = np.array([lowcut, highcut]) / nyq

    if automatic_order:
        # Compute the transition bandwidth using the given ratio
        transition = transition_ratio * (highcut - lowcut)
        # Ensure stopband frequencies are within valid limits
        stop_low = max(lowcut - transition, 0.1)  # Avoid a zero or negative frequency
        stop_high = min(highcut + transition, nyq * 0.99)
        Ws = np.array([stop_low, stop_high]) / nyq

        # Determine the minimum filter order required to meet the criteria:
        # passband ripple (gpass) must be within 1.5 dB and stopband attenuation (gstop) is at least 20 dB.
        order, Wn = buttord(wp=Wn, ws=Ws, gpass=1.5, gstop=20)
        if debug:
            print(f"Auto-calculated order: {order}")

    # Validate final filter order to ensure it's practical
    if order < 2 or order > 30:
        raise ValueError(f"Invalid filter order: {order} (must be between 2 and 30)")

    if debug:
        print(f"Using order: {order}, Filter Cutoffs: {Wn * nyq} Hz")
    else:
        print("Using order:", order, "Cutoffs:", Wn * nyq, "Hz")

    try:
        sos = butter(order, Wn, btype='band', output='sos')
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
    except Exception as e:
        raise RuntimeError(f"Filter application failed: {str(e)}")

def plot_comparison(time_axis, orig_eeg, filt_eeg, orig_emg, filt_emg, time_unit='minutes'):
    """
    Plots a side-by-side comparison of the original and band-pass filtered data
    for both EEG and EMG channels.

    Parameters:
      time_axis (array): Common time axis for plotting.
      orig_eeg (array): Original EEG data.
      filt_eeg (array): Filtered EEG data.
      orig_emg (array): Original EMG data.
      filt_emg (array): Filtered EMG data.
      time_unit (str): Time unit label for x-axis.
    """
    plt.figure(figsize=(12, 12))
    
    # Original EEG Plot
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, orig_eeg, label="Original EEG", linewidth=0.3)
    plt.title("Original EEG")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Filtered EEG Plot
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, filt_eeg, label="Filtered EEG", linewidth=0.3)
    plt.title("Filtered EEG")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Original EMG Plot
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, orig_emg, label="Original EMG", linewidth=0.1)
    plt.title("Original EMG")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Filtered EMG Plot
    plt.subplot(4, 1, 4)
    plt.plot(time_axis, filt_emg, label="Filtered EMG", linewidth=0.1)
    plt.title("Filtered EMG")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def bandpass_plot_data(edf_file, eeg_low, eeg_high, emg_low, emg_high):
    # Read the EDF file
    signals, meta_data, meta_data2 = highlevel.read_edf(edf_file)
    print("Meta Data:", meta_data, "Meta Data 2:", meta_data2)
    fs = meta_data[0]['sample_rate']
    if fs is None:
        fs = meta_data[0]['sample_frequency']
    print("Sample frequency: ", fs, "Hz")

    # Bandpass filter the EEG and EMG channels
    channel1 = signals[0]
    channel2 = signals[1]
    # resample to 256Hz
    resample_hz = 256
    from scipy.signal import resample

    num_samples = int(len(channel1) * resample_hz / fs)
    channel1 = resample(channel1, num_samples)
    channel2 = resample(channel2, num_samples)
    eeg_lowcut = eeg_low
    eeg_highcut = eeg_high
    emg_lowcut = emg_low
    emg_highcut = emg_high
    filt_channel1 = bandpass_filter_channel(channel1, fs, eeg_lowcut, eeg_highcut)
    filt_channel2 = bandpass_filter_channel(channel2, fs, emg_lowcut, emg_highcut)

    # Plot the comparison of original and filtered data
    total_samples = len(channel1)
    time_axis = [i / fs / 60 for i in range(total_samples)]
    plot_comparison(time_axis, channel1, filt_channel1, channel2, filt_channel2, time_unit='minutes')



    
    
