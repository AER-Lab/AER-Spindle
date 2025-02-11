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
def bandpass_filter_channel(data, fs, lowcut, highcut, gpass=3, gstop=40, transition_ratio=0.2, debug=False):
    """
    Bandpass filter with automatic order selection using buttord.
    
    Parameters:
        data (array): Input signal
        fs (float): Sampling frequency (Hz)
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)
        gpass (float): Passband ripple (dB) - default 3dB
        gstop (float): Stopband attenuation (dB) - default 40dB
        transition_ratio (float): Transition band width as ratio of passband width
        debug (bool): Show debug info
    
    Returns:
        np.ndarray: Filtered signal
    """
    # Validation
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    nyquist = 0.5 * fs
    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(f"Invalid cutoff frequencies. Must satisfy: 0 < {lowcut} < {highcut} < {nyquist}")

    # Calculate transition bandwidth (20% of passband width by default)
    passband_width = highcut - lowcut
    transition = transition_ratio * passband_width
    
    # Set stopband edges with safety margins
    stop_low = max(lowcut - transition, 0.1)  # Prevent 0Hz
    stop_high = min(highcut + transition, nyquist * 0.99)  # Prevent Nyquist
    
    # Normalize frequencies for digital filter design
    Wp = np.array([lowcut, highcut]) / nyquist
    Ws = np.array([stop_low, stop_high]) / nyquist

    # Calculate optimal order and natural frequency
    order, Wn = buttord(Wp, Ws, gpass, gstop)
    print("Order: ", order)
    
    if debug:
        print(f"Optimal order: {order}")
        print(f"Natural frequency: {Wn * nyquist} Hz")
        print(f"Stopbands: [{stop_low:.1f}, {stop_high:.1f}] Hz")

    # Design SOS filter (better numerical stability)
    sos = butter(order, Wn, btype='band', output='sos')
    
    # Zero-phase filtering (forward + backward)
    filtered = sosfiltfilt(sos, data)
    
    return filtered


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
    fs = meta_data[0]['sample_rate']
    print("Sample frequency: ", fs, "Hz")

    # Bandpass filter the EEG and EMG channels
    channel1 = signals[0]
    channel2 = signals[1]
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



    
    
