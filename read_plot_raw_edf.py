import pandas as pd
import matplotlib.pyplot as plt
from pyedflib import highlevel
from scipy.signal import butter, filtfilt

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
def bandpass_filter_channel1(data, fs, lowcut, highcut, order=5):
    """
    Bandpass filter for channel1 (EEG).
    
    Parameters:
      data (array): The EEG channel1 data.
      fs (float): Sampling frequency in Hz.
      lowcut (float): Lower cutoff frequency.
      highcut (float): Higher cutoff frequency.
      order (int): Order of the filter (default=5).
    
    Returns:
      array: Filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def bandpass_filter_channel2(data, fs, lowcut, highcut, order=5):
    """
    Bandpass filter for channel2 (EEG).
    
    Parameters:
      data (array): The EEG channel2 data.
      fs (float): Sampling frequency in Hz.
      lowcut (float): Lower cutoff frequency.
      highcut (float): Higher cutoff frequency.
      order (int): Order of the filter (default=5).
    
    Returns:
      array: Filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

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
    plt.figure(figsize=(12, 10))
    
    # EEG Comparison Plot
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, orig_eeg, label="Original EEG", linewidth=0.3)
    plt.plot(time_axis, filt_eeg, label="Filtered EEG", linewidth=0.3)
    plt.title("EEG Comparison")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # EMG Comparison Plot
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, orig_emg, label="Original EMG", linewidth=0.1)
    plt.plot(time_axis, filt_emg, label="Filtered EMG", linewidth=0.1)
    plt.title("EMG Comparison")
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def bandpass_plot_data(edf_file, eeg_low, eeg_high, emg_low, emg_high, order_magnitude=5):
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
    order = order_magnitude
    filt_channel1 = bandpass_filter_channel1(channel1, fs, eeg_lowcut, eeg_highcut, order)
    filt_channel2 = bandpass_filter_channel2(channel2, fs, emg_lowcut, emg_highcut, order)

    # Plot the comparison of original and filtered data
    total_samples = len(channel1)
    time_axis = [i / fs / 60 for i in range(total_samples)]
    plot_comparison(time_axis, channel1, filt_channel1, channel2, filt_channel2, time_unit='minutes')



    
    
