import pandas as pd
import matplotlib.pyplot as plt
from pyedflib import highlevel

# Define function to read and plot EDF file data
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



