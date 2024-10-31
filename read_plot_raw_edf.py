import pandas as pd
import matplotlib.pyplot as plt
from pyedflib import highlevel

# 1 = REM, 2 WAKe, 3 NREM
def read_plot_raw_edf(edf_file):
    # Read EDF file
    try:
        # Read EDF file
        signals, meta_data, meta_data2 = highlevel.read_edf(edf_file)
    except Exception as e:
        print(f"An error occurred while reading the EDF file: {e}, Either this is not an EDF file or the file is corrupted.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(signals).T
    print("Dataframe shape: ", df.shape)
    # check how many channels/columns are in the dataframe /2 for pairs
    channel_pairs = round(df.shape[1]/2,0)
    if channel_pairs == 1:
        print("There is ", channel_pairs,"pair of channels in the dataframe")
    else:
        print("There are ", channel_pairs,"pairs of channels in the dataframe")
    
    sample_frequency = meta_data[0]['sample_rate']
    print("Sample frequency: ", sample_frequency, "Hz")

    # for each pair, get the first and second channel and extract the data to plot
    for i in range(0, df.shape[1], 2):
        print("Pair: ", i, "Channel 1: ", i, "Channel 2: ", i+1)
        # get the first channel
        channel1 = df.iloc[:, i]
        # get the second channel
        channel2 = df.iloc[:, i+1]
        # plot the first channel
        plt.figure(figsize=(12, 8))  # create a new figure for each pair
        plt.subplot(2, 1, 1)
        plt.plot(channel1)
        plt.title(f"Channel {i}")
        plt.xlabel('Time (s)')
        plt.ylabel(f"Amplitude")
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        plt.ylim(ymin, ymax + 0.025 * yrange)
        # plot the second channel
        plt.subplot(2, 1, 2)
        plt.plot(channel2)
        plt.title(f"Channel {i+1}")
        plt.xlabel('Time (s)')
        plt.ylabel(f"Amplitude")
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        plt.ylim(ymin, ymax + 0.025 * yrange)
        plt.tight_layout()
        plt.show()  # show the plot for the current pair