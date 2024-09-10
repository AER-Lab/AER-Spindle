import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyedflib import highlevel

# 1 = REM, 2 WAKe, 3 NREM
def read_plot_raw_edf(edf_file):
    # Read EDF file
    signals, meta_data, meta_data2 = highlevel.read_edf(edf_file)

    # Convert to DataFrame
    df = pd.DataFrame(signals).T
    # Add header for each column, Column 0 = EEG1, Column 1 = EEG2, Column 2 = EMG
    df.columns = ['EEG1', 'EMG']
    
    # Extract sampling frequency from signal headers
    sample_frequency = meta_data[0]['sample_rate']

    # Read from meta data
    # print("All headers: ", meta_data, "\n", meta_data2)

    # for each channel, print the label, dimension, sample rate
    for i in range(len(meta_data)):
        print(f"Channel {i} - Label: {meta_data[i]['label']}, Sample rate: {meta_data[i]['sample_rate']} Hz")

    # Time vector for plotting
    n_samples = signals[0].shape[0]
    time = np.arange(n_samples) / sample_frequency

    print("Total number of samples: ", n_samples)
    print("Sample frequency: ", sample_frequency, "Hz, ")
    print("Number of samples per channel: ", n_samples, "Divided by sample frequency: ",sample_frequency , "is ",n_samples/sample_frequency)
    print("Total time is: ", n_samples/sample_frequency, " seconds", "||", n_samples/sample_frequency/60, " minutes were recorded", "||", n_samples/sample_frequency/3600, " hours were recorded")
    
    # Convert DataFrame to NumPy array for indexing
    signals_np = df.to_numpy()
    
    # Plot each signal
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(df.columns):
        plt.subplot(len(df.columns), 1, i + 1)
        plt.plot(time, signals_np[:, i])
        # plot a line in the middle of the plot
        plt.axhline(y=signals_np[:, i].mean(), color='k', linestyle='solid', linewidth=0.5)
        # draw vertical lines at the beginning and end of the plot
        plt.axvline(x=0, color='k', linestyle='solid', linewidth=0.5)
        plt.axvline(x=n_samples/sample_frequency, color='k', linestyle='solid', linewidth=0.5)
        # draw vertical line at the middle of the plot to indicate half of the recording
        plt.axvline(x=n_samples/sample_frequency/2, color='r', linestyle='solid', linewidth=0.5)
        plt.title(f"{column} - {meta_data[i]['dimension']}")
        plt.xlabel('Time (s)')
        plt.ylabel(f"Amplitude ({meta_data[i]['dimension']})")
    
    plt.tight_layout()
    plt.show()
