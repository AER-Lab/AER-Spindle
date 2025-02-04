import os
import glob
import numpy as np
import pandas as pd
import torch
from preprocess_plot_spectrograms import preprocess_and_plot_edf
import torch

    
def map_labels(label):
    if label == 'W' or label == 'W*' or label==2:
        return 0
    elif label == 'NR' or label == 'NR*' or label==3:
        return 1
    elif label == 'R'or label == 'R*' or label==1:
        return 2
    else:
        raise ValueError(f"Unexpected label: {label}")
    

def load_data_and_labels(folder_path, params):
    all_spectrograms = []
    all_labels = []
    edf_files = glob.glob(os.path.join(folder_path, '*.edf'))
    print("Found {} EDF [EEG/EMG] files in the folder".format(len(edf_files)))

    for eeg_file in edf_files:
        base_name = os.path.splitext(os.path.basename(eeg_file))[0]
        label_file = os.path.join(folder_path, base_name + '.csv')
        data = preprocess_and_plot_edf(eeg_file, params)
        print("Data Shape for {}: ".format(base_name), data.shape)
        # Append the 4D array directly to the list
        all_spectrograms.append(data)
            
        # Read and map labels, time and label
        labels_df = pd.read_csv(label_file, header=None, names=['time', 'label'])
        labels = labels_df['label'].apply(map_labels).values



        num_epochs = data.shape[0]
        # assert that num of epochs matches the num of labels
        if len(labels) != num_epochs:
            raise ValueError("Number of epochs does not match the number of labels", "Labels:",len(labels), "Epoch:", num_epochs, "File:", base_name)

 
        all_labels.append(labels)
    
    # Concatenate along the first axis
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print("All spectrograms shape: ", all_spectrograms.shape)
    print("All labels shape: ", all_labels.shape)
    
    # Convert to PyTorch tensors
    all_spectrograms = torch.tensor(all_spectrograms, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return all_spectrograms, all_labels

if __name__ == "__main__":
    # Optional: any testing or standalone logic goes here
    print("Module imported, but not running load_data_and_labels")