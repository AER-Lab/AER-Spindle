# Function to load all data
import os
import glob
import numpy as np
import pandas as pd
import torch
from preprocess_plot_spectrograms import preprocess_and_plot_edf
import torch
from torch.utils.data import Dataset


# TODO [COMPLETE]: replace torch with pytorch dataset -
#? Class implementation of EEGDataset as a subclass of torch.utils.data.Dataset from the PyTorch library
class EEGDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        return torch.tensor(spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#? Function to map labels used for other dataset.
# def map_labels(label):
#     # WAKE
#     if label == 2:
#         return 2
#     # NREM
#     if label == 3:
#         return 3
    
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
            
        # Read and map labels
        #? used for other dataset from SPINDLE paper

        # labels_df = pd.read_csv(label_file, header=None, names=['time', 'label1', 'label2'])
        labels_df = pd.read_csv(label_file, header=None, names=['time', 'label1'])
        labels1 = labels_df['label1'].apply(map_labels).values
        #? used for other dataset from SPINDLE paper

        # labels2 = labels_df['label2'].apply(map_labels).values
        # if len(labels1) != len(labels2):
        #     raise ValueError("Label columns have different lengths")
        
        num_epochs = data.shape[0]
        # assert that num of epochs matches the num of labels
        if len(labels1) != num_epochs:
            raise ValueError("Number of epochs does not match the number of labels", "Labels:",len(labels1), "Epoch:", num_epochs, "File:", base_name)
        #? used for other dataset from SPINDLE paper
        # if len(labels1) > num_epochs:
        #     labels1 = labels1[:num_epochs]
        #     labels2 = labels2[:num_epochs]
        # elif len(labels1) < num_epochs:
        #     labels1 = np.pad(labels1, (0, num_epochs - len(labels1)), 'constant', constant_values=3)
        #     labels2 = np.pad(labels2, (0, num_epochs - len(labels2)), 'constant', constant_values=3)
        
        # all_labels.append(np.stack([labels1, labels2], axis=1))
        all_labels.append(labels1)
    
    # Concatenate along the first axis
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print("All spectrograms shape: ", all_spectrograms.shape)
    print("All labels shape: ", all_labels.shape)
    
    # Convert to PyTorch tensors
    all_spectrograms = torch.tensor(all_spectrograms, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return all_spectrograms, all_labels

# previous implementation without GLOB
    
    # for file_name in os.listdir(folder_path):
    #     if file_name.endswith('.edf'):
    #         base_name = os.path.splitext(file_name)[0]
    #         eeg_file = os.path.join(folder_path, file_name)
    #         label_file = os.path.join(folder_path, base_name + '.csv')
    #         print("Reading files: ", eeg_file, label_file)
    #         data = preprocess_and_plot_edf(eeg_file, params)
    #         print("Data Shape for {}: ".format(file_name), data.shape)
            
    #         # Append the 4D array directly to the list
    #         all_spectrograms.append(data)
            
    #         # Read and map labels
    #         # labels_df = pd.read_csv(label_file, header=None, names=['time', 'label1', 'label2'])
    #         labels_df = pd.read_csv(label_file, header=None, names=['time', 'label1'])
    #         labels1 = labels_df['label1'].apply(map_labels).values
    #         # labels2 = labels_df['label2'].apply(map_labels).values
            
    #         # if len(labels1) != len(labels2):
    #         #     raise ValueError("Label columns have different lengths")
            
    #         num_epochs = data.shape[0]
    #         # assert that num of epochs matches the num of labels
    #         if len(labels1) != num_epochs:
    #             raise ValueError("Number of epochs does not match the number of labels", "Labels:",len(labels1), "Epoch:", num_epochs, "File:", file_name)
    #         # if len(labels1) > num_epochs:
    #         #     labels1 = labels1[:num_epochs]
    #         #     labels2 = labels2[:num_epochs]
    #         # elif len(labels1) < num_epochs:
    #         #     labels1 = np.pad(labels1, (0, num_epochs - len(labels1)), 'constant', constant_values=3)
    #         #     labels2 = np.pad(labels2, (0, num_epochs - len(labels2)), 'constant', constant_values=3)
            
    #         # all_labels.append(np.stack([labels1, labels2], axis=1))
    #         all_labels.append(labels1)
    
    # # Concatenate along the first axis
    # all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    # all_labels = np.concatenate(all_labels, axis=0)
    
    # print("All spectrograms shape: ", all_spectrograms.shape)
    # print("All labels shape: ", all_labels.shape)
    
    # # Convert to PyTorch tensors
    # all_spectrograms = torch.tensor(all_spectrograms, dtype=torch.float32)
    # all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    # return all_spectrograms, all_labels