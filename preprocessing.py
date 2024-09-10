import numpy as np
import pyedflib
import matplotlib.pyplot as plt
from spindle_preproc import SpindlePreproc
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import setup_logger_to_std
# CNN Network for training..
from spindle_graph import SpindleGraph
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from generate_dummy_data import generate_dummy_data
from plot_spectrogram import plot_spectrogram
from classification_accuracy import classification_accuracy
from load_data_and_labels import load_data_and_labels
from preprocess_plot_spectrograms import preprocess_and_plot_edf

# learning rate of the paper
# learning_rate = 0.00005
#? TEST learning rate OVERFITTING
learning_rate = 1e-3
# Nuber of full iterations over the training data, 5 as in the paper
# epoch_num = 5
epoch_num = 100
# dropout rate 50% as in the paper
# drop_out_rate = 0.5
#?TEST dropout rate OVERFITTING
drop_out_rate = 0
# each mini-batch contains 100 samples as in the paper
batch_size_num = 100
# number of classes/expected categories WAKE, NREM, REM, Artifact
num_classes = 4





# # preprocessing parameters. Below are defaults used as in the paper
SPINDLE_PREPROCESSING_PARAMS = {
    'name': 'SpindlePreproc',
    'target_srate': 128,
    'spectrogram-stride': 16,
    'time_interval': 4,
    'num_neighbors': 4,
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 12},
    'EMG-filtering': {'lfreq': 2, 'hfreq': 30}
}


# Training function
def train_model(model, criterion, optimizer, train_loader, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # print("input shape", inputs.shape)
            # print("label shape", labels.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.size(0) != labels.size(0):
                continue
            # Convert one-hot encoded labels to class indices
            labels = torch.argmax(labels, dim=1)
            # Ensure labels are 1D tensor of class indices
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #Implement the classification accuracy  which takes model predictions and ground truth data and tells us how well the model is preforming correct predictions/total predictions
            accuracy = classification_accuracy(outputs, labels)

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy.item()}")
    return model

# Save model weights
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)


    

folder_path_to_data = r'C:\Users\geosaad\Desktop\Main-Scripts\PYTHON\Spindle-DeepSleep\Spindle\data\test_data'
# Load all data
all_spectrograms, all_labels = load_data_and_labels(folder_path_to_data, SPINDLE_PREPROCESSING_PARAMS)

print("ALL SPECTROGRAMS_Spectrogram shape",all_spectrograms.shape)
print("Labels shape",all_labels.shape)



# Create DataLoader -DATA
dataset = TensorDataset(all_spectrograms, all_labels)
train_loader = DataLoader(dataset, batch_size=batch_size_num, shuffle=True)


# this is the CNN MODEL SpindleGraph
# SPINDLE CNN MODEL
model = SpindleGraph(input_dim=(3,24,160), nb_class=num_classes, dropout_rate=drop_out_rate)


criterion = nn.CrossEntropyLoss()
# model parameters
print("Model parameters", model.parameters())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#? Train the model
# train_model(model, criterion, optimizer, train_loader, epochs=epoch_num)

print("Model trained successfully", model)

# ? Save the model weights for CNN model
# save_model_weights(model, 'DUMMY_model.pth')

# Number of epochs in the data

def preprocess_edf_signals(edf_file, epoch_duration=4):
    all_signals, signal_header, header = pyedflib.highlevel.read_edf(edf_file)
    sample_rate = signal_header[0]['sample_rate']
    num_samples_per_epoch = int(sample_rate * epoch_duration)
    
    # Calculate the number of epochs
    num_epochs = all_signals[0].shape[0] // num_samples_per_epoch
    
    # Reshape the signals to match the epochs
    eeg1 = all_signals[0][:num_epochs * num_samples_per_epoch]
    eeg2 = all_signals[1][:num_epochs * num_samples_per_epoch]
    emg = all_signals[2][:num_epochs * num_samples_per_epoch]
    
    # Stack the data to match the input shape (num_epochs, channels, height, width)
    input_data = np.stack([eeg1, eeg2, emg], axis=1)
    
    return input_data, num_epochs, sample_rate



class_mapping = {
    1: "WAKE",
    2: "NREM",
    3: "REM",
    4: "Artifact"  # This is for error/unknown cases
}

# Load the model
model = SpindleGraph(input_dim=(3,24,160), nb_class=num_classes, dropout_rate=drop_out_rate)
model.load_state_dict(torch.load('DUMMY_model.pth'))
model.eval()


# predict
def predict_sleep_stages(model, input_data, num_epochs):
    """
    Predict sleep stages from input data using the provided model.

    Args:
        model: A trained PyTorch model for predicting sleep stages.
        input_data: A numpy array or torch tensor of shape [num_epochs, channels, height, width].
        num_epochs: Number of epochs in the input data.

    Returns:
        List of predicted sleep stages for each epoch.
    """
    model.eval()
    print("Input data shape", input_data.shape)
    eeg1_spectrogram = input_data[0]
    eeg2_spectrogram = input_data[1]
    emg_spectrogram = input_data[2]

    print("EEG1 Spectrogram Shape:", eeg1_spectrogram.shape)
    print("EEG2 Spectrogram Shape:", eeg2_spectrogram.shape)
    print("EMG Spectrogram Shape:", emg_spectrogram.shape)

    num_epochs_eeg1 = eeg1_spectrogram.shape[0]
    num_epochs_eeg2 = eeg2_spectrogram.shape[0]
    num_epochs_emg = emg_spectrogram.shape[0]
    print("Number of epochs EEG1:", num_epochs_eeg1)
    print("Number of epochs EEG2:", num_epochs_eeg2)
    print("Number of epochs EMG:", num_epochs_emg)
    all_sleep_stages = []
    with torch.no_grad():  # Disable gradient computation
        for epoch_idx in range(num_epochs):
            epoch_data = torch.tensor(input_data[epoch_idx], dtype=torch.float32)
            if epoch_data.ndimension() == 3:
                epoch_data = epoch_data.unsqueeze(0)  # Add batch dimension
            predictions = model(epoch_data)  # Forward pass
            class_indices = torch.argmax(predictions, dim=1)
            sleep_stages = [class_mapping.get(idx.item(), "Artifact") for idx in class_indices]
            all_sleep_stages.append(sleep_stages[0])  # Assuming one prediction per epoch
    return all_sleep_stages


# #? Example usage with dummy data
num_epochs_dummy = 21600
channels = 3
height = 24
width = 160
dummy_data = generate_dummy_data(num_epochs_dummy, channels, height, width)
print("Dummy data shape:", dummy_data.shape)
predicted_stages_dummy = predict_sleep_stages(model, dummy_data, num_epochs_dummy)
print("Dummy data sleep stages:", predicted_stages_dummy)
print("Number of dummy data sleep stages:", len(predicted_stages_dummy))



example_file = r'C:\Users\geosaad\Desktop\Main-Scripts\PYTHON\Spindle-DeepSleep\Spindle\data\test_data\A1.edf'

def load_data_for_prediction(folder_path, params):
    all_spectrograms = []
    all_metadata = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.edf'):
            base_name = os.path.splitext(file_name)[0]
            eeg_file = os.path.join(folder_path, file_name)
            
            # Extract metadata from EDF file
            with pyedflib.EdfReader(eeg_file) as f:
                sampling_rate = f.getSampleFrequency(0)  # Assuming all channels have the same sampling rate
                duration = f.getFileDuration()  # Total duration in seconds
                num_samples = f.getNSamples()[0]  # Total number of samples in the first channel
                n_channels = f.signals_in_file
                channel_labels = f.getSignalLabels()
            
            # Calculate the number of epochs
            epoch_duration = params.get('epoch_duration', 4)  # Duration of each epoch in seconds
            num_epochs = int(duration / epoch_duration)
            
            # Process the EEG file to generate spectrograms
            eeg1_spectrogram, eeg2_spectrogram, emg_spectrogram = preprocess_and_plot_edf(eeg_file, params)
            
            # Stack spectrograms along a new dimension to combine them into a single array
            spectrograms = np.stack([eeg1_spectrogram, eeg2_spectrogram, emg_spectrogram], axis=1)
            
            # Store spectrograms and metadata
            all_spectrograms.append(spectrograms)
            metadata = {
                'file_name': file_name,
                'sampling_rate': sampling_rate,
                'duration': duration,
                'num_samples': num_samples,
                'num_epochs': num_epochs,
                'n_channels': n_channels,
                'channel_labels': channel_labels
            }
            all_metadata.append(metadata)
    
    # Concatenate all spectrograms from all files
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    
    # Convert to PyTorch tensor
    all_spectrograms = torch.tensor(all_spectrograms, dtype=torch.float32)
    
    return all_spectrograms, all_metadata

# Load the data for prediction
spectrograms, metadata = load_data_for_prediction(folder_path_to_data, SPINDLE_PREPROCESSING_PARAMS)

print("File spectrograms shape:", spectrograms.shape)
print("Metadata:", metadata)



def predict_sleep_stages(spectrograms, model):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    with torch.no_grad():  # Disable gradient calculation
        num_datapoint = spectrograms.shape[0]  # Number of epochs
        for i in range(num_datapoint):
            epoch_spectrogram = torch.tensor(spectrograms[i]).unsqueeze(0)  # Add batch dimension
            output = model(epoch_spectrogram)
            
            # Get the predicted class (assuming output is logits, apply softmax if necessary)
            predicted_class = output.argmax(dim=1).item()
            predicted_label = class_mapping.get(predicted_class + 1, "Artifact")  # +1 to match class_mapping
            predictions.append(predicted_label)
    
    return predictions

# Load the model
model = SpindleGraph(input_dim=(3,24,160), nb_class=len(class_mapping), dropout_rate=drop_out_rate)
model.load_state_dict(torch.load('DUMMY_model.pth'))  # Replace 'DUMMY_model.pth' with the actual model path
model.eval()

# Predict sleep stages
predicted_labels = predict_sleep_stages(spectrograms, model)


# Example print of the first 10 predictions
print(f"Predicted Labels: {predicted_labels[:10]}")

