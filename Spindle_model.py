import numpy as np
import matplotlib.pyplot as plt
from pyedflib import highlevel
import pandas as pd
import torch
from spindle_preproc import SpindlePreproc
from utils import plot_spectrogram
from load_data_and_labels import load_data_and_labels
from torch.utils.data import DataLoader, TensorDataset
from train_model import train_model
from spindle_graph import SpindleGraph
import torch.nn as nn
import torch.optim as optim
from load_data_prediction import load_data_prediction
from predict_sleep_stages import predict_sleep_stages
from read_plot_raw_edf import read_plot_raw_edf
import os


# Training Path
path_for_training = r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\test'
folder_file_prediction = r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA'



learning_rate = 0.00005
epoch_num = 5
drop_out_rate = 0.5
batch_size_num = 100
num_classes = 4



#1.
# Read EDF and plot RAW signals. Print details about channels
# read_plot_raw_edf(path_to_edf)

# params for preprocessor based on paper
SPINDLE_PREPROCESSING_PARAMS = {
    'name': 'SpindlePreproc',
    # Sample rate to resample the data to standardize the input - ADA 256
    'target_srate': 128,
    # stride is the number of samples to move the window forward by
    'spectrogram-stride': 16,
    # time_interval is the duration of each epoch in seconds
    'time_interval': 4,
    # num_neighbors is the number of neighboring epochs to include for context. 4 means, 2 on each side (including the current epoch, so total 5)
    'num_neighbors': 4,
    # EEG-filtering and EMG-filtering are dictionaries with 'lfreq' and 'hfreq' keys
    # EEG filter between 0.5 and 24 Hz, EMG filter between 0.5 and 30
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 12},
    'EMG-filtering': {'lfreq': 0.5, 'hfreq': 30}
}


# load the data for preprocessing and plot spectrograms
# 2.
data, all_labels = load_data_and_labels(path_for_training, SPINDLE_PREPROCESSING_PARAMS)
print("-----------------------------------------------------------------------------------------------")
target_tensor = torch.ones(21600,3, 24,160)


data_shape = data.shape
# print("Data shape: ", data_shape)
# data shape is (21600, 3, 24, 160)
# we want to extract 3 channels, 24 time points, 160 frequency points
data_shape = data_shape[1:]

target_tensor_shape = target_tensor.shape

# Create DataLoader -DATA
dataset = TensorDataset(data, all_labels)
train_loader = DataLoader(dataset, batch_size=batch_size_num, shuffle=True)
# 3.
# # model should use data_shape as input_dim
model = SpindleGraph(input_dim=data_shape, nb_class=num_classes, dropout_rate=drop_out_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 4.
train_model(model, criterion, optimizer, train_loader, epochs=epoch_num)

print("Model trained successfully", model)
# 5.
# Save model weights
model_name = 'SPINDLE_model-ALIKA_Mouse8.pth'
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)
# ? Save the model weights for CNN model
save_model_weights(model, model_name)
print("Model weights saved successfully", model)

# 6.
# Load model weights
def load_model_weights(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

# ? Load the model weights for CNN model
model = load_model_weights(model, model_name)
print("Model weights loaded successfully", model)



# Prediction Path
# example_file_prediction = r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\post-918.edf'

# Assuming load_data_prediction and predict_sleep_stages are already defined elsewhere


array_files = ['Alika-Mouse1','Alika-Mouse2','Alika-Mouse3','Alika-Mouse4','Alika-Mouse5','Alika-Mouse6', 'post-1100', 'post-1324', 'post-1325' ,'post-1348', 'post-1356', 'post-1']

# Loop over the files specified in array_files
for file_base in array_files:
    # Construct the full path for each EDF file
    example_file_prediction = os.path.join(folder_file_prediction, f"{file_base}.edf")

    # Load the data for prediction
    data = load_data_prediction(example_file_prediction, SPINDLE_PREPROCESSING_PARAMS)

    # Predict sleep stages
    predictions = predict_sleep_stages(data, model)

    # Save the predictions to a CSV file
    df = pd.DataFrame(predictions, columns=['Prediction'])
    prediction_csv_path = os.path.join(folder_file_prediction, f"{file_base}_predictions.csv")
    df.to_csv(prediction_csv_path, index=False, header=False)
    print(f"Predictions saved to {prediction_csv_path}")
