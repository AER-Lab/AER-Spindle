import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont
from read_plot_raw_edf import read_plot_raw_edf
from load_data_Training import load_data_and_labels
from train_model import train_model
from torch.utils.data import DataLoader, TensorDataset
from Spindle import SpindleGraph
import torch.nn as nn
import torch.optim as optim
import torch
import os
import pandas as pd
from load_data_prediction import load_data_prediction
from predict_sleep_stages import predict_sleep_stages




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

learning_rate = 0.00005
epoch_num = 5
drop_out_rate = 0.5
batch_size_num = 100
num_classes = 4

expected_data_shape = (2,24,160)

# Function to handle folder selection and running stats/plots
def Read_plot_EDF():
    edf_file = filedialog.askopenfilename(title="Select EDF File", filetypes=[("EDF files", "*.edf")])
    if edf_file:
        # Check if the selected file ends with '.edf'
        if edf_file.lower().endswith(".edf"):
            messagebox.showinfo("EDF File", f"Plotting Raw Data: {edf_file}")
            # Call your function with the selected EDF file
            read_plot_raw_edf(edf_file)
        else:
            messagebox.showwarning("Invalid File", "Please select a valid .edf file.")
    else:
        messagebox.showwarning("No File Selected", "Please select an .edf file to continue.")

def Training():
    # Step 1: User selects folder
    folder_path = filedialog.askdirectory(title="Select Folder for Training")
    
    if folder_path:
        # Step 2: Prompt user to rename the model (or use a default name)
        model_name = simpledialog.askstring("Model Name", "Enter model name (default: SPINDLE_model-test.pth):", initialvalue="SPINDLE_model-test.pth")
        
        if model_name is None:
            model_name = 'SPINDLE_model-test.pth'  # If user cancels, use default

        messagebox.showinfo("Training", f"Processing & Training the model using files from: {folder_path} with model name: {model_name}")

        # Step 3: Load data and labels
        data, all_labels = load_data_and_labels(folder_path, SPINDLE_PREPROCESSING_PARAMS)

        # Step 4: Print data shape and prepare it for the model
        print("Data Shape:", data.shape)
        data_shape = data.shape[1:]

        # Step 5: Create DataLoader
        dataset = TensorDataset(data, all_labels)
        train_loader = DataLoader(dataset, batch_size=batch_size_num, shuffle=True)

        # Step 6: Initialize model, loss, and optimizer
        model = SpindleGraph(input_dim=data_shape, nb_class=num_classes, dropout_rate=drop_out_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create progress bar window
        progress_window = tk.Toplevel()
        progress_window.title("Training Progress")
        progress_window.geometry("400x100")
        
        # Progress bar widget
        progress = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=20)
        
        # Label to display progress percentage
        progress_label = tk.Label(progress_window, text="0% completed")
        progress_label.pack()

        # Step 7: Train the model with progress bar
        train_model(model, criterion, optimizer, train_loader, epochs=epoch_num, progress=progress, label=progress_label)

        # Step 8: Save the model weights
        torch.save(model.state_dict(), model_name)
        print(f"Model weights saved successfully as {model_name}")
        
        # Close the progress window
        progress_window.destroy()
        
    else:
        messagebox.showwarning("No Folder Selected", "Please select a folder to continue.")

# Function to handle running the larger function
def Prediction():
    # Step 1: Ask user to select the model weights file
    model_weights_file = filedialog.askopenfilename(title="Select Model Weights File", filetypes=[("PyTorch Model Files", "*.pth")])
    
    if not model_weights_file:
        messagebox.showwarning("No File Selected", "Please select a model weights file.")
        return
    
    # Step 2: Ask user to select the folder containing the prediction files
    folder_file_prediction = filedialog.askdirectory(title="Select Folder for Prediction Files")
    
    if not folder_file_prediction:
        messagebox.showwarning("No Folder Selected", "Please select a folder for prediction files.")
        return

    messagebox.showinfo("Prediction", "Running Predictions...")

    # Step 3: Load model weights
    def load_model_weights(model, filename):
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model

    # Initialize the model (adjust according to your model structure)
    model = SpindleGraph(input_dim=expected_data_shape, nb_class=4, dropout_rate=0.5)  # Customize input_dim as per your needs

    # Step 4: Load the model weights
    model = load_model_weights(model, model_weights_file)
    print("Model weights loaded successfully:", model)

    # Step 5: Set up progress bar for prediction files
    progress_window = tk.Toplevel()
    progress_window.title("Prediction Progress")
    progress_window.geometry("400x150")

    progress = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=10)

    progress_label = tk.Label(progress_window, text="0% completed")
    progress_label.pack()

    # Label for estimated time remaining
    time_label = tk.Label(progress_window, text="Estimated time remaining: Calculating...")
    time_label.pack()

    # List of EDF files for prediction
    edf_files = [f for f in os.listdir(folder_file_prediction) if f.endswith('.edf')]
    total_files = len(edf_files)

    # Start time for calculating time estimates
    start_time = time.time()

    # Function to update progress and estimated time
    def update_progress(file_idx):
        # Calculate progress percentage
        progress_value = (file_idx / total_files) * 100
        progress['value'] = progress_value
        progress_label.config(text=f"{progress_value:.2f}% completed")
        
        # Calculate elapsed time and estimate time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / file_idx
        remaining_files = total_files - file_idx
        estimated_time_remaining = avg_time_per_file * remaining_files

        # Convert estimated time remaining to minutes or hours
        if estimated_time_remaining < 60:
            time_remaining_text = f"{estimated_time_remaining:.2f} seconds"
        elif estimated_time_remaining < 3600:
            time_remaining_text = f"{estimated_time_remaining / 60:.2f} minutes"
        else:
            time_remaining_text = f"{estimated_time_remaining / 3600:.2f} hours"

        time_label.config(text=f"Estimated time remaining: {time_remaining_text}")
        progress.update()

    # Step 6: Loop through each EDF file
    for idx, file_base in enumerate(edf_files, start=1):
        file_base = file_base.split('.')[0]
        example_file_prediction = os.path.join(folder_file_prediction, f"{file_base}.edf")

        # Load data for prediction
        data = load_data_prediction(example_file_prediction, SPINDLE_PREPROCESSING_PARAMS)

        # Step 7: Predict sleep stages
        predictions = predict_sleep_stages(data, model)

        # Step 8: Save predictions to a CSV file
        df = pd.DataFrame(predictions, columns=['Prediction'])
        prediction_csv_path = os.path.join(folder_file_prediction, f"{file_base}_predictions.csv")
        df.to_csv(prediction_csv_path, index=False, header=False)
        print(f"Predictions saved to {prediction_csv_path}")

        # Update progress and estimated time after each file
        update_progress(idx)

    # Close progress window once predictions are complete
    progress_window.destroy()
    messagebox.showinfo("Prediction", "Predictions completed successfully.")
    

# Create the main window
root = tk.Tk()
root.title("AER-Lab Model Building")
root.geometry("600x1000")
root.configure(bg='#2E4053')

# Custom Font
custom_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

# Create a header label
header_label = tk.Label(root, text="Mouse Sleep States Analysis", font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
                        fg="#F7DC6F", bg='#2E4053')
header_label.pack(pady=10)

# Create buttons with custom styling
button_style = {"font": custom_font, "bg": "#1ABC9C", "fg": "white", "relief": tk.RAISED, "bd": 5, "width": 25, "height": 8}

read_raw_edf_button = tk.Button(root, text="Read & Plot RAW EDF Signal", command=Read_plot_EDF, **button_style)
read_raw_edf_button.pack(pady=10)

Training_button = tk.Button(root, text="Training", command=Training, **button_style)
Training_button.pack(pady=10)

Prediction_button = tk.Button(root, text="Prediction", command=Prediction, **button_style)
Prediction_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
