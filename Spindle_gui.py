import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont
from read_plot_raw_edf import bandpass_plot_data
from load_data_Training import load_data_and_labels
from train_model import train_model
from torch.utils.data import DataLoader, TensorDataset
from Spindle import SpindleGraph
import torch.nn as nn
import torch.optim as optim
import torch
import os
import pandas as pd
from compare_prediction_accuracy import compare_files
from load_data_prediction import load_data_prediction
from predict_sleep_stages import predict_sleep_stages
import time
from correct_states import process_files




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


# Tooltip class
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# Function to handle folder selection and running stats/plots
def Read_plot_EDF():
    edf_file = filedialog.askopenfilename(title="Select EDF File", filetypes=[("EDF files", "*.edf")])
    # Prompt user for filter parameters
    eeg_low = simpledialog.askfloat("Filter Parameters", "Enter EEG low-pass cutoff (Hz):", initialvalue=0.5)
    eeg_high = simpledialog.askfloat("Filter Parameters", "Enter EEG high-pass cutoff (Hz):", initialvalue=12)
    emg_low = simpledialog.askfloat("Filter Parameters", "Enter EMG low-pass cutoff (Hz):", initialvalue=25)
    emg_high = simpledialog.askfloat("Filter Parameters", "Enter EMG high-pass cutoff (Hz):", initialvalue=50)
    filter_order = simpledialog.askinteger("Filter Parameters", "Enter filter order (default: 5):", initialvalue=5)

    if all(v is not None for v in [eeg_low, eeg_high, emg_low, emg_high, filter_order]):
        # Call function with selected file and parameters
        bandpass_plot_data(edf_file, eeg_low, eeg_high, emg_low, emg_high, filter_order)
    else:
        messagebox.showwarning("Invalid Parameters", "Please enter valid filter parameters.")

def Training():
    # Step 1: User selects folder
    folder_path = filedialog.askdirectory(title="Select Folder for Training")
    
    if folder_path:
        # Step 2: Prompt user for model name and destination folder
        model_name = simpledialog.askstring("Model Name", "Enter model name (default: SPINDLE_model-test):", initialvalue="SPINDLE_model-test")
        if model_name is None:
            model_name = 'SPINDLE_model-test'  # If user cancels, use default
            
        destination_folder = filedialog.askdirectory(title="Select Destination Folder for Model")
        if not destination_folder:
            messagebox.showwarning("No Folder Selected", "Please select a destination folder to save the model.")
            return
        
        model_name = os.path.join(destination_folder, model_name + ".pth")


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

        start_time = time.time()

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

         # Label for estimated time remaining
        time_label = tk.Label(progress_window, text="Estimated time remaining: Calculating...")
        time_label.pack()

        # Step 7: Train the model with progress bar
        train_model(model, criterion, optimizer, train_loader, epochs=epoch_num, progress=progress, label=progress_label, time_label=time_label, start_time=start_time)

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
    folder_file_prediction = filedialog.askdirectory(title="Select Folder to predict files")
    
    if not folder_file_prediction:
        messagebox.showwarning("No Folder Selected", "Please select a folder for prediction files.")
        return


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
        print("Predicting for:", example_file_prediction)

        # Load data for prediction
        data = load_data_prediction(example_file_prediction, SPINDLE_PREPROCESSING_PARAMS)

        # Step 7: Predict sleep stages
        predictions = predict_sleep_stages(data, model)

        # Step 8: Save predictions to a CSV file
        epochs = [i * SPINDLE_PREPROCESSING_PARAMS['time_interval'] for i in range(len(predictions))]

        df = pd.DataFrame({'Epochs': epochs, 'Prediction': predictions})
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
root.geometry("750x1020")
root.configure(bg='#2E4053')

# Custom Font
custom_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

# Create a header label
header_label = tk.Label(root, text="Sleep/Wake States Annotations", font=tkfont.Font(family="Helvetica", size=22, weight="bold"),
                        fg="#F7DC6F", bg='#2E4053')
header_label.pack(pady=10)

# Create buttons with custom styling -old
button_style = {"font": custom_font, "bg": "#1ABC9C", "fg": "white", "relief": tk.RAISED, "bd": 5, "width": 17, "height": 1}



# Fonts for steps and instructions
step_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
instructions_font = tkfont.Font(family="Helvetica", size=10)

# Step 1: Visualize Data
step1_label = tk.Label(root, text="Step 1: Bandpass Filter & Plot EEG/EMG Data (Optional)", font=step_font, fg="#F7DC6F", bg="#2E4053")
instructions1_label = tk.Label(root, text="Select an EDF file with 1 EEG and 1 EMG channel, then choose the filter parameters. \n", 
                                font=instructions_font, fg="white", bg="#2E4053", wraplength=500)
step1_label.pack(pady=(20, 5))
instructions1_label.pack(pady=(5, 10))

read_raw_edf_button = tk.Button(root, text="Visualize Data", command=Read_plot_EDF, **button_style)
read_raw_edf_button.pack(pady=10)

# Step 2: Train Model
step2_label = tk.Label(root, text="Step 2: Train your model using spindle parameters (Optional)", font=step_font, fg="#F7DC6F", bg="#2E4053")
instructions2_label = tk.Label(root, text="Train a model on EDF files with matching CSV annotations. \n EDF & CSV files should have corresponding names [file_1.edf, file_1.csv]. \n CSVs nust have labels in the second column  [(W, NR, R) or (2, 3, 1)]"
                                           "\n 1) Select a folder containing the edf/csv files \n 2) Specify a name \n 3) Select a destination folder for the trained model weight file.", 
                                font=instructions_font, fg="white", bg="#2E4053", wraplength=500)
step2_label.pack(pady=(20, 5))
instructions2_label.pack(pady=(5, 10))

Training_button = tk.Button(root, text="Train Model", command=Training, **button_style)
Training_button.pack(pady=10)

current_directory = os.getcwd()

# Step 3: Run Predictions
step3_label = tk.Label(root, text="Step 3: Predict Sleep/Wake states", font=step_font, fg="#F7DC6F", bg="#2E4053")
instructions3_label = tk.Label(root, text=f"Use either your trained model or the 'AER Lab' model \n {current_directory}\\Spindle_MM.pth.\n"
                                           "1) Select the model weights file. \n 2) Choose the folder with EDF files for predictions.", 
                                font=instructions_font, fg="white", bg="#2E4053", wraplength=500)
step3_label.pack(pady=(20, 5))
instructions3_label.pack(pady=(5, 10))

Prediction_button = tk.Button(root, text="Run Predictions", command=Prediction, **button_style)
Prediction_button.pack(pady=10)

# Step 4: Evaluate
step4_label = tk.Label(root, text="Step 4: Evaluate", font=step_font, fg="#F7DC6F", bg="#2E4053")
instructions4_label = tk.Label(root, text="Correct state predictions using specific rules and compare with annotations."
                                        "\n\nTo correct states: \n1) Select input folder with _predictions.csv files \n2) Select output folder for corrected states.", font=instructions_font, fg="white", bg="#2E4053", wraplength=500)
step4_label.pack(pady=(20, 5))
instructions4_label.pack(pady=(5, 10))


def correct_states_handler():
    input_folder = filedialog.askdirectory(title="Select Input Folder with Predictions")
    if not input_folder:
        messagebox.showwarning("No Input Folder", "Please select an input folder.")
        return
        
    output_folder = filedialog.askdirectory(title="Select Output Folder for Corrected States")
    if not output_folder:
        messagebox.showwarning("No Output Folder", "Please select an output folder.")
        return
        
    process_files(input_folder, output_folder)
    messagebox.showinfo("Success", "States corrected successfully!")

correct_states_button = tk.Button(root, text="Correct states", command=correct_states_handler, **button_style)
correct_states_button.pack(pady=10)



def compare_predictions():
    folder_path = filedialog.askdirectory(title="Select folder containing predictions and manual annotations")
    if folder_path:
        compare_files(folder_path)
    else:
        messagebox.showwarning("No Folder Selected", "Please select a folder to compare files.")

instructions5_label = tk.Label(root, text="\nTo compare predictions: \n1) Select input folder containing predictions and manual annotations", font=instructions_font, fg="white", bg="#2E4053", wraplength=500)
instructions5_label.pack(pady=(5, 10))

compare_button = tk.Button(root, text="Compare predictions", command=compare_predictions, **button_style)
compare_button.pack(pady=10)



# Run the GUI loop
root.mainloop()