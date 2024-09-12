import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

# Function to handle folder selection and running stats/plots
def run_stats_and_plots():
    folder_path = filedialog.askdirectory()
    if folder_path:
        messagebox.showinfo("Stats/Plots", f"Running stats and plots on files in: {folder_path}")
        # Example function calls:
        # process_folder(folder_path)
        # generate_plots(folder_path)
    else:
        messagebox.showwarning("No Folder Selected", "Please select a folder to continue.")

# Function to handle running the larger function
def run_large_function():
    messagebox.showinfo("Large Function", "Running the larger function...")
    # Example function call:
    # large_function()

# Create the main window
root = tk.Tk()
root.title("Mouse Sleep States GUI")
root.geometry("400x200")
root.configure(bg='#2E4053')

# Custom Font
custom_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

# Create a header label
header_label = tk.Label(root, text="Mouse Sleep States Analysis", font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
                        fg="#F7DC6F", bg='#2E4053')
header_label.pack(pady=10)

# Create buttons with custom styling
button_style = {"font": custom_font, "bg": "#1ABC9C", "fg": "white", "relief": tk.RAISED, "bd": 5, "width": 20, "height": 2}

stats_button = tk.Button(root, text="Run Stats/Plots", command=run_stats_and_plots, **button_style)
stats_button.pack(pady=10)

large_function_button = tk.Button(root, text="Run Large Function", command=run_large_function, **button_style)
large_function_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
