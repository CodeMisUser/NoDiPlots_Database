
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *       
import matplotlib.pyplot as plt
from readQ import setup_Qfile
from readQ import nodi_plot 
import numpy as np
import pandas as pd 

# Function to select a file
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.config(text=file_path)  # Display file name in the label
        global selected_file, Time, Data, Count, file_length
        selected_file = file_path  # Store the selected file path
        [Time,Data,Count] = setup_Qfile(selected_file,"end")
        # insert code for displaying the file length
        file_length = np.round(Time[-1],2)
        file_label_length.config(text="File length: " + str(file_length) + " (s)")

# Function to run plotting
def run_plotting():
    global Time, Data, Count, file_length
    Time_temp = Time
    Data_temp = Data
    Count_temp = Count
    try:
        # insert code for checking the correct bounds
        if checkbox_var_length.get() is False:
            start = int(input_entry_start.get())  # Get user input as string
            end = int(input_entry_end.get())  # Get user input as string
            if (start < end) and (end < file_length):
                Time_temp = [t for t in Time if start <= t <= end]
                Data_temp = [m for t, m in zip(Time, Data) if start <= t <= end]
                Count_temp = len(Time_temp)
            
        input_value_plot = dropdown_var_plot.get()  # Get user input as string
        input_grid_size = int(dropdown_var_grid_size.get())

        save_data = nodi_plot(Time_temp,Data_temp,Count_temp,input_value_plot,input_grid_size)
        if checkbox_var_save.get() is True:
            df = pd.DataFrame(save_data)
            df.to_csv(selected_file + "_" + input_value_plot.replace(" ", "") + "_pulses_" + str(Count_temp) + str(input_grid_size) + "_file.csv")

    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number.")

# Create the main window
root = tk.Tk()
root.title("NoDi* Plotter")

root.geometry("600x400")

# Label and button for file selection
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=10)

file_button = tk.Button(root, text="Select File", command=select_file)
file_button.pack(pady=1)

file_label_length = tk.Label(root, text="File length: (empty) (s)")
file_label_length.pack(pady=1)

# Input for user to enter a number
input_length_label = tk.Label(root, text="Enter desired length(s): (Start) (End)")
input_length_label.pack(pady=5)

# Frame to hold the entry fields and tickbox in a row
input_frame = tk.Frame(root)
input_frame.pack(pady=1)

# First input entry
input_entry_start = tk.Entry(input_frame, width=15)
input_entry_start.pack(side=tk.LEFT, padx=5)

# Second input entry
input_entry_end = tk.Entry(input_frame, width=15)
input_entry_end.pack(side=tk.LEFT, padx=5)

# Tickbox (checkbox)
checkbox_var_length = tk.BooleanVar()
checkbox_length = tk.Checkbutton(input_frame, text="Whole File", variable=checkbox_var_length)
checkbox_length.pack(side=tk.LEFT, padx=5)

# plotting dropbdown
dropdown_label = tk.Label(root, text="Select a plot option:")
dropdown_label.pack(pady=5)

# Input for user to enter a number
dropdown_var_plot = tk.StringVar(root)
dropdown_var_plot.set("Q(i+1) and Q(i)")  # Set default option

options = ["Q(i+1) and Q(i)", "T(i+1) and T(i)", "T(i) and Q(i)", "Q(i)/T(i) and T(i)"]  # List of options
dropdown_menu_plot = tk.OptionMenu(root, dropdown_var_plot, *options)
dropdown_menu_plot.pack(pady=1)

# plotting dropbdown
dropdown_label_grid_size = tk.Label(root, text="Select a grid size:")
dropdown_label_grid_size.pack(pady=5)

# Input for user to enter a number
dropdown_var_grid_size = tk.StringVar(root)
dropdown_var_grid_size.set("100")  # Set default option

options_grid = ["25", "100"]  # List of options
dropdown_menu_grid_size = tk.OptionMenu(root, dropdown_var_grid_size, *options_grid)
dropdown_menu_grid_size.pack(pady=1)

# Frame to hold the entry fields and tickbox in a row
input_frame_2 = tk.Frame(root)
input_frame_2.pack(pady=10)

# Button to run plotting
plot_button = tk.Button(input_frame_2, text="Create Plots", command=run_plotting)
plot_button.pack(side=tk.LEFT,padx=5)

checkbox_var_save = tk.BooleanVar()
checkbox_save = tk.Checkbutton(input_frame_2,text="Save Array", variable=checkbox_var_save)
checkbox_save.pack(side=tk.LEFT,padx=5)

# Run the GUI loop
root.mainloop()
