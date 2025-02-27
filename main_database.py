
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *       
import matplotlib.pyplot as plt
from readQ import setup_Qfile
from readQ import nodi_plot 
from readQ import setup_csvfile 
import numpy as np
import pandas as pd
from tkinter import font as tkFont

# Function to select a file
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.config(text=file_path)  # Display file name in the label
        global selected_file, Time, Data, Count, file_length
        selected_file = file_path  # Store the selected file path
        if file_path.endswith('.Q'):
            [Time,Data,Count] = setup_Qfile(selected_file,"end")
        elif file_path.endswith('.csv'):
            [Time,Data,Count] = setup_csvfile(selected_file,"end")
        else:
            msg = print('file #s could not be opened - check folder!')
            [Time,Data,Count] = [0,0,0]
        # insert code for displaying the file length
        file_length = np.round((max(Time) - min(Time)),2)
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
            start = float(input_entry_start.get())   # Get user input as string
            end = float(input_entry_end.get()) # Get user input as string
            if (start < end) and (end < file_length):
                
                Time_temp = np.array([t for t in Time if start <= t <= end])
                Data_temp = np.array([m for t, m in zip(Time, Data) if start <= t <= end])
                Count_temp = np.size(Data_temp)

        input_value_plot = dropdown_var_plot.get()  # Get user input as string
        input_grid_size = int(dropdown_var_grid_size.get())

        x_scale = input_entry_xscale.get()
        y_scale = input_entry_yscale.get()

        save_data = nodi_plot(Time_temp,Data_temp,Count_temp,input_value_plot,input_grid_size,x_scale,y_scale)
        if checkbox_var_save.get() is True:
            df = pd.DataFrame(save_data)
            df.to_csv(selected_file + "_" + input_value_plot.replace(" ", "") + "_pulses_" + str(Count_temp) + str(input_grid_size) + "_file.csv")

    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid file length.")

# Create the main window
root = tk.Tk()
root.title("PSA Plotter")
font_size = 16
button_size = tkFont.Font(family='Helvetica', size=12)

root.geometry("800x350")

# Label and button for file selection
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=5)

file_button = tk.Button(root, text="Select File", command=select_file, font=(button_size))
file_button.pack(pady=1)


# Label for file length
file_label_length = tk.Label(root, text="File length: (empty) (s)")
file_label_length.pack(pady=1)


# Label and Entry for file length selecion
input_length_label = tk.Label(root, text="Enter desired length(s): (Start) (End)")
input_length_label.pack(pady=5)

input_frame = tk.Frame(root)
input_frame.pack(pady=1)

input_entry_start = tk.Entry(input_frame, width=15)
input_entry_start.pack(side=tk.LEFT, padx=5)

input_entry_end = tk.Entry(input_frame, width=15)
input_entry_end.pack(side=tk.LEFT, padx=5)

checkbox_var_length = tk.BooleanVar()
checkbox_length = tk.Checkbutton(input_frame, text="Whole File", variable=checkbox_var_length, font=(button_size))
checkbox_length.pack(side=tk.LEFT, padx=5)


# Label and Entry for plot option
input_frame_2 = tk.Frame(root)
input_frame_2.pack(pady=1)

dropdown_label = tk.Label(input_frame_2, text="Select a plot option:", font=(font_size))
dropdown_label.pack(side=tk.LEFT, padx=5)

dropdown_var_plot = tk.StringVar(input_frame_2)
dropdown_var_plot.set("Q(i+1) and Q(i)")  # Set default option

options = ["Q(i+1) and Q(i)", "T(i+1) and T(i)", "T(i) and Q(i)", "Q(i)/T(i) and T(i)","Timescale","Q Density","T Density"]  # List of options
dropdown_menu_plot = tk.OptionMenu(input_frame_2, dropdown_var_plot, *options)
dropdown_menu_plot.pack(side=tk.LEFT, padx=5)
dropdown_menu_plot.config(font=button_size) # set the button font

# Label and entry for grid size
input_frame_3 = tk.Frame(root)
input_frame_3.pack(pady=1)

dropdown_label_grid_size = tk.Label(input_frame_3, text="Select a grid size:", font=(font_size))
dropdown_label_grid_size.pack(side=tk.LEFT, padx=5)

dropdown_var_grid_size = tk.StringVar(input_frame_3)
dropdown_var_grid_size.set("100")  # Set default option

options_grid = ["25", "100"]  # List of options
dropdown_menu_grid_size = tk.OptionMenu(input_frame_3, dropdown_var_grid_size, *options_grid)
dropdown_menu_grid_size.pack(side=tk.LEFT, padx=5)
dropdown_menu_grid_size.config(font=button_size) # set the button font


# Label and entry for scaling factor
input_frame_5 = tk.Frame(root)
input_frame_5.pack(pady=1)

scale_factor_label = tk.Label(input_frame_5, text="Enter X and Y scale", font=(font_size))
scale_factor_label.pack(side=tk.LEFT, padx=5)

input_entry_xscale = tk.Entry(input_frame_5, width=15)
input_entry_xscale.pack(side=tk.LEFT, padx=5)

input_entry_yscale = tk.Entry(input_frame_5, width=15)
input_entry_yscale.pack(side=tk.LEFT, padx=5)


# Label and entry for plotting
input_frame_4 = tk.Frame(root)
input_frame_4.pack(pady=10)

plot_button = tk.Button(input_frame_4, text="Create Plots", command=run_plotting, font=(button_size))
plot_button.pack(side=tk.LEFT,padx=5)

checkbox_var_save = tk.BooleanVar()
checkbox_save = tk.Checkbutton(input_frame_4,text="Save Array", variable=checkbox_var_save,font=(button_size))
checkbox_save.pack(side=tk.LEFT,padx=5)


[wid.config(font=(None,font_size)) for wid in root.winfo_children() if isinstance(wid, Label) ]
# Run the GUI loop
root.mainloop()
