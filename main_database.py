
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from readQ import setup_Qfile
from readQ import nodi_plot 

# Function to select a file
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.config(text=file_path)  # Display file name in the label
        global selected_file
        selected_file = file_path  # Store the selected file path

# Function to run plotting
def run_plotting():
    try:
        #input_value = float(input_entry.get())  # Get user input and convert to float
        #your_plotting_function(input_value)  # Run the plotting function with input
        input_value = input_entry.get()  # Get user input as string
        input_value_2 = dropdown_var.get()  # Get user input as string
        [Time,Data,Count] = setup_Qfile(selected_file,input_value)
        nodi_plot(Time,Data,Count,input_value_2)
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number.")

# Create the main window
root = tk.Tk()
root.title("NoDi* Plotter")

root.geometry("600x400")

# Label and button for file selection
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=10)

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(pady=5)

# Input for user to enter a number
input_label = tk.Label(root, text="Enter desired length:")
input_label.pack(pady=5)

input_entry = tk.Entry(root)
input_entry.pack(pady=5)

dropdown_label = tk.Label(root, text="Select a plot option:")
dropdown_label.pack(pady=5)

# Input for user to enter a number
dropdown_var = tk.StringVar(root)
dropdown_var.set("Q(i+1) and Q(i)")  # Set default option

options = ["Q(i+1) and Q(i)", "T(i+1) and T(i)", "T(i) and Q(i)", "Q(i)/T(i) and T(i)"]  # List of options
dropdown_menu = tk.OptionMenu(root, dropdown_var, *options)
dropdown_menu.pack(pady=10)

# Button to run plotting
plot_button = tk.Button(root, text="Create Plots", command=run_plotting)
plot_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
