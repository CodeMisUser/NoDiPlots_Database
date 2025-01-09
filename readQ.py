
#import packages
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib.ticker import LogFormatter
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

import pandas as pd
from datetime import datetime
import re

def setup_Qfile(sourcefile,command):

    # the value of command determines the length of the reading  
    if not command.isdigit():
        f = open(sourcefile, 'rb')
        content = f.read()
        num_bytes = len(content)
        count = int(num_bytes/12)
        f.close()
    else:
        count = int(command)
    
    return readQ(sourcefile,count)

def  readQ(sourcefile,Count):
    #file gets opened
    f = open(sourcefile, 'rb')
    if f == -1:
        msg = print('file #s could not be opened - check folder!')
        return [0,0,0]
    counter=0
    # reset Matrix to Null
    Time = np.zeros((Count))
    # reset Matrix to Null
    Data = np.zeros((Count))
    # read file unitl count is reached
    while (counter < Count):
        # Charge
        q = np.fromfile(f,dtype='float32',count=1) # 4 Bytes
        # Time
        t = np.fromfile(f,dtype='float64',count=1); # 8 Bytes
        # if data avaiable
        if q.size > 0 and t.size > 0:
            # store results
            Time[counter]=t[0]
            Data[counter]=q[0]
            counter=counter+1
    #file gets closed
    f.close()
    # returns the three important data types
    return[Time,Data,counter]

def setup_csvfile(sourcefile,command):
    # Read the semicolon-separated file into a DataFrame
    data = pd.read_csv(sourcefile, delimiter=';', skiprows=1, header=None, names=["UTC_Time", "Value"])
    values = data["Value"].astype(float)
    Data = values.to_numpy(dtype='float64') 

    # the value of command determines the length of the reading  
    if not command.isdigit():
        count = len(Data)
    else:
        count = int(command)
    
    Time = np.zeros((count))
    counter = 0        
    temp = data["UTC_Time"]
    while (counter < count):
        first_split = temp[counter].split(":")
        second_split = first_split[0].split(" ")
        seconds = re.sub(r'\.\.+', '.', str(first_split[2]))
        Time[counter] = float(second_split[1])*3600 + float(first_split[1])*60 + float(seconds)
        counter=counter+1

    # Convert to arrays
    return [Time,Data,count]

def nodi_plot(Time,Data,Count,Plot_command, grid_size):
    #responsible for the creation of the nodi* plots
    # sets up the matrix for the magnitude calculation
    magnitude_i = np.zeros(Count)
    magnitude_i_1 = np.zeros(Count)
    # sets up the matrix for the time calculation
    time_i = np.zeros(Count)
    time_i_1 = np.zeros(Count)
    # sets up the counter for the calculation; set to 2 as calculation takes PD before and after into account
    counter = 1
    counter2 = 0
    # performs the calculation for each PD pulse

    while(counter < (Count-1)):
        # magnitude
        magnitude_i[counter2] = Data[counter] - Data[counter-1]
        magnitude_i_1[counter2] = Data[counter+1] - Data[counter]
        # time
        time_i[counter2] = Time[counter] - Time[counter-1]
        time_i_1[counter2] = Time[counter+1] - Time[counter]        
        counter = counter + 1
        counter2 = counter2 + 1

    # clean the lists
    zero_indices_time = np.where(time_i <= 0)
    zero_indices_time_i = np.where(time_i_1 <= 0)

    # Remove entries from both lists at the zero indices
    magnitude_i = np.delete(magnitude_i, zero_indices_time)
    time_i = np.delete(time_i, zero_indices_time)

    magnitude_i_1 = np.delete(magnitude_i_1, zero_indices_time_i)
    time_i_1 = np.delete(time_i_1, zero_indices_time_i)

    # create 2D plot
    # sets up the plot labels
    if Plot_command == 'Q(i+1) and Q(i)':
        x = magnitude_i/(10**-12)
        y = magnitude_i_1/(10**-12)
        word = 'Magnitude'
        xlabel = r'$\Delta q_{{i}} \, \text{[pC]}$'
        ylabel = r'$\Delta q_{{i+1}} \, \text{[pC]}$'
        nodi_plot_2D_3D(x,y,word,xlabel,ylabel,grid_size,Time,Count)
    elif Plot_command == 'T(i+1) and T(i)':
        x = time_i
        y = time_i_1
        word = 'Time'
        xlabel = r'$\Delta t_{{i}} \, \text{[s]}$'
        ylabel = r'$\Delta t_{{i+1}} \, \text{[s]}$'
        nodi_plot_2D_3D(x,y,word,xlabel,ylabel,grid_size,Time,Count)
    elif Plot_command == 'T(i) and Q(i)':
        x = magnitude_i/(10**-12)
        y = time_i
        word = 'Time and Magnitude'
        xlabel = r'$\Delta q_{{i}} \, \text{[pC]}$'
        ylabel = r'$\Delta t_{{i}} \, \text{[s]}$'
        nodi_plot_2D_3D(x,y,word,xlabel,ylabel,grid_size,Time,Count)
    elif Plot_command == 'Q(i)/T(i) and T(i)':
        x = time_i
        y = (magnitude_i / time_i)/(10**-12)
        word = 'Magnitude and Time'
        xlabel = r'$\Delta t_{{i}} \, \text{[s]}$'
        ylabel = r'$\Delta q_{{i}}/ \Delta t_{{i}} \, \text{[pC/s]}$'
        nodi_plot_2D_3D(x,y,word,xlabel,ylabel,grid_size,Time,Count)
    elif Plot_command == 'Timescale':
        x = Time
        y = Data/(10**-12)
        word = 'PD Magnitude vs. Time (PD Count: ' + str(Count) + ')'
        xlabel = 'Time [s]'
        ylabel = 'Magnitude [pC]'
        time_scale_plot(x,y,word,xlabel,ylabel)
    elif Plot_command == 'Q Density':
        x = Data/(10**-12)
        xlabel = 'Magnitude Differences [pC]'
        ylabel = 'Count'
        word = 'Log Scaled Density Function of Magnitude Differences'
        plot_density_distribution(x,word,xlabel,ylabel)
    elif Plot_command == 'T Density':
        x = Time
        xlabel = 'Time Differences [s]'
        ylabel = 'Count'
        word = 'Log Scaled Density Function of Time Differences'
        plot_density_distribution(x,word,xlabel,ylabel)
    else:
        msg = print('Command is wrong!')
        return 0
    
def nodi_plot_2D_3D(x,y,word,xlabel,ylabel,grid_size,Time,Count):
    # Create a 100x100 grid of equally spaced containers
    x_bins = np.linspace(min(x), max(x), grid_size)
    y_bins = np.linspace(min(y), max(y), grid_size)

    # 2D histogram to count the number of points in each bin
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Calculate the number of minutes (assuming time is in seconds)
    num_seconds = (max(Time) - min(Time))/60

     # Calculate the mean stack count per minute
    mean_stack_count_per_min = np.sum(H) / num_seconds

    # Scale the stack heights by the mean stack count per minute
    H_scaled = (H / mean_stack_count_per_min)

    titlesize = 40
    labelsize = 40
    ticksize = 40

    # Plot the log-scaled density as a heatmap
    plt.imshow(H_scaled.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap='viridis',norm=LogNorm())
    ax = plt.gca()

    # Set the y-axis to use exponential notation
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize=ticksize)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(ticksize)
    ty = ax.yaxis.get_offset_text()
    ty.set_fontsize(ticksize)

    # Add a colorbar with logarithmic scale
    cbar = plt.colorbar()
    cbar.set_label('Mean Stack Count Per Minute', size = labelsize)
    cbar.ax.tick_params(labelsize=ticksize)

    #cbar.set_label('Normalized Values')

    # Add labels and title
    plt.xlabel(xlabel, size = labelsize)
    plt.ylabel(ylabel, size = labelsize)
    file_length = np.round((max(Time) - min(Time)),2)
    plt.title('PD Count: ' + str(Count) + ' Length: ' + str(file_length) + ' s', size = titlesize)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.subplots_adjust(left=0.125, bottom=0.125, right=0.988, top=0.86, wspace=None, hspace=None)
    plt.show()

    # Create 3D plot

    titlesize = 30
    labelsize = 20
    ticksize = 20

    # Create a meshgrid for the surface plot
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Plotting the 3D heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, np.transpose(H_scaled), cmap='viridis', edgecolor='none',norm=LogNorm())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize=ticksize)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(ticksize)
    ty = ax.yaxis.get_offset_text()
    ty.set_fontsize(ticksize)

    # Set labels
    ax.set_xlabel(xlabel, size = labelsize)
    ax.set_ylabel(ylabel, size = labelsize)
    #ax.set_title('3D Heatmap of Normalized ' + word +' Differences (PD Count: ' + str(Count) + ')', size = titlesize)

    # Create an axes for the colorbar
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # [left, bottom, width, height]

    # Create a colorbar with log normalization
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=LogNorm())
    mappable.set_array(H_scaled)

    # Create colorbar using the new axes
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label('Mean Stack Count Per Minute', size = labelsize)
    cbar.ax.tick_params(labelsize=ticksize)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    return H_scaled.T

def time_scale_plot(x,y,word,xlabel,ylabel):
    # plotting the timescale
    titlesize = 40
    labelsize = 40
    ticksize = 40
    x = x - min(x)
    plt.title(word,size = titlesize)
    plt.xlabel(xlabel, size = labelsize)
    plt.ylabel(ylabel, size = labelsize)    
    plt.plot(x, y, marker='o', linestyle='-', color='b', linewidth=1, markersize=2)

     # Adding grid lines for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
    
    # Customizing tick parameters for a cleaner look
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.94, wspace=None, hspace=None)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    
    return [x, y]

def plot_density_distribution(data,word,xlabel,ylabel):

    if len(data) < 2:
        print("Not enough data to create density plots.")
        return

    # Calculate differences
    diffs = np.diff(data)   

    #zero_indices = np.where(diffs == 0)
    #diffs = np.delete(diffs, zero_indices)
    titlesize = 40
    labelsize = 40
    ticksize = 40
    # Create a density plot for time differences
    plt.hist(diffs, bins=1000, density=False, color='skyblue', edgecolor='black')
    plt.yscale('log')
    plt.title(word, size=titlesize)
    plt.xlabel(xlabel, size = labelsize)
    plt.ylabel(ylabel, size = labelsize)
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=None, hspace=None)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

    zero_indices_diffs = np.where(diffs < 0)
    diffs = np.delete(diffs, zero_indices_diffs)

    plt.hist(diffs, bins=1000, density=False, color='skyblue', edgecolor='black')
    plt.yscale('log')
    plt.title(word, size=titlesize)
    plt.xlabel(xlabel, size = labelsize)
    plt.ylabel(ylabel, size = labelsize)
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=None, hspace=None)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

    return diffs



