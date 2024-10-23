
#import packages
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib.ticker import LogFormatter
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

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
    counter=1
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

def nodi_plot(Time,Data,Count,Command):
    #responsible for the creation of the nodi* plots
    # sets up the matrix for the magnitude calculation
    magnitude_i = np.zeros(Count)
    magnitude_i_1 = np.zeros(Count)
    # sets up the matrix for the time calculation
    time_i = np.zeros(Count)
    time_i_1 = np.zeros(Count)
    # sets up the counter for the calculation; set to 2 as calculation takes PD before and after into account
    counter = 2
    # performs the calculation for each PD pulse
    while(counter < (Count-1)):
        # magnitude
        magnitude_i[counter] = Data[counter] - Data[counter-1]
        magnitude_i_1[counter] = Data[counter+1] - Data[counter]
        # time
        time_i[counter] = Time[counter] - Time[counter-1]
        time_i_1[counter] = Time[counter+1] - Time[counter]
        # update counter
        counter = counter + 1

    # clean the lists
    zero_indices_time = np.where(time_i == 0)
    zero_indices_time_i = np.where(time_i_1 == 0)

    # Remove entries from both lists at the zero indices
    magnitude_i = np.delete(magnitude_i, zero_indices_time)
    time_i = np.delete(time_i, zero_indices_time)

    magnitude_i_1 = np.delete(magnitude_i_1, zero_indices_time_i)
    time_i_1 = np.delete(time_i_1, zero_indices_time_i)

    # create 2D plot
    # sets up the plot labels
    if Command == 'Q(i+1) and Q(i)':
        x = magnitude_i
        y = magnitude_i_1
        word = 'Magnitude'
        xlabel = r'$\Delta q_{{i}} \, \text{(C)}$'
        ylabel = r'$\Delta q_{{i+1}} \, \text{(C)}$'
    elif Command == 'T(i+1) and T(i)':
        x = time_i
        y = time_i_1
        word = 'Time'
        xlabel = r'$\Delta t_{{i}} \, \text{(s)}$'
        ylabel = r'$\Delta t_{{i+1}} \, \text{(s)}$'
    elif Command == 'T(i) and Q(i)':
        x = magnitude_i
        y = time_i
        word = 'Time and Magnitude'
        xlabel = r'$\Delta q_{{i}} \, \text{(C)}$'
        ylabel = r'$\Delta t_{{i}} \, \text{(s)}$'
    elif Command == 'Q(i)/T(i) and T(i)':
        x = time_i
        y = magnitude_i / time_i
        word = 'Magnitude and Time'
        xlabel = r'$\Delta t_{{i}} \, \text{(s)}$'
        ylabel = r'$\Delta q_{{i}}/ \Delta t_{{i}} \, \text{(C/s)}$'
    else:
        msg = print('Command is wrong!')
        return 0
    # Create a 100x100 grid of equally spaced containers
    grid_size = 100
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

    # Plot the log-scaled density as a heatmap
    plt.imshow(H_scaled.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap='viridis',norm=LogNorm())
    ax = plt.gca()

    # Set the y-axis to use exponential notation
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add a colorbar with logarithmic scale
    cbar = plt.colorbar()
    cbar.set_label('Log-Scaled Normalized Density')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Log-Scaled ' + word + ' Differences Heatmap (PD Count: ' + str(Count) + ')')
    plt.show()

    # Create 3D plot
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

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Density')
    ax.set_title('3D Heatmap of ' + word +' Differences (PD Count: ' + str(Count) + ')')

    # Create an axes for the colorbar
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # [left, bottom, width, height]

    # Create a colorbar with log normalization
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=LogNorm())
    mappable.set_array(H_scaled)

    # Create colorbar using the new axes
    cbar = plt.colorbar(mappable, cax=cax, label='Count')
    cbar.set_label('Log Scaled Density')

    plt.show()
#endregion

#command = '80000'
#command = 'end'
#filename = 'C:/Users/lukas/Downloads/Thesis/Waveforms/0001/unit1.1.Q'
#[Time,Data,Count] = setup_Qfile(filename,command)

#nodi_plot(Time,Data,Count,'QT')

# Create the plot
#plt.plot(Time, Data, label='PD Magnitude', marker='o')  # Line plot with markers

# Adding titles and labels
#plt.title('Simple Line Plot')               # Title of the plot
#plt.xlabel('Time (s)')                  # Label for X-axis
#plt.ylabel('PD Magnitude (pC)')                  # Label for Y-axis

# Add a legend
#plt.legend()

# Show grid lines for better readability
#plt.grid(True)

# Display the plot
#plt.show()


