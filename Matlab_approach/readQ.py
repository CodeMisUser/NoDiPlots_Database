
#region Functions
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib.ticker import LogFormatter
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

def setup_Qfile(sourcefile,command):
    # number of datas to zero
    
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
    #function open one or more .q files and move infromation back
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
    # read file unitl count is reached or no more infromation is avaiable
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
        else:
            # otherwise stop
            break
    #file gets closed
    f.close()
    return[Time,Data,counter]

def nodi_plot(Time,Data,Count,Command):
    magnitude_i = np.zeros(Count)
    magnitude_i_1 = np.zeros(Count)

    time_i = np.zeros(Count)
    time_i_1 = np.zeros(Count)

    counter = 2

    while(counter < (Count-1)):
        magnitude_i[counter] = Data[counter] - Data[counter-1]
        magnitude_i_1[counter] = Data[counter+1] - Data[counter]

        time_i[counter] = Time[counter] - Time[counter-1]
        time_i_1[counter] = Time[counter+1] - Time[counter]

        counter = counter + 1

    # Create a scatter plot with density as color
    #xy = np.vstack([magnitude_i, magnitude_i_1])
    #z = gaussian_kde(xy)(xy)  # Kernel density estimation for color

    #plt.scatter(magnitude_i,magnitude_i_1,c=z)
    #plt.show()

    # create 2D plot

    # Create a 100x100 grid of equally spaced containers
    grid_size = 100
    if Command == 'Q':
        x = magnitude_i
        y = magnitude_i_1
        word = 'Magnitude'
        xlabel = 'Q\'i (C)'
        ylabel = 'Q\'i+1 (C)'
    elif Command == 'T':
        x = time_i
        y = time_i_1
        word = 'Time'
        xlabel = 'T\'i (S)'
        ylabel = 'T\'i+1 (S)'
    elif Command == 'QT':
        x = magnitude_i
        y = time_i
        word = 'Magnitude and Time'
        xlabel = 'Q\'i (C)'
        ylabel = 'T\'i (S)'
    else:
        msg = print('Command is wrong!')
        return 0

    x_bins = np.linspace(min(x), max(x), grid_size)
    y_bins = np.linspace(min(y), max(y), grid_size)

    # 2D histogram to count the number of points in each bin
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Calculate the number of minutes (assuming time is in minutes)
    num_minutes = (max(Time) - min(Time))/60

    # Calculate the mean stack count per minute
    mean_stack_count_per_min = np.sum(H) / num_minutes

    # Scale the stack heights by the mean stack count per minute
    H_scaled = (H / mean_stack_count_per_min)

    # Apply a logarithmic scale to the stack counts for color coding
    #for y in range(99):
    #    for x in range(99):
    #    
    #        if H_scaled[x,y] != 0:
    #            H_scaled[x,y] = np.log10(H_scaled[x,y])
    #        else:
    #            H_scaled[x,y] = np.nan
    
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
command = 'end'
filename = 'C:/Users/lukas/Downloads/Thesis/Waveforms/0001_matlab_DC/unit1.1.Q'
[Time,Data,Count] = setup_Qfile(filename,command)

nodi_plot(Time,Data,Count,'QT')

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


