
#region Functions
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib.ticker import LogFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

def setup_Qfile(sourcefile,command):
    # number of datas to zero
    
    if not command.isdigit():
        f = open(sourcefile, 'rb')
        content = f.read()
        num_bytes = len(content)
        maxanzds = int(num_bytes/12)
        f.close()
    else:
        maxanzds = int(command)
    
    return readQ(sourcefile,maxanzds)

def  readQ(sourcefile,maxanzds):
    #function open one or more .q files and move infromation back
    #file gets opened
    f = open(sourcefile, 'rb')
    if f == -1:
        msg = print('file #s could not be opened - check folder!')
        return [0,0,0]
    anzGelesen=1
    # reset Matrix to Null
    vZeit = np.zeros((maxanzds))
    # reset Matrix to Null
    vLadung = np.zeros((maxanzds))
    # read file unitl maxanzds is reached or no more infromation is avaiable
    while (anzGelesen < maxanzds):
        # Charge
        q = np.fromfile(f,dtype='float32',count=1) # 4 Bytes
        # Time
        t = np.fromfile(f,dtype='float64',count=1); # 8 Bytes
        # if data avaiable
        if q.size > 0 and t.size > 0:
            # store results
            vZeit[anzGelesen]=t[0]
            vLadung[anzGelesen]=q[0]
            anzGelesen=anzGelesen+1
        else:
            # otherwise stop
            break
    #file gets closed
    f.close()
    return[vZeit,vLadung,anzGelesen]

def nodi_plot(vZeit,vLadung,anzGelesen):
    magnitude_i = np.zeros(anzGelesen)
    magnitude_i_1 = np.zeros(anzGelesen)

    time_i = np.zeros(anzGelesen)
    time_i_1 = np.zeros(anzGelesen)

    count = 2

    while(count < (anzGelesen-1)):
        magnitude_i[count] = vLadung[count] - vLadung[count-1]
        magnitude_i_1[count] = vLadung[count] - vLadung[count+1]

        time_i[count] = vZeit[count] - vZeit[count-1]
        time_i_1[count] = vZeit[count] - vZeit[count+1]

        count = count + 1

    # Create a scatter plot with density as color
    #xy = np.vstack([magnitude_i, magnitude_i_1])
    #z = gaussian_kde(xy)(xy)  # Kernel density estimation for color

    #plt.scatter(magnitude_i,magnitude_i_1,c=z)
    #plt.show()

    # Create a 100x100 grid of equally spaced containers
    grid_size = 100
    x_bins = np.linspace(min(magnitude_i), max(magnitude_i), grid_size)
    y_bins = np.linspace(min(magnitude_i_1), max(magnitude_i_1), grid_size)

    # 2D histogram to count the number of points in each bin
    H, xedges, yedges = np.histogram2d(magnitude_i, magnitude_i_1, bins=[x_bins, y_bins])

    # Calculate the number of minutes (assuming time is in minutes)
    num_minutes = (max(vZeit) - min(vZeit))/60

    # Calculate the mean stack count per minute
    mean_stack_count_per_min = np.sum(H) / num_minutes

    # Scale the stack heights by the mean stack count per minute
    H_scaled = H / mean_stack_count_per_min

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
    
    # Add a colorbar with logarithmic scale
    cbar = plt.colorbar()
    cbar.set_label('Log-Scaled Normalized Density')

    # Add labels and title
    plt.xlabel('Difference with Previous Magnitude')
    plt.ylabel('Difference with Next Magnitude')
    plt.title('Log-Scaled Magnitude Differences Heatmap (100x100 grid)')

    plt.show()

    # Create 3D plot

    # Create a meshgrid for the surface plot
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Plotting the 3D heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, H_scaled, cmap='viridis', edgecolor='none',norm=LogNorm())

    # Set labels
    ax.set_xlabel('Difference with Previous Magnitude')
    ax.set_ylabel('Difference with Next Magnitude')
    ax.set_zlabel('Count')
    ax.set_title('3D Heatmap of Magnitude Differences')

    # Create an axes for the colorbar
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # [left, bottom, width, height]

    # Create a colorbar with log normalization
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=LogNorm())
    mappable.set_array(H_scaled)

    # Create colorbar using the new axes
    cbar = plt.colorbar(mappable, cax=cax, label='Count')
    cbar.set_label('Log Scale')

    plt.show()



#endregion

#command = '80000'
command = 'end'
filename = 'C:/Users/lukas/Downloads/Thesis/Waveforms/0001_matlab/unit1.1.Q'
[vZeit,vLadung,anzGelesen] = setup_Qfile(filename,command)

nodi_plot(vZeit,vLadung,anzGelesen)

# Create the plot
#plt.plot(vZeit, vLadung, label='PD Magnitude', marker='o')  # Line plot with markers

# Adding titles and labels
#plt.title('Simple Line Plot')               # Title of the plot
#plt.xlabel('X-axis Label')                  # Label for X-axis
#plt.ylabel('Y-axis Label')                  # Label for Y-axis

# Add a legend
#plt.legend()

# Show grid lines for better readability
#plt.grid(True)

# Display the plot
#plt.show()


