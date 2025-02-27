import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator) 

# Data
labels = [
    "Day 1 - One", "Day 1 - Two", "Day 2 - One", "Day 2 - Two",
    "Day 3 - One", "Day 3 - Two", "Day 4 - One", "Day 4 - Two"
]
values1 = [-116, -35, -40, -152, -132, -40, -44, -44]
values2 = [5.59E+04, 5.29E+04, 3.37E+04, 4.85E+04, 
           3.39E+04, 3.66E+04, 2.86E+04, 3.45E+04]

titlesize = 30
labelsize = 24
ticksize = 24
fig, ax1 = plt.subplots(figsize=(12, 8))

# Primary axis
ax1.plot(labels, values1, marker='o', linestyle='-', color='b', label="PD Apparent Charge Magnitude [pC]")
ax1.set_ylabel("PD Apparent Charge Magnitude [pC]", color='b', fontsize=24)
ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
ax1.tick_params(axis='x', labelsize=12)

# Secondary axis

ax2 = ax1.twinx() 
ax2.set_ylabel('Y2-axis', color = 'green') 
plot_2 = ax2.plot(labels, values2, marker='s', linestyle='--', color='r', label="Repetition Rate [1/s]")
ax2.set_ylabel("Repetition Rate [1/s]", color='r', fontsize=24)
ax2.tick_params(axis='y', labelcolor='r', labelsize=20)
ax2.yaxis.set_major_formatter(FormatStrFormatter('% 1.1e')) 


# Formatting
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=24)
ax1.set_title("PD Charge Magnitude and Repetition Rate", fontsize=30)
ax1.grid(True)

# Show plot
plt.tight_layout()
plt.show()

# Data
labels = [
    "Day 1", "Day 2", "Day 3 - One", "Day 3 - Two", "Day 4 - One", "Day 4 - Two", "Day 7"
]
values1 = [-14.3, -11, -81, -23, -77, -19, -20]
values2 = [4.17E+04, 1.14E+03, 1.59E+05, 1.60E+05, 1.60E+05, 1.60E+05, 1.60E+05]

titlesize = 30
labelsize = 24
ticksize = 24
fig, ax1 = plt.subplots(figsize=(12, 8))

# Primary axis
ax1.plot(labels, values1, marker='o', linestyle='-', color='b', label="PD Apparent Charge Magnitude [pC]")
ax1.set_ylabel("PD Apparent Charge Magnitude [pC]", color='b', fontsize=24)
ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
ax1.tick_params(axis='x', labelsize=12)

# Secondary axis

ax2 = ax1.twinx() 
ax2.set_ylabel('Y2-axis', color = 'green') 
plot_2 = ax2.plot(labels, values2, marker='s', linestyle='--', color='r', label="Repetition Rate [1/s]")
ax2.set_ylabel("Repetition Rate [1/s]", color='r', fontsize=24)
ax2.tick_params(axis='y', labelcolor='r', labelsize=20)
ax2.yaxis.set_major_formatter(FormatStrFormatter('% 1.1e')) 


# Formatting
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=24)
ax1.set_title("PD Charge Magnitude and Repetition Rate", fontsize=30)
ax1.grid(True)

# Show plot
plt.tight_layout()
plt.show()
