import os.path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from omicron import load_and_preprocess, plot_histogram

BASE_PATH = 'C:/Users/hansm/Documents/Prosjekter/FastTrans/eksperiment/'
# Format of FILES: (filename relative to BASE_PATH,
#                   approx frequency,
#                   correct phase and voltage?)
FILES = [
         #('data/20200228/omicron/0001/unit1.1', 30, True, 'spectrum_5'),
         #('data/20200608/omicron/0001/unit1.1', 30, True, 'test'),
         ('data/20200608/omicron/0001/unit1.2', 30, True, 'pmt_2'),
         #('data/20200423/omicron/0001/unit1.1', 30, True),
         #('data/20200402/omicron/0001/unit1.2', 30, True, 'pmt'),
         #('data/20200422/omicron/0001/unit1.1', 30, True, 'mini_circuit'),
        ]

FIG_BASE_PATH = 'C:/Users/hansm/Documents/Prosjekter/FastTrans/arbeidsnotat/pd_detection/fig/'
        

def main():
    for bname, freq_hint, correct, name in FILES:
        fname = os.path.join(BASE_PATH, bname)
        _, pulses, rms_voltage, frequency = (load_and_preprocess(fname, freq_hint, correct, correct))
        print(type(pulses))
    
    # filter out pulses
    charge_limits = (3.2e-13, 4000e-12)
    pulses_new = pulses[pulses.charge > charge_limits[0]]
    print(pulses_new)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_histogram(pulses_new, ax=ax, charge_range=(10e-12, 500e-12),
                   bipolar=False, log_charge=False)
    fig.tight_layout()
    fig_fname = os.path.join(FIG_BASE_PATH,name+".png")
    plt.savefig(fig_fname)
    plt.show()
    

if __name__ == "__main__":
    main()
    