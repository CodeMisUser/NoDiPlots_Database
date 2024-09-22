# -*- coding: utf-8 -*-

from __future__ import division

from glob import glob
import os.path
import warnings

import numpy as np
from scipy import signal
import pandas as pd


def _load_voltage(fname):
    return np.memmap(fname, np.float32, 'r')

def _load_charge(fname):
    dtype = np.dtype([('charge', np.float32), ('time', np.float64)])
    return np.memmap(fname, dtype, 'r')

def _load_phase(fname):
    return np.memmap(fname, np.float64, 'r')

def _to_dataframe(time, phase, charge):
    return pd.DataFrame(dict(time=np.asarray(time),
                             phase=np.asarray(phase),
                             charge=np.asarray(charge)))


def get_units(dirname):
    fnames = glob(os.path.join(dirname, '*.V'))
    units = [os.path.splitext(os.path.basename(fn))[0] for fn in fnames]
    return units


def load_pulses_chunked(basename, chunksize=1e5):
    charge_mm = _load_charge(basename + '.Q')
    phase_mm = _load_phase(basename + '.PH')
    assert charge_mm.shape == phase_mm.shape

    num_records = charge_mm.shape[0]
    loc = 0
    while loc < num_records:
        s = slice(loc, loc + chunksize)
        time = charge_mm['time'][s]
        charge = charge_mm['charge'][s]
        phase = phase_mm[s]
        yield _to_dataframe(time, phase, charge)
        loc += chunksize


def load_voltage(basename):
    voltage = np.asarray(_load_voltage(basename + '.V'))
    time = np.arange(voltage.shape[0]) * 48e-6   # Sampling interval is 48 us
    return pd.DataFrame(dict(voltage=voltage), index=time)


def load_pulses(basename, load_phase=True):
    charge_mm = _load_charge(basename + '.Q')
    time = charge_mm['time'][:]
    charge = charge_mm['charge'][:]
    series = {'charge': charge}
    if load_phase:
        phase_mm = _load_phase(basename + '.PH')
        phase = phase_mm[:]
        series['phase'] = phase
    return pd.DataFrame(series, index=time)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #DIRECTORY = ('/media/sf_c_josteinf/Documents/Insulation Quality'
                 #'/stream_recordings/2015-06-15/0002')
    DIRECTORY = ('C:/Users/henstr/Documents/Henrik/data/exported_matlab_files/5kv')

    units = get_units(DIRECTORY)
    print(units)

    for unit in units:
        v = _load_voltage(os.path.join(DIRECTORY, unit + '.V'))
        npts = 1000
        interval = 48e-6
        ve = v[-npts:]

        # Load data
        p = _load_phase(os.path.join(DIRECTORY, unit + '.PH'))
        qt = _load_charge(os.path.join(DIRECTORY, unit + '.Q'))
        q = qt['charge']
        t = qt['time']
        timedelta = t[-1]
        print('Timedelta: {:.4g}s'.format(timedelta))

        # Settings for histograms
        histogram_options = dict(clim=(1e-6, 1e-1),
                                 charge_range=(50e-12, 30e-9))

        # Create a histogram
        fig, ax = plt.subplots()
        ai = _get_axis_image(ax, timedelta=timedelta, **histogram_options)
        hist = _build_histogram(p, q, timedelta=timedelta, **histogram_options)
        _set_histogram(ai, hist)


        # Create a sequence of split histograms
        window_width = 30
        frame_step = 30
        frames_per_second = 5

        fig, ax = plt.subplots()
        ai = _get_axis_image(ax, timedelta=window_width,
                                 **histogram_options)
        timespans = np.array([[frame_step*i, frame_step*i + window_width]
                               for i in range(int(t[-1] / frame_step))])
        for i, (hist, timedelta) in enumerate(partial_histograms(p, q, t,
                                                            timespans,
                                                            **histogram_options)):
            _set_histogram(ai, hist)
            fig.savefig(os.path.join(DIRECTORY, '{:04d}.png'.format(i)),
                        dpi=192)
        plt.close(fig)

    plt.show()


