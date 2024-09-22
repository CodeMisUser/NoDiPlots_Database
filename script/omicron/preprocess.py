import numpy as np
import pandas as pd
from scipy import fftpack, signal
import numba

from .io_ import load_pulses, load_voltage 
#from .hilbert import hilbert, clean_scipy_fftpack_cache 
from .hilbert import hilbert
from . import calibration



def correct_phase(pulses, offset=0):
    pulses_new = pulses.copy()
    pulses_new.phase = pulses.phase - offset
    return pulses_new


def determine_phase(a, offset=True):
    """
    Determine phase of a signal consisting of a single Fourier component
    using the Hilbert transform (time complexity O(N log N)).

    Parameters
    ----------
    a : Real-valued signal containing a single Fourier component
    Returns
    -------
    phase : Phase in the interval [0, 1).
    """
    a = np.asarray(a)

    # Remove DC offset
    if offset:
        a[:] = remove_offset(a)
    from time import time
    start = time()
    ha = hilbert(a)
    end = time()
    #print('Time elapsed in Hilbert transform: {:.3g} s'.format(end - start))
    phase = ((np.angle(ha) + np.pi/2) / (2*np.pi)) % 1.
    return phase


def add_phase(df, signal_name='voltage'):
    phase = determine_phase(df[signal_name])
    df = df.assign(phase=phase)
    return df


def determine_frequency(t, p):
    t = np.asarray(t)
    p = np.asarray(p)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError('Both time and phase array must be one-dimensional')
    if t.shape[0] != p.shape[0]:
        raise ValueError('Time and phase array must have equal length')
    # Find the discontinuities in phase
    tz = (t[:-1])[(p[1:] - p[:-1]) < -0.9]
    if tz.shape[0] < 2:
        raise ValueError("There are less than two periods of the signal")
    # Find the length of each period
    periods = tz[1:] - tz[:-1]
    #if ((periods.max() - periods.min()) / periods.min()) > 0.02:
    #    raise ValueError("Length of periods differ by more than 2%")
    # Return the average frequency
    return 1 / periods.mean()


@numba.njit
def _downsample_smooth(a, factor, kernel, a_out):
    for i in range(a_out.shape[0]):
        v = 0.0
        j0 = i * factor + factor // 2 - kernel.shape[0] // 2
        for j in range(kernel.shape[0]):
            v += kernel[j] * a[j0 + j]
        a_out[i] = v
    return a_out


def downsample_smooth(a, factor, kernel_width=None):
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError('a must be one-dimensional')
    if np.abs(factor - int(factor)) > 1e-5:
        print("Warning: downsample_smooth: factor should be an integer")
    factor = int(factor)
    kernel_width = kernel_width if not kernel_width is None else factor
    if np.abs(kernel_width - int(kernel_width)) > 1e-5:
        print("Warning: downsample_smooth: kernel_width should be an integer")
    kernel_width = int(kernel_width)
    kernel = signal.hann(kernel_width)
    kernel = kernel / kernel.sum()
    a_out = np.zeros(a.shape[0] // factor)
    _downsample_smooth(a, factor, kernel, a_out)
    return a_out


def remove_offset(a):
    offset = (a.min() + a.max()) / 2
    return a - offset


def load_and_preprocess(fname, frequency_hint,
                        correct_phase=True, correct_voltage=True):
    voltage = load_voltage(fname)
    # Downsample voltage if possible
    downsample_factor = 50 / frequency_hint   # downsample relative to 50Hz
    if downsample_factor >= 2:
        v = downsample_smooth(voltage.voltage, downsample_factor)
        t = downsample_smooth(voltage.index, downsample_factor, 10)
        del voltage
        voltage = pd.DataFrame({'voltage': v}, index=t)
    # Remove DC offset
    voltage.voltage = remove_offset(voltage.voltage)
    # Determine phase for all voltage samples
    voltage = voltage.assign(phase=determine_phase(voltage.voltage))
    # Determine average frequency and rms voltage
    frequency = determine_frequency(voltage.index, voltage.phase)
    rms_voltage = np.sqrt(np.mean(voltage.voltage**2))
    if correct_voltage:
        rms_voltage = rms_voltage / calibration.get_magnitude(frequency)
    # Load partial discharges and assign phase
    pulses = load_pulses(fname, load_phase=not correct_phase)
    if correct_phase:
        pulse_phase = (np.interp(pulses.index, voltage.index, voltage.phase)
                       - calibration.get_phase(frequency)) % 1
        pulses = pulses.assign(phase=pulse_phase)

    return voltage, pulses, rms_voltage, frequency
