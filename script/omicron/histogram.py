# -*- coding: utf-8 -*-

from __future__ import division

from glob import glob
import os.path
import warnings

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt



def _build_histogram(pulses, phase_bins=720, charge_bins=None,
                     charge_range=(1e-12, 200e-12), log_charge=True,
                     timedelta=None, clim=None, bipolar=True):
    charge_bins = charge_bins or (1000 if bipolar else 500)
    charge_limits = (charge_bins//2 + 1) if bipolar else (charge_bins + 1)
    if log_charge:

        bins_charge = np.logspace(np.log10(charge_range[0]),
                                np.log10(charge_range[1]),
                                charge_limits)
    else:
        bins_charge = np.linspace(charge_range[0], charge_range[1],
                                  charge_limits)
    bins = (np.linspace(0, 1, phase_bins+1), bins_charge)

    #d = np.hstack((np.asarray(phase[slice_]), np.asarray(charge[slice_])))
    #print(d.shape, d.dtype)
    
    phase = pulses.phase
    charge = pulses.charge
    
    if not bipolar:
        charge = np.abs(charge)

    hist_pos, _, ce = np.histogram2d(np.asarray(phase, dtype=np.float64),
                                     np.asarray(charge, dtype=np.float64),
                                     bins=bins)
    delta_charge = ce[1:] - ce[:-1]
    hist_pos /= delta_charge * 1e12 * (360 / phase_bins)

    if bipolar:
        hist_neg, _, ce = np.histogram2d(np.asarray(phase, dtype=np.float64),
                                        -np.asarray(charge, dtype=np.float64),
                                        bins=bins)
        delta_charge = ce[1:] - ce[:-1]
        hist_neg /= delta_charge * 1e12 * (360 / phase_bins)

        hist = np.hstack((hist_pos[:, ::-1], hist_neg))
    else:
        hist = hist_pos[:, ::-1]

    if timedelta:
        hist /= timedelta
    hist = hist.T
    return hist


def _get_ticks_logarithmic(start, end):
    digits = 1, 3, 5, 7
    tick_locs = []
    for exponent in range(int(np.floor(np.log10(start))),
                          int(np.ceil(np.log10(end)))):
        for d in digits:
            v = d * 10**exponent
            if v <= start:
                continue
            elif v > end:
                break
            else:
                tick_locs.append("%.3g" % v)        #"%.3g" to avoid too many decimals on axis
    return (np.array(tick_locs)).astype(np.float)   #astype(np.float) to avoid too many decimals on axis


def _get_axis_image(ax, charge_range=(1e-12, 200e-12), log_charge=True,
                    clim=(1e1, 1e5), timedelta=None,
                    bipolar=True, phase_bins=720, charge_bins=1000, cb=True):
    clim = np.log10(clim[0]), np.log10(clim[1])
    # plt.style.use('classic')     # modify colorbar and plot appearance
    if log_charge:
        yticks = np.array(_get_ticks_logarithmic(charge_range[0], charge_range[1]))
        yticks_loc = ((np.log10(np.array(yticks)) - np.log10(charge_range[0]))
                      / (np.log10(charge_range[1]) - np.log10(charge_range[0])))
    else:
        yticks = np.linspace(charge_range[0], charge_range[1], 6)
        yticks_loc = ((yticks - charge_range[0])
                      / (charge_range[1] - charge_range[0]))

    # Create AxisImage artist
    a = np.empty((charge_bins, phase_bins))
    a[...] = np.nan
    if bipolar:
        ai = ax.imshow(a, aspect='auto', clim=clim, extent=(0, 360, -1, 1))
    else:
        ai = ax.imshow(a, aspect='auto', clim=clim, extent=(0, 360, 0, 1))

    # Set ticks and grid lines
    ax.set_xticks(np.array([0, 90, 180, 270, 360]))

    # Avoid duplicate zeros; zero/cutoff is added below
    tickmask = yticks > charge_range[0]
    yticks = yticks[tickmask]
    yticks_loc = yticks_loc[tickmask]

    # Choose between pC and nC as units for charge
    charge_scale = 1e12 #if charge_range[1] < 1.5e-9 else 1e9
    yticks *= charge_scale
    yticks=np.around(yticks,1)                                              #remove too many decimals

    if bipolar:
        yticks = (list(-yticks[::-1])
                  + ['\u00b1 {:.1f}'.format(charge_range[0] * 1e12)]
                  + list(yticks))
        yticks_loc = list(-yticks_loc[::-1]) + [0] + list(yticks_loc)
    else:
        yticks = (['{:.1f}'.format(charge_range[0] * 1e12)]
                  + list(yticks))
        yticks_loc = [0] + list(yticks_loc)
    ax.set_yticks(yticks_loc)
    ax.set_yticklabels(yticks)
    ax.grid(True)

    # Set axis labels
    ax.set_xlabel('Phase angle [$^\circ$]')
    ax.set_ylabel('Apparent charge [{}C]'
                  ''.format('p' if charge_scale == 1e12 else 'n'))

    # Add a sine curve
    t = np.linspace(0, 360, 360)
    waveform = (0.5*np.sin(np.deg2rad(t)) if bipolar else
                (0.25*np.sin(np.deg2rad(t)) + 0.5))
    ax.plot(t, waveform, 'k-')

    # Add colorbar
    if cb:
        cbar = ax.get_figure().colorbar(ai)
        cticks = np.arange(int(np.ceil(clim[0])),
                           int(np.ceil(clim[1])), dtype=float)
        if len(cticks) < 2:
            cticks = np.array([clim[0], clim[1]], dtype=float)
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(np.round(10**cticks,2))                                #np.round to avoid too many decimals on axis
        cbar.set_label('PD repetition rate [PDs/s$^\circ$*pC]' if not timedelta is None
                       else 'PD repetition rate [PDs/$^\circ$pC]')

    return ai


def partial_histograms(p, q, t, timespans, **histogram_options):
    assert timespans.ndim == 2
    assert timespans.shape[1] == 2
    if ('timedelta' in histogram_options and
        not histogram_options['timedelta'] is None):
        warnings.warn('timedelta is already set when it could be calculated')


    indices = t.searchsorted(timespans)
    for i in range(indices.shape[0]):
        start = indices[i, 0]
        end = indices[i, 1]
        s = slice(start, end)
        timedelta = t[end if end < len(t) else end-1] - t[start]
        ho = histogram_options.copy()
        if ((not 'timedelta' in ho) or (not ho['timedelta'] is None)):
            ho['timedelta'] = timedelta
        yield _build_histogram(p[s], q[s], **ho), timedelta


def _set_histogram(ai, hist):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        loghist = np.log10(hist)
    ai.set_data(loghist)
    ai.get_figure().canvas.draw()


def plot_histogram(pulses, ax=None, charge_range=(1e-12, 200e-12),
                   log_charge=True, clim=None, timedelta=None,
                   bipolar=True, phase_bins=720, charge_bins=None):
    charge_bins = charge_bins or (1000 if bipolar else 500)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None
    # plt.jet() #change color bar appearance to classical "jet" appearance
    hist = _build_histogram(pulses, charge_range=charge_range,
                            log_charge=log_charge,
                            phase_bins=phase_bins, bipolar=bipolar,
                            charge_bins=charge_bins, timedelta=timedelta)
    if clim is None:
        mask = np.isfinite(hist) & (hist > 0)
        clim = (np.min(hist[mask]), np.max(hist[mask]))
    
    cb=True
    ai = _get_axis_image(ax, charge_range, log_charge, clim, timedelta,
                         bipolar, phase_bins, charge_bins, cb)
    _set_histogram(ai, hist)
    if fig:
        fig.tight_layout()
        
    

    return hist
