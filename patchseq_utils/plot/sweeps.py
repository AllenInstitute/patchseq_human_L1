from ipfx.script_utils import dataset_for_specimen_id
from ipfx.time_series_utils import subsample_average
from ipfx.qc_feature_extractor import sweep_qc_features
from ipfx.utilities import drop_failed_sweeps
from . import scalebars
from .scalebars import add_scalebar
from .. import lims
import numpy as np
import matplotlib.pyplot as plt


def lims_sweep_table(specimen_id, sweep_type="Long Square", qc_sweeps=True, spiking=None, depolarizing=None):
    sweeps = lims.get_sweep_info(specimen_id, spiking=spiking, passed_only=qc_sweeps,
                               depolarizing=depolarizing, sweep_type=sweep_type)
    return sweeps

def get_dataset_sweeps(specimen_id, path=None, lims_sweep_info=False, qc_sweeps=True, sweeps_query=None, stimulus="Long Square"):
    if path is None:
        dataset = dataset_for_specimen_id(specimen_id)
    else:
        dataset = dataset_for_specimen_id(specimen_id, data_source='filesystem', path=path)
    if lims_sweep_info:
        sweep_table = lims_sweep_table(specimen_id, sweep_type=stimulus, qc_sweeps=qc_sweeps)
        sweep_table['spiking'] = sweep_table["num_spikes"] > 0
        if qc_sweeps:
            sweep_table = sweep_table.loc[lambda df: df['workflow_state'].str.contains('passed')]
    else:
        if qc_sweeps:
            drop_failed_sweeps(dataset)
        else:
            dataset.sweep_info = sweep_qc_features(dataset)
        sweep_table = dataset.filtered_sweep_table(stimuli=[stimulus]).copy()
        # test for zero crossing during stim only as shortcut for spiking sweeps
        dataset.sweep_set(sweep_table["sweep_number"]).select_epoch('stim')
        sweep_table['spiking'] = sweep_table["sweep_number"].apply(
            lambda n: any(dataset.sweep(n).v > 0))
    if sweeps_query is not None:
        sweep_table = sweep_table.query(sweeps_query)
    return dataset, sweep_table
    
def plot_spikes(dataset, sweeps, n_max=1, scalebar=False, offset=0, **kwargs):
    sweeps = sweeps.query("spiking == True")
    if len(sweeps)==0:
        return
    sweepset = dataset.sweep_set(sweeps["sweep_number"].iloc[:n_max])
    plot_sweeps_thumb(sweepset, scalebar=scalebar, offset=offset, **kwargs)


def plot_sag(dataset, sweeps, n_max=3, scalebar=False, offset=0, **kwargs):
    sweeps = sweeps.query("stimulus_amplitude <= -20 & stimulus_amplitude >= -80")
    sweeps = sweeps.sort_values("stimulus_amplitude", ascending=True)
    if len(sweeps)==0:
        return
    sweepset = dataset.sweep_set(sweeps["sweep_number"].iloc[:n_max])
    plot_sweeps_thumb(sweepset, scalebar=scalebar, offset=offset, **kwargs)
    
def plot_hero(dataset, sweeps, n_max=1, scalebar=False, offset=0, **kwargs):
    sweeps = sweeps.query("stimulus_amplitude > 0 & spiking")
    sweeps = sweeps.sort_values("stimulus_amplitude", ascending=True)
    amp = sweeps["stimulus_amplitude"]
    # hero sweep
    sweeps = sweeps[(amp > amp.min() + 39) & (amp < amp.min() + 61)]
    if len(sweeps)==0:
        return
    sweepset = dataset.sweep_set(sweeps["sweep_number"].values[:n_max])
    plot_sweeps_thumb(sweepset, scalebar=scalebar, offset=offset, **kwargs)

def plot_rheo(dataset, sweeps, scalebar=False, offset=0, plot_peri=False, **kwargs):
    sweeps = sweeps.query("stimulus_amplitude > 0")
    sweeps = sweeps.sort_values("stimulus_amplitude", ascending=True)
    if sweeps["spiking"].sum() > 0:
        rheo_i = sweeps.index.get_loc(sweeps[sweeps["spiking"]].index[0])
        if plot_peri:
            numbers = sweeps["sweep_number"].values[max(rheo_i-1, 0):rheo_i+1]
        else:
            numbers = [sweeps["sweep_number"].values[rheo_i]]
        sweepset = dataset.sweep_set(numbers)
    elif len(sweeps) > 0:
        # plot highest amp sweep
        sweepset = dataset.sweep_set(sweeps["sweep_number"].values[-1])
    else:
        return
        
    plot_sweeps_thumb(sweepset, scalebar=scalebar, offset=offset, **kwargs)

def plot_sweep_panel(dataset, sweeps, scalebar=False, plot_peri=False, figsize=(4,4), ax=None, **args):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # ax.set_prop_cycle('color',plt.cm.cool(np.linspace(0,1,4)))
    ax.set_prop_cycle('alpha',np.linspace(1,0.4,4))
    plt.sca(ax)
    plot_sag(dataset, sweeps, n_max=1, **args)
    plot_rheo(dataset, sweeps, plot_peri=plot_peri, **args)
    plot_hero(dataset, sweeps, scalebar=scalebar, offset=20, **args)
        
def plot_sweeps_thumb(sweepset, scalebar=False, xmin=-0.05, xmax=1.15, offset=0, 
                      dy=None, loc=None, **kwargs):
    sweepset.select_epoch("recording")
    sweepset.align_to_start_of_epoch("stim")
    nblock = 10
    delta = 0
    for i, v in enumerate(sweepset.v):
        delta += offset
        t = subsample_average(sweepset.t[i], nblock)
        plt.plot(t, subsample_average(v, nblock) + delta, **kwargs)
    plt.axis('off')
    ax = plt.gca()
    plt.xlim(xmin, xmax)
    if dy:
        y0, y1 = ax.get_ylim()
        plt.ylim(y1-dy, y1)
    if scalebar:
        add_scalebar(ax, loc=loc, matchx=False, matchy=False, sizex=0.5, 
                    sizey=10, textprops=dict(size='large', weight='bold'))
        #  labelx='0.5 s', labely='10 mV',
    # if scale_factor:
    #     w = x1-x0
    #     h = y1-y0
    #     set_size(w/scale_factor, h/scale_factor, ax=ax)

from ipfx.spike_detector import detect_putative_spikes

def plot_sweep_spikes(sweep, nspikes=[0], offset=3.5, pre=1, post=2, dy=None, scalebar=False, **kwargs):
    sweep.select_epoch("stim")
    spikes = detect_putative_spikes(sweep.v, sweep.t)
    nblock = 3
    spikes = (spikes//nblock).astype(int)
    t = subsample_average(sweep.t, nblock)*1e3
    v = subsample_average(sweep.v, nblock)
    fs = 1/(t[1]-t[0])
    window = np.arange(-pre*fs, post*fs, dtype=int)

    delta = 0
    for i in spikes[nspikes]:
        plt.plot(t[i+window] - t[i] + delta, v[i+window], **kwargs)
        delta += offset
    plt.axis('off')
    ax = plt.gca()
    if dy:
        y0, y1 = ax.get_ylim()
        plt.ylim(y1-dy, y1)
    if scalebar:
        add_scalebar(ax, matchx=False, matchy=False, sizex=0.5, labelx='0.5 s',
                    sizey=10, labely='10 mV', textprops=dict(size='large', weight='bold'))

def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)