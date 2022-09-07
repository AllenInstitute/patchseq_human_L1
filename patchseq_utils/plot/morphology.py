from math import cos
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from . import make_upright as mu 
from .. import lims
import allensdk.core.swc as swc
from os.path import splitext

def plot_cell_lims(specimen_id, rotate=True, scale_factor=None, scalebar=True, color=None):
    """Plot morphology from LIMS by specimen_id
    
    Parameters
    ----------
    specimen_id : int
    rotate : bool or float, optional
        if True, rotate morphology according to transform stored in LIMS
        if numerical value is provided, rotate morphology by that angle (degrees)
        if False, don't rotate
    color : matplotlib colorspec, optional
        base color to use for axon and apicals, with darker shade for dendrites
        default uses a standard color scheme
    scale_factor : int, optional
        microns/inch, or autoscales by default
    scalebar : bool, optional
        add 100 um scale bar, by default True
    """
    if rotate==True:
        nrn = mu.make_upright_morphology(specimen_id)
    else:
        path =  lims.get_swc_path(specimen_id, manual_only=True)
        nrn = swc.read_swc(path)
        if rotate:
            th = np.radians(rotate)
            aff = [np.cos(th), -np.sin(th), 0,
                   np.sin(th), np.cos(th), 0,
                   0, 0, 1,
                   0, 0, 0]
            nrn.apply_affine(aff)
    plot_morph(nrn, scale_factor=scale_factor, scalebar=scalebar, color=color)

def swc_to_svg(swc_path, out_path=None, scale_factor=None, scalebar=True, color=None, transparent=True):
    """Plot morphology to SVG from SWC file
    
    Parameters
    ----------
    swc_path : str, path to swc file
    out_path : str, optional
        path to save svg to, defaults to renamed swc path
    color : matplotlib colorspec, optional
        base color to use for axon and apicals, with darker shade for dendrites
        default uses a standard color scheme
    scale_factor : int, optional
        microns/inch, or autoscales by default
    scalebar : bool, optional
        add 100 um scale bar, by default True
    transparent : bool, default True
        whether to save svg with transparent background
    """
    out_path = out_path or splitext(swc_path)[0] + ".svg"
    nrn = swc.read_swc(swc_path)
    plot_morph(nrn, scale_factor=scale_factor, scalebar=scalebar, color=color)
    plt.savefig(out_path, transparent=transparent)

def plot_morph(nrn, scale_factor=None, scalebar=True, color=None, show_axon=True):
    """Plot morphology from AllenSDK SWC object
    
    Parameters
    ----------
    nrn : AllenSDK SWC instance
    scale_factor : int, optional
        microns/inch, or autoscales by default
    scalebar : bool, optional
        add 100 um scale bar, by default True
    """
    fig, ax = plt.subplots()
    if color:
        # axon, dend, apical
        base = matplotlib.colors.to_rgb(color)
        dark = 0.5*np.array(base)
        MORPH_COLORS = {2: base, 3: dark, 4: base}
    else:
        MORPH_COLORS = {2: "steelblue", 3: "firebrick", 4: "salmon", }
    if not show_axon:
        MORPH_COLORS.pop(2)
    for compartment, color in MORPH_COLORS.items():
        lines_x = []
        lines_y = []
        for c in nrn.compartment_list_by_type(compartment):
            if c["parent"] == -1:
                continue
            p = nrn.compartment_index[c["parent"]]
            lines_x += [p["x"], c["x"], None]
            lines_y += [p["y"], c["y"], None]
        plt.plot(lines_x, lines_y, c=color, linewidth=1)
    for c in nrn.compartment_list_by_type(1):
        plt.plot(c["x"], c["y"], 'ok', markersize=5)
    ax.set_aspect("equal")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = x1-x0
    h = y1-y0
    if scalebar:
        bar_length = 100
        plt.plot([x1-bar_length, x1], [y0, y0], 'k', linewidth=4)
        plt.text(x1-bar_length, y0+h/30, '100 $\mu$m')
    if scale_factor:
        set_size(w/scale_factor, h/scale_factor, ax=ax)
    ax.axis('off')
    # ax.invert_yaxis()
    return fig

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