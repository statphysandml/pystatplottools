# Based on from the Code from http://bkanuka.com/posts/native-latex-plots/ (Practical Machine Learning)


import os
import numpy as np
import matplotlib as mpl
mpl.use('pgf')


def dev_mode():
    return False


def figsize(scale, ratio=None):
    fig_width_pt = 267.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    if ratio is None:
        fig_height = fig_width * golden_mean  # height in inches
    else:
        fig_height = fig_width*ratio
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 7,  # LaTeX default is 10pt font.
    "font.size": 7,
    "legend.fontsize": 5, # changed from 8 # Make the legend/label fonts a little smaller
    "xtick.labelsize": 7,# changed from 8
    "ytick.labelsize": 7, # changed from 8
    "figure.figsize": figsize(1),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{mathrsfs}",
        r"\usepackage{esint} %for additional integral symbols",
        r"\usepackage{bbm} % Number fields symbols"
    ]
}

pgf_with_latex_scaled = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 7,  # LaTeX default is 10pt font.
    "font.size": 7,
    "legend.fontsize": 5, # changed from 8 # Make the legend/label fonts a little smaller
    "xtick.labelsize": 7,# changed from 8
    "ytick.labelsize": 7, # changed from 8
    "figure.figsize": figsize(1),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{mathrsfs}",
        r"\usepackage{esint} %for additional integral symbols",
        r"\usepackage{bbm} % Number fields symbols"
    ]
}

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


# I make my own newfig and savefig functions
def newfig(width, **kwargs):
    kwargs.pop('figsize', None)  # removes figsize from kwargs if it exists in kwargs
    ratio = kwargs.pop('ratio', None)
    if ratio is None:
        mpl.rcParams.update(pgf_with_latex)
    else:
        mpl.rcParams.update(pgf_with_latex_scaled)
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize(width, ratio), **kwargs)
    return fig, ax


def surfacenewfig(width, **kwargs):
    kwargs.pop('figsize', None)  # removes figsize from kwargs if it exists in kwargs
    ratio = kwargs.pop('ratio', None)
    if ratio is None:
        mpl.rcParams.update(pgf_with_latex)
    else:
        mpl.rcParams.update(pgf_with_latex_scaled)
    plt.clf()
    fig = plt.figure(figsize=figsize(width, ratio), **kwargs)
    ax = fig.gca(projection='3d')
    return fig, ax


def newfiggrid(width, **kwargs):
    ratio = kwargs.pop('ratio', None)
    if ratio is None:
        mpl.rcParams.update(pgf_with_latex)
    else:
        mpl.rcParams.update(pgf_with_latex_scaled)
    plt.clf()
    fig = plt.figure(figsize=figsize(width, ratio))
    return fig


def savefig(savedir_figures, path_to_out_file=None, type="pdf"):
    extent = "tight"
    print(path_to_out_file)
    if path_to_out_file and type == "pdf":
        path_to_out_dir = os.path.dirname(os.path.join(savedir_figures, path_to_out_file))
        if not os.path.exists(path_to_out_dir):
            os.makedirs(path_to_out_dir)
        filename = os.path.join(savedir_figures, path_to_out_file)
        plt.savefig('{}.pdf'.format(filename), bbox_inches=extent,  pad_inches=0.03)
        plt.close()
    elif path_to_out_file and type == "png":
        path_to_out_dir = os.path.dirname(os.path.join(savedir_figures, path_to_out_file))
        if not os.path.exists(path_to_out_dir):
            os.makedirs(path_to_out_dir)
        filename = os.path.join(savedir_figures, path_to_out_file)
        plt.savefig('{}.png'.format(filename), bbox_inches=extent, pad_inches=0.03, dpi=300)
        plt.close()
    else:
        assert 0, 'You have to enter a path, if you want to watch the figure change to figure development'
