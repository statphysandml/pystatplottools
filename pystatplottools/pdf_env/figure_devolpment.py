# Based on from the Code from http://bkanuka.com/posts/native-latex-plots/ (Practical Machine Learning)


import matplotlib.pyplot as plt
# Functions equivalent to figure_management


def newfig(width, **kwargs):
    ratio = kwargs.pop('ratio', None)
    fig, ax = plt.subplots(**kwargs)
    return fig, ax


def surfacenewfig(width, **kwargs):
    ratio = kwargs.pop('ratio', None)
    fig  = plt.figure(**kwargs)
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.gca(projection='3d')
    return fig, ax


def savefig(savedir_figures, path_to_out_file=None, type="pdf", extent=0):
    plt.show()
    plt.close()
