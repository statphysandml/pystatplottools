import numpy as np
# import pdf_plotting_env.figure_management as fma
import pdf_plotting_env.figure_devolpment as fma
import matplotlib.pyplot as plt

def simple_plot():
    fig, ax = fma.newfig(0.5)
    plt.plot(np.arange(0, 100, 1), np.arange(0, 100, 1),'-',lw=1,label='m=0')
    fma.savefig("./", 'sample_plot_simple')
    plt.close()

def plot_with_grid():
    fig, ax = fma.newfig(0.5, nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].plot(np.arange(0, 100, 1), np.arange(0, 100, 1),'-',lw=1,label='m=0')
    ax[1].plot(np.arange(0, 100, 1), np.arange(0, 100, 1),'-',lw=1,label='m=0')
    fma.savefig("./", 'sample_plot_grid')
    plt.close()

simple_plot()
plot_with_grid()