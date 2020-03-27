from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class DiagramBaseClass:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.data = kwargs.pop("data")

    # Helper Functions

    @staticmethod
    def get_min_max_of_index(data, index):
        return data[index].min(), data[index].max()

    @staticmethod
    def get_exp_norm_and_levs(lev_min, lev_max, lev_num):
        lev_exp = np.linspace(np.log10(lev_min), np.log10(lev_max), lev_num)  # 0.5
        levs = np.power(10, lev_exp)

        from matplotlib import colors
        return colors.LogNorm(), levs

    @staticmethod
    def get_lin_norm_and_levs(lev_min, lev_max, lev_num):
        levs = np.linspace(lev_min, lev_max, lev_num)  # 0.5

        from matplotlib import colors
        return colors.Normalize(vmin=lev_min, vmax=lev_max), levs

    # Design

    @staticmethod
    def set_ax_labels(ax, x_label, y_label):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    @staticmethod
    def set_ax_label_visibility(ax, x_ticks_visible, y_ticks_visible):
        import matplotlib.pyplot as plt
        if x_ticks_visible is False:
            plt.setp(ax.get_xticklabels(), visible=False)
        if y_ticks_visible is False:
            plt.setp(ax.get_yticklabels(), visible=False)

    @staticmethod
    def set_ticks(x_ticks, y_ticks):
        assert False, "To be implemented"