from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class DiagramBaseClass:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.data = kwargs.pop("data")
        self._ax = kwargs.pop("ax")

    @property
    def ax(self):
        return self._ax

    # Helper Functions

    @staticmethod
    def get_min_max_of_index(data, index):
        return data[index].min(), data[index].max()

    @staticmethod
    def get_log_norm_and_levs(lev_min, lev_max, lev_num):
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

    def set_ax_labels(self, x_label=None, y_label=None):
        if x_label is not None:
            self._ax.set_xlabel(x_label)
        if y_label is not None:
            self._ax.set_ylabel(y_label)

    def set_ax_label_visibility(self, x_ticks_visible=True, y_ticks_visible=True):
        import matplotlib.pyplot as plt
        if x_ticks_visible is False:
            plt.setp(self._ax.get_xticklabels(), visible=False)
        if y_ticks_visible is False:
            plt.setp(self._ax.get_yticklabels(), visible=False)

    def add_fancy_box(self, legend_name):
        from pystatplottools.visualization.utils import add_fancy_legend_box
        return add_fancy_legend_box(ax=self._ax, name=legend_name)

    def set_ticks(self, x_ticks, y_ticks):
        assert False, "To be implemented"