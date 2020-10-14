from pystatplottools.ppd_plotting_env.diagram_base_class import DiagramBaseClass
import numpy as np


class Histogram(DiagramBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_index = kwargs.pop("x_index", None)

    def plot(self, ax):
        cols = self.data.columns.values
        data_col = np.argwhere(cols != "bin")[0][0]
        self.histogram(ax, self.data[cols[data_col]], self.data["bin"])

    @staticmethod
    def histogram(ax, hist, rel_bins, color="darkblue", label=None):
        width = rel_bins[1] - rel_bins[0]
        hist = ax.bar(rel_bins, hist, width=width * 0.9, color=color, label=label)
        return hist
