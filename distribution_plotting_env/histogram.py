from distribution_plotting_env.diagram_base_class import DiagramBaseClass


class Histogram(DiagramBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_index = kwargs.pop("x_index")

    def plot(self, ax):
        self.histogram(ax, )

    @staticmethod
    def histogram(ax, hist, rel_bins, color="darkblue", label=None):
        width = rel_bins[1] - rel_bins[0]
        hist = ax.bar(rel_bins[:-1], hist, width=width * 0.9, color=color, label=label)
        return hist
