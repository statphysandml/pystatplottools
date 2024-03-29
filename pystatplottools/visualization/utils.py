

def figure_decorator(func):
    def decorated(*args, **kwargs):
        filename = kwargs.get("filename")
        directory = kwargs.pop("directory", None)
        fig = kwargs.get("fig", None)
        ax = kwargs.get("ax", None)

        fma = kwargs.pop("fma", None)

        figsize = kwargs.get("figsize", (10, 7))
        width = kwargs.get("width", 1.0)
        type = kwargs.get("type", "png")
        ratio = kwargs.get("ratio", None)
        dim = kwargs.get("dim", None)

        import matplotlib.pyplot as plt
        ownfig = False
        if fig is None and fma is None:
            # Generate figure
            ownfig = True
            if dim is not None:
                fig, ax = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)
        elif fig is None and fma is not None:
            # Generate fma like figure
            # width is for the latex pdf file and figsize for plotting directly in jupyter notebook!
            if dim is not None:
                fig, ax = fma.newfig(nrows=dim[0], ncols=dim[1], width=width, figsize=figsize, ratio=ratio)
            else:
                fig, ax = fma.newfig(width=width, figsize=figsize, ratio=ratio)

        kwargs["fig"] = fig
        kwargs["ax"] = ax

        if directory is None:
            directory = func(*args, **kwargs)
        else:
            func(*args, **kwargs)

        if ownfig:
            plt.tight_layout()
            plt.show()
            plt.close()
        elif fma is not None:
            # print(directory, filename, type)
            fma.savefig(directory, filename, type=type)

        return fig, ax
    return decorated


def add_fancy_legend_box(ax, name):
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], linestyle="--", color="white")]
    legend = ax.legend(custom_lines, [name], loc="upper right", framealpha=1, fancybox=False,
                       handlelength=0.0, handletextpad=0)
    return legend
