from pdf_plotting_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


def probability_historam_plot():
    z_index = "probability"

    from examples.compute_distributionDD import compute_probability_distribution1D
    linearized_statistics = compute_probability_distribution1D(z_index_name=z_index)

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from distribution_plotting_env.histogram import Histogram

    histogram = Histogram(data=linearized_statistics.loc["a"][["bin", "df1"]])

    fig, ax = fma.newfig(1.4)
    histogram.set_ax_labels(ax, x_label="xLabel", y_label="Probability")
    histogram.plot(ax)
    # hist = histogram.histogram(ax, hist, rel_bins, color="darkblue", label=None)

    plt.tight_layout()
    fma.savefig("./", "probability_histogram")


if __name__ == '__main__':
    probability_historam_plot()