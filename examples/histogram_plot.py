from pdf_plotting_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


def probability_historam_plot():
    z_index = "probability"

    from examples.compute_distributionDD import compute_probability_distribution1D
    linearized_statistics = compute_probability_distribution1D(z_index_name=z_index)

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from distribution_plotting_env.histogram import Histogram

    histogram = Histogram(data=linearized_statistics.loc["df1"])

    from distribution_plotting_env.contour2D import Contour2D
    contour2D = Contour2D(
        data=linearized_statistics.loc["df2"],
        compute_x_func=lambda x: x["a"],  # possibility to rescale x and y axis or perform other operation for x axis
        # like computing a mass difference
        compute_y_func=lambda x: x["b"],
        z_index=z_index
    )

    fig, ax = fma.newfig(1.4)
    histogram.set_ax_labels(ax, x_label="xLabel", y_label="Probability")
    # hist = histogram.histogram(ax, hist, rel_bins, color="darkblue", label=None)

    plt.tight_layout()
    fma.savefig("./", "probability_histogram")


if __name__ == '__main__':
    probability_historam_plot()