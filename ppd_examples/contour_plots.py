from pystatplottools.ppd_pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


def probability_contour_plot():
    z_index = "probability"

    from ppd_examples.compute_distribution2D import compute_probability_distribution2D
    linearized_statistics = compute_probability_distribution2D(z_index_name=z_index)

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from pystatplottools.ppd_plotting_env.contour2D import Contour2D
    contour2D = Contour2D(
        data=linearized_statistics.loc["df2"],
        compute_x_func=lambda x: x["a"],  # possibility to rescale x and y axis or perform other operation for x axis
        # like computing a mass difference
        compute_y_func=lambda x: x["b"],
        z_index=z_index
    )

    fig, ax = fma.newfig(1.4)
    contour2D.set_ax_labels(ax, x_label="xLabel", y_label="yLabel")
    cf = contour2D.contourf(
        ax=ax,
        cbar_scale="Lin",
        lev_num=40,
    )
    contour2D.add_colorbar(fig=fig, cf=cf, z_label="Probability"
                           # cax=cbar_ax,
                           # z_ticks=[-1.4, -1, -0.5, 0, 0.5, 1, 1.4],
                           # z_tick_labels=['$-1.4$', '$-1$', '$-0.5$', '$0$', '$0.5$',
                           #               '$1$', '$1.4$']
                           )
    plt.tight_layout()
    fma.savefig("./", "probability_contour")


def plot_statistics_of_columns():
    from ppd_examples.compute_distribution2D import compute_statistics_of_column_distribution2D
    linearized_statistics = compute_statistics_of_column_distribution2D()

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from pystatplottools.ppd_plotting_env.contour2D import Contour2D
    contour2D = Contour2D(
        data=linearized_statistics.loc["df2"],
        compute_x_func=lambda x: x["a"],  # possibility to rescale x and y axis or perform other operation for x axis
        # like computing a mass difference
        compute_y_func=lambda x: x["b"],
        z_index="idx"
    )

    fig, ax = fma.newfig(1.4)

    contour2D.set_ax_labels(ax, x_label="xLabel", y_label="yLabel")
    cf = contour2D.contourf(
        ax=ax,
        cbar_scale="Exp",
        lev_num=40,
    )
    contour2D.add_colorbar(fig=fig, cf=cf, z_label="Idx")

    plt.tight_layout()
    fma.savefig("./", "mean_contour")


if __name__ == '__main__':
    probability_contour_plot()
    plot_statistics_of_columns()