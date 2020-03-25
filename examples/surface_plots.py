from pdf_plotting_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


def probability_surface_plot():
    z_index = "probability"

    from examples.compute_distribution2D import compute_probability_distribution2D
    linearized_statistics = compute_probability_distribution2D(z_index_name=z_index)

    dataframe_indices = linearized_statistics.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    from distribution_plotting_env.contour2D import Contour2D
    contour2D = Contour2D(
        data=linearized_statistics.loc["df1"],
        compute_x_func=lambda x: x["a"],
        compute_y_func=lambda x: x["b"],
        z_index=z_index
    )

    fig, ax = fma.surfacenewfig(1.4)
    contour2D.set_ax_labels(ax, x_label="xLabel", y_label="yLabel")
    cf = contour2D.surface(
        ax=ax,
        cbar_scale="Lin",
        lev_num=40,
    )
    contour2D.add_colorbar(fig=fig, cf=cf, z_label="Probability")
    plt.tight_layout()
    fma.savefig("./", "probability_surface")


if __name__ == '__main__':
    probability_surface_plot()
