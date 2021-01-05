from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
fma, plt = loading_figure_mode(develop=True)


def joint_distribution_contour_plot():
    z_index = "probability"

    from examples.compute_distributionDD import compute_joint_distributionDD
    joint_distribution = compute_joint_distributionDD(z_index_name=z_index)

    dataframe_indices = joint_distribution.index.unique(0)
    print("Considered dataframes", dataframe_indices.values)

    fig, ax = fma.newfig(1.4)

    from pystatplottools.plotting.contour2D import Contour2D
    contour2D = Contour2D(
        ax=ax,
        data=joint_distribution.loc["df2"],
        compute_x_func=lambda data: data["a"] - 0.1,
        y="b",
        z_index=z_index
    )

    contour2D.set_ax_labels(x_label="xLabel", y_label="yLabel")

    cf = contour2D.contourf(
        cbar_scale="Lin",
        lev_num=40,
    )

    contour2D.add_colorbar(fig=fig, cf=cf, z_label="Probability")

    plt.tight_layout()
    fma.savefig("./", "probability_contour")


if __name__ == '__main__':
    joint_distribution_contour_plot()
