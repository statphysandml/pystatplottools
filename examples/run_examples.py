from examples.compute_distribution2D import compute_probability_distribution2D, compute_statistics_of_column_distribution2D
from examples.compute_distributionDD import compute_statistics_of_column_distribution1D, \
    compute_statistics_of_column_distributionDD, \
    compute_probability_distribution1D, \
    compute_probability_distribution1D_alternative, \
    compute_probability_distributionDD
from examples.histogram_plot import probability_historam_plot
from examples.contour_plots import probability_contour_plot, plot_statistics_of_columns
from examples.surface_plots import probability_surface_plot

from examples.compute_expectation_values import compute_expectation_values_distribution1D


if __name__ == '__main__':
    compute_probability_distribution2D()
    compute_statistics_of_column_distribution2D()

    compute_probability_distribution1D()
    compute_statistics_of_column_distribution1D()
    compute_probability_distribution1D_alternative()
    compute_probability_distributionDD()
    compute_statistics_of_column_distributionDD()

    compute_expectation_values_distribution1D()

    probability_historam_plot()

    probability_contour_plot()
    plot_statistics_of_columns()

    probability_surface_plot()
