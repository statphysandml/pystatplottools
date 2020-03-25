from examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


from distributions.distributionDD import DistributionDD


def compute_probability_distribution2D(z_index_name="probability"):
    dist2d = DistributionDD(data=data)

    # Generate for each given column a two d distribution based on x_index and y_index as columns
    # This is done separately for each initial dataset
    binned_statistics = dist2d.compute_binned_statistics(
        axes_indices=["a", "b"],
        range_min=[-5.0, -4.0],
        range_max=[5.0, 4.0],
        nbins=[100, 100],
        statistic='probability'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b"],
                                                                       binned_statistics=binned_statistics,
                                                                       output_column_names=z_index_name)
    return linearized_statistics


def compute_statistics_of_column_distribution2D():
    dist2d = DistributionDD(data=data)

    # Generate for each given column a two d distribution based on x_index and y_index as columns
    # This is done separately for each initial dataset
    binned_statistics = dist2d.compute_binned_statistics(
        axes_indices=["a", "b"],
        columns=["c", "idx"],
        range_min=[-5.0, -4.0],
        range_max=[5.0, 4.0],
        nbins=[100, 100],
        statistic='mean'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b"],
                                                                       binned_statistics=binned_statistics)

    return linearized_statistics


if __name__ == '__main__':
    compute_probability_distribution2D()
    compute_statistics_of_column_distribution2D()