from ppd_examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


from pystatplottools.ppd_distributions.distributionDD import DistributionDD


# Computation of a 1-dimensional matrix for each dataframe where each entry contains the relative number of events
# within the defined bins by axes_indices. The data is sorted by the columns. Each dataframe has its own column
def compute_probability_distribution1D_sort_by_column(z_index_name="probability"):
    dist1d = DistributionDD(data=data)

    binned_statistics = dist1d.compute_1Dhistograms(
        columns=["a", "b", "c"],
        range_min=[-5.0, -4.0, -4.5],
        range_max=[5.0, 4.0, 4.5],
        nbins=[10, 12, 8],
        statistic='probability'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_histograms(histogram_data=binned_statistics, order_by_bin=True)
    return linearized_statistics


# Computation of a 1-dimensional matrix for each dataframe where each entry contains the relative number of events
# within the defined bins by axes_indices. The data is sorted according to the given dataframes.
# This method is restriced to a single column
def compute_probability_distribution1D(z_index_name="probability"):
    dist1d = DistributionDD(data=data)

    # Todo: Add exception - if axes_indices is not defined, an histogram is computed for each col - this can
    #  also be another function!! and therefore also useful for DD distributions - based on subspaces

    binned_statistics = dist1d.binned_statistics_over_axes(
        axes_indices=["a"],
        range_min=[-5.0],
        range_max=[5.0],
        nbins=[10],
        statistic='probability'
    )
    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a"],
                                                                       binned_statistics=binned_statistics,
                                                                       output_column_names=z_index_name)

    return linearized_statistics

# Computation of a 1-dimensional matrix for each dataframe where each entry contains given statistics over the provided
# columns computed within the defined bins by axes_indices. The data is sorted according to the given dataframes
def compute_statistics_of_column_distribution1D(z_index_name="probability"):
    dist1d = DistributionDD(data=data)

    binned_statistics = dist1d.binned_statistics_over_axes(
        axes_indices=["a"],
        columns=["b", "c", "idx"],
        range_min=[-5.0],
        range_max=[4.0],
        nbins=[10],
        statistic='mean'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a"],
                                                                       binned_statistics=binned_statistics)
    return linearized_statistics


# Computation of a D-dimensional matrix for each dataframe where each entry contains the relative number of events
# within the defined bins by axes_indices. The data is sorted according to the given dataframes.
def compute_probability_distributionDD(z_index_name="probability"):
    distdd = DistributionDD(data=data)

    binned_statistics = distdd.binned_statistics_over_axes(
        axes_indices=["a", "b", "c"],
        range_min=[-5.0, -4.0, -6.0],
        range_max=[4.0, 6.0, 5.0],
        nbins=[10, 12, 8],
        statistic='probability'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b", "c"],
                                                                       binned_statistics=binned_statistics,
                                                                       output_column_names=z_index_name)
    return linearized_statistics


# Computation of a D-dimensional matrix for each dataframe where each entry contains given statistics over the provided
# columns computed within the defined bins by axes_indices. The data is sorted according to the given dataframes.
def compute_statistics_of_column_distributionDD(z_index_name="probability"):
    distdd = DistributionDD(data=data)

    binned_statistics = distdd.binned_statistics_over_axes(
        axes_indices=["a", "b", "c"],  # Bins
        columns=["idx"],  # Columns to compute the statictis
        range_min=[-5.0, -4.0, -6.0], # Ranges of the bins
        range_max=[4.0, 6.0, 5.0],
        nbins=[10, 12, 8],
        statistic='mean' # Computes statistics
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b", "c"],
                                                                       binned_statistics=binned_statistics)
    return linearized_statistics


if __name__ == '__main__':
    compute_probability_distribution1D_sort_by_column()

    compute_probability_distribution1D()
    compute_statistics_of_column_distribution1D()

    compute_probability_distributionDD()
    compute_statistics_of_column_distributionDD()