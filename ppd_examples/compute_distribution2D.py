from ppd_examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


from pystatplottools.ppd_distributions.distributionDD import DistributionDD


def compute_probability_distribution2D(z_index_name="probability"):
    dist2d = DistributionDD(data=data)

    binned_statistics = dist2d.binned_statistics_over_axes(
        axes_indices=["a", "b"],
        range_min=[-5.0, -4.0],
        range_max=[5.0, 4.0],
        nbins=[100, 100],
        statistic='probability'
    )

    # 'hist' contains the probabilities for the different 2D bins

    # {'df1': {'a': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'rel_bins_index': ['a', 'b']}},
    #  'df2': {'a': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'rel_bins_index': ['a', 'b']}}}

    # Transforms binned_statistics into a linear list of mid boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b"],
                                                                       binned_statistics=binned_statistics,
                                                                       output_column_names=z_index_name)
    #       a       b       probability
    # df1   -4.95   -3.96   0.0
    #       ...     ...     ...
    #       4.95    3.95    0.0
    # df2   -4.95   -3.96   0.0
    #       ...     ...     ...
    #       -4.95   -3.96   0.0

    return linearized_statistics


def compute_statistics_of_column_distribution2D():
    dist2d = DistributionDD(data=data)

    # Generate for each given column a two d distribution based on x_index and y_index as columns
    # This is done separately for each initial dataset
    binned_statistics = dist2d.binned_statistics_over_axes(
        axes_indices=["a", "b"],
        columns=["c", "idx"],
        range_min=[-5.0, -4.0],
        range_max=[5.0, 4.0],
        nbins=[100, 100],
        statistic='mean'
    )

    # 'hist' contains the computed statistics ('mean') of each bin over the given index in the front (in this cas for 'c' or 'idx')

    # {'df1': {'c': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'rel_bins_index': ['a', 'b']},
    #          'idx': {'hist': array([[0., 0., 0., ..., 0., 0., 0.],
    #                                 [0., 0., 0., ..., 0., 0., 0.]]),
    #                  'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                               array([-4., -3.92, ..., 3.92, 4.])],
    #                  'rel_bins_index': ['a', 'b']},
    #  'df2': {'c': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'rel_bins_index': ['a', 'b']},
    #          'idx': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                  'rel_bins': [array([-5., -4.9, ..., 4.9, 5.]),
    #                               array([-4., -3.92, ..., 3.92, 4.])],
    #                  'rel_bins_index': ['a', 'b']}}}

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_statistics = DistributionDD.linearize_binned_statistics(axes_indices=["a", "b"],
                                                                       binned_statistics=binned_statistics)

    return linearized_statistics


if __name__ == '__main__':
    compute_probability_distribution2D()
    compute_statistics_of_column_distribution2D()