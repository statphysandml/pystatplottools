from pystatplottools.distributions.distributionDD import DistributionDD


from examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


data_in_range_mask = DistributionDD.compute_data_mask_based_on_ranges(
    data=data[["a", "b", "c"]].values,
    ranges_min=[-5.0, -4.0, -4.5],
    ranges_max=[5.0, 4.0, 4.5]
)
data = data[data_in_range_mask]


def compute_joint_distributionDD(z_index_name="probability"):
    from pystatplottools.distributions.joint_distribution import JointDistribution
    joint_distribution = JointDistribution(data=data)

    joint_distribution.compute(
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
    #                'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'binedgesindex': ['a', 'b']}},
    #  'df2': {'a': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'binedgesindex': ['a', 'b']}}}

    # Transforms binned_statistics into a linear list of centers for the different bins
    # and the respective statistics for the values
    linearized_joint_distribution = joint_distribution.linearize(output_statistics_name=z_index_name,
                                                                 dataframes_as_columns=False, bin_alignment="center")
    #       a       b       probability
    # df1   -4.95   -3.96   0.0
    #       ...     ...     ...
    #       4.95    3.95    0.0
    # df2   -4.95   -3.96   0.0
    #       ...     ...     ...
    #       -4.95   -3.96   0.0

    linearized_joint_distribution_dataframes_as_columns_true = joint_distribution.linearize(
        output_statistics_name=z_index_name, dataframes_as_columns=True, bin_alignment="center")

    #               a       b       df1     df2
    # probability   -4.95   -3.96   0.0     0.0
    #               ...     ...     ...     ...
    #               4.95    3.95    0.0     0.0

    # joint_distribution["df1"]["probability"].keys()
    # dict_keys(['hist', 'binedges', 'binedgesindex'])

    return linearized_joint_distribution


def compute_binned_statisticsDD():
    from pystatplottools.distributions.binned_statistics import BinnedStatistics
    binned_statistics = BinnedStatistics(data=data)

    # Generate for each given column a two d distribution based on x_index and y_index as columns
    # This is done separately for each initial dataset
    binned_statistics.compute(
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
    #                'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'binedgesindex': ['a', 'b']},
    #          'idx': {'hist': array([[0., 0., 0., ..., 0., 0., 0.],
    #                                 [0., 0., 0., ..., 0., 0., 0.]]),
    #                  'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                               array([-4., -3.92, ..., 3.92, 4.])],
    #                  'binedgesindex': ['a', 'b']},
    #  'df2': {'c': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                             array([-4., -3.92, ..., 3.92, 4.])],
    #                'binedgesindex': ['a', 'b']},
    #          'idx': {'hist': array([[0., ..., 0.],
    #                               ...,
    #                               [0., ..., 0.]]),
    #                  'binedges': [array([-5., -4.9, ..., 4.9, 5.]),
    #                               array([-4., -3.92, ..., 3.92, 4.])],
    #                  'binedgesindex': ['a', 'b']}}}

    # Transforms binned_statistics into a linear list of centers for the different bins
    # and the respective statistics for the values
    linearized_statistics = binned_statistics.linearize()

    return linearized_statistics


def compute_marginal_distributionDD():
    from pystatplottools.distributions.marginal_distribution import MarginalDistribution
    marginal_distribution = MarginalDistribution(data=data)

    marginal_distribution.compute(
        axes_indices=["a", "b", "c"],
        range_min=[-5.0, -4.0, -4.5],
        range_max=[5.0, 4.0, 4.5],
        nbins=[10, 12, 8],
        statistic='probability'
    )

    # Transforms binned_statistics into a linear list of left boundaries for the different bins
    # and the respective statistics for the values
    linearized_marginal_distribution = marginal_distribution.linearize(order_by_bin=True)

    linearized_marginal_distribution_not_order_by_bin = marginal_distribution.linearize(order_by_bin=False)

    return linearized_marginal_distribution


def compute_sparse_joint_distribution():
    from pystatplottools.distributions.sparse_joint_distribution import SparseJointDistribution
    sparse_joint_distribution = SparseJointDistribution(data=data)

    sparse_joint_distribution.compute(
        axes_indices=["a", "b", "c"],
        statistic="probability",
        transform='lin',
        nbins=[100, 100, 100],
        range_min=[-5.0, -4.0, -4.5],
        range_max=[5.0, 4.0, 4.5],
        bin_scales='linear',
        # For one_dim_bin_index=False, bins are stored as tuples with an entry to each dimension
        # -> longer computation time
        # For one_dim_bin_index=True, bins are linearized - nbins^dim must be a valid number
        # -> faster computation time
        one_dim_bin_index=False
    )

    linearized_sparse_distribution, bin_information = sparse_joint_distribution.linearize(order_by_bin=True)

    linearized_sparse_distribution_not_ordered_by_bin, bin_information = sparse_joint_distribution.linearize(order_by_bin=False)

    # Possibility to extract the actual binedges for each row
    actual_binedges, bin_names = DistributionDD.compute_multi_index_bin(
        linearized_sparse_distribution, bin_information=bin_information)

    # Bins with zero probability are not contained in the linearlized_marginal_distribution dataframe. This is different
    # to the linearized marginal distribution that can be obtained by marginal_distribution() and
    # linearize_marginal_distribution()

    linearized_joint_distribution = sparse_joint_distribution.linearized_sparse_distribution_to_linearized_joint_distribution(
        linearized_sparse_distribution=linearized_sparse_distribution, bin_information=bin_information, bin_alignment="center"
    )

    return linearized_joint_distribution


if __name__ == '__main__':
    linearized_joint_distribution = compute_joint_distributionDD()
    linearized_statistics = compute_binned_statisticsDD()
    linearized_marginal_distribution = compute_marginal_distributionDD()
    linearized_sparse_joint_distribution = compute_sparse_joint_distribution()

    linearized_joint_distribution = DistributionDD.transpose_linearized_statistics(
        axes_indices=["a", "b"], data=linearized_joint_distribution)

    linearized_joint_distribution_ab = DistributionDD.marginalize(
        initial_axes_indices=["a", "b", "c"],
        remaining_axes_indices=["a", "b"],
        data=linearized_sparse_joint_distribution
    )

    linearized_joint_distribution = DistributionDD.drop_zero_statistics(axes_indices=["a", "b"], data=linearized_joint_distribution)

    print((linearized_joint_distribution - linearized_joint_distribution_ab).sum())

    pass
