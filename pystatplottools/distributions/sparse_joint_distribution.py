import numpy as np
import pandas as pd


from pystatplottools.distributions.distributionDD import DistributionDD


class SparseJointDistribution(DistributionDD):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
            self,
            axes_indices=None,
            statistic='count',
            transform='lin',
            nbins=[],  # Refers to the number of bins for each column
            range_min=None,  # Refers to the ranges of the bins for each column
            range_max=None,  # Refers to the ranges of the bins for each column
            bin_scales='linear',
            one_dim_bin_index=False
    ):
        """
        Computes a flattened histogram over the data of self.data[axes_indices] based on the other parameters. Enables
        the computation of high-dimensional histograms since a grid of bins is not explicitly computed in the
        higher-dimensional space. This is the essential difference to np.histrogramdd and joint_distribution(). Each
        data point is assigned to a bin in a histogram that has the same dimension as self.data by the computation of a
        higher dimensional index. Afterwards a single index of each data point is computed with np.ravel_multi_index
        based on this higher-dimensional index. The functions returns a histogram that is computed based on the single
        index of each data point.
        :param axes_indices:
        :param statistic:
        :param transform:
        :param nbins:
        :param range_min:
        :param range_max:
        :param bin_scales:
        :param one_dim_bin_index: True or False - For true, the linearization is much faster. However, the number of
                bins is counted with one scalar. The maximum bin index increases according to n_bins_per_dim^dim.
        :return:
        """

        if axes_indices is None:
            columns = self.data.columns
        else:
            columns = axes_indices

        columns, row_values, histogram_prep = self._prepare(
            axes_indices=columns, columns=columns, transform=transform, nbins=nbins,
            range_min=range_min, range_max=range_max, bin_scales=bin_scales
        )

        binned_statistics = dict()
        for row in row_values:
            data_in_range_mask = DistributionDD.compute_data_mask_based_on_ranges(
                data=self.data[columns].loc[row].values,
                ranges_min=[histogram_prep[ax_idx][row]['range_min'] for ax_idx in columns],
                ranges_max=[histogram_prep[ax_idx][row]['range_max'] for ax_idx in columns]
            )

            data = self.data[columns].loc[row].values[data_in_range_mask]
            actual_bin_edges = np.array([histogram_prep[ax_idx][row]["binedges"] for ax_idx in columns])
            actual_bin_scales = np.array([histogram_prep[ax_idx][row]["binscale"] for ax_idx in columns])

            if np.all([np.array_equal(bin, actual_bin_edges[idx + 1]) for idx, bin in enumerate(actual_bin_edges[:-1])]):
                # All dimensions have the same range_min, range_max
                dat = np.zeros(np.shape(data), dtype=np.int16)
                for bin_edge in actual_bin_edges[0][1:-1]:
                    dat[np.nonzero(data >= bin_edge)] += 1
                data = dat
            else:
                for dim in range(data.shape[1]):
                    dat = np.zeros(data.shape[0], dtype=np.int16)
                    for bin_edge in actual_bin_edges[dim][1:-1]:
                        dat[np.nonzero(data[:, dim] >= bin_edge)] += 1
                    data[:, dim] = dat

            data = np.array(data, dtype=np.int16)

            if one_dim_bin_index is True:
                indices = np.ravel_multi_index(data.transpose(), dims=nbins)
                hist, binedges = np.histogram(indices,
                                              bins=np.concatenate([np.unique(indices), np.array([np.max(indices) + 1])]),
                                              density=False)

                if statistic == 'probability':
                    hist = hist / len(data)

                binarea = np.prod([bin[-1] - bin[0] for bin in actual_bin_edges])

                binned_statistics[row] = {statistic: {'hist': hist, 'binarea': binarea, 'binedges': [binedges],
                                                      'actual_bin_edges': actual_bin_edges, 'binnames': columns,
                                                      'binscale': actual_bin_scales}}
            else:
                unique_non_zero_bins, non_zero_counts = np.unique(data, axis=0, return_counts=True)
                # Remainder/Explanation
                # hist == non_zero_counts
                # binedges[:-1] == np.ravel_multi_index(unique_non_zero_bins.transpose(), dims=nbins)
                if statistic == 'probability':
                    non_zero_counts = non_zero_counts / len(data)
                binarea = np.prod([bin[-1] - bin[0] for bin in actual_bin_edges])
                binedges = [np.concatenate([unique_non_zero_bins, np.array([np.max(unique_non_zero_bins, axis=0) + 1])])]
                binned_statistics[row] = {statistic: {'hist': non_zero_counts, 'binarea': binarea, 'binedges': binedges,
                                                      'actual_bin_edges': actual_bin_edges, 'binnames': columns,
                                                      'binscale': actual_bin_scales}}

        self._distribution = binned_statistics

    def linearize(self, order_by_bin=False):
        """
        Generates a linearized dataframe from histogram_data
        :param order_by_bin: True/False
        :return: If order_by_bin = True: A multiindex dataframe with index (col, bin_idx) and columns (binedges, df...).
                                         The binedges refer to col. The binedges refer to ravelled indices of
                                         a higher-dimensional histogram if histogram_data has been generated by histrogramDD.
                 if order_by_bin = False: A multiindex dataframe with index (df, col, bin_idx) and columns (binedges, data),
                                         where data contains the counts for probabilities for each col
                 In addition, a dictionary is returned that contains for each dataframe information about the bin edges
                 and the bin names in each dimension
        """

        key = list(self._distribution.keys())[0]
        hist_key = list(self._distribution[key].keys())[0]
        bin_information = {"bins": self._distribution[key][hist_key]["actual_bin_edges"],
                           "binnames": self._distribution[key][hist_key]["binnames"],
                           "binscale": self._distribution[key][hist_key]["binscale"]}

        for hist in self._distribution.values():
            for bins_a, bins_b in zip(hist[hist_key]["actual_bin_edges"], bin_information["bins"]):
                assert np.array_equal(bins_a, bins_b),\
                    "Bins of the different dataframes are not equal. Linearization to histogram not possible."

        return DistributionDD._linearize_histograms(
            histogram_data=self._distribution, order_by_bin=order_by_bin, bin_alignment="left"), bin_information
    
    @staticmethod
    def linearized_sparse_distribution_to_linearized_joint_distribution(
            linearized_sparse_distribution, bin_information, bin_alignment="center"):
        unravelled_bins, bin_names = DistributionDD.compute_multi_index_bin(
            linearized_sparse_distribution=linearized_sparse_distribution, bin_information=bin_information, bin_alignment=bin_alignment)
        bin_dataframe = pd.concat([pd.DataFrame(data=unravelled_bins, columns=bin_names)], keys=[linearized_sparse_distribution.index[0][0]])
        if "bin" in linearized_sparse_distribution.keys():
            linearized_joint_distribution = pd.concat([bin_dataframe, linearized_sparse_distribution.drop("bin", axis=1)], axis=1)
        else:
            linearized_joint_distribution = pd.concat(
                [bin_dataframe,
                 linearized_sparse_distribution.drop(["bin_" + str(bin) for bin in bin_information["binnames"]], axis=1)],
                axis=1)
        linearized_joint_distribution.index.names = linearized_sparse_distribution.index.names
        linearized_joint_distribution.columns.names = ["axes_and_dfs"]
        # linearized_joint_distribution = linearized_joint_distribution.sort_values(by=bin_names)
        # linearized_joint_distribution = linearized_joint_distribution.groupby(level=0).apply(
        #     lambda x: x.reset_index(drop=True))
        return linearized_joint_distribution
