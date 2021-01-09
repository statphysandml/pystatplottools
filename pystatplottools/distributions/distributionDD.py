import numpy as np
import pandas as pd


def transform_log10(data, columns):
    # Compute log10 of data
    log_data = data[columns].apply(["log10"])
    log_data.columns = log_data.columns.droplevel(1) + "_log10"
    cols_to_use = data.columns.difference(log_data.columns)
    data = pd.concat([data[cols_to_use], log_data], axis=1, verify_integrity=True)
    return data, [col + '_log10' for col in columns]


class DistributionDD:

    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        self.name = kwargs.pop("name", "Unknown")
        self._distribution = None
        self._axes_indices = None

    @property
    def distribution(self):
        return self._distribution

    @staticmethod
    def transpose_linearized_statistics(axes_indices, data):
        # Verify if bins of axes_indices are the same
        if data.columns.names[0] == "axes_and_statistics":
            new_data_columns_names = ["axes_and_dfs"]
            new_data_index_names = ["statistics", "idx"]

        else:
            new_data_columns_names = ["axes_and_statistics"]
            new_data_index_names = ['dfs', 'idx']

        non_axes_cols = [item for item in data.columns if item not in axes_indices]
        upper_level_row_indices = list(data.index.unique(level=0))
        bins = data.loc[upper_level_row_indices[0]][axes_indices].values
        num_data_per_upper_level_row = len(bins)
        for key in upper_level_row_indices:
            upper_level_df_bins = data.loc[key][axes_indices].values
            assert np.array_equal(bins, upper_level_df_bins),\
                "Bins of the different upper level row dataframes do not coincide. Transposing is not possible."

        data = data[non_axes_cols].values.transpose().reshape(len(upper_level_row_indices), -1)
        bins = np.tile(bins.transpose(), (1, len(non_axes_cols)))
        index = pd.MultiIndex.from_product([non_axes_cols, np.arange(num_data_per_upper_level_row)], names=new_data_index_names)
        data = pd.DataFrame(np.concatenate([bins, data], axis=0).transpose(), index=index, columns=axes_indices + upper_level_row_indices)
        data.columns.names = new_data_columns_names
        return data

    @staticmethod
    def marginalize(initial_axes_indices, remaining_axes_indices, data):
        non_axes_cols = [item for item in data.columns if item not in initial_axes_indices]
        upper_row_level_name = data.index.names[0]
        data = data.reset_index(level=0).groupby(by=[upper_row_level_name] + remaining_axes_indices)[non_axes_cols].agg("sum").reset_index()
        data = data.set_index(upper_row_level_name)
        data = data.groupby(upper_row_level_name).apply(lambda x: x.reset_index(drop=True))
        data.index.names = [upper_row_level_name, "idx"]
        return data

    @staticmethod
    def drop_zero_statistics(axes_indices, data):
        non_axes_cols = [item for item in data.columns if item not in axes_indices]
        data_index_names = data.index.names
        data = data[np.any(data[non_axes_cols].values != 0, axis=1)].groupby(data_index_names[0]).apply(lambda x: x.reset_index(drop=True))
        data.index.names = data_index_names
        return data

    @staticmethod
    def compute_multi_index_bin(linearized_sparse_distribution, bin_information, bin_alignment="center"):
        bins = bin_information["bins"]
        shape = tuple([len(bin) - 1 for bin in bins])
        if "bin" in linearized_sparse_distribution.keys():
            multi_dimensionsal_indices = np.unravel_index(indices=linearized_sparse_distribution.bin.values, shape=shape)
        else:
            multi_dimensionsal_indices = linearized_sparse_distribution[["bin_" + str(bin) for bin in bin_information["binnames"]]].values.transpose()

        binedges = np.zeros((len(linearized_sparse_distribution), len(bins)))
        for dim, (bin, dim_index) in enumerate(zip(bins, multi_dimensionsal_indices)):
            if bin_alignment == "center":
                aligned_bins = ((bin[:-1] + bin[1:]) / 2.0)[dim_index]
            elif bin_alignment == "right:":
                aligned_bins = bin[1:][dim_index]
            else:
                aligned_bins = bin[:-1][dim_index]
            binedges[:, dim] = aligned_bins

        return binedges, bin_information["binnames"]

    def extract_min_max_range_values(self, columns=None):
        if columns is None:
            range_min, range_max = list(map(list, self.data.agg(["min", "max"]).values))
        else:
            range_min, range_max = list(map(list, self.data[columns].agg(["min", "max"]).values))
        return range_min, range_max

    # Private methods

    def _compute_binned_statistics(self, axes_indices=None,
                                    columns=None,
                                    statistic='count',
                                    transform='lin',
                                    nbins=[],
                                    range_min=None,
                                    range_max=None,
                                    bin_scales='linear',
                                    with_binnumber=False):
        """
        Computes a joint probability distribution/histogram of axes_indices over binned values of the data frame
        self.data or computes binned statistics over binned samples. The function makes of numpy's binned_statistic_dd
        function
        :param axes_indices: list of column indices - random variables of the joint distribution / the binned dimensions
        :param columns: the statistic is computed over these values
        :param statistic: statistic to be computed
        :param transform: lin or log10 - possibility to transform the column data before the computation of the statistics
        :param nbins: scalar of list - number of bins in each dimensions
        :param range_min: scalar or list - minimum of the ranges of the bins in the different dimension of axes_indices
        :param range_max: scalar or list - maximum of ranges of the bins in the different dimension of axes_indices
        :param bin_scales: "linear" or "logarithmic" - possibility to introduce another scale for the bins
        :param with_binnumber: assign binned_numbers are also returned for each value in self.data
        :return: dictionary that contains for each dataframe and each column a histogram. For statistic equal to
        "probability" or "count", the histograms can be accessed via [df][statistic] for columns via [df][column]
        """
        
        if axes_indices is None:
            # Compute 1D statistics (marginal distribution) separately for each column
            axes_indices = columns  # To generate the correct bins
            seperate_statistics = True
        else:
            seperate_statistics = False

        if columns is None:
            # Probability distribution or standard Histogram
            columns = [axes_indices[0]]

        columns, row_values, histogram_prep = self._prepare(
            axes_indices=axes_indices, columns=columns, transform=transform, nbins=nbins,
            range_min=range_min, range_max=range_max, bin_scales=bin_scales
        )

        from scipy.stats import binned_statistic_dd

        bin_statistic = statistic

        sample_indices = axes_indices  # Default assumes computation with axes indices
        binned_statistics = dict()
        for row in row_values:
            binned_statistics[row] = dict()
            for col in columns:
                if statistic == "probability":
                    bin_statistic = lambda x: len(x) * 1.0 / len(self.data.loc[row])
                if seperate_statistics:  # For 1D histrograms
                    sample_indices = [col]
                data_in_range_mask = DistributionDD.compute_data_mask_based_on_ranges(
                    data=self.data.loc[row][sample_indices].values,
                    ranges_min=[histogram_prep[ax_idx][row]['range_min'] for ax_idx in sample_indices],
                    ranges_max=[histogram_prep[ax_idx][row]['range_max'] for ax_idx in sample_indices]
                )
                assert np.sum(data_in_range_mask) != 0, "No data point is in the given range in at least one dimension"
                if np.sum(data_in_range_mask) < len(self.data.loc[row][sample_indices]):
                    print("Not all data points are in the given ranges")
                hist, binedges, binnumber = binned_statistic_dd(
                    sample=self.data.loc[row][sample_indices].values[data_in_range_mask], values=self.data.loc[row][col].values[data_in_range_mask], statistic=bin_statistic,
                    bins=[histogram_prep[ax_idx][row]['binedges'] for ax_idx in sample_indices])
                binscale = [histogram_prep[ax_idx][row]['binscale'] for ax_idx in sample_indices]

                out_key = col
                if (statistic == "probability" or statistic == "count") and seperate_statistics is False:
                    out_key = statistic

                if with_binnumber:
                    binned_statistics[row][out_key] = {'hist': hist, 'binedges': binedges, 'binedgesindex': sample_indices, 'binnumber': binnumber, 'binscale': binscale}
                else:
                    binned_statistics[row][out_key] = {'hist': hist, 'binedges': binedges, 'binedgesindex': sample_indices, 'binscale': binscale}

        return binned_statistics

    def _prepare(self, axes_indices=None, columns=None, transform='lin', nbins=[], range_min=None, range_max=None,
                bin_scales='linear'):

        if transform == "log10":
            # Compute log10 of data
            data, columns = transform_log10(data=self.data, columns=columns)

        # includes the indices of the different considered datasets
        row_values = list(self.data.index.unique(0))

        if range_min is None:
            ranges_min = self.data[axes_indices].groupby(level=0).agg(['min'])
        else:
            ranges_min = DistributionDD._tile_scalar(val=range_min, row_values=row_values, axes_indices=axes_indices, identifier='min')

        if range_max is None:
            ranges_max = self.data[axes_indices].groupby(level=0).agg(['max'])
        else:
            ranges_max = DistributionDD._tile_scalar(val=range_max, row_values=row_values, axes_indices=axes_indices, identifier='max')

        nb = DistributionDD._tile_scalar(val=nbins, row_values=row_values, axes_indices=axes_indices, identifier='nbins')
        binscales = DistributionDD._tile_scalar(val=bin_scales, row_values=row_values, axes_indices=axes_indices, identifier='binscale')

        histogram_prep = pd.concat([ranges_min, ranges_max, nb, binscales], axis=1, sort=True)
        histogram_prep = histogram_prep.groupby(level=0, axis=1).apply(
            lambda x: DistributionDD._get_bin_properties_of_collection(x))

        return columns, row_values, histogram_prep

    # The operation assumes that the order of the columns in data are the same as in histogram_prep
    @staticmethod
    def compute_data_mask_based_on_ranges(data, ranges_min, ranges_max):
        if len(data) == data.size:
            # Data is one dimensional
            data = data.reshape(len(data), 1)
        mask1 = data <= np.tile(np.array(ranges_max), (len(data), 1))
        mask2 = data >= np.tile(np.array(ranges_min), (len(data), 1))
        return np.all(mask1 & mask2, axis=1)

    @staticmethod
    def _linearize_histograms(histogram_data, order_by_bin=False, bin_alignment="center"):
        """
        Generates a linearized dataframe from histogram_data
        :param histogram_data: ToDo
        :param order_by_bin: True/False
        :return: If order_by_bin = True: A multiindex dataframe with index (col, bin_idx) and columns (binedges, df...).
                                         The binedges refer to col. The binedges refer to ravelled indices of
                                         a higher-dimensional histogram if histogram_data has been generated by histrogramDD.
                 if order_by_bin = False: A multiindex dataframe with index (df, col, bin_idx) and columns (binedges, data),
                                         where data contains the counts for probabilities for each col
        """
        keys = list(histogram_data.keys())
        columns = list(histogram_data[keys[0]].keys())

        linearized_data = []

        if order_by_bin:
            for col in columns:

                # Compute union of all bins of all dataframe

                # Only has an effect if binedges do not coincide for all keys
                shape = list(histogram_data[keys[0]][col]["binedges"][0].shape)
                shape[0] = 0
                # Multidimensional bin index
                if len(shape) > 1:
                    bins = np.empty(shape=shape, dtype=np.float32)
                    for key in keys:
                        bins = [x for x in set(tuple(x) for x in bins) | set(
                            tuple(x) for x in histogram_data[key][col]["binedges"][0][:-1])]
                    bins.sort()
                    binnames = histogram_data[keys[0]][col]['binnames']
                # One-dimensional bin index
                else:
                    from functools import reduce
                    binarrays = [histogram_data[key][col]["binedges"][0][:-1] for key in keys]
                    bins = reduce(np.union1d, binarrays)

                # Generate a pandas dataframe for each dataset (row_value)

                results = {}
                # Same bins for each dataframe
                if np.array_equal(np.array(bins), histogram_data[keys[0]][col]['binedges'][0][:-1]):
                    from pystatplottools.utils.bins_and_alignment import align_bins
                    bins = align_bins(bins=histogram_data[keys[0]][col]['binedges'][0],
                                      bin_alignment=bin_alignment,
                                      bin_scale=histogram_data[keys[0]][col]['binscale'][0])
                    if len(shape) > 1:
                        results = {"bin_" + str(bin): bins[:, idx] for idx, bin in enumerate(binnames)}
                    else:
                        results = {"bin": bins}
                    for key in keys:
                        results[key] = histogram_data[key][col]['hist'].reshape(-1)
                # Different bins for each dataframe
                else:
                    assert bin_alignment == "left", \
                        "For irregular bins or bins that refer to indices, only left alignment is reasonable."
                    if len(shape) > 1:  # -> bins
                        for key in keys:
                            result = np.zeros(len(bins))
                            considered_bins = [tuple(x) for x in histogram_data[key][col]['binedges'][0][:-1]]
                            result[np.array([item in considered_bins for item in bins]).nonzero()[0]] = histogram_data[key][col]['hist'].reshape(-1)
                            results[key] = result

                        bins = np.array(bins)
                        for idx, bin in enumerate(binnames):
                            results["bin_" + str(bin)] = bins[:, idx]
                        results = {k: results[k] for k in ["bin_" + str(bin) for bin in binnames] + keys}
                    else:
                        results = {"bin": bins}
                        for key in keys:
                            result = np.zeros(len(bins))
                            result[np.in1d(bins, histogram_data[key][col]['binedges'][0][:-1]).nonzero()[0]] = histogram_data[key][col]['hist'].reshape(-1)
                            results[key] = result

                linearized_data.append(pd.DataFrame(results))
            linearized_data = pd.concat(linearized_data, keys=columns)
            linearized_data.columns.names = ["bin_num_and_dfs"]
            linearized_data.index.names = ["statistics", "idx"]
            return linearized_data
        else:
            for key in keys:
                results = []
                for idx, col in enumerate(columns):
                    from pystatplottools.utils.bins_and_alignment import align_bins
                    bins = align_bins(bins=histogram_data[key][col]['binedges'][0],
                                      bin_alignment=bin_alignment,
                                      bin_scale=histogram_data[key][col]['binscale'][0])
                    if len(list(bins.shape)) > 1:
                        binnames = histogram_data[keys[0]][col]['binnames']
                        result = {"bin_" + str(bin): bins[:, idx] for idx, bin in enumerate(binnames)}
                        result["data"] = histogram_data[key][col]['hist'].reshape(-1)
                        results.append(pd.DataFrame(result))
                    else:
                        results.append(pd.DataFrame({"bin": bins, "data": histogram_data[key][col]['hist'].reshape(-1)}))
                linearized_data.append(pd.concat(results, keys=columns))
            linearized_data = pd.concat(linearized_data, keys=keys)
            linearized_data.index.names = ["dfs", "statistics", "idx"]
            linearized_data.columns.names = ["bin_num_and_statistics"]
            return linearized_data

    @staticmethod
    def _linearize_binned_statistics(axes_indices, binned_statistics, output_statistics_names=None,
                                      dataframes_as_columns=True, bin_alignment="center"):
        keys = list(binned_statistics.keys())
        columns = list(binned_statistics[keys[0]].keys())

        if isinstance(output_statistics_names, str):
            output_statistics_names = [output_statistics_names]

        if output_statistics_names is not None:
            assert len(columns) == len(output_statistics_names), \
                "Number of columns and number of output_statistics_names do not coincide"

        binedges = binned_statistics[keys[0]][columns[0]]['binedges']

        # Check whether all rows and columns have the same rel_bin
        for key in keys:
            for col in columns:
                binn = binned_statistics[key][col]['binedges']
                assert False not in [np.array_equal(bis, bi) for bis, bi in zip(binedges, binn)], \
                    "Rows and columns do not share the same bins - linearisation is not possible"

        bins = binned_statistics[keys[0]][columns[0]]
        bins["nbins"] = np.array([len(bins['binedges'][i])-1 for i in range(len(axes_indices))])

        resulting_bins = []
        for i in range(len(axes_indices)):
            from pystatplottools.utils.bins_and_alignment import align_bins
            ax_bins = align_bins(bins=bins["binedges"][i], bin_alignment=bin_alignment, bin_scale=bins["binscale"][i])
            resulting_bins.append(np.tile(np.repeat(ax_bins, np.prod(bins["nbins"][i + 1:])), np.prod(bins["nbins"][:i])))

        # Determine columns names
        cols_out = []
        if output_statistics_names is None:
            for col in columns:
                if col in axes_indices:
                    col_out = str(col) + "_ext"
                else:
                    col_out = col
                cols_out.append(col_out)
        else:
            cols_out = output_statistics_names

        if dataframes_as_columns:
            linearized_data = []
            for idx, col in enumerate(columns):
                results = {ax_idx: resulting_bin for ax_idx, resulting_bin in zip(axes_indices, resulting_bins)}
                for key in keys:
                    results[key] = binned_statistics[key][col]['hist'].reshape(-1)
                linearized_data.append(pd.DataFrame(results))
            linearized_data = pd.concat(linearized_data, keys=cols_out)
            linearized_data.columns.names = ["axes_and_dfs"]
            linearized_data.index.names = ["statistics", "idx"]
            return linearized_data
        else:
            linearized_data = []
            for key in keys:
                results = {ax_idx: resulting_bin for ax_idx, resulting_bin in zip(axes_indices, resulting_bins)}
                for idx, col in enumerate(columns):
                    results[cols_out[idx]] = binned_statistics[key][col]['hist'].reshape(-1)
                linearized_data.append(pd.DataFrame(results))
            linearized_data = pd.concat(linearized_data, keys=keys)
            linearized_data.columns.names = ["axes_and_statistics"]
            linearized_data.index.names = ["dfs", "idx"]
            return linearized_data

    ''' Further Helper Functions '''

    @staticmethod
    def _tile_scalar(val, row_values, axes_indices, identifier='nbins'):
        n = len(axes_indices)
        if hasattr(val, "__len__") and type(val) != str:
            assert len(val) == n, "Number of " + identifier + " and dimension " + str(n) + "do not coincide."
            scalars = val
        else:
            scalars = [val for _ in range(n)]
        nb = [scalars for _ in range(len(row_values))]
        nb_tuples = list(zip(axes_indices, [identifier for _ in range(n)]))
        nb_col_index = pd.MultiIndex.from_tuples(tuples=nb_tuples)
        nb = pd.DataFrame(nb, index=row_values, columns=nb_col_index)
        return nb

    @staticmethod
    def _get_bin_properties_of_collection(x):
        col = x.columns.get_level_values(0).values[0]
        results = dict()
        for idx in x.index.values:
            assert x.loc[idx][col, 'min'] != x.loc[idx][col, 'max'] or nbins == 1,\
                "Computation of a histogram/distribution for " + str(col) + " currently not possible since all values are equal." \
                "Consider to take out " + str(col) + " from your distribution."
            results[idx] = DistributionDD._get_bin_properties(range_min=x.loc[idx][col, 'min'],
                                                              range_max=x.loc[idx][col, 'max'],
                                                              nbins=x.loc[idx][col, 'nbins'],
                                                              binscale=x.loc[idx][col, 'binscale'])
        return results

    @staticmethod
    def _get_bin_properties(range_min, range_max, nbins, binscale='linear'):
        if binscale == "linear":
            binedges = np.linspace(range_min, range_max, nbins + 1)
        elif binscale == "logarithmic":
            binedges = np.logspace(np.log10(range_min), np.log10(range_max), nbins + 1)
        else:
            assert False, 'No scale given in _get_bin_properties'
        return {'range_max': range_max,
                'range_min': range_min,
                'nbins': nbins,
                'binedges': binedges,
                'binscale': binscale}
