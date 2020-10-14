import numpy as np
import pandas as pd

from pystatplottools.ppd_distributions.distributionbaseclass import DistributionBaseClass


# ToDo: Check what is done with samples that are outside of the given ranges!!

class DistributionDD(DistributionBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def binned_statistics_over_axes(self, axes_indices,
                                    columns=None,
                                    statistic='count',
                                    transform='lin',
                                    nbins=[], # Refers to the number of bins in each dimension
                                    # Range accepts None, Scalar and List
                                    range_min=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
                                    range_max=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
                                    bin_scales='Linear',
                                    with_binnumber=False
                                    ):
        nbins = DistributionBaseClass.scalar_to_list_if_scalar_and_not_none(val=nbins, n=len(axes_indices))

        return self.compute_binned_statistics(axes_indices=axes_indices,
                                              columns=columns,
                                              statistic=statistic,
                                              transform=transform,
                                              nbins=nbins,
                                              range_min=range_min,
                                              range_max=range_max,
                                              bin_scales=bin_scales,
                                              with_binnumber=with_binnumber
                                              )

    def compute_1Dhistograms(self,
                           columns,
                           statistic='count',
                           transform='lin',
                           nbins=[], # Refers to the number of bins for each column
                           range_min=None, # Refers to the ranges of the bins for each column
                           range_max=None, # Refers to the ranges of the bins for each column
                           bin_scales='Linear',
                           with_binnumber=False
                           ):

        nbins = DistributionBaseClass.scalar_to_list_if_scalar_and_not_none(val=nbins, n=len(columns))

        return self.compute_binned_statistics(columns=columns,
                                       statistic=statistic,
                                       transform=transform,
                                       nbins=nbins,
                                       range_min=range_min,
                                       range_max=range_max,
                                       bin_scales=bin_scales,
                                       with_binnumber=with_binnumber
                                       )

    def compute_DDhistograms(self,
                             transform='lin',
                             nbins=[],  # Refers to the number of bins for each column
                             range_min=None,  # Refers to the ranges of the bins for each column
                             range_max=None,  # Refers to the ranges of the bins for each column
                             bin_scales='Linear',
                             with_binedges=False
                            ):

        columns = self.data.columns
        nbins = DistributionBaseClass.scalar_to_list_if_scalar_and_not_none(val=nbins, n=len(columns))
        range_min = DistributionBaseClass.scalar_to_list_if_scalar_and_not_none(val=range_min, n=len(columns))
        range_max = DistributionBaseClass.scalar_to_list_if_scalar_and_not_none(val=range_max, n=len(columns))

        columns, row_values, histogram_prep = self.prepare(
            axes_indices=columns, columns=columns, transform=transform, nbins=nbins,
            range_min=range_min, range_max=range_max, bin_scales=bin_scales
        )

        binned_statistics = dict()
        for row in row_values:
            data_in_range_mask = DistributionDD.compute_data_mask_based_on_ranges(
                data=self.data.loc[row].values,
                ranges_min=[histogram_prep[ax_idx][row]['range_min'] for ax_idx in columns],
                ranges_max=[histogram_prep[ax_idx][row]['range_max'] for ax_idx in columns]
            )

            data = self.data.loc[row].values[data_in_range_mask]
            bins = np.array([hist[row]["bin_edges"] for hist in histogram_prep])

            if np.all([np.array_equal(bin, bins[idx + 1]) for idx, bin in enumerate(bins[:-1])]):
                # All dimensions have the same range_min, range_max
                dat = np.zeros(np.shape(data), dtype=np.int8)
                for bin_edge in bins[0, 1:-1]:
                    dat[np.nonzero(data >= bin_edge)] += 1
                data = dat
            else:
                for dim in range(data.shape[1]):
                    dat = np.zeros(data.shape[0], dtype=np.int8)
                    for bin_edge in bins[dim, 1:-1]:
                        dat[np.nonzero(data[:, dim] >= bin_edge)] += 1
                    data[:, dim] = dat

            indices = np.ravel_multi_index(data.transpose(), dims=nbins)
            hist, binedges = np.histogram(indices, bins=np.concatenate([np.unique(indices), np.array([np.max(indices) + 1])]), density=False)
            binarea = np.prod(bins[:, 1] - bins[:, 0])

            if with_binedges:
                binned_statistics[row] = {'hist': hist, 'binarea': binarea, 'binedges': binedges}
            else:
                binned_statistics[row] = {'hist': hist, 'binarea': binarea}

        return binned_statistics

    def compute_binned_statistics(self, axes_indices=None,
                                    columns=None,
                                    statistic='count',
                                    transform='lin',
                                    nbins=[],
                                    range_min=None,
                                    range_max=None,
                                    bin_scales='Linear',
                                    with_binnumber=False):

        if axes_indices is None:
            # Compute 1D statistics separate for each column
            axes_indices = columns # To generate the correct bins
            seperate_statistics = True
        else:
            seperate_statistics = False

        if columns is None:
            columns = [axes_indices[0]]

        columns, row_values, histogram_prep = self.prepare(
            axes_indices=axes_indices, columns=columns, transform=transform, nbins=nbins,
            range_min=range_min, range_max=range_max, bin_scales=bin_scales
        )

        from scipy.stats import binned_statistic_dd

        bin_statistic = statistic

        sample_indices = axes_indices # Default assumes computation with axes indices
        binned_statistics = dict()
        for row in row_values:
            binned_statistics[row] = dict()
            for col in columns:
                if statistic == "probability":
                    bin_statistic = lambda x: len(x) * 1.0 / len(self.data.loc[row])
                if seperate_statistics:  # For 1D histrograms
                    sample_indices = col
                data_in_range_mask = DistributionDD.compute_data_mask_based_on_ranges(
                    data=self.data.loc[row][sample_indices].values,
                    ranges_min=[histogram_prep[ax_idx][row]['range_min'] for ax_idx in sample_indices],
                    ranges_max=[histogram_prep[ax_idx][row]['range_max'] for ax_idx in sample_indices]
                )
                assert np.sum(data_in_range_mask) != 0, "No data point is in the given range in at least one dimension"
                if np.sum(data_in_range_mask) < len(self.data.loc[row][sample_indices]):
                    print("Not all data points are in the given ranges")
                hist, rel_bins, binnumber = binned_statistic_dd(
                    sample=self.data.loc[row][sample_indices].values[data_in_range_mask], values=self.data.loc[row][col].values[data_in_range_mask], statistic=bin_statistic,
                    bins=[histogram_prep[ax_idx][row]['bin_edges'] for ax_idx in sample_indices])
                if with_binnumber:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': sample_indices, 'binnumber': binnumber}
                else:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': sample_indices}

        return binned_statistics

    def prepare(self, axes_indices=None,
                                    columns=None,
                                    transform='lin',
                                    nbins=[],
                                    range_min=None,
                                    range_max=None,
                                    bin_scales='Linear'):

        if transform == "log10":
            # Compute log10 of data
            columns = DistributionBaseClass.transform_log10(data=self.data, columns=columns)

        # includes the indices of the different considered datasets
        row_values = list(self.data.index.unique(0))

        if range_min is None:
            histogram_prep = self.data[axes_indices].groupby(level=0).agg(['min', 'max'])
        else:
            ranges_min = DistributionBaseClass.tile_scalar(val=range_min, row_values=row_values, columns=axes_indices, identifier='min')
            ranges_max = DistributionBaseClass.tile_scalar(val=range_max, row_values=row_values, columns=axes_indices, identifier='max')
            histogram_prep = pd.concat([ranges_min, ranges_max], axis=1, sort=True)

        nb = DistributionBaseClass.reorder_nbins(nbins=nbins, row_values=row_values, columns=axes_indices)

        histogram_prep = pd.concat([histogram_prep, nb], axis=1, sort=True)

        # bin_scales = [bin_scales for _ in range(len(axes_indices))]  # Should be applied if bin_scales is a string
        histogram_prep = histogram_prep.groupby(level=0, axis=1).apply(
            lambda x: DistributionBaseClass.get_bin_properties_of_collection(x, bin_scales))

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
    def linearize_histograms(histogram_data, order_by_bin=False):
        keys = list(histogram_data.keys())
        columns = list(histogram_data[keys[0]].keys())

        linearized_data = []

        if order_by_bin:
            for col in columns:
                bins = histogram_data[keys[0]][col]['rel_bins'][0]
                results = {"bin": (bins[:-1] + bins[1:]) / 2}
                for key in keys:
                    results[key] = histogram_data[key][col]['hist'].reshape(-1)
                linearized_data.append(pd.DataFrame(results))
            return pd.concat(linearized_data, keys=columns)
        else:
            for key in keys:
                results = [] # {"column": [], "index": [], "bin": [], "data": []}  # ax_idx: resulting_bin for ax_idx, resulting_bin in zip(axes_indices, resulting_bins)}
                for idx, col in enumerate(columns):
                    bins = histogram_data[key][col]['rel_bins'][0]
                    bins = (bins[:-1] + bins[1:]) / 2
                    results.append(pd.DataFrame({"bin": bins, "data": histogram_data[key][col]['hist'].reshape(-1)}))
                linearized_data.append(pd.concat(results, keys=columns))
            return pd.concat(linearized_data, keys=keys)

    @staticmethod
    def linearize_binned_statistics(axes_indices, binned_statistics, output_column_names=None):
        keys = list(binned_statistics.keys())
        columns = list(binned_statistics[keys[0]].keys())

        if isinstance(output_column_names, str):
            output_column_names = [output_column_names]

        if output_column_names is not None:
            assert len(columns) == len(output_column_names), \
                "Number of columns and number of output_column_names do not coincide"

        bins = binned_statistics[keys[0]][columns[0]]['rel_bins']
        # Check whether all rows and columns have the same rel_bin
        for key in keys:
            for col in columns:
                binn = binned_statistics[key][col]['rel_bins']
                assert False not in [np.array_equal(bis, bi) for bis, bi in zip(bins, binn)], \
                    "Rows and columns do not share the same bins - Linearisation is not possible"

        bin_indices = np.array(binned_statistics[keys[0]][columns[0]]["rel_bins_index"])
        bins = {bin_idx: bin_content for bin_idx, bin_content in zip(bin_indices, bins)}
        bin_sizes = np.array([len(bins[ax_idx])-1 for ax_idx in axes_indices])

        resulting_bins = []
        for i, ax_idx in enumerate(axes_indices):
            resulting_bins.append(np.tile(
                np.repeat((bins[ax_idx][:-1] + bins[ax_idx][1:]) / 2, np.prod(bin_sizes[i + 1:])),
                np.prod(bin_sizes[:i]))
            )

        linearized_data = []

        for key in keys:
            results = {ax_idx: resulting_bin for ax_idx, resulting_bin in zip(axes_indices, resulting_bins)}
            for idx, col in enumerate(columns):
                if output_column_names is None:
                    if col in axes_indices:
                        col_out = str(col) + "_ext"
                    else:
                        col_out = col
                else:
                    col_out = output_column_names[idx]
                results[col_out] = binned_statistics[key][col]['hist'].reshape(-1)
            linearized_data.append(pd.DataFrame(results))

        return pd.concat(linearized_data, keys=keys)

