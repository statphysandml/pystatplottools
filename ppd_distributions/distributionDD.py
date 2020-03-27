import numpy as np
import pandas as pd

from ppd_distributions.distributionbaseclass import DistributionBaseClass


class DistributionDD(DistributionBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def binned_statistics_over_axes(self, axes_indices,
                                    columns=None,
                                    statistic='count',
                                    transform='lin',
                                    nbins=[], # Refers to the number of bins in each dimension
                                    range_min=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
                                    range_max=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
                                    bin_scales='Linear',
                                    with_binnumber=False
                                    ):
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
        return self.compute_binned_statistics(columns=columns,
                                       statistic=statistic,
                                       transform=transform,
                                       nbins=nbins,
                                       range_min=range_min,
                                       range_max=range_max,
                                       bin_scales=bin_scales,
                                       with_binnumber=with_binnumber
                                       )

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

        if transform == "log10":
            # Compute log10 of data
            columns = self.transform_log10(columns=columns)

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

        from scipy.stats import binned_statistic_dd

        bin_statistic = statistic

        sample_indices = axes_indices # Default assumes computation with axes indices
        binned_statistics = dict()
        for row in row_values:
            binned_statistics[row] = dict()
            for col in columns:
                if statistic is "probability":
                    bin_statistic = lambda x: len(x) * 1.0 / len(self.data.loc[row])
                if seperate_statistics:  # For 1D histrograms
                    sample_indices = col
                hist, rel_bins, binnumber = binned_statistic_dd(
                    sample=self.data.loc[row][sample_indices].values, values=self.data.loc[row][col], statistic=bin_statistic,
                    bins=[histogram_prep[ax_idx][row]['bin_edges'] for ax_idx in sample_indices])
                if with_binnumber:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': sample_indices, 'binnumber': binnumber}
                else:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': sample_indices}

        return binned_statistics

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

