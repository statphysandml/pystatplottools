import numpy as np
import pandas as pd

from distributions.distributionbaseclass import DistributionBaseClass


class DistributionDD(DistributionBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.expectation_values = None

    def compute_binned_statistics(self, axes_indices,
                                  columns=None,
                                  range_min=None,
                                  range_max=None,
                                  nbins=[],
                                  bin_scales='Linear',
                                  transform="lin",
                                  with_binnumber=False,
                                  statistic='probability'):
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

        binned_statistics = dict()
        for row in row_values:
            binned_statistics[row] = dict()
            for col in columns:
                if statistic is "probability":
                    bin_statistic = lambda x: len(x) * 1.0 / len(self.data.loc[row])
                hist, rel_bins, binnumber = binned_statistic_dd(
                    sample=self.data.loc[row][axes_indices].values, values=self.data.loc[row][col], statistic=bin_statistic,
                    bins=[histogram_prep[ax_idx][row]['bin_edges'] for ax_idx in axes_indices])
                if with_binnumber:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': axes_indices, 'binnumber': binnumber}
                else:
                    binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins, 'rel_bins_index': axes_indices}

        return binned_statistics

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

