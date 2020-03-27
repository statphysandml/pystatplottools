from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class DistributionBaseClass:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.data = kwargs.pop("data")
        self.name = kwargs.pop("name", "Unknown")

    def transform_log10(self, columns):
        # Compute log10 of data
        log_data = self.data[columns].apply(["log10"])
        log_data.columns = log_data.columns.droplevel(1) + "_log10"
        cols_to_use = self.data.columns.difference(log_data.columns)
        self.data = pd.concat([self.data[cols_to_use], log_data], axis=1, verify_integrity=True)
        return [col + '_log10' for col in columns]

    @staticmethod
    def reorder_nbins(nbins, row_values, columns):
        assert len(columns) == len(nbins), "Number of axes indices and nbins do not coincide"
        n = len(columns)
        nb = [[nbin for nbin in nbins] for _ in range(len(row_values))]
        nb_tuples = list(zip(columns, ['nb' for _ in range(n)]))
        nb_col_index = pd.MultiIndex.from_tuples(tuples=nb_tuples)
        nb = pd.DataFrame(nb, index=row_values, columns=nb_col_index)
        return nb

    @staticmethod
    def tile_scalar(val, row_values, columns, identifier='nb'):
        n = len(columns)
        if hasattr(val, "__len__") and len(val) == n:
            scalars = val
        else:
            scalars = [val for _ in range(n)]
        nb = [scalars for _ in range(len(row_values))]
        nb_tuples = list(zip(columns, [identifier for _ in range(n)]))
        nb_col_index = pd.MultiIndex.from_tuples(tuples=nb_tuples)
        nb = pd.DataFrame(nb, index=row_values, columns=nb_col_index)
        return nb

    @staticmethod
    def get_bin_properties_of_collection(x, bin_scale):
        # col = x.index.get_level_values(0)

        col = x.columns.get_level_values(0).values[0]
        results = dict()
        for idx in x.index.values:
            results[idx] = DistributionBaseClass.get_bin_properties(range_min=x.loc[idx][col, 'min'],
                                                                    range_max=x.loc[idx][col, 'max'],
                                                                    nbins=x.loc[idx][col, 'nb'], scale=bin_scale)
        return results

    @staticmethod
    def get_bin_properties(range_min, range_max, nbins, scale='Linear'):
        if scale is "Linear":
            bin_edges = np.linspace(range_min, range_max, nbins + 1)
        elif scale is "Logarithmic":
            bin_edges = np.logspace(np.log10(range_min), np.log10(range_max), nbins + 1)
        else:
            assert False, 'No scale given in get_bin_properties'
        return {'range_max': range_max,
                'range_min': range_min,
                'nbins': nbins,
                'bin_edges': bin_edges,
                'scale': scale}
