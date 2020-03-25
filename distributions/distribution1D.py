import numpy as np
import pandas as pd

from distributions.distributionbaseclass import DistributionBaseClass


def quant25(x):
    return x.quantile(0.25)


def quant75(x):
    return x.quantile(0.75)


def secondmoment(x):
    return pow(x, 2).mean()


def fourthmoment(x):
    return pow(x, 4).mean()


def compute_specificheat(dist, N):
    dist.expectation_values["SpecificHeat", "mean"] = pow(dist.expectation_values['Beta', 'mean'], 2)/N*(dist.expectation_values['Energy', 'secondmoment']-pow(dist.expectation_values['Energy', 'mean'], 2))


def compute_binder_cumulant(dist):
    dist.expectation_values["BinderCumulant", "mean"] = 1 - dist.expectation_values['Mean', 'fourthmoment']/(3*pow(dist.expectation_values['Mean', 'secondmoment'], 2))


class Distribution1D(DistributionBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.histograms = None
        self.expectation_values = None

    def compute_expectation_values(self,
                                   columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss',
                                            'clean_vs_recon_propagator_loss', 'noisy_vs_recon_propagator_loss',
                                            'parameter_norm'],
                                   exp_values=['mean', 'max', 'min', 'median', 'quant25', 'quant75', 'std'],
                                   transform="lin"):

        # Replace quantiles by corresponding functions
        quant25list = np.argwhere(np.array(exp_values) == 'quant25').flatten()
        if len(quant25list) > 0:
            exp_values[quant25list[0]] = quant25

        quant75list = np.argwhere(np.array(exp_values) == 'quant75').flatten()
        if len(quant75list) > 0:
            exp_values[quant75list[0]] = quant75

        secondMoment = np.argwhere(np.array(exp_values) == 'secondMoment').flatten()
        if len(secondMoment) > 0:
            exp_values[secondMoment[0]] = secondmoment

        fourthMoment = np.argwhere(np.array(exp_values) == 'fourthMoment').flatten()
        if len(secondMoment) > 0:
            exp_values[fourthMoment[0]] = fourthmoment

        if transform == "log10":
            # Compute log10 of data
            columns = self.transform_log10(columns=columns)
            # Adapt columns to apply measures on


        # Compute or extend expectation values
        if self.expectation_values is None:
            self.expectation_values = self.data.groupby(level=0)[columns].agg(exp_values)
        else:
            new_expectation_values = self.data.groupby(level=0)[columns].agg(exp_values)
            # Extract duplicate columns
            cols_to_use = self.expectation_values.columns.difference(new_expectation_values.columns)
            self.expectation_values = pd.concat([self.expectation_values[cols_to_use], new_expectation_values], axis=1,
                                                verify_integrity=True).sort_index(axis=1)

    def compute_histograms(self,
                           columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss',
                                     'clean_vs_recon_propagator_loss', 'noisy_vs_recon_propagator_loss',
                                     'parameter_norm'],
                           range_min=None,
                           range_max=None,
                           nbins=10,
                           bin_scale='Linear',
                           transform="lin",
                           kind='histogram'):
        if transform == "log10":
            # Compute log10 of data
            columns = self.transform_log10(columns=columns)

        n = len(columns)
        row_values = list(self.data.index.unique(0))

        if range_min is None:
            histogram_prep = self.data[columns].groupby(level=0).agg(['min', 'max'])

        # ToDo: Implement possibility to define own min max values for each column!

        # histogram_prep['spectral_function_loss_log10', 'nb'] = 10

        nb = DistributionBaseClass.reorder_nbins(nbins=[nbins], row_values=row_values, columns=columns)

        histogram_prep = pd.concat([histogram_prep, nb], axis=1, sort=True)

        histogram_prep = histogram_prep.groupby(level=0, axis=1).apply(lambda x: DistributionBaseClass.get_bin_properties_of_collection(x, bin_scale))

        binned_statistics = dict()
        for row in row_values:
            binned_statistics[row] = dict()
            for col in columns:
                hist, rel_bins = Distribution1D.get_histogram1d(
                    data=self.data.loc[row, col].values,
                    bin_properties=histogram_prep.loc[col][row],
                    kind=kind)
                binned_statistics[row][col] = {'hist': hist, 'rel_bins': rel_bins}

        self.histograms = binned_statistics

        # range_min = entity_to_list(range_min, n)
        # range_max = entity_to_list(range_max, n)
        # nbins = entity_to_list(nbins, n)
        #
        # bin_properties = [DistributionBaseClass.get_bin_properties(
        #     range_min=r_min,
        #     range_max=r_max,
        #     nbins=nb,
        #     scale=bin_scale) for idx, (r_min, r_max, nb) in enumerate(zip(range_min, range_max, nbins))]

    @staticmethod
    def get_histogram1d(data, bin_properties, kind='histogram'):
        z, rel_bins = Distribution1D.generate_histogram1d_prob(
            data=data,
            bins=bin_properties['bin_edges'],
            kind=kind,
            scale=bin_properties['scale']
        )
        return z, rel_bins

    @staticmethod
    def generate_histogram1d_prob(data, nbins=None, range=None, bins=None, kind="count", scale="Linear"):
        if scale is "Linear":
            if bins is not None:
                hist, rel_bins = np.histogram(data, bins=bins)
            else:
                hist, rel_bins = np.histogram(data, bins=nbins, range=range)
        elif scale is "Logarithmic":
            if bins is not None:
                hist, rel_bins = np.histogram(data, bins=bins)
            else:
                hist, rel_bins = np.histogram(data, bins=np.logspace(np.log10(range[0][0]), np.log10(range[0][1]),
                                                                     nbins[0] + 1))
        hist[-1] += len(data) - np.sum(hist)

        if kind == "probability_dist":
            assert (np.sum(hist * 1.0 / np.sum(hist)) > 0.999) and (
                    np.sum(hist * 1.0 / np.sum(hist)) < 1.001), "wrong probability" + str(
                np.sum(hist / np.sum(hist)))
            return hist * 1.0 / np.sum(hist), rel_bins
        elif kind == "histogram":
            return hist, rel_bins

    @staticmethod
    def plot_histogram(ax, hist, rel_bins, color="darkblue", label=None):
        import matplotlib.pyplot as plt

        #if hasattr(self, 'bin_widths'):
        #    print('Probability sum with bin widths', np.sum(self.probabilities*self.bin_widths))
        #else:
        #    print('Probability sum', np.sum(self.probabilities))
#        if self.bin_properties['scale'] is "Logarithmic":
#            log_width_time_deltas = self.bin_edges[1:]-self.bin_edges[:-1]
#            ax.bar(self.bin_edges[:-1]+log_width_time_deltas/2, self.probabilities,
#                      width=0.9*log_width_time_deltas, color=color, label=label)
            #ax.scatter(self.bin_edges[:-1]+log_width_time_deltas/2, self.probabilities, color=color)
#            ax.set_xscale("log")
#        else:
        width = rel_bins[1]-rel_bins[0]
        ax.bar(rel_bins[:-1], hist, width=width*0.9, color=color, label=label)

        return ax


if __name__ == "__main__":
    from loading import Loading
    loading = Loading("EvaluationDynamic", "TestDataSet2BWCascade")

    dist1d = Distribution1D(data=loading.get_data())
    dist1d.compute_expectation_values(columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss'],
                                      exp_values=['mean', 'max', 'min'], transform='log10')
    # dist1d.compute_expectation_values(columns=['clean_vs_noisy_propagator_loss', 'clean_vs_recon_propagator_loss',
    #                                            'noisy_vs_recon_propagator_loss', 'parameter_norm'],
    #                                   exp_values=['mean', 'median', 'quant25', 'quant75', 'std'])

    dist1d.compute_histograms(columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss'], transform='log10')
