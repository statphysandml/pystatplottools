import numpy as np
import pandas as pd

from ppd_distributions.distributionbaseclass import DistributionBaseClass


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

if __name__ == "__main__":
    pass
    # from loading import Loading
    # loading = Loading("EvaluationDynamic", "TestDataSet2BWCascade")
    #
    # dist1d = Distribution1D(data=loading.get_data())
    # dist1d.compute_expectation_values(columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss'],
    #                                   exp_values=['mean', 'max', 'min'], transform='log10')
    # dist1d.compute_expectation_values(columns=['clean_vs_noisy_propagator_loss', 'clean_vs_recon_propagator_loss',
    #                                            'noisy_vs_recon_propagator_loss', 'parameter_norm'],
    #                                   exp_values=['mean', 'median', 'quant25', 'quant75', 'std'])