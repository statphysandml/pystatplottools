import numpy as np
import pandas as pd

from pystatplottools.ppd_distributions.distributionbaseclass import DistributionBaseClass


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


class ExpectationValue(DistributionBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.computed_expectation_values = None
        self.bootstrap_errors = None

    def compute_expectation_value(self,
                                  columns=['spectral_function_loss', 'clean_vs_noisy_propagator_loss',
                                           'clean_vs_recon_propagator_loss', 'noisy_vs_recon_propagator_loss',
                                           'parameter_norm'],
                                  exp_values=['mean', 'max', 'min', 'median', 'quant25', 'quant75', 'std'],
                                  transform="lin"):

        self.computed_expectation_values = ExpectationValue._evaluate_expectation_values(data=self.data,
                                                      computed_expectation_values=self.computed_expectation_values,
                                                      columns=columns, exp_values=exp_values, transform=transform)

    @property
    def expectation_values(self):
        return self.computed_expectation_values

    def compute_error_with_bootstrap(self, n_means_boostrap, number_of_measurements, columns, exp_values, running_parameter="default",
                                     transform="lin"):
        split_data = [pd.concat([tup[1], tup[1]]).reset_index(drop=True) for tup in list(self.data.groupby(running_parameter))]

        bootstrap_df = pd.concat(split_data, keys=self.data.index.unique(0)).sort_index(level=0)
        means = []
        for _ in range(n_means_boostrap):
            sampled_df = bootstrap_df.groupby(running_parameter).apply(lambda x: x.sample(n=number_of_measurements, replace=False))
            sampled_df = sampled_df.droplevel(level=1)
            sampled_expectation_values = ExpectationValue._evaluate_expectation_values(data=sampled_df, computed_expectation_values=None, columns=columns, exp_values=exp_values, transform=transform)

            means.append(sampled_expectation_values)

        self.bootstrap_errors = pd.concat(means).groupby(running_parameter).apply(lambda x: x.std())

    @staticmethod
    def _evaluate_expectation_values(data, computed_expectation_values, columns,
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
            columns = DistributionBaseClass.transform_log10(data=data, columns=columns)
            # Adapt columns to apply measures on

        # Compute or extend expectation values
        if computed_expectation_values is None:
            computed_expectation_values = data.groupby(level=0)[columns].agg(exp_values)
        else:
            new_computed_expectation_values = data.groupby(level=0)[columns].agg(exp_values)
            # Extract duplicate columns
            cols_to_use = computed_expectation_values.columns.difference(new_computed_expectation_values.columns)
            computed_expectation_values = pd.concat([computed_expectation_values[cols_to_use], new_computed_expectation_values],
                                                axis=1,
                                                verify_integrity=True).sort_index(axis=1)

        return computed_expectation_values