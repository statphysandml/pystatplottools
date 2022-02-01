import numpy as np
import pandas as pd


def mean(x):
    return x.mean()


def std(x):
    if np.iscomplexobj(x.iloc[0]):
        return x.agg(lambda y: np.real(y.to_numpy()).std() + 1.0j * np.imag(y.to_numpy()).std())
    else:
        return x.std()


def var(x):
    if np.iscomplexobj(x.iloc[0]):
        return x.apply(lambda y: np.real(y.to_numpy()).std() + 1.0j * np.imag(y.to_numpy()).var())
    else:
        return x.var()


def quant25(x):
    return x.quantile(0.25)


def quant75(x):
    return x.quantile(0.75)


def secondmoment(x):
    return pow(x, 2).mean()


def fourthmoment(x):
    return pow(x, 4).mean()


class ExpectationValue:
    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        self.computed_expectation_values = None
        self.errors = None

    def compute_expectation_value(self, columns,
                                  exp_values=['mean', 'max', 'min', 'median', 'quant25', 'quant75', 'std'],
                                  transform="lin"):

        self.computed_expectation_values = ExpectationValue.__evaluate_expectation_values(data=self.data,
                                                      computed_expectation_values=self.computed_expectation_values,
                                                      columns=columns, exp_values=exp_values, transform=transform)

    @property
    def expectation_values(self):
        return self.computed_expectation_values

    def jackknife_error(self, number_of_measurements, columns, exp_values, running_parameter="default",
                                     transform="lin"):
        import copy
        ep_std = ExpectationValue(data=self.data)
        ep_std.compute_expectation_value(columns=columns, exp_values=['mean'])
        ep_mean = copy.deepcopy(ep_std.expectation_values)
        data_mask = np.ones(len(self.data), dtype=np.bool)
        data_mask[0::number_of_measurements] = False
        estimated_error = np.zeros_like(ep_mean.values)
        for leave_out_index in range(1, number_of_measurements):
            ep_std.data = self.data.loc[data_mask, columns]
            ep_std.compute_expectation_value(columns=columns, exp_values=['mean'])
            estimated_error += np.power(ep_std.expectation_values.values - ep_mean.values, 2)
            data_mask[leave_out_index - 1::number_of_measurements] = True
            data_mask[leave_out_index::number_of_measurements] = False
        self.errors = pd.DataFrame(data=np.sqrt(estimated_error), index=ep_mean.index, columns=ep_mean.columns)

    def bootstrap_error(self, n_means_boostrap, number_of_measurements, columns, exp_values, running_parameter="default",
                                     transform="lin"):
        split_data = [pd.concat([tup[1], tup[1]]).reset_index(drop=True) for tup in list(self.data.groupby(running_parameter))]

        bootstrap_df = pd.concat(split_data, keys=self.data.index.unique(0)).sort_index(level=0)
        means = []
        for _ in range(n_means_boostrap):
            sampled_df = bootstrap_df.groupby(running_parameter).apply(lambda x: x.sample(n=number_of_measurements, replace=False))
            sampled_df = sampled_df.droplevel(level=1)
            sampled_expectation_values = ExpectationValue.__evaluate_expectation_values(data=sampled_df, computed_expectation_values=None, columns=columns, exp_values=exp_values, transform=transform)

            means.append(sampled_expectation_values)

        self.errors = pd.concat(means).groupby(running_parameter).apply(lambda x: std(x))

    def statistical_error(self, number_of_measurements, columns):
        ep_std = ExpectationValue(data=self.data)
        ep_std.compute_expectation_value(columns=columns, exp_values=['std'])
        self.errors = ep_std.expectation_values / np.sqrt(number_of_measurements)

    @staticmethod
    def __evaluate_expectation_values(data, computed_expectation_values, columns,
                                     exp_values=['mean', 'max', 'min', 'median', 'quant25', 'quant75', 'std'],
                                     transform="lin"):

        # Replace function names by executable functions
        for func_string, func in zip(
                ["mean", "std", "var", "quant25", "quant75", "secondmoment", "fourthmoment"],
                [mean, std, var, quant25, quant75, secondmoment, fourthmoment]):
            exp_values = ExpectationValue.replace_function_by_string(func_string=func_string, func=func, exp_values=exp_values)

        if transform == "log10":
            # Compute log10 of data
            from pystatplottools.distributions.distributionDD import transform_log10
            data, columns = transform_log10(data=data, columns=columns)
            # Adapt columns to apply measures on

        # Compute or extend expectation values
        if computed_expectation_values is None:
            computed_expectation_values = data[columns].groupby(level=0, axis=0).agg(exp_values).sort_index(axis=1)
        else:
            new_computed_expectation_values = data[columns].groupby(level=0, axis=0).agg(exp_values)
            # Extract duplicate columns
            cols_to_use = computed_expectation_values.columns.difference(new_computed_expectation_values.columns)
            computed_expectation_values = pd.concat([computed_expectation_values[cols_to_use], new_computed_expectation_values],
                                                    axis=1, verify_integrity=True).sort_index(axis=1)

        return computed_expectation_values

    @staticmethod
    def drop_multiindex_levels_with_unique_entries(data):
        # Drop multiindex levels with unique entries
        to_drop = []
        for lev in range(data.columns.nlevels):
            if len(data.columns.unique(lev)) == 1:
                to_drop.append(lev)
        if len(data.columns) > 1:
            data.columns = data.columns.droplevel(to_drop)
        else:
            data.columns = data.columns.droplevel(to_drop[1:])
        return data

    @staticmethod
    def replace_function_by_string(func_string, func, exp_values):
        funclist = np.argwhere(np.array(exp_values) == func_string).flatten()
        if len(funclist) > 0:
            exp_values[funclist[0]] = func
        return exp_values