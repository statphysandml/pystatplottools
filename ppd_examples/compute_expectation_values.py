from ppd_examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


from pystatplottools.ppd_distributions.expectation_value import ExpectationValue


def compute_expectation_values_distribution1D():
    ep = ExpectationValue(data=data)

    ep.compute_expectation_value(columns=['a', 'b'],  # , 'Energy'],
                                      exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    # ep.compute_expectation_values(columns=['Beta'], exp_values=['mean'])

    expectation_values = ep.expectation_values
    return expectation_values


if __name__ == '__main__':
    compute_expectation_values_distribution1D()