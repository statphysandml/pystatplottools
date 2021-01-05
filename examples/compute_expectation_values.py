from pystatplottools.expectation_values.expectation_value import ExpectationValue


def compute_expectation_values_distribution1D():
    from examples.mock_data import load_multivariate_mock_data
    data = load_multivariate_mock_data()
    
    ep = ExpectationValue(data=data)

    ep.compute_expectation_value(columns=['a', 'b'],
                                 exp_values=['mean', 'max', 'min', 'secondmoment', 'fourthmoment'])

    expectation_values = ep.expectation_values
    return expectation_values


if __name__ == '__main__':
    compute_expectation_values_distribution1D()