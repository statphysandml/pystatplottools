from ppd_examples.mock_data import load_multivariate_mock_data
data = load_multivariate_mock_data()


from ppd_distributions.expectation_value import ExpectationValue


def compute_expectation_values_distribution1D():
    ep = ExpectationValue(data=data)

    # assert False, "ToDo"
    # ep.compute_expectation_values(columns=['Mean', 'AbsMean'],  # , 'Energy'],
    #                                   exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    # ep.compute_expectation_values(columns=['Beta'], exp_values=['mean'])