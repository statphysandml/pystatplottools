import numpy as np
import pandas as pd


def load_multivariate_mock_data():
    num = 10000

    data1 = np.vstack(list(np.random.multivariate_normal(mean=[0, 1, -1.0], cov=[[1, 0, 0], [0, 4, 0], [0, 0, 10]],
                                                         size=num).transpose()) + [np.random.rand(num)]).transpose()
    df1 = pd.DataFrame(data=data1, index=None, columns=["a", "b", "c", "idx"])

    data2 = np.vstack(list(np.random.multivariate_normal(mean=[2.0, 0.0, 1.0], cov=[[2, 1, 4], [1, 4, 1], [4, 1, 10]],
                                                         size=num).transpose()) + [np.random.rand(num)]).transpose()
    df2 = pd.DataFrame(data=data2, index=None, columns=["a", "b", "c", "idx"])

    return pd.concat([df1, df2], keys=["df1", "df2"])