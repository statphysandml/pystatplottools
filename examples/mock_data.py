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


def load_characteristics():
    from examples.pytorch_in_memory_dataset import load_rectangle_data_memory_dataset
    data_loader = load_rectangle_data_memory_dataset()
    dataset = data_loader.get_dataset_inspector()
    data, targets = dataset.data.cpu().numpy(), dataset.targets.cpu().numpy()

    import numpy as np
    distances_to_edge_in_x_direction = [np.argwhere(dat.sum(axis=0) != 0).flatten()[0] for dat in data]
    distances_to_edge_in_y_direction = [np.argwhere(dat.sum(axis=1) != 0).flatten()[0] for dat in data]

    import pandas as pd
    characteristics = pd.DataFrame({"A": np.prod(targets, axis=1), "height": targets[:, 0], "width": targets[:, 1],
                                    "mean": data.mean(axis=(1, 2)), "color": data.max(axis=(1, 2)),
                                    "dx": distances_to_edge_in_x_direction, "dy": distances_to_edge_in_y_direction})

    # Set multiindex ("A", sample_num)
    characteristics = characteristics.set_index("A")
    characteristics = characteristics.groupby("A").apply(lambda x: x.reset_index(drop=True))
    return characteristics