import torch_geometric.data as geometric_data


class GeometricDataLoader(geometric_data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)

    def get_dataset_inspector(self):
        op = getattr(self.dataset, "get_datagenerator", None)
        if callable(op):
            return self.dataset.get_datagenerator()
        else:
            return self.dataset
