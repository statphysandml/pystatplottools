from torch.utils import data
from torch.utils.data import _utils
default_collate = _utils.collate.default_collate


# Expects batch of configs when calling __getitem__ of dataset
class BatchDataLoader(data.DataLoader):
    def __init__(self, dataset, raw_samples=False, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):

        batch_size = dataset.datagenerator.batch_size
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
        # Returns the sample from the BatchDataGenerator as it is. Can be used if the BatchDataGenerator produces already the correct datatype for training
        self.raw_samples = raw_samples

    def __iter__(self):
        return HelperIterBatchDataLoader(dataset=self.dataset, collate_fn=self.collate_fn, pin_memory=self.pin_memory, n=self.__len__(), raw_samples=self.raw_samples)

    # def __next__(self):
    #     while self.i < self.n:
    #         batch = self.dataset[0]
    #         batch = [self.collate_fn(bat) for bat in batch]
    #         if self.pin_memory:
    #             batch = _utils.pin_memory.pin_memory_batch(batch)
    #         yield batch
    #         self.i += 1


class HelperIterBatchDataLoader:
    def __init__(self, dataset, collate_fn, pin_memory, n, raw_samples):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.i = 0
        self.n = n
        self.raw_samples = raw_samples

    def _next(self):
        while self.i < self.n:
            batch = self.dataset[self.i]

            if self.raw_samples or hasattr(batch, "batch"):  # Second condition is for geometric datasets
                # No collate_fn takes place
                pass
            else:
                batch = [self.collate_fn(bat) for bat in batch]
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            self.i += 1
            yield batch

    def __iter__(self):
        assert False, "__iter__ in HelperIterBatchDataloader of pystatplottools not implemented."

    def __next__(self):
        val = self._next()
        return next(val)

    def __len__(self):
        return self.n


def data_loader_factory(data_loader_name="default"):
    if data_loader_name == "BatchDataLoader":
        return BatchDataLoader
    elif data_loader_name == "GeometricDataLoader":
        from torch_geometric.loader import DataLoader
        return DataLoader
    else:
        return data.DataLoader
