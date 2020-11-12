from torch.utils import data
from torch.utils.data import _utils
default_collate = _utils.collate.default_collate


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, status="default", shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

    def get_dataset_inspector(self):
        op = getattr(self.dataset, "get_datagenerator", None)
        if callable(op):
            return self.dataset.get_datagenerator()
        else:
            return self.dataset


class BatchDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, status="default", shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
        self.status = status

    def __iter__(self):
        return HelperIterBatchDataLoader(dataset=self.dataset, collate_fn=self.collate_fn, pin_memory=self.pin_memory, n=self.__len__(), status=self.status)

    def get_dataset_inspector(self):
        op = getattr(self.dataset, "get_datagenerator", None)
        if callable(op):
            return self.dataset.get_datagenerator()
        else:
            return self.dataset

    # def __next__(self):
    #     while self.i < self.n:
    #         batch = self.dataset[0]
    #         batch = [self.collate_fn(bat) for bat in batch]
    #         if self.pin_memory:
    #             batch = _utils.pin_memory.pin_memory_batch(batch)
    #         yield batch
    #         self.i += 1


class HelperIterBatchDataLoader:

    def __init__(self, dataset, collate_fn, pin_memory, n, status):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.i = 0
        self.n = n
        self.status = status

    def _next(self):
        while self.i < self.n:
            batch = self.dataset[0]
            if self.status == "target_param":
                batch = (batch[0], [self.collate_fn(bat) for bat in batch[1]])
            elif self.status == "target_config":
                batch = ([self.collate_fn(bat) for bat in batch[0]], batch[1])
            elif self.status == "batch":
                pass
            else:
                batch = [self.collate_fn(bat) for bat in batch]
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            self.i += 1
            yield batch

    def __next__(self):
        val = self._next()
        return next(val)

    def __len__(self):
        return self.n


# Generate a data loader based on the given parameters
def generate_data_loader(data_generator, data_generator_args, data_loader, data_loader_params, n, seed, device, set_seed=True):
    # from DataGeneration.datagenerator import SpectralReconstructionGenerator

    data_generator = data_generator(
        seed=seed,
        set_seed=set_seed,
        batch_size=data_loader_params['batch_size'],
        device=device,
        **data_generator_args,
    )

    if "status" in data_loader_params:
        data_loader_params["status"] = data_generator_args["data_type"]
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datasets import Dataset
    dataset = Dataset(datagenerator=data_generator, n=n)

    return data_loader(dataset, **data_loader_params)


def data_loader_factory(data_loader_name="default"):
    if data_loader_name == "BatchDataLoader":
        return BatchDataLoader
    else:
        return DataLoader
