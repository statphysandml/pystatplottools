import numpy as np
import json
import os
import shutil


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


def data_loader_factory(data_loader_name="default"):
    if data_loader_name == "BatchDataLoader":
        return BatchDataLoader
    else:
        return DataLoader


# Generate a data loader based on the given parameters with an OnTheFlyDataaset
def generate_on_the_fly_data_loader(data_generator, data_generator_args, data_loader, data_loader_params, n, seed, device, set_seed=True):
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
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datasets import OnTheFlyDataset
    dataset = OnTheFlyDataset(datagenerator=data_generator, n=n)

    return data_loader(dataset, **data_loader_params)


# Generic data loader class
def load_data_loader(batch_size, root=None, slices=None, shuffle=True, device=None, num_workers=0, rebuild=False,
                     data_generator_args=None, batch_data_generator=None):
    # batch_data_generator="BatchIsingGraphDataGenerator"

    # Load data generator (samples will be generated in real time) and not stored permanently
    if root is None:  # <-> No information given about where the dataset is stored or where it should be generated
        assert data_generator_args is not None, "Cannot generate dataloader without a root directory or a data_generator_args object"
        data_generator_args["data_type"] = "batch"
        n = data_generator_args["number_of_files"] * data_generator_args["total_number_of_data_per_file"]

        data_loader_params = {
            'status': "batch",
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0, }

        from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory

        data_loader_func = data_loader_factory(data_loader_name="BatchDataLoader")
        data_generator_func = data_generator_factory(data_generator_name=batch_data_generator)

        return generate_on_the_fly_data_loader(
            data_generator=data_generator_func,
            data_generator_args=data_generator_args,
            data_loader=data_loader_func,
            data_loader_params=data_loader_params,
            n=n,
            seed=0,
            device=device
        )

    # Data generator args is stored in the /raw/ directory
    if data_generator_args is not None:
        data_generator_args["data_type"] = "batch"
        n = data_generator_args["number_of_files"] * data_generator_args["total_number_of_data_per_file"]

        # Generate /raw/ directory for storage of the data_generator_args
        if not os.path.isdir(root + "/raw/"):
            os.makedirs(root + "/raw/")

        new_config_data = {
                "data_generator_name": batch_data_generator,
                "data_generator_args": data_generator_args,
                "seed": 0,
                "set_seed": True,
                "batch_size": batch_size,
                "n": n
            }

        # Check if given data_generator_args is the same as already stored - in this case the data doesn't have to be generated again
        skip_rebuilding = False
        if os.path.isfile(root + "/raw/config.json"):
            with open(root + "/raw/config.json") as json_file:
                config_data = json.load(json_file)
                if new_config_data == config_data:
                    skip_rebuilding = True

        if os.path.isdir(root + "/processed/") and not skip_rebuilding:
            print("Write new data_config into file - Data will be generated based on the new data_config file, "
                  "set data_generator_args to None to avoid a reprocessing of the dataset")
            shutil.rmtree(root + "/processed/")

        with open(root + "/raw/" + 'config.json', 'w') as outfile:
            json.dump(new_config_data, outfile, indent=4)

        return load_ising_data_loader(
            root=root,
            batch_size=batch_size,
            slices=None,
            shuffle=True,
            num_workers=0,
            rebuild=False
        )
    else:
        assert os.path.isdir(root + "/raw/"), "Cannot load data, data generation config file or data_generator_args is not provided."

    # Data is loaded based on the given config file from the /raw/ directory
    if rebuild and os.path.isdir(root + "/processed/"):
        # Remove processed directory
        shutil.rmtree(root + "/processed/")

    data_loader_params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers}

    # Load Dataset
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datasets import CustomInMemoryDataset
    dataset = CustomInMemoryDataset(root=root, data_loader_name="BatchIsingDataGenerator")

    if slices is not None:
        dataset = dataset[slices[0]:slices[1]]
        dataset.n = slices[1] - slices[0]

    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.dataloaders import data_loader_factory
    data_loader_func = data_loader_factory(data_loader_name="GeometricDataLoader")
    return data_loader_func(dataset, **data_loader_params)
