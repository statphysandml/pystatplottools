import os
from abc import ABC

import torch
import json

import torch.utils.data as torch_data
from torch.utils.data import _utils
default_collate = _utils.collate.default_collate
import numpy as np

import torch_geometric.data as geometric_data


'''Particular class that overloads the dataset generator of pytorch - for a better performance during training'''

# Only used for a real time sampling -> True?
class Dataset(torch_data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, datagenerator, n):
        super().__init__()
        # Number of samples to be generated
        self.n = n

        # Auxiliary datagenerator
        self.datagenerator = datagenerator

    def __len__(self):
        # Denotes the total number of samples
        return self.n

    def __getitem__(self, index):
        # Generates one sample of data
        return self.datagenerator.sampler()

    def get_datagenerator(self):
        return self.datagenerator


class InMemoryDataset(object):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.root = root
        self.n = None
        self.datagenerator = None

    @property
    def raw_file_names(self):
        return "config.json"

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def __len__(self):
        # Denotes the total number of samples
        return self.n

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.processed_file_names))

    def get_datagenerator(self):
        return self.datagenerator

    def build_datagenerator(self, data_loader_name):
        with open(self.raw_dir + "/" + self.raw_file_names) as json_file:
            config_data = json.load(json_file)
            config_data["data_generator_args"]["data_type"] = "batch"
            self.n = config_data["n"]

            from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory
            # Auxiliary datagenerator
            data_generator = data_generator_factory(data_generator_name=data_loader_name)
            self.datagenerator = data_generator(
                seed=config_data["seed"],
                set_seed=config_data["set_seed"],
                batch_size=config_data["batch_size"],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **config_data["data_generator_args"],
            )


class GeometricInMemoryDataset(InMemoryDataset, geometric_data.InMemoryDataset):
    def __init__(self, root, data_loader_name, transform=None, pre_transform=None, pre_filter=None, plain=False):
        self.data_generator_name = data_loader_name

        super().__init__(root, transform, pre_transform, pre_filter)

        self.plain = plain
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.datagenerator is None:
            self.build_datagenerator(self.data_generator_name)

    def download(self):
        pass

    def process(self):
        self.build_datagenerator(self.data_generator_name)

        data_list = []
        for _ in range(int(self.n / self.datagenerator.batch_size)):
            data_list += self.datagenerator.sampler().to_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CustomInMemoryDataset(InMemoryDataset, torch_data.Dataset):
    def __init__(self, root, data_loader_name, transform=None, pre_transform=None, pre_filter=None, plain=False):
        # Number of samples to be generated

        super().__init__(root, transform, pre_transform, pre_filter)

        self.plain = plain
        self.data_generator_name = data_loader_name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.process()

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, self.processed_file_names))

        if self.datagenerator is None:
            self.build_datagenerator(self.data_generator_name)

    def __getitem__(self, index):
        if self.plain:
            return self.data[index].view(-1), self.targets[index]
        else:
            return self.data[index], self.targets[index]

    def get_plain_data(self):
        return self.data.view(self.n, -1)

    def get_random_batch(self, batch_size):
        rn = np.random.choice(self.n, batch_size)
        return self.data[rn], self.targets[rn]

    def get_random_sample(self):
        rn = np.random.randint(0, self.n)
        return self.__getitem__(rn)

    def download(self):
        pass

    def process(self):
        if self._check_exists():
            return

        # os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing...')

        self.build_datagenerator(self.data_generator_name)

        data_list = []
        target_list = []
        # For BatchDataGenerators (the situation might be different for others!!)
        for _ in range(int((self.n + self.datagenerator.batch_size) / self.datagenerator.batch_size)):
            sample, target = self.datagenerator.sampler()
            data_list.append(sample)  # [s for s in sample]
            target_list.append(target)  # [t for t in target]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data_list = torch.tensor(data_list, dtype=torch.float32, device="cpu")
        self.data = torch.cat(data_list)[:self.n]
        self.targets = torch.cat(target_list)[:self.n]

        with open(os.path.join(self.processed_folder, self.processed_file_names), 'wb') as f:
            torch.save((self.data, self.targets), f)

        print('Done!')
