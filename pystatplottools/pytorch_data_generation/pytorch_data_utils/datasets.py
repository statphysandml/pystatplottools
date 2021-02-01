import os
import os.path as osp

import torch
import json

import torch.utils.data as torch_data
from torch.utils.data import _utils
default_collate = _utils.collate.default_collate
import numpy as np


from pystatplottools.utils.multiple_inheritance_base_class import MHBC


'''
Three datasets:
- InRealTimeDataset for the generation in real time
- InMemoryDataset
- GeometricInMemoryDataset (see pytorch_geometric_utils.datasets)
'''


# Base Class for InMemoryDatasets
class InMemoryDatasetBaseClass(MHBC):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        # For child classes with more than one parent class
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

    # Can be used as a replacement for __init__ if base constructors do not coincide in the case of multiple inheritance
    def initialize(self, data_generator_factory, root=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.n = None
        self.data_generator_factory = data_generator_factory

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

    # def get_datagenerator(self):
    #     return self.datagenerator

    def build_datagenerator(self, raw_dir, raw_file_name, data_generator_name=None):
        with open(raw_dir + "/" + raw_file_name) as json_file:
            config_data = json.load(json_file)

            if data_generator_name is None:
                # Stored data generator used for generation
                data_generator = self.data_generator_factory(data_generator_name=config_data["data_generator_name"])
            else:
                # Enables using a different data generator for testing and evaluation purposes - not generation
                data_generator = self.data_generator_factory(data_generator_name=data_generator_name)

            if "n" in config_data["data_generator_args"].keys():
                self.n = config_data["data_generator_args"]["n"]
            else:
                self.n = len(self.datagenerator)

            # Auxiliary datagenerator
            return data_generator(
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **config_data["data_generator_args"],
            )


# Loads dataset from file if generated or generates in the process function
# -> in the latter case, the datagenerator needs a sampler function
class InMemoryDataset(torch_data.Dataset, InMemoryDatasetBaseClass):
    def __init__(self, root, data_generator_factory, sample_data_generator_name=None, transform=None, pre_transform=None, pre_filter=None, plain=False, collate_fn=default_collate):
        super().initialize(data_generator_factory, root, transform, pre_transform, pre_filter)

        self.plain = plain
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.collate_fn = collate_fn

        self.process()

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, self.processed_file_names))

        self.n = len(self.data)

        if sample_data_generator_name is not None:
            self.sample_data_generator_name = sample_data_generator_name
            self.datagenerator = self.build_datagenerator(self.raw_dir, self.raw_file_names, sample_data_generator_name)
        else:
            self.datagenerator = None

    def __getitem__(self, index):
        if self.plain:
            return self.data[index].view(-1), self.targets[index]
        else:
            return self.data[index], self.targets[index]

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

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

        datagenerator = self.build_datagenerator(self.raw_dir, self.raw_file_names)

        data_list = []
        target_list = []

        iterator = 0
        while len(data_list) < self.n:
            sample, target = datagenerator.sampler()
            if isinstance(sample, list) or hasattr(datagenerator, "batch_size"):
                data_list += [self.collate_fn(sampl) for sampl in sample]
                target_list += [self.collate_fn(targ) for targ in target]
                iterator += len(sample)
            else:
                data_list.append(self.collate_fn(sample))
                target_list.append(self.collate_fn(target))
                iterator += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data = torch.stack(data_list[:self.n])
        self.targets = torch.stack(target_list[:self.n])

        with open(os.path.join(self.processed_folder, self.processed_file_names), 'wb') as f:
            torch.save((self.data, self.targets), f)

        print('Done!')


'''Particular class that overloads the dataset generator of pytorch - for a better performance during training'''


# Only used for a real time sampling -> True?
class InRealTimeDataset(torch_data.Dataset):
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