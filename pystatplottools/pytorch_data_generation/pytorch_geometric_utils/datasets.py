import torch
import torch_geometric.data as geometric_data


from pystatplottools.pytorch_data_generation.pytorch_data_utils.datasets import InMemoryDatasetBaseClass


class GeometricInMemoryDataset(InMemoryDatasetBaseClass, geometric_data.InMemoryDataset):
    def __init__(self, root, data_generator_factory, sample_data_generator_name=None, transform=None, pre_transform=None, pre_filter=None, plain=False):
        super().initialize(data_generator_factory, root, transform, pre_transform, pre_filter)
        geometric_data.InMemoryDataset.__init__(self, root, transform, pre_transform, pre_filter)

        self.plain = plain

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.n = len(self.data.y)

        if sample_data_generator_name is not None:
            self.sample_data_generator_name = sample_data_generator_name
            self.datagenerator = self.build_datagenerator(self.raw_dir, self.raw_file_names, sample_data_generator_name)
        else:
            self.datagenerator = None

    def download(self):
        pass

    def process(self):
        self.datagenerator = self.build_datagenerator(self.raw_dir, self.raw_file_names)

        data_list = []

        while len(data_list) < self.n:
            data_list += self.datagenerator.sampler().to_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Is it maybe better to collate while generation -> to save memory?
        data, slices = self.collate(data_list[:self.n])

        with open(self.processed_paths[0], 'wb') as f:
            torch.save((data, slices), f)