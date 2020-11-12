import numpy as np
import json
import os
import shutil
from torchvision import utils

import torch
from torch_geometric.data import Data, Batch


from pystatplottools.ppd_pytorch_data_generation.ising_model.batchisingdatagenerator import BatchIsingDataGenerator


class BatchIsingGraphDataGenerator(BatchIsingDataGenerator):
    def __init__(self, **kwargs):  # points_G, points_rho, noise_with = 0.00001, a_min_bw = 1.0, ):
        kwargs["data_type"] = "target_param"
        super().__init__(**kwargs)
        self.dimensions = kwargs.pop("dimensions")
        self.config_size = len(self.data["Config"].iat[0])
        self.self_loops = True
        self.edge_indices = self.generate_edge_indices()
        self.num_features = 1

    def get_config_dimensions(self):
        return self.dimensions

    def sample_target_param(self):
        config, beta = super().sample_target_param()
        config = torch.tensor(config, dtype=torch.float32, device=self.device)
        config = Batch.from_data_list([Data(x=conf, edge_index=self.edge_indices, y=float(bet[0])) for (conf, bet) in zip(config, beta)])
        config.x = config.x.view(-1, self.num_features)
        return config

    def generate_edge_indices(self):
        node_indices = np.arange(self.config_size)
        edge_indices = []
        for i in node_indices:
            offset = 1
            # std::vector < T* > nn_of_site;
            # // std::cout << "i: " << i << std::endl;

            for dim in self.dimensions:
                # print(i-i%(offset * dim)+(i+offset) % (offset * dim), " - ", i-i%(offset * dim)+(i-offset+offset * dim)%(offset * dim))
                edge_indices.append([node_indices[i], node_indices[i - i % (offset * dim) + (i + offset) % (offset * dim)]])
                edge_indices.append([node_indices[i], i - i % (offset * dim) + (i - offset + offset * dim) % (offset * dim)])
                offset = offset * dim
            if self.self_loops:
                edge_indices.append([i, i])
        edge_indices = np.array(edge_indices).T
        return torch.tensor(edge_indices, dtype=torch.long, device=self.device)

    def convert_to_config_data(self, batch):
        config = batch.x.view(-1, np.prod(self.dimensions))
        beta = batch.y.to(config.device)
        return config, beta


def load_ising_graph_data_loader(batch_size, slices=None, shuffle=True, device=None, num_workers=0, rebuild=False, data_generator_args=None,
                                 root="./../../../data/IsingGraph"):
    # Load data generator (samples will be generated in real time) and not stored permanently
    if root is None:
        assert data_generator_args is not None, "Cannot generate dataloader without a root directory or a data_generator_args object"
        data_generator_args["data_type"] = "batch"
        n = data_generator_args["number_of_files"] * data_generator_args["total_number_of_data_per_file"]

        data_loader_params = {
            'status': "batch",
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0, }

        from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.dataloaders import generate_data_loader, \
            data_loader_factory
        from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory

        data_loader_func = data_loader_factory(data_loader_name="BatchDataLoader")
        data_generator_func = data_generator_factory(data_generator_name="BatchIsingGraphDataGenerator")

        return generate_data_loader(
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

        if not os.path.isdir(root + "/raw/"):
            os.makedirs(root + "/raw/")

        new_config_data = {
                "data_generator_name": "BatchIsingGraphDataGenerator",
                "data_generator_args": data_generator_args,
                "seed": 0,
                "set_seed": True,
                "batch_size": batch_size,
                "n": n
            }

        skip_rebuilding = False

        # Check if given data_generator_args is the same as already stored - in this case the data doesn't have to be generated again
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

        return load_ising_graph_data_loader(
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
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datasets import GeometricInMemoryDataset
    dataset = GeometricInMemoryDataset(root=root, data_loader_name="BatchIsingGraphDataGenerator")

    if slices is not None:
        dataset = dataset[slices[0]:slices[1]]
        dataset.n = slices[1] - slices[0]

    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.dataloaders import data_loader_factory
    data_loader_func = data_loader_factory(data_loader_name="GeometricDataLoader")
    return data_loader_func(dataset, **data_loader_params)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N_per_temperature = 1000  # May be chosen not too large for an in memory dataset
    number_of_files = 21

    data_generator_args = {
        "path": "/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/",
        "chunksize": N_per_temperature,
        "total_number_of_data_per_file": N_per_temperature,
        "number_of_files": 21,
        "dimensions": [16, 16]
    }

    # # Generate dataset
    train_loader = load_ising_graph_data_loader(data_generator_args=data_generator_args, batch_size=128,
                                                device=device, root="./../../../data/IsingGraph")

    # Load dataset
    # train_loader = load_ising_graph_data_loader(data_generator_args=None, batch_size=128,
    #                                                 device=device, root="./../../../data/IsingGraph")

    dataset_inspector = train_loader.get_dataset_inspector()
    config_dim = dataset_inspector.get_config_dimensions()

    # Plot single sample
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    batch = dataset_inspector.sample_target_param()
    config, beta = dataset_inspector.convert_to_config_data(batch)
    dataset_inspector.im_single_config(ax=ax, tensor_dat=config[0], config_dim=config_dim)

    plt.show()

    # Plot a bunch of samples
    dataset_inspector.im_batch(config[:36], config_dim=config_dim)

    # Plot a bunch of samples as a grid
    dataset_inspector.im_batch_grid(config, config_dim=config_dim)

    # from program.plain_networks import GCNConvNet
    # model = GCNConvNet().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    # import torch.nn.functional as F
    # model.train()
    # for epoch in range(200):
    #     for batch, target in train_loader:
    #         batch = batch.to(device)
    #         optimizer.zero_grad()
    #         # batch.x = torch.cat([batch.x, -batch.x], dim=1)
    #         out = model(batch)
    #         loss = F.nll_loss(out, torch.zeros(len(batch.x), dtype=torch.long, device=device))
    #         loss.backward()
    #         optimizer.step()
