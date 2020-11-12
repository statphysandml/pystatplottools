import os
import shutil
import json


from pystatplottools.ppd_pytorch_data_generation.ising_model.isingdatagenerator import IsingDataGenerator


class BatchIsingDataGenerator(IsingDataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_target_config(self):
        if self.iterator >= len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection()  # load data

        self.iterator += self.batch_size

        if self.iterator > len(self.data):
            top = len(self.data)
        else:
            top = self.iterator

        return list(self.data["Beta"].iloc[self.iterator - self.batch_size:top].values.reshape((-1, 1))), \
               list(self.data["Config"].iloc[self.iterator - self.batch_size:top])

    def sample_target_param(self):
        if self.iterator >= len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection()  # load data

        self.iterator += self.batch_size

        if self.iterator > len(self.data):
            self.iterator = len(self.data)

        return list(self.data["Config"].iloc[self.iterator - self.batch_size:self.iterator]), \
               list(self.data["Beta"].iloc[self.iterator - self.batch_size:self.iterator].values.reshape((-1, 1)))


def generate_ising_data_loader(batch_size, N_per_temperature, device):
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.dataloaders import generate_data_loader

    n = 21 * N_per_temperature

    data_loader_params = {'batch_size': batch_size,
                          'shuffle': True,
                          'num_workers': 0}

    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory
    from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.dataloaders import data_loader_factory
    data_loader_func = data_loader_factory(data_loader_name="BatchDataLoader")
    data_generator_func = data_generator_factory(data_generator_name="BatchIsingDataGenerator")

    data_generator_args = {
        "data_type": "target_param",
        "path": "/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/",
        "chunksize": N_per_temperature,
        "total_number_of_data_per_file": N_per_temperature,
    }

    return generate_data_loader(
        data_generator=data_generator_func,
        data_generator_args=data_generator_args,
        data_loader=data_loader_func,
        data_loader_params=data_loader_params,
        n=n,
        seed=0,
        device=device
    )


def load_ising_data_loader(batch_size, slices=None, shuffle=True, device=None, num_workers=0, rebuild=False, data_generator_args=None,
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


if __name__ == '__main__':
    # pass
    from pystatplottools.ppd_utils.utils import set_up_directories
    #
    data_root, results_root = set_up_directories(data_dir="IsingModel", results_dir="IsingModel")
    #
    from pystatplottools.ppd_utils.utils import device
    #
    device = device()
    #
    N_per_temperature = 1000  # May be chosen not too large for an in memory dataset
    number_of_files = 11
    #
    # data_generator_args = {
    #     "path": "/home/lukas/extrapolationphasestructure/Lukas/Code/data/IsingModelMetropolis",
    #     "chunksize": N_per_temperature,
    #     "total_number_of_data_per_file": N_per_temperature,
    #     "number_of_files": 11,
    #     "dimensions": [4, 4]
    # }
    #
    # train_loader = load_is(data_generator_args=data_generator_args, batch_size=128,
    #                        device=device, root=data_root)
