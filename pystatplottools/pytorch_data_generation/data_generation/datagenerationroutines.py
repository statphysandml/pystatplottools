import os
import shutil
import json


# Generate a data loader based on the given parameters with an OnTheFlyDataaset
# (samples will be generated in real time and not stored permanently)
def load_in_real_time_data_loader(
        batch_size, data_generator_args, device=None, n=None, seed=0, set_seed=True,
        data_generator_name=None, data_generator_factory=None, data_generator_func=None,
        data_loader_params=None, data_loader_name="DataLoader", raw_samples=False, shuffle=True, num_workers=0):

    data_generator_args, data_loader_params = prepare_data_generator(
        batch_size=batch_size, data_generator_args=data_generator_args,
        data_loader_params=data_loader_params, data_loader_name=data_loader_name,
        raw_samples=raw_samples, shuffle=shuffle, num_workers=num_workers
    )

    data_generator, n = generate_data_generator(
        data_generator_args=data_generator_args, device=device, n=n, seed=seed, set_seed=set_seed,
        data_generator_name=data_generator_name, data_generator_factory=data_generator_factory,
        data_generator_func=data_generator_func
    )

    from pystatplottools.pytorch_data_generation.pytorch_data_utils.datasets import InRealTimeDataset
    dataset = InRealTimeDataset(datagenerator=data_generator, n=n)

    from pystatplottools.pytorch_data_generation.pytorch_data_utils.dataloaders import data_loader_factory
    data_loader = data_loader_factory(data_loader_name=data_loader_name)

    return data_loader(dataset, **data_loader_params)


# Data generator args are or will be stored in the /raw/ directory
# -> Data generation is prepared by storing a config.json file in the /raw/ directory
def prepare_in_memory_dataset(
        root, data_generator_args, batch_size=None, n=None, seed=None, set_seed=True,
        data_generator_name=None, data_generator_factory=None, data_generator_func=None):
    # Generate /raw/ directory for storage of the data_generator_args
    if not os.path.isdir(root + "/raw/"):
        os.makedirs(root + "/raw/")

    data_generator_args, _ = prepare_data_generator(
        batch_size=batch_size, data_generator_args=data_generator_args
    )

    if batch_size is not None:
        data_generator_args["batch_size"] = batch_size

    # To get n and to verify if data_generator can generate batches
    # (skip_loading_data_in_init is an argument to skip loading the data in the data generator)
    datagenerator, n = generate_data_generator(
        data_generator_args={**data_generator_args, "skip_loading_data_in_init": True}, device="cpu", n=n, seed=seed, set_seed=set_seed,
        data_generator_name=data_generator_name, data_generator_factory=data_generator_factory,
        data_generator_func=data_generator_func
    )

    if batch_size is not None and not hasattr(datagenerator, "batch_size"):
        assert False, "Data generator is expected to generate batches instead of single samples. " \
                      "Setting a batch_size is not reasonable in this case. Consider to remove your argument for " \
                      "batch size or define a datagenerator that has an attribute batch_size and generated batches " \
                      "with batch size batch_size"

    data_generator_args = {
        "seed": seed,
        "set_seed": set_seed,
        "n": n,
        **data_generator_args
    }

    new_config_data = {
        "data_generator_name": data_generator_name,
        "data_generator_args": data_generator_args,
    }

    # Check if given data_generator_args are the same as already stored - in this case the data is not generated again
    skip_rebuilding = False
    if os.path.isfile(root + "/raw/config.json"):
        with open(root + "/raw/config.json") as json_file:
            config_data = json.load(json_file)
            if new_config_data == config_data:
                skip_rebuilding = True

    if os.path.isdir(root + "/processed/") and not skip_rebuilding:
        print("Write new data_config into file - Data will be generated based on the new data_config file")
        shutil.rmtree(root + "/processed/")

    with open(root + "/raw/" + 'config.json', 'w') as outfile:
        json.dump(new_config_data, outfile, indent=4)


# Generic data loader class
def load_in_memory_dataset(root, batch_size, data_generator_factory, slices=None, shuffle=True, num_workers=0,
                     rebuild=False, sample_data_generator_name=None):
    assert os.path.isfile(root + "/raw/config.json"), "Cannot load data, data generation config file is not provided."

    # Data loader is generated based on data in memory (or will be generated based on /raw/config.json for the first
    # time and stored afterwards)

    if rebuild and os.path.isdir(root + "/processed/"):
        # Remove processed directory
        shutil.rmtree(root + "/processed/")

    # Load Dataset
    from pystatplottools.pytorch_data_generation.pytorch_data_utils.datasets import InMemoryDataset
    dataset = InMemoryDataset(root=root, sample_data_generator_name=sample_data_generator_name,
                              data_generator_factory=data_generator_factory)

    if slices is not None:
        dataset = dataset[slices[0]:slices[1]]
        dataset.n = slices[1] - slices[0]

    from pystatplottools.pytorch_data_generation.pytorch_data_utils.dataloaders import data_loader_factory
    data_loader_func = data_loader_factory(data_loader_name="DataLoader")  # InMemoryDataset cannot handle .pt loaded data

    data_loader_params = {
        'batch_size': batch_size,  # -> Set since BatchDataLoader is not used
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return data_loader_func(dataset, **data_loader_params)


''' Helper functions '''


def prepare_data_generator(
        batch_size, data_generator_args, data_loader_params=None, data_loader_name="DataLoader",
        raw_samples=False, shuffle=True, num_workers=0):

    # Data loader params
    if data_loader_params is None:
        data_loader_params = {
            'raw_samples': raw_samples,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
    else:
        assert raw_samples is False and shuffle and num_workers == 0,\
            "When data_loader_params are defined, raw_samples, shuffle and num_workers are not taken into account."

    # Remove batch_size in data_loader params
    if "batch_size" in data_generator_args:
        del data_generator_args["batch_size"]
        print(
            "Batch_size argument has been deleted in data_generator_args since this parameter needs to be passed separately for this function")
    if "batch_size" in data_loader_params:
        del data_loader_params["batch_size"]
        print(
            "Batch_size argument has been deleted in data_loader_params since this parameter needs to be passed separately for this function")

    if data_loader_name == "BatchDataLoader":
        data_generator_args["batch_size"] = batch_size
    else:
        data_loader_params["batch_size"] = batch_size

    return data_generator_args, data_loader_params


def generate_data_generator(data_generator_args, device=None, n=None, seed=0, set_seed=True,
                            data_generator_name=None, data_generator_factory=None, data_generator_func=None):

    # Data generator func
    assert (data_generator_name is None and data_generator_func is not None) or (
                data_generator_name is not None and data_generator_func is None), \
        "One and only one of the arguments data_generator_name and data_generator_func is allowed to be defined."

    if data_generator_func is None:
        data_generator_func = data_generator_factory(data_generator_name=data_generator_name)

    data_generator = data_generator_func(
        seed=seed,
        set_seed=set_seed,
        device=device,
        **data_generator_args,
    )

    if n is None:
        try:
            n = len(data_generator)
        except TypeError:
            assert False, "n needs to be defined if the data generator has no __len__() function"
    else:
        try:
            n_actual = len(data_generator)
            if n > n_actual:
                assert False, "n cannot be larger than the number of samples of the generator"
        except TypeError:
            pass

    return data_generator, n
