import numpy as np


from examples.rectangle_data_generator import RectangleGenerator, data_generator_factory


if __name__ == '__main__':

    ''' Normal DataLoader and RectangleGenerator with an InRealTimeDataset '''

    data_generator = RectangleGenerator(
        # DataGeneratoreBaseClass Args
        seed=None,
        set_seed=True,
        # RectangleGenerator Args
        dim=(10, 12)
    )

    # Possible usage:
    # sample = data_generator.sampler()

    from pystatplottools.pytorch_data_generation.pytorch_data_utils.datasets import InRealTimeDataset
    dataset = InRealTimeDataset(datagenerator=data_generator, n=10000)

    from torch.utils import data
    data_loader_params = {
        'batch_size': 512,
        'shuffle': True,
        'num_workers': 0
    }
    data_loader = data.DataLoader(dataset=dataset, **data_loader_params)

    import time
    t = time.time()

    # Load training data
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))

    # Load training data - Second epoch - Different/New samples in each epoch!
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))

    elapsed_time = np.round_(time.time() - t, 3)

    ''' Same as above with the BatchRectangleGenerator and the load_in_real_time_data_loader '''

    data_generator_args = {
        # RectangleGenerator Args
        "dim": (10, 12)
    }

    data_loader_params = {
        # 'raw_samples': True,  # Returns the sample from the BatchDataGenerator as it is. Can be used if the BatchDataGenerator produces already the correct datatype for training
        'shuffle': True,  # Used correctly by the Dataloader??
        'num_workers': 0}

    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import load_in_real_time_dataset
    dataset = load_in_real_time_dataset(
        data_generator_args=data_generator_args,
        batch_size=512,
        data_generator_name="BatchRectangleGenerator",
        data_generator_factory=data_generator_factory,
        seed=None,
        set_seed=True,
        n=10000
    )

    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import load_in_real_time_data_loader
    data_loader = load_in_real_time_data_loader(
        dataset=dataset,
        data_loader_params=data_loader_params,
        data_loader_name="BatchDataLoader"
    )

    t = time.time()

    # Load training data
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))

    # Load training data - Second epoch - Different/New samples in each epoch!
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))

    elapsed_time_batch = np.round_(time.time() - t, 3)

    print("RectangleGenerator:", elapsed_time, " sec., BatchRectangleGenerator:", elapsed_time_batch, " sec.")