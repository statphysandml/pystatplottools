

def prepare_rectangle_data_in_memory_dataset():
    rel_gen_data_path = "./data/RectangleData"

    ''' Data generation with storage of a permament file '''

    data_generator_args = {
        # RectangleGenerator Args
        "dim": [10, 12]
    }
    # Prepare in memory dataset
    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import prepare_in_memory_dataset
    from examples.rectangle_data_generator import data_generator_factory

    # Write a config.json file providing all necessary information for generating the
    # dataset
    prepare_in_memory_dataset(
        root=rel_gen_data_path,
        batch_size=89,
        n=50000,
        data_generator_args=data_generator_args,
        data_generator_name="BatchRectangleGenerator",
        data_generator_factory=data_generator_factory
    )


def load_rectangle_data_memory_dataset():
    # Load the in memory dataset
    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import load_in_memory_dataset
    from examples.rectangle_data_generator import data_generator_factory
    # When this function is called the first time, the dataset is generated, otherwise, only loaded
    dataset = load_in_memory_dataset(
        root="./data/RectangleData", data_generator_factory=data_generator_factory,
        rebuild=False,
        sample_data_generator_name=None,  # "RectangleGenerator"  # optional: for a generation of new samples
    )
    return dataset


if __name__ == '__main__':
    from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
    fma, plt = loading_figure_mode(develop=True)
    plt.style.use('seaborn-dark-palette')

    directory = "./results/RectangleData"

    ''' In Memory Data Loader '''

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prepare_rectangle_data_in_memory_dataset()
    dataset = load_rectangle_data_memory_dataset()

    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import load_in_memory_data_loader
    data_loader = load_in_memory_data_loader(dataset=dataset, batch_size=120, slices=(0, 4000), shuffle=True,
                                             num_workers=10)

    # Load training data
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(target))
        if batch_idx == 0:  # Useful for verifying the shuffle parameter of the data loader
            print(data)

    # Load training data - Second epoch
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(target))
        if batch_idx == 0:
            print(data)
