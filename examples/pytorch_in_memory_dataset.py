

def prepare_rectangle_data_in_memory_dataset():
    rect_data_dir = "RectangleData"
    root = "./data/RectangleData"

    from pystatplottools.utils.utils import set_up_directories
    data_root, results_root = set_up_directories(data_dir=rect_data_dir, results_dir=rect_data_dir)

    ''' Data generation with storage of a permament file '''

    data_generator_args = {
        # RectangleGenerator Args
        "dim": [10, 12]
    }
    # Prepare in memory dataset
    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import prepare_in_memory_dataset
    from examples.rectangle_data_generator import data_generator_factory

    prepare_in_memory_dataset(
        root=root,
        batch_size=89,
        n=50000,
        data_generator_args=data_generator_args,
        data_generator_name="BatchRectangleGenerator",
        data_generator_factory=data_generator_factory
    )


def load_rectangle_data_memory_dataset():
    # Load in memory dataset
    from pystatplottools.pytorch_data_generation.data_generation.datagenerationroutines import load_in_memory_dataset
    from examples.rectangle_data_generator import data_generator_factory
    data_loader = load_in_memory_dataset(
        root="./data/RectangleData", batch_size=89, data_generator_factory=data_generator_factory, slices=None, shuffle=True,
        num_workers=0, rebuild=False
        # sample_data_generator_name="RectangleGenerator"  # optional: for a generation of new samples
    )
    return data_loader


if __name__ == '__main__':
    from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
    fma, plt = loading_figure_mode(develop=True)
    plt.style.use('seaborn-dark-palette')

    directory = "./results/RectangleData"

    ''' In Memory Data Loader '''

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prepare_rectangle_data_in_memory_dataset()
    data_loader = load_rectangle_data_memory_dataset()

    # Load training data
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))

    # Load training data - Second epoch
    for batch_idx, batch in enumerate(data_loader):
        data, target = batch
        print(batch_idx, len(data))
