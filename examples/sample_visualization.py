

if __name__ == '__main__':
    from pystatplottools.pdf_env.loading_figure_mode import loading_figure_mode
    fma, plt = loading_figure_mode(develop=True)
    plt.style.use('seaborn-dark-palette')

    # Load data loader
    from examples.pytorch_in_memory_dataset import prepare_rectangle_data_in_memory_dataset,\
        load_rectangle_data_memory_dataset
    prepare_rectangle_data_in_memory_dataset()
    data_loader = load_rectangle_data_memory_dataset()

    directory = "./results/RectangleData"

    ''' Sample Visualization '''

    from pystatplottools.visualization import sample_visualization

    config_dim = (10, 12)

    # Random sample
    config, label = data_loader.dataset.get_random_sample()

    # Alternative
    # dataset_inspector = data_loader.get_dataset_inspector()
    # config, label = dataset_inspector.sampler()

    # Random batch with size 108
    batch, batch_label = data_loader.dataset.get_random_batch(108)

    # Single Sample
    sample_visualization.fd_im_single_sample(sample=config, label=label, config_dim=config_dim, minmax=(0, 1),
                                             fma=fma, filename="single_sample", directory=directory, figsize=(4, 4))

    # Batch with labels
    sample_visualization.fd_im_batch(batch, batch_labels=batch_label, num_samples=24, dim=(4, 6), config_dim=config_dim,
                                     ab=(batch[:24].cpu().numpy().min(), batch[:24].cpu().numpy().max()), fma=fma,
                                     filename="batch", directory=directory, width=3.0)

    # Batch grid
    sample_visualization.fd_im_batch_grid(batch, config_dim=config_dim, ab=(0, 1),
                                          fma=fma, filename="batch_grid", directory=directory)
