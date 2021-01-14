import numpy as np


from pystatplottools.visualization.utils import figure_decorator


def im_single_sample(ax, sample, config_dim, label=None, minmax=None, ab=None, num_std=None, cmap='viridis', **kwargs):

    if label is not None:
        try:
            import torch
            if isinstance(label, torch.Tensor):
                label = ', '.join(list(label.cpu().numpy().reshape(-1).astype(str)))
            else:
                label = ', '.join(list(np.array(label).reshape(-1).astype(str)))
        except ModuleNotFoundError:
            pass

    if 'norm' not in kwargs.keys() and (minmax is None and ab is None and num_std is None):
        minmax = (0, 1)  # Default

    img = prepare_visualizatiion(
        repr=sample,
        config_dim=config_dim,
        minmax=minmax,
        ab=ab,
        num_std=num_std
    )[0][0]

    if 'norm' in kwargs.keys():
        ax.imshow(img, cmap=cmap, **kwargs)
    else:
        # By prepare_visualizatiion, the data is expected to be in the interval [0, 1]
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0, **kwargs)

    if label is not None:
        from matplotlib.pyplot import Line2D
        custom_lines = [Line2D([0], [0], color="white", lw=4)]
        ax.legend(
            custom_lines, [str(label)], loc="upper right", handlelength=0
        )


@figure_decorator
def fd_im_single_sample(sample, config_dim, label=None, minmax=None, ab=None, num_std=None, cmap='viridis',
                        filename=None, directory=None, title=None, fig=None, ax=None, fma=None, figsize=(10, 7),
                        width=1.3, type="png", ratio=None, **kwargs):
    im_single_sample(ax=ax, sample=sample, config_dim=config_dim, label=label, minmax=minmax, ab=ab, num_std=num_std,
                     cmap=cmap, **kwargs)


def im_batch(ax, batch_repr, config_dim, batch_labels=None, num_samples=None, minmax=None, ab=None, num_std=None,
             dim=(6, 6), cmap='viridis', **kwargs):

    if 'norm' not in kwargs.keys() and (minmax is None and ab is None and num_std is None):
        minmax = (0, 1)  # Default

    batch = prepare_visualizatiion(
        repr=batch_repr,
        config_dim=config_dim,
        num_samples=num_samples,
        minmax=minmax,
        ab=ab,
        num_std=num_std
    )

    if batch_labels is None:
        for idx, dat in enumerate(batch):
            # (Samples are already properly rescaled)
            im_single_sample(ax=ax[np.unravel_index(idx, (dim[0], dim[1]))], sample=dat, config_dim=config_dim,
                             minmax=None, ab=(0, 1), num_std=None, cmap=cmap, **kwargs)
    else:
        for idx, (dat, label) in enumerate(zip(batch, batch_labels)):
            # (Samples are already properly rescaled)
            im_single_sample(ax=ax[np.unravel_index(idx, (dim[0], dim[1]))], sample=dat, config_dim=config_dim,
                             label=label, minmax=None, ab=(0, 1), num_std=None, cmap=cmap, **kwargs)


@figure_decorator
def fd_im_batch(batch_repr, config_dim, batch_labels=None, num_samples=None, minmax=None, ab=None, num_std=None,
                cmap='viridis', filename=None, directory=None, title=None, fig=None, ax=None, fma=None, figsize=(10, 7),
                width=1.3, type="png", ratio=None, dim=(6, 6), **kwargs):
    im_batch(ax=ax, batch_repr=batch_repr, config_dim=config_dim, batch_labels=batch_labels, num_samples=num_samples,
             minmax=minmax, ab=ab, num_std=num_std, dim=dim, cmap=cmap, **kwargs)


def im_batch_grid(ax, batch_repr, config_dim, num_samples=None, nrow=12, minmax=None, ab=None, num_std=None, title=None,
                  **kwargs):

    if 'norm' not in kwargs.keys() and (minmax is None and ab is None and num_std is None):
        minmax = (0, 1)  # Default

    batch = prepare_visualizatiion(
        repr=batch_repr,
        config_dim=config_dim,
        num_samples=num_samples,
        minmax=minmax,
        ab=ab,
        num_std=num_std
    )

    from torchvision import utils
    import torch
    out = utils.make_grid(torch.tensor(batch), nrow=nrow)
    grid = out.cpu().numpy().transpose((1, 2, 0))
    if 'norm' in kwargs.keys():
        ax.imshow(grid, **kwargs)
    else:
        # By prepare_visualizatiion, the data is expected to be in the interval [0, 1]
        ax.imshow(grid, vmin=0.0, vmax=1.0, **kwargs)

    if title is not None:
        ax.set_title(label=title, fontsize=12)


@figure_decorator
def fd_im_batch_grid(batch_repr, config_dim, num_samples=None, nrow=12, minmax=None, ab=None, num_std=None,
                     filename=None, directory=None, title=None, fig=None, ax=None, fma=None, figsize=(10, 7),
                     width=1.3, type="png", ratio=None, **kwargs):
    im_batch_grid(ax=ax, batch_repr=batch_repr, config_dim=config_dim, num_samples=num_samples, nrow=nrow,
                  minmax=minmax, ab=ab, num_std=num_std, title=title, **kwargs)


''' Helper functions '''


def prepare_visualizatiion(repr, config_dim, num_samples=16, minmax=None, ab=None, num_std=None):
    """
    :param repr:
    :param config_dim:
    :param num_samples:
    :param minmax: tuple - Data is scaled from the interval [data.min(), data.max()] to the interval [min, max]
    :param ab: Data is scaled from the interval [a, b] to the interval [0, 1]
    :param num_std: scalar - Data is scaled from the interval
        [data.mean() - num_std, data.mean() + num_std] to the interval [0, 1]. Data outside the range is clipped to 0 and 1
    :return:
    """
    try:
        import torch
        if isinstance(repr, torch.Tensor):
            repr = repr.cpu().numpy()
        else:
            repr = repr
    except ModuleNotFoundError:
        pass

    batch = repr.reshape(-1, 1, config_dim[0], config_dim[1])
    if num_samples is not None:
        batch = batch[:num_samples]

    if num_std is not None:
        assert minmax is None and ab is None, "Only one not None parameter of num_std, ab and minmax is reasonable."
        batch = _rescale_num_std_to_zero_one(
            data=batch, num_std=num_std
        )
    elif ab is not None:
        assert minmax is None and num_std is None, "Only one not None parameter of num_std, ab and minmax is reasonable."
        batch = _a_b_to_zero_one(data=batch, ab=ab)
    elif minmax is not None:
        assert num_std is None and ab is None, "Only one not None parameter of (num_std, ab and minmax) is reasonable."
        batch = _minmax_rescale_data(
            data=batch, minmax=minmax
        )
    return batch


def _a_b_to_zero_one(data, ab):
    a, b = ab
    return (data - a) / (b - a)


def _minmax_rescale_data(data, minmax):
    if data.max() == data.min():
        print("Unique color for data sample - better use ab parameter than minmax.")
        return data
    else:
        return (data - data.min()) / (data.max() - data.min()) * (
                minmax[1] - minmax[0]
        ) + minmax[0]


def _rescale_num_std_to_zero_one(data, num_std):
    mini = data.mean() - num_std * data.std()
    maxi = data.mean() + num_std * data.std()
    rescaled_data = (data - mini) / (maxi - mini)
    return np.clip(rescaled_data, 0.0, 1.0)


# Optional function for converting given output from the train_loader to a corresponding batch

def batch_converter(batch):
    if isinstance(batch, list) or isinstance(batch, tuple):
        batch_y = batch[1]
        batch = batch[0]
    else:
        batch_y = batch.y
    return batch, batch_y
