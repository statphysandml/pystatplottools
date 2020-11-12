import glob, os
import pandas as pd
import numpy as np

from pystatplottools.ppd_pytorch_data_generation.data_generation_base_classes.datageneratorbaseclass import DataGeneratorBaseClass
from pystatplottools.ppd_loading.loading import ConfigurationLoader

# Note that the class currently works only if the number of fixed parameters equals for all types of the same function


class IsingDataGenerator(ConfigurationLoader, DataGeneratorBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.iterator = 0

        self.data_type = kwargs.pop("data_type")

        if self.data_type == "target_config":
            self.inp_size = 1
            self.tar_size = len(self.data["Config"].iat[0])
            self.sampler = self.sample_target_config
        elif self.data_type == "target_param":
            self.inp_size = len(self.data["Config"].iat[0])
            self.tar_size = 1
            self.sampler = self.sample_target_param

        self.inspect = lambda axes, net, data_loader, device: self.inspect_magnetization(axes=axes, net=net,
                                                                                         data_loader=data_loader,
                                                                                         device=device)

    def sample_target_config(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection()  # load data

        self.iterator += 1
        return np.array([self.data["Beta"].iloc[self.iterator - 1]]), self.data["Config"].iloc[self.iterator - 1]

    def sample_target_param(self):
        if self.iterator == len(self.data):
            self.iterator = 0  # Reset iterator
            self.data = self.get_next_chunk_collection()  # load data

        self.iterator += 1
        return self.data["Config"].iloc[self.iterator - 1], np.array([self.data["Beta"].iloc[self.iterator - 1]])

    @staticmethod
    def retrieve_data(data_loader, device='cpu', net=None):
        beta_data = []
        config_data = []
        mean = []

        # from NetworkModels.inn import INN
        # if net is not None and isinstance(net, INN):
        #     import torch
        #     num = 400
        #     for i in range(100):
        #         betas = torch.tensor(np.transpose([np.repeat(np.linspace(0.1, 0.7, 25, dtype=np.float32), num)]))
        #         betas = betas.to(device)
        #         output = net.backward_step(betas)
        #         config_data += list(output[:, :16].detach().cpu().numpy())
        #         mean += list(np.mean(np.sign(output[:, :16].detach().cpu().numpy()), axis=1))
        #         # config_data += list(np.mean(output[:, :16].detach().cpu().numpy(), axis=1))
        #         beta_data += [beta[0].detach().cpu().numpy() for beta in betas]
        #
        #     # for batch_idx, (beta, config) in enumerate(data_loader):
        #     #     beta, config = beta.to(device), config.to(device)
        #     #
        #     #
        #     #     config_data.append(np.mean(output[0][:16].detach().cpu().numpy()))
        #     #     beta_data.append(beta[0].detach().cpu().numpy()[0])
        # else:
        for batch_idx, (betas, config) in enumerate(data_loader):
            config_data += [conf.detach().numpy() for conf in config]
            mean += [np.mean(conf.detach().numpy()) for conf in config]
            beta_data += [beta[0].detach().numpy() for beta in betas]

        config_data = np.array(config_data)
        beta_data = np.array(beta_data)
        mean = np.array(mean)
        beta_data = np.array([f"{bet:.3f}" for bet in beta_data])

        config_data = np.sign(config_data)

        absmean = np.abs(np.mean(np.sign(config_data), axis=1))
        # Todo: Add a computation of the Energy based on nearest neighbour

        data = pd.DataFrame(
            {"beta": beta_data, "Beta": np.array(beta_data, dtype=np.float32), "Mean": mean, "AbsMean": absmean})

        # Add AbsMean, Energy, and Higher Moments

        data.index.name = "Num"
        data.set_index(["beta"], append=True, inplace=True)
        data = data.reorder_levels(["beta", "Num"])
        data.sort_values(by="beta", inplace=True)
        return data, config_data

    # @staticmethod
    # def inspect_magnetization(axes, data_loader, device='cpu', net=None):
    #     from program.plot_routines.distribution1D import Distribution1D
    #     data, _ = IsingDataGenerator.retrieve_data(data_loader=data_loader, device=device, net=net)
    #     dist1d = Distribution1D(data=data)
    #     dist1d.compute_histograms(columns=['Mean'], kind="probability_dist")
    #
    #     from program.plot_routines.contour2D import add_fancy_box
    #
    #     for i, idx in enumerate(data.index.unique(0)):
    #         histodat = dist1d.histograms[idx]['Mean']
    #         Distribution1D.plot_histogram(ax=axes[int(i * 1.0 / 7)][i % 7], **histodat)
    #         add_fancy_box(axes[int(i * 1.0 / 7)][i % 7], idx)
    #
    #     import matplotlib.pyplot as plt
    #     plt.tight_layout()

    # @staticmethod
    # def inspect_observables(axes, data_loader, device='cpu', net=None):
    #     # data, filenames = IsingDataGenerator.load_all_configurations(
    #     #     "/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/HeatbathSmall")
    #     # data = pd.concat(data, keys=[f"{float(file[file.find('=') + 1:-4]):.3f}" for file in filenames])
    #     # data = data.sort_index(level=0)
    #
    #     data, config_data = IsingDataGenerator.retrieve_data(data_loader=data_loader, device=device, net=net)
    #
    #     from program.plot_routines.distribution1D import Distribution1D
    #     dist1d = Distribution1D(data=data)
    #
    #     dist1d.compute_expectation_values(columns=['Mean', 'AbsMean'],  # , 'Energy'],
    #                                       exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    #     dist1d.compute_expectation_values(columns=['Beta'], exp_values=['mean'])
    #
    #     from program.plot_routines.distribution1D import compute_binder_cumulant
    #     # compute_specificheat(dist=dist1d, N=16)
    #     compute_binder_cumulant(dist=dist1d)
    #
    #     betas = dist1d.expectation_values['Beta']['mean']
    #
    #     axes[0][0].set_xlabel("$\\beta")
    #     axes[0][1].set_xlabel("$\\beta")
    #     axes[1][0].set_xlabel("$\\beta")
    #     axes[1][1].set_xlabel("$\\beta")
    #
    #     axes[0][0].set_ylabel("$\langle |m| \\rangle$")
    #     axes[0][1].set_ylabel("$c$")
    #     axes[1][0].set_ylabel("$U_L$")
    #     axes[1][1].set_ylabel("$\langle E \\rangle$")
    #
    #     axes[0][0].plot(betas, dist1d.expectation_values.loc[:, 'AbsMean']['mean'])
    #     axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'Mean']['mean'])
    #     # axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'SpecificHeat']['mean'].values)
    #     axes[1][0].plot(betas, dist1d.expectation_values.loc[:, 'BinderCumulant']['mean'])
    #     # axes[1][1].plot(betas, dist1d.expectation_values.loc[:, 'Energy']['mean'])
    #
    #     import matplotlib.pyplot as plt
    #     plt.tight_layout()

    @staticmethod
    def im_batch(data, config_dim, dim=(6, 6), fig=None, axes=None):
        import matplotlib.pyplot as plt
        if fig is None:
            ownfig = True
            fig, axes = plt.subplots(dim[0], dim[1], figsize=(10, 10))
            assert len(data) <= dim[0] * dim[1], "Number of samples should be smaller than or equal to the dimension product for a nice visualization"
        else:
            ownfig = False

        for idx, dat in enumerate(data):
            IsingDataGenerator.im_single_config(ax=axes[np.unravel_index(idx, (dim[0], dim[1]))], tensor_dat=dat,
                                                config_dim=config_dim)

        if ownfig:
            plt.tight_layout()
            plt.show()
        return fig, axes

    @staticmethod
    def im_batch_grid(batch, config_dim, fig=None, ax=None):
        import matplotlib.pyplot as plt
        if fig is None:
            ownfig = True
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ownfig = False

        batc = batch.data.view(-1, 1, config_dim[0], config_dim[0])
        from torchvision import utils
        out = utils.make_grid(batc)
        inp = out.cpu().numpy().transpose((1, 2, 0))
        ax.imshow(inp)

        if ownfig:
            plt.tight_layout()
            plt.show()
        return fig, ax

    @staticmethod
    def im_single_config(ax, tensor_dat, config_dim):
        inp = tensor_dat.cpu().numpy().reshape(config_dim[0], config_dim[1])
        ax.imshow(inp)


if __name__ == '__main__':
    pass

    # data_generator = IsingDataGenerator(
    #     data_type="target_config",
    #     path="/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/Heatbath",
    #     chunksize=500,
    #     total_number_of_data_per_file=200000,
    #     set_seed=True,
    #     seed=None
    # )
    #
    # print(data_generator.sampler())
    #

    ''' Example for loading a data_loader for Ising configurations that starts from a certain chunk '''

    # from DataGeneration.datasetgenerator import generate_data_loader
    #
    # n = 5000000
    #
    # # The chunksize might be chosen to be batchsize/number of files -> for each batch a new chunk is loaded per reader
    # # Example: 25 readers according to 25 files -> with a chunksize of 40 , one obtains in total 1000 samples per chunkstep
    #
    # data_generator_args = {
    #     "data_type": "target_config",
    #     "path": "/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/HeatbathSmall",
    #     "chunksize": 40,
    #     "total_number_of_data_per_file": 20000,
    #     "current_chunk_iterator_val": 2
    # }
    #
    # data_loader = generate_data_loader(
    #     data_generator=IsingDataGenerator,
    #     data_generator_args=data_generator_args,
    #     data_loader_params={'batch_size': 10,
    #                         'shuffle': False,
    #                         'num_workers': 0},
    #     n=n,
    #     seed=0,
    #     set_seed=False)
    #
    # data_points = []
    # data_colors = []
    #
    # for batch_idx, (beta, config) in enumerate(data_loader):
    #     # data_points += [coor.detach().numpy() for coor in coordinates]
    #     # data_colors += [color[0].detach().numpy() for color in colors]
    #     print(batch_idx)
    #     # print(beta, config)

    ''' Newly commented'''

    # from program.data_generation.data_generation_base_classes.dataloaders import generate_data_loader
    # # from program.data_generation.ising_model.isingdatagenerator import IsingDataGenerator
    #
    # n = 21 * 10000
    #
    # data_loader_params = {'batch_size': 1000,
    #     'shuffle': True,
    #     'num_workers': 0}
    #
    # from program.data_generation.data_generation_base_classes.datageneratorbaseclass import data_generator_factory
    # from program.data_generation.data_generation_base_classes.dataloaders import data_loader_factory
    #
    # data_loader_func = data_loader_factory(data_loader_name="BatchDataLoader")
    # data_generator_func = data_generator_factory(data_generator_name="BatchIsingDataGenerator")
    #
    # data_generator_args = {
    #     "data_type": "target_config",
    #     "path": "/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/",
    #     "chunksize": 10000,
    #     "total_number_of_data_per_file": 10000,
    #     "number_of_files": 21
    #     # "batch_size": 10
    # }
    #
    # data_loader = generate_data_loader(
    #     data_generator=data_generator_func,
    #     data_generator_args=data_generator_args,
    #     data_loader=data_loader_func,
    #     data_loader_params=data_loader_params,
    #     n=n,
    #     seed=0,
    #     device="cpu"
    # )
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(3, 7, figsize=(14, 7))
    # IsingDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    # plt.show()
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(2, 2)
    #
    # IsingDataGenerator.inspect_observables(axes=axes, data_loader=data_loader)
    #
    # plt.show()

    ''' Example for computing and plotting observables '''

    # Computing observables

    # data, filenames = IsingDataGenerator.load_all_configurations("/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/")
    # data = pd.concat(data, keys=[f"{float(file[file.find('=') + 1:-4]):.3f}" for file in filenames])
    # data = data.sort_index(level=0)
    #
    # from ising_model_data_loader.PlotRoutines.distribution1D import Distribution1D
    # dist1d = Distribution1D(data=data)
    #
    # dist1d.compute_expectation_values(columns=['Mean', 'AbsMean', 'Energy'],
    #                                   exp_values=['mean', 'max', 'min', 'secondMoment', 'fourthMoment'])
    # dist1d.compute_expectation_values(columns=['Beta'], exp_values=['mean'])
    #
    # from ising_model_data_loader.PlotRoutines.distribution1D import compute_specificheat, compute_binder_cumulant
    # compute_specificheat(dist=dist1d, N=16)
    # compute_binder_cumulant(dist=dist1d)
    #
    # from ising_model_data_loader.PlotRoutines.plotting_environment.loading_figure_mode import loading_figure_mode
    # fma = loading_figure_mode("saving")
    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-dark-palette')
    #
    # # Plot observables
    #
    # fig, axes = fma.newfig(1.7, nrows=2, ncols=2)
    #
    # betas = dist1d.expectation_values['Beta']['mean']
    #
    # axes[0][0].set_xlabel("$\\beta")
    # axes[0][1].set_xlabel("$\\beta")
    # axes[1][0].set_xlabel("$\\beta")
    # axes[1][1].set_xlabel("$\\beta")
    #
    # axes[0][0].set_ylabel("$\langle |m| \\rangle$")
    # axes[0][1].set_ylabel("$c$")
    # axes[1][0].set_ylabel("$U_L$")
    # axes[1][1].set_ylabel("$\langle E \\rangle$")
    #
    # axes[0][0].plot(betas, dist1d.expectation_values.loc[:, 'AbsMean']['mean'])
    # axes[0][1].plot(betas, dist1d.expectation_values.loc[:, 'SpecificHeat']['mean'].values)
    # axes[1][0].plot(betas, dist1d.expectation_values.loc[:, 'BinderCumulant']['mean'])
    # axes[1][1].plot(betas, dist1d.expectation_values.loc[:, 'Energy']['mean'])
    #
    # plt.tight_layout()
    #
    # fma.savefig("./../Examples/", "observables")
    #
    # # Example for computing and plotting histrograms for the resulting distribution per temper  ature for the magnetization
    #
    # dist1d.compute_histograms(columns=['Mean', 'AbsMean', 'Energy'], kind="probability_dist", nbins=10)
    #
    # fig, axes = fma.newfig(2.3, nrows=5, ncols=5, ratio=1)
    #
    # for i, idx in enumerate(data.index.unique(0).sort_values()):
    #     histodat = dist1d.histograms[idx]['Mean']
    #     axes[int(i * 1.0 / 5)][i % 5].set_xlim(-1, 1)
    #     Distribution1D.plot_histogram(ax=axes[int(i * 1.0 / 5)][i % 5], **histodat)
    #     from ising_model_data_loader.PlotRoutines.contour2D import add_fancy_box
    #     add_fancy_box(axes[int(i * 1.0 / 5)][i % 5], idx)
    #
    # plt.tight_layout()
    # fma.savefig("./../Examples/", "groundtruth")

    # from DataGeneration.datasetgenerator import generate_data_loader
    #
    # n = 2000*25
    #
    # data_generator_args = {
    #     "data_type": "target_config",
    #     "path": "/remote/lin34/kades/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/HeatbathSmall",
    #     "chunksize": 1,
    #     "total_number_of_data_per_file": 2000
    # }
    #
    # data_loader = generate_data_loader(
    #     data_generator=IsingDataGenerator,
    #     data_generator_args=data_generator_args,
    #     data_loader_params={'batch_size': 1,
    #                         'shuffle': True,
    #                         'num_workers': 1},
    #     n=n,
    #     seed=0,
    #     set_seed=False)
    #
    # # import matplotlib.pyplot as plt
    # #
    # # fig, axes = plt.subplots(5, 5)
    #
    # # IsingDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    #
    # # plt.show()
    #
    # from plotting_environment.loading_figure_mode import loading_figure_mode
    #
    # fma = loading_figure_mode("saving")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('seaborn-dark-palette')
    #
    # fig, axes = fma.newfig(3, nrows=5, ncols=5, ratio=1)
    #
    # IsingDataGenerator.inspect_magnetization(axes=axes, data_loader=data_loader)
    #
    # fma.savefig(".", "test")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.close()
