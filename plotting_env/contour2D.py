import numpy as np
import pandas as pd

from plotting_env.diagram_base_class import DiagramBaseClass

# def mass_difference(data):
#     return data.mass2 - data.mass1


def add_fancy_box(ax, legend_name):
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], linestyle="--", color="white")]
    legend = ax.legend(custom_lines, [legend_name], loc="upper right", framealpha=1, fancybox=False, handlelength=0.0,
                       handletextpad=0)


class Contour2D(DiagramBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.compute_x_func = kwargs.pop("compute_x_func")
        self.compute_y_func = kwargs.pop("compute_y_func")
        self.z_index = kwargs.pop("z_index")

        self.data_x = self.compute_x_func(self.data)
        self.data_y = self.compute_y_func(self.data)

        width = int(np.sqrt(len(self.data_x)))
        self.data_x = self.data_x.values.reshape((width, width)).T
        self.data_y = self.data_y.values.reshape((width, width)).T
        # self.data_x, self.data_y = np.meshgrid((data.mass2 - data.mass1).values, data.gamma2.values)
        self.data_z = self.data[self.z_index].values.reshape((width, width)).T

        self.scaling = int(kwargs.pop("scaling", 1))
        self.data_x = self.data_x[::self.scaling, ::self.scaling]
        self.data_y = self.data_y[::self.scaling, ::self.scaling]
        self.data_z = self.data_z[::self.scaling, ::self.scaling]

    def contourf(self, ax, lev_num=10, cbar_scale=None, levs=None, norm=None, cmap='RdGy_r'):
        norm, levs, isnan_mask = self.determine_norm_and_lev(lev_num=lev_num, cbar_scale=cbar_scale,
                                                             levs=levs, norm=norm)
        cf = ax.contourf(self.data_x, self.data_y, self.data_z, levels=levs, norm=norm, cmap=cmap)
        self.data_z[isnan_mask] = np.nan # To ensure that the same plot is generated if the function is called several time
        return cf

    def determine_norm_and_lev(self, lev_num=10, cbar_scale=None, levs=None, norm=None):
        isnan_mask = np.isnan(self.data_z)

        if cbar_scale is None:
            assert (levs is not None) and (norm is not None), "levs and norm need to be defined if cbar_scale is None"
        elif cbar_scale == "Exp":
            norm, levs = self.get_exp_norm_and_levs_of_z(lev_num=lev_num)
            self.data_z[isnan_mask] = self.data_z.min()
        elif cbar_scale == "Lin":
            norm, levs = self.get_lin_norm_and_levs_of_z(lev_num=lev_num)
            self.data_z[isnan_mask] = 0
        return norm, levs, isnan_mask

    def surface(self, ax, lev_num=10, cbar_scale=None, levs=None, norm=None, cmap='RdGy_r'):

        norm, levs, isnan_mask = self.determine_norm_and_lev(lev_num=lev_num, cbar_scale=cbar_scale,
                                                             levs=levs, norm=norm)
        surf = ax.plot_surface(self.data_x, self.data_y, self.data_z, cmap=cmap,
                               linewidth=0, antialiased=False)
        self.data_z[
            isnan_mask] = np.nan  # To ensure that the same plot is generated if the function is called several time
        return surf

    def get_min_max_of_z(self):
        return Contour2D.get_min_max_of_index(data=self.data, index=self.z_index)

    def get_exp_norm_and_levs_of_z(self, lev_num):
        lev_min, lev_max = self.get_min_max_of_z()
        return Contour2D.get_exp_norm_and_levs(lev_min=lev_min, lev_max=lev_max, lev_num=lev_num)

    def get_lin_norm_and_levs_of_z(self, lev_num):
        lev_min, lev_max = self.get_min_max_of_z()
        return Contour2D.get_lin_norm_and_levs(lev_min=lev_min, lev_max=lev_max, lev_num=lev_num)

    # Colorbar

    @staticmethod
    def add_colorbar(fig, cf, z_label, z_ticks=None, z_tick_labels=None, cax=None):
        if z_ticks is None:
            cbar = fig.colorbar(cf, cax=cax)
        else:
            cbar = fig.colorbar(cf, ticks=z_ticks, cax=cax)
        cbar.ax.set_ylabel(z_label)

        if z_tick_labels is not None:
            cbar.ax.set_yticklabels(z_tick_labels)


# #### Code for spectral reconstrubtion #####
#
# if __name__ == '__main__':
#
#     from PlotRoutines.loading import Loading
#
#     loading = Loading("EvaluationDynamic", "TestDataSet2BWContourCascade")
#     data = loading.get_data()
#
#     contour2D = Contour2D(
#         data=data.loc[data.index.unique(0)[0]],
#         compute_x_func=mass_difference,
#         compute_y_func=lambda x: x.gamma2,
#         z_index='spectral_function_loss'
#     )
#
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     # cbar_ax = fig.add_axes([0.92, 0.1, 0.012, 0.8])
#
#     contour2D.contourf(
#         fig=fig,
#         ax=ax,
#         # cax=cbar_ax,
#         lev_num=40,
#         x_label="x",
#         y_label="y",
#         z_label="z",
#         # x_ticks_visible=False,
#         z_ticks=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1],
#         z_tick_labels=['$10^{-3}$', '$3\\times 10^{-3}$', '$10^{-2}$', '$3 \\times 10^{-2}$', '$10^{-1}$',
#                     '$3\\times 10^{-1}$', '$1$']
#     )
#
#     add_fancy_box(ax, "Fancy Box")
#
#     plt.show()