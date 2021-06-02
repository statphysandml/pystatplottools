import numpy as np

from pystatplottools.plotting.diagram_base_class import DiagramBaseClass


class Contour2D(DiagramBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x = kwargs.pop("x", None)
        self.y = kwargs.pop("y", None)
        self.z_index = kwargs.pop("z_index")

        if self.x is None:
            self.compute_x_func = kwargs.pop("compute_x_func", lambda data: data[self.x])
            self.data_x = self.compute_x_func(self.data)
        else:
            self.data_x = self.data[self.x]

        if self.y is None:
            self.compute_y_func = kwargs.pop("compute_y_func", lambda data: data[self.y])
            self.data_y = self.compute_y_func(self.data)
        else:
            self.data_y = self.data[self.y]

        width = len(self.data_x.unique())
        height = len(self.data_y.unique())

        self.data_x = self.data_x.values.reshape((width, height)).T
        self.data_y = self.data_y.values.reshape((width, height)).T
        self.data_z = self.data[self.z_index].values.reshape((width, height)).T

        self.scaling = int(kwargs.pop("scaling", 1))
        self.data_x = self.data_x[::self.scaling, ::self.scaling]
        self.data_y = self.data_y[::self.scaling, ::self.scaling]
        self.data_z = self.data_z[::self.scaling, ::self.scaling]

    def contourf(self, lev_num=10, cbar_scale=None, levs=None, norm=None, cmap='RdGy_r'):
        norm, levs, isnan_mask = self.determine_norm_and_lev(lev_num=lev_num, cbar_scale=cbar_scale,
                                                             levs=levs, norm=norm)
        cf = self._ax.contourf(self.data_x, self.data_y, self.data_z, levels=levs, norm=norm, cmap=cmap)
        self.data_z[isnan_mask] = np.nan  # To ensure that the same plot is generated if the function is called several time
        return cf

    def pcolormesh(self, lev_num=10, cbar_scale=None, levs=None, norm=None, cmap='RdGy_r'):
        norm, levs, isnan_mask = self.determine_norm_and_lev(lev_num=lev_num, cbar_scale=cbar_scale,
                                                             levs=levs, norm=norm)

        from pystatplottools.utils.bins_and_alignment import from_coordinates_to_bin_boundaries
        x, y = from_coordinates_to_bin_boundaries([self.data_x[0], self.data_y[:, 0]], bin_alignment="center")

        pc = self._ax.pcolormesh(x, y, self.data_z, cmap=cmap, norm=norm)
        self.data_z[isnan_mask] = np.nan  # To ensure that the same plot is generated if the function is called several time
        return pc

    def determine_norm_and_lev(self, lev_num=10, cbar_scale=None, levs=None, norm=None):
        isnan_mask = np.isnan(self.data_z)

        if cbar_scale is None:
            assert (levs is not None) and (norm is not None), "levs and norm need to be defined if cbar_scale is None"
        elif cbar_scale == "Log":
            assert (levs is None) and (norm is None), "levs and norm cannot be defined when cbar_scale is defined"
            norm, levs = self.get_log_norm_and_levs_of_z(lev_num=lev_num)
            self.data_z[isnan_mask] = self.data_z.min()
        elif cbar_scale == "Lin":
            assert (levs is None) and (norm is None), "levs and norm cannot be defined when cbar_scale is defined"
            norm, levs = self.get_lin_norm_and_levs_of_z(lev_num=lev_num)
            self.data_z[isnan_mask] = 0
        return norm, levs, isnan_mask

    def surface(self, lev_num=10, cbar_scale=None, levs=None, norm=None, cmap='RdGy_r'):

        norm, levs, isnan_mask = self.determine_norm_and_lev(lev_num=lev_num, cbar_scale=cbar_scale,
                                                             levs=levs, norm=norm)
        surf = self._ax.plot_surface(self.data_x, self.data_y, self.data_z, cmap=cmap,
                               linewidth=0, antialiased=False)
        self.data_z[
            isnan_mask] = np.nan  # To ensure that the same plot is generated if the function is called several time
        return surf

    def get_min_max_of_z(self):
        return Contour2D.get_min_max_of_index(data=self.data, index=self.z_index)

    def get_log_norm_and_levs_of_z(self, lev_num):
        lev_min, lev_max = self.get_min_max_of_z()
        return Contour2D.get_log_norm_and_levs(lev_min=lev_min, lev_max=lev_max, lev_num=lev_num)

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
            assert len(z_ticks) == len(z_tick_labels), "Len of z_ticks and z_tick_labels must be the same"
            if hasattr(cf, "levels"):
                assert np.min(z_ticks) > cf.levels.min(), "z_ticks must be in the range of cf.levels, otherwise the tick labels are assigned wrongly - min(z_ticks) too small."
                assert np.max(z_ticks) < cf.levels.max(), "z_ticks must be in the range of cf.levels, otherwise the tick labels are assigned wrongly - max(z_ticks) too large."
            else:
                assert False, "The assignment of the tick labels to the ticks might be wrong. - Alternative: Add colorbar by yourself."
            cbar.ax.set_yticklabels(z_tick_labels)
        return cbar