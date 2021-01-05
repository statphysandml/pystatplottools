from pystatplottools.distributions.distributionDD import DistributionDD


class BinnedStatistics(DistributionDD):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
            self, axes_indices,
            columns,
            statistic='count',
            transform='lin',
            nbins=[],  # Refers to the number of bins in each dimension
            # Range accepts None, Scalar and List
            range_min=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
            range_max=None,  # Refers to the ranges of the bins in the different dimension of axes_indices
            bin_scales='linear',
            with_binnumber=False
    ):
        """
        Computes binned statistics over binned samples in axes_indices for each data frame in self.data.
        The function makes of numpy's binned_statistic_dd function.
        :param axes_indices: list of column indices -  binned dimensions
        :param columns: the statistic is computed over these values
        :param statistic: statistic to be computed
        :param transform: lin or log10 - possibility to transform the column data before the computation of the statistics
        :param nbins: scalar of list - number of bins in each dimensions
        :param range_min: None, scalar or list - minimum of the ranges of the bins in the different dimension of axes_indices
        :param range_max: None, scalar or list - maximum of ranges of the bins in the different dimension of axes_indices
        :param bin_scales: scalar or list - "linear" or "logarithmic" - possibility to introduce another scale for the bins
        :param with_binnumber: assign binned_numbers are also returned for each value in self.data
        :return: dictionary that contains for each dataframe and each column a histogram with the computed statistics.
            The histograms can be accessed via [df][column]
        """

        self._axes_indices = axes_indices

        self._distribution = self._compute_binned_statistics(axes_indices=self._axes_indices,
                                                columns=columns,
                                                statistic=statistic,
                                                transform=transform,
                                                nbins=nbins,
                                                range_min=range_min,
                                                range_max=range_max,
                                                bin_scales=bin_scales,
                                                with_binnumber=with_binnumber
                                                )

    def linearize(self, output_statistics_names=None, dataframes_as_columns=False, bin_alignment="center"):
        """
        Generates a linearlized dataframe from a binned_statistics.
        :param axes_indices: random variables of the binned values
        :param binned_statistics: binned statistics generated by binned_statistics
        :param output_statistics_name: optional definition of the names of the columns for which the statistics has been
            computed
        :return: A dataframe with multiindex (df, idx) and columns (bin_center..., statistics values)
        """
        return DistributionDD._linearize_binned_statistics(
            axes_indices=self._axes_indices, binned_statistics=self._distribution,
            output_statistics_names=output_statistics_names, dataframes_as_columns=dataframes_as_columns,
            bin_alignment=bin_alignment
        )