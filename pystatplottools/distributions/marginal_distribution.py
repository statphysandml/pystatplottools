from pystatplottools.distributions.distributionDD import DistributionDD


class MarginalDistribution(DistributionDD):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
            self, axes_indices,
            statistic='count',
            transform='lin',
            nbins=[],  # Refers to the number of bins for each column
            range_min=None,  # Refers to the ranges of the bins for each column
            range_max=None,  # Refers to the ranges of the bins for each column
            bin_scales='linear',
            with_binnumber=False
    ):
        """
        Computes the marginal distribution of the data given by axes_indices. A marginal distribution is computed for
        each random variable in axes_indices.
        :param axes_indices: list of column indices
        :param statistic: "probability" or "count" - statistic to be computed
        :param nbins: scalar of list - number of bins for each random variable
        :param range_min: None, scalar or list - minimum of the range of the bins for each random variable
        :param range_max: None, scalar or list - maximum of the range of the bins for each random variable
        :param bin_scales: scalar or list - "linear" or "logarithmic" - possibility to introduce another scale for the bins
        :param with_binnumber: assign binned_numbers are also returned for each value in self.data
        :return: dictionary that contains for each dataframe and each axes_index a histogram. The histograms can be
            accessed via [df][axes_index]
        """

        self._axes_indices = axes_indices

        self._distribution = self._compute_binned_statistics(columns=self._axes_indices,
                                       statistic=statistic,
                                       transform=transform,
                                       nbins=nbins,
                                       range_min=range_min,
                                       range_max=range_max,
                                       bin_scales=bin_scales,
                                       with_binnumber=with_binnumber
                                       )

    def linearize(self, order_by_bin=False, bin_alignment="center"):
        """
        Generates a linearized dataframe from marginal_distribution
        :param marginal_distribution: marginal distribution generated by marginal_distribution()
        :param order_by_bin: True/False
        :return: If order_by_bin = True: A multiindex dataframe with index (col, bin_idx) and columns (binedges, df...).
                                         The binedges are refer to col.
                 if order_by_bin = False: A multiindex dataframe with index (df, col, bin_idx) and columns (binedges, data),
                                         where data contains the counts/probabilities for each col
        """
        return DistributionDD._linearize_histograms(
            histogram_data=self._distribution, order_by_bin=order_by_bin, bin_alignment=bin_alignment)
