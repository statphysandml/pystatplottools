

def align_bins(bins, bin_alignment, bin_scale=None):
    if bin_alignment == "center":
        if bin_scale == "logarithmic":
            import numpy as np
            # Ensure that np.log(bins) == regular grid
            return np.exp((np.log(bins[1:]) + np.log(bins[:-1])) / 2.0)
        else:
            return (bins[:-1] + bins[1:]) / 2
    elif bin_alignment == "right":
        return bins[1:]
    else:  # left alignnment
        return bins[:-1]


def revert_align_bins(data_range, bin_alignment, bin_scale=None):
    import numpy as np
    if bin_scale == "logarithmic":
        del_data = np.log(data_range[1]) - np.log(data_range[0])
        if bin_alignment == "center":
            new_data_range = np.append(np.log(data_range), np.log(data_range[-1]) + del_data)
            new_data_range = new_data_range - del_data / 2.0
            new_data_range = np.exp(new_data_range)
            return new_data_range
        elif bin_alignment == "right":
            new_data_range = np.insert(np.log(data_range), 0, np.log(data_range[0]) - del_data)
            return np.exp(new_data_range)
        else:
            new_data_range = np.append(np.log(data_range), np.log(data_range[-1]) + del_data)
            return np.exp(new_data_range)
    else:
        del_data = data_range[1] - data_range[0]
        if bin_alignment == "center":
            new_data_range = np.append(data_range, data_range[-1] + del_data)
            return new_data_range - del_data / 2.0
        elif bin_alignment == "right":
            return np.insert(data_range, 0, data_range[0] - del_data)
        else:
            return np.append(data_range, data_range[-1] + del_data)


def from_coordinates_to_bin_boundaries(data_ranges, bin_alignment, bin_scale=None):
    new_data_ranges = []
    for data_range in data_ranges:
        new_data_ranges.append(revert_align_bins(data_range=data_range, bin_alignment=bin_alignment, bin_scale=bin_scale))
    return new_data_ranges


def from_bin_boundaries_to_coordinates(data_ranges, bin_alignment, bin_scale=None):
    new_data_ranges = []
    for data_range in data_ranges:
        new_data_ranges.append(align_bins(bins=data_range, bin_alignment=bin_alignment, bin_scale=bin_scale))
    return new_data_ranges
