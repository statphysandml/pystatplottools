from abc import ABCMeta
import numpy as np


class DataGeneratorBaseClass:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):  # points_G, points_rho, noise_with = 0.00001, a_min_bw = 1.0, ):
        self.device = kwargs.pop('device', "cpu")
        self.batch_size = kwargs.pop('batch_size', 1)

        # Determines if a seed is set or not
        set_seed = kwargs.get('set_seed', True)

        # Set the random seed for the generation (setting the keylist before setting the random_seed is necessary to
        # generate the same/or not the same parameters)

        random_seed = kwargs.get('seed', None)
        if random_seed is not None and set_seed is True:
            print("Random seed is set by seed", random_seed)
            np.random.seed(random_seed)
        elif set_seed is True:
            print("random seed is set by np.random.seed()")
            np.random.seed()

        self.inp_size = None
        self.tar_size = None

    def input_size(self):
        return self.inp_size

    def target_size(self):
        return self.tar_size

    # def get_chunk_iterator(self):
    #     return None

    def get_std_for_parameter_ranges(self, ratio):
        return None


def data_generator_factory(data_generator_name="IsingDataGenerator"):
    if data_generator_name == "BatchIsingDataGenerator":
        from pystatplottools.ppd_pytorch_data_generation.ising_model.batchisingdatagenerator import BatchIsingDataGenerator
        return BatchIsingDataGenerator
    elif data_generator_name == "BatchIsingGraphDataGenerator":
        from pystatplottools.ppd_pytorch_data_generation.ising_model.isinggraphdatagenerator import BatchIsingGraphDataGenerator
        return BatchIsingGraphDataGenerator
    else:
        from pystatplottools.ppd_pytorch_data_generation.ising_model.isingdatagenerator import IsingDataGenerator
        return IsingDataGenerator