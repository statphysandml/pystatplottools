from abc import ABCMeta
import numpy as np


class DataGeneratorBaseClass:
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.device = kwargs.pop('device', "cpu")

        # Determines if a seed is set or not
        set_seed = kwargs.get('set_seed', True)

        # Set the random seed for the generation (setting the keylist before setting the random_seed is necessary to
        # generate the same/or not the same parameters)

        random_seed = kwargs.get('seed', None)
        if random_seed is not None and set_seed is True:
            print("Random seed is set by seed", random_seed)
            np.random.seed(random_seed)
        elif set_seed is True:
            print("Random seed is set by np.random.seed()")
            np.random.seed()

        self.inp_size = None
        self.tar_size = None

    def input_size(self):
        return self.inp_size

    def target_size(self):
        return self.tar_size
