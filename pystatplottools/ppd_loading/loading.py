import pandas as pd
import numpy as np
import json
import os
import glob


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


class ConfigurationLoader:
    def __init__(self, **kwargs):
        print("###### The configurations are flipped to ensure a balanced dataset between the two phases of the magnetization - this is only valid for"
              " models without an external field! ######")

        # Defines how many samples of a single file will be loaded in one chunk. If multiple files are loaded, the
        # resulting total number of samples in one chunk is determined by chunksize * number_of_files = chunksize * #readers
        self.path = kwargs.pop('path')

        self.chunksize = kwargs.pop("chunksize", 100)
        self.total_number_of_data_per_file = kwargs.pop("total_number_of_data_per_file")
        self.running_parameter = kwargs.pop("running_parameter", None)
        current_chunk_iterator_val = kwargs.pop("current_chunk_iterator_val", None)

        assert self.total_number_of_data_per_file % self.chunksize == 0, "Total number of data is not dividable by given chunksize"
        self.total_chunks = int(self.total_number_of_data_per_file / self.chunksize)  # The number of


        # Enables to continue read the file from a given chunk iterator val
        if current_chunk_iterator_val is not None:
            self.skiprows = range(1, current_chunk_iterator_val * self.chunksize + 1)
        else:
            self.skiprows = 0

        # A reader is assigned to each file
        self.readers, self.filenames = ConfigurationLoader.load_configuration_readers(
            path=self.path, chunksize=self.chunksize, skiprows=self.skiprows)

        self.chunk_iterator = 0
        self.data = None
        self.data = self.get_next_chunk_collection()

    def get_next_chunk_collection(self, resample=True):
        if self.data is not None and self.total_chunks == 1:
            # No need to reload the data - the data is just resampled
            if resample:
                self.data = self.data.sample(frac=1).reset_index(drop=True)
            return self.data
        if self.chunk_iterator >= self.total_chunks:
            # Readers and filenames are reloaded for the next iteration
            self.readers, self.filenames = ConfigurationLoader.load_configuration_readers(path=self.path,
                                                                                  chunksize=self.chunksize)
            self.chunk_iterator = 0

        data = []
        for idx, reader in enumerate(self.readers):
            chunk = next(reader)
            # df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
            if "Unnamed" in chunk.columns[-1]:
                chunk.drop(chunk.columns[len(chunk.columns) - 1], axis=1, inplace=True)
            running_parameter_val = np.float32(self.filenames[idx][self.filenames[idx].find("=")+1:self.filenames[idx].find(".dat")])
            chunk = chunk.assign(**{self.running_parameter.capitalize(): running_parameter_val})
            data.append(chunk)

        data = ConfigurationLoader.merge_file_datastreams(data=data, resample=resample)

        data = ConfigurationLoader.transform_config_data(data=data)

        self.chunk_iterator += 1
        # print("Chunk i", self.chunk_iterator)

        ''' A trick is applied to obtain a magnetization that is zero for all temperatures with an external field -
        this trick should of course not be applied if the external field is finite!!
        '''

        mean_values = data.groupby(self.running_parameter.capitalize())["Mean"].apply(lambda x: x.mean())
        percentages = (1 + mean_values) / 2.0

        for beta in data.Beta.unique():
            random_index = None
            num_to_be_changed_rows = int(len(data) / 21 * (percentages - 0.5).loc[beta])
            if num_to_be_changed_rows < 0:
                random_index = data.query(self.running_parameter.capitalize() + " == " + str(beta) + " and Mean < 0").sample(abs(num_to_be_changed_rows)).index
            elif num_to_be_changed_rows > 0:
                random_index = data.query(self.running_parameter.capitalize() + " == " + str(beta) + " and Mean > 0").sample(abs(num_to_be_changed_rows)).index
            if random_index is not None:
                data.loc[random_index, "Config"] = data.loc[random_index].Config.apply(lambda x: -1.0 * x)
            data.Mean = data.Config.apply(lambda x: x.mean())

        return data

    def get_chunk_iterator(self):
        return self.chunk_iterator

    @staticmethod
    def load_configuration_readers(path="./../../../Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/Heatbath/",
                                   chunksize=100, skiprows=0):
        readers = []
        filenames = []

        current_directory = os.path.abspath(os.getcwd())

        os.chdir(path)
        for file in glob.glob("*.dat"):
            reader = pd.read_csv(file, delimiter="\t", header=0, chunksize=chunksize, skiprows=skiprows, index_col=False)
            readers.append(reader)
            filenames.append(file)

        os.chdir(current_directory)
        return readers, filenames  # , chunk_order

    @staticmethod
    def load_all_configurations(path="./../../../Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/Heatbath/",
                                identifier="None", running_parameter="beta", skiprows=0):
        data = []
        filenames = []

        current_directory = os.path.abspath(os.getcwd())

        os.chdir(path)

        data_files = glob.glob(identifier + "*.dat")
        # Load data of multiple files based on running parameter
        if running_parameter != "default":
            for file in data_files:
                dat = pd.read_csv(file, delimiter="\t", header=0, skiprows=skiprows, index_col=False)

                if "Unnamed" in dat.columns[-1]:
                    dat.drop(dat.columns[len(dat.columns) - 1], axis=1, inplace=True)
                running_parameter_val = np.float32(file[file.find("=")+1:file.find(".dat")])
                dat = dat.assign(**{running_parameter.capitalize(): running_parameter_val})
                data.append(dat)
                filenames.append(file)

        # Load single data file
        else:
            file = data_files[0]
            dat = pd.read_csv(file, delimiter="\t", header=0, skiprows=skiprows, index_col=False)

            if "Unnamed" in dat.columns[-1]:
                dat.drop(dat.columns[len(dat.columns) - 1], axis=1, inplace=True)
            dat = dat.assign(**{running_parameter.capitalize(): running_parameter})
            data.append(dat)
            filenames.append(file)

        data = ConfigurationLoader.merge_file_datastreams(data=data, by_col_index=running_parameter)
        data = ConfigurationLoader.transform_config_data(data=data)

        if running_parameter == "default":
            del data["Default"]
        os.chdir(current_directory)

        return data, filenames  # , chunk_order

    @staticmethod
    def merge_file_datastreams(data, by_col_index=None, resample=False):
        if by_col_index is None:
            data = pd.concat(data)
            data = data.reset_index(drop=True)
            if resample:
                data = data.sample(frac=1).reset_index(drop=True)
        else:
            if isinstance(data[0].loc[0, by_col_index.capitalize()], str):
                keys = [data[0].loc[0, by_col_index.capitalize()]]
            else:
                keys = [f"{x.loc[0, by_col_index.capitalize()]:.6f}" for x in data]
            data = pd.concat(data, keys=keys).sort_index(level=0)
            data.index.set_names([by_col_index, 'sample_num'], inplace=True)
        return data

    @staticmethod
    def transform_config_data(data):
        if "Config" in data and not isinstance(data["Config"].iloc[0], np.float):
            data["Config"] = data["Config"].apply(lambda x: np.float32(x.split()))

        if "ComplexConfig" in data:
            complex_config = data.ComplexConfig.apply(lambda x: np.float32(x.split()))
            data.insert(1, "StateReal", complex_config.apply(lambda x: x[0]))
            data.insert(3, "StateImag", complex_config.apply(lambda x: x[1]))
            # data.drop("ComplexConfig", axis=1, inplace=True)

        # if "Drift" in data:
        #     complex_drift = data.Drift.apply(lambda x: np.float32(x.split()))
        #     data.insert(1, "DriftReal", complex_drift.apply(lambda x: x[0]))
        #     data.insert(3, "DriftImag", complex_drift.apply(lambda x: x[1]))
            # data.drop("Drift", axis=1, inplace=True)

        if "RepaConfig" in data:
            repa_config = data.RepaConfig.apply(lambda x: np.float32(x.split()))
            for i in range(len(repa_config.iloc[0])):
                data.insert(len(data.columns), "State" + str(i + 1), repa_config.apply(lambda x: x[i]))

        return data


if __name__ == '__main__':
    configuration_loade_args = {
        # "path": "/home/lukas/Lattice_Git/PROJECTS/LUKAS/BoltzmannMachine/data/IsingModel/",
        "path": "/home/lukas/LatticeModelImplementations/main_project/data/IsingModel",
        "chunksize": 100,
        "running_parameter": "beta",
        "total_number_of_data_per_file": 1000
        # "batch_size": 10
    }
    # loader = ConfigurationLoader(**configuration_loade_args)

    data, filenames = ConfigurationLoader.load_all_configurations(
        path="/home/lukas/LatticeModelImplementations/main_project/data/IsingModel",
        running_parameter="beta")

    pass