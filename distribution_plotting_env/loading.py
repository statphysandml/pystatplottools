import pandas as pd
import numpy as np
import json


def load_data(mode, filename, directory, root_dir, rel_path="./"):
    df = pd.read_csv(rel_path + root_dir + "/" + directory + "/" + mode + "_" + filename + ".dat", header=0, delimiter="\t",
                     index_col=False)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    # if "Config" in df:
        # df.Config = df.Config.apply(lambda x: np.float32(x.split()))
        # df.insert(1, "State", df.Config.apply(lambda x: x[0]))

    if "ComplexConfig" in df:
        complex_config = df.ComplexConfig.apply(lambda x: np.float32(x.split()))
        df.insert(1, "StateReal", complex_config.apply(lambda x: x[0]))
        df.insert(3, "StateImag", complex_config.apply(lambda x: x[1]))
        # df.drop("ComplexConfig", axis=1, inplace=True)
    if "Drift" in df:
        complex_drift = df.Drift.apply(lambda x: np.float32(x.split()))
        df.insert(1, "DriftReal", complex_drift.apply(lambda x: x[0]))
        df.insert(3, "DriftImag", complex_drift.apply(lambda x: x[1]))
        # df.drop("Drift", axis=1, inplace=True)
    if "RepaConfig" in df:
        repa_config = df.RepaConfig.apply(lambda x: np.float32(x.split()))
        for i in range(len(repa_config.iloc[0])):
            df.insert(len(df.columns), "State" + str(i + 1), repa_config.apply(lambda x: x[i]))
    return df


def load_json(mode, filename, directory, root_dir, rel_path="./"):
    with open(rel_path + root_dir + "/" + directory + "/" + mode + "_params_" + filename + ".json") as json_file:
        return json.load(json_file)
