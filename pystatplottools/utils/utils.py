import os
import json


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def set_up_directories(
    data_dir, results_dir, data_base_dir="./data/", results_base_dir="./results/"
):
    data_root = data_base_dir + data_dir + "/"
    results_root = results_base_dir + results_dir + "/"

    if not os.path.isdir(data_base_dir):
        os.makedirs(data_base_dir)

    if not os.path.isdir(results_base_dir):
        os.makedirs(results_base_dir)

    if not os.path.isdir(results_root):
        os.makedirs(results_root)
    return data_root, results_root


def device(device_id=0):
    import torch
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def add_index_level(data, key="df"):
    import pandas as pd
    return pd.concat([data], keys=[key])


def drop_index_level(data, single_key="df"):
    import pandas as pd
    return pd.concat([data.reset_index().drop("level_1", axis=1)], keys=[single_key])
