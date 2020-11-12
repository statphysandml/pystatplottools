import torch
import os


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
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")