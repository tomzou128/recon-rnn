import importlib
import os

# find the dataset definition by name, for example ScanNetDataset (scannet.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    # if dataset_name == 'scannet':
    #     return getattr(module, "ScanNetDataset")
    # elif dataset_name == 'ARKit':
    #     return getattr(module, "ARKitDataset")
    if dataset_name == 'scannet_depth':
        return getattr(module, "ScanNetDatasetDepth")
    elif dataset_name == 'scannet_depth_rnn':
        return getattr(module, "ScanNetDatasetDepthRNN")
