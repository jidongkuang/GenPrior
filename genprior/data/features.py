"""Dataset for precomputed skeleton features."""

import numpy as np
from torch.utils.data import Dataset


class FeatureFeeder(Dataset):
    SPLIT_KEYS = {
        "train": ("train_data", "train_label"),
        "test_zsl": ("zsl_data", "zsl_label"),
        "test_gzsl": ("gzsl_data", "gzsl_label"),
    }

    def __init__(self, data_path, split_type="train"):
        try:
            data_key, label_key = self.SPLIT_KEYS[split_type]
        except KeyError as error:
            raise ValueError(f"unsupported split type: {split_type}") from error

        with np.load(data_path) as archive:
            self.data = archive[data_key]
            self.label = archive[label_key]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return np.asarray(self.data[index]), self.label[index], index
