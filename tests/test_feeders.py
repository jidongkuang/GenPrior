import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior.data import ntu, pku


class FeederOutputTest(unittest.TestCase):
    def test_feature_archive_split_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "features.npz"
            np.savez(
                archive_path,
                train_data=np.array([[1, 2]]),
                train_label=np.array([3]),
                zsl_data=np.array([[4, 5]]),
                zsl_label=np.array([6]),
                gzsl_data=np.array([[7, 8]]),
                gzsl_label=np.array([9]),
            )

            feeder = ntu.Feeder(archive_path, "test_zsl")
            sample, label, index = feeder[0]
            np.testing.assert_array_equal(sample, np.array([4, 5]))
            self.assertEqual(label, 6)
            self.assertEqual(index, 0)

    def test_feeders_return_only_sample_label_and_index(self):
        for feeder_class in (ntu.Feeder, pku.Feeder):
            with self.subTest(feeder=feeder_class.__module__):
                feeder = feeder_class.__new__(feeder_class)
                feeder.data = np.arange(12).reshape(3, 4)
                feeder.label = np.array([2, 4, 6])

                sample, label, index = feeder[1]

                np.testing.assert_array_equal(sample, np.array([4, 5, 6, 7]))
                self.assertEqual(label, 4)
                self.assertEqual(index, 1)


if __name__ == "__main__":
    unittest.main()
