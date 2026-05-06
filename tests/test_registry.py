import unittest

from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE

from src.utils.registry import get_model_instance


class RegistryTest(unittest.TestCase):
    def test_get_model_instance_supports_initial_models(self):
        iforest = get_model_instance("iforest", random_state=7)
        inne = get_model_instance("inne", random_state=7)
        hbos = get_model_instance("hbos", random_state=7)

        self.assertIsInstance(iforest, IForest)
        self.assertEqual(iforest.random_state, 7)
        self.assertIsInstance(inne, INNE)
        self.assertEqual(inne.random_state, 7)
        self.assertIsInstance(hbos, HBOS)

    def test_unknown_model_reports_valid_options(self):
        with self.assertRaisesRegex(ValueError, "iforest"):
            get_model_instance("unknown")


if __name__ == "__main__":
    unittest.main()
