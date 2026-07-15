import tomllib
import unittest

from wkde_cad.empirical.benchmark import experiment


class ConfigScriptTest(unittest.TestCase):
    def test_empirical_benchmark_default_outputs_are_under_outputs_root(self):
        with open(experiment.DEFAULT_CONFIG, "rb") as config_file:
            cfg = tomllib.load(config_file)

        self.assertEqual(
            cfg["experiment"]["output_dir"],
            "outputs/empirical/benchmark/results/logistic",
        )
        self.assertEqual(
            cfg["model_selection"]["output_dir"],
            "outputs/empirical/benchmark/model_selection",
        )
        self.assertEqual(cfg["conformal"]["split_calib"], 0.5)


if __name__ == "__main__":
    unittest.main()
