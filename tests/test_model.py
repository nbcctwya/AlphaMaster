from pathlib import Path
import inspect
import unittest

import torch
import yaml

import qlib
from alphamaster.dataset import marketDataHandler
from alphamaster.model import MASTER, MASTERTrainer


ROOT = Path(__file__).resolve().parents[1]


class ModelMigrationTest(unittest.TestCase):
    def test_qlib_comes_from_conda_environment(self):
        qlib_path = Path(inspect.getfile(qlib)).resolve()
        self.assertIn("site-packages/qlib", qlib_path.as_posix())
        self.assertNotIn("qlib-MASTER", qlib_path.as_posix())

    def test_master_is_plain_pytorch_and_trainer_has_no_qlib_base(self):
        self.assertTrue(issubclass(MASTER, torch.nn.Module))
        self.assertEqual(MASTERTrainer.__bases__, (object,))

    def test_master_shape_and_market_feature_count(self):
        model = MASTER(
            d_feat=158,
            d_model=256,
            t_nhead=4,
            s_nhead=2,
            gate_input_start_index=158,
            gate_input_end_index=221,
            T_dropout_rate=0.5,
            S_dropout_rate=0.5,
            beta=10,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(7, 8, 221))
        self.assertEqual(output.shape, (7,))

        handler = marketDataHandler.__new__(marketDataHandler)
        handler.market_indices = ["sh000300", "sh000852", "sh000905"]
        fields, names = handler.get_feature_config()
        self.assertEqual(len(fields), 63)
        self.assertEqual(len(names), 63)

    def test_baseline_configuration_is_preserved(self):
        expected_segments = {
            "train": ["2009-01-01", "2020-12-31"],
            "valid": ["2021-01-01", "2022-12-31"],
            "test": ["2023-01-01", "2025-12-31"],
        }
        for market in ("csi300", "sp500"):
            with self.subTest(market=market):
                config = yaml.safe_load((ROOT / "configs" / f"master_{market}.yaml").read_text())
                kwargs = config["task"]["model"]["kwargs"]
                dataset_kwargs = config["task"]["dataset"]["kwargs"]
                segments = dataset_kwargs["segments"]
                actual_segments = {
                    key: [str(value) for value in values] for key, values in segments.items()
                }
                self.assertEqual(actual_segments, expected_segments)
                self.assertEqual(dataset_kwargs["step_len"], 8)
                self.assertEqual(kwargs["d_feat"], 158)
                self.assertEqual(kwargs["d_model"], 256)
                self.assertEqual(kwargs["t_nhead"], 4)
                self.assertEqual(kwargs["s_nhead"], 2)
                self.assertEqual(kwargs["gate_input_start_index"], 158)
                self.assertEqual(kwargs["gate_input_end_index"], 221)
                self.assertEqual(kwargs["T_dropout_rate"], 0.5)
                self.assertEqual(kwargs["S_dropout_rate"], 0.5)
                self.assertEqual(kwargs["n_epochs"], 40)
                self.assertEqual(kwargs["lr"], 0.000008)
                self.assertEqual(kwargs["train_stop_loss_thred"], 0.95)


if __name__ == "__main__":
    unittest.main()
