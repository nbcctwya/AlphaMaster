"""Short-range integration check against the installed Qlib CN provider."""

from pathlib import Path
import argparse
import copy

import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.backtest import backtest

from alphamaster.model import DailyBatchSamplerRandom, MASTERTrainer


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="also run one training epoch")
    parser.add_argument("--backtest", action="store_true", help="also run a short Qlib backtest")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load((ROOT / "configs/master_csi300.yaml").read_text())
    qlib.init(**config["qlib_init"])
    dataset_config = config["task"]["dataset"]
    kwargs = dataset_config["kwargs"]
    handler_kwargs = kwargs["handler"]["kwargs"]
    handler_kwargs.update(
        start_time="2023-01-01",
        end_time="2023-06-30",
        fit_start_time="2023-01-01",
        fit_end_time="2023-03-31",
    )
    market_kwargs = kwargs["market_data_handler_config"]
    market_kwargs.update(
        start_time="2023-01-01",
        end_time="2023-06-30",
        fit_start_time="2023-01-01",
        fit_end_time="2023-03-31",
    )
    kwargs["segments"] = {
        "train": ("2023-03-01", "2023-03-31"),
        "valid": ("2023-04-01", "2023-04-28"),
        "test": ("2023-05-01", "2023-05-31"),
    }
    dataset = init_instance_by_config(dataset_config)
    sample = dataset.prepare("test", col_set=["feature", "label"], data_key="infer")
    first = sample[0]
    print(f"samples={len(sample)}")
    print(f"sample_shape={first.shape}")
    print(f"index_names={sample.get_index().names}")
    if first.shape[-1] != 222:
        raise AssertionError(f"expected 221 features + label, got {first.shape[-1]}")
    sampler = DailyBatchSamplerRandom(sample, shuffle=False)
    for batch in sampler:
        dates = sample.get_index()[batch].get_level_values("datetime")
        if dates.nunique() != 1:
            raise AssertionError("MASTER daily batch contains multiple trading dates")
    ordered_index = sample.get_index()[sampler.ordered_indices()]
    if not ordered_index.is_monotonic_increasing:
        raise AssertionError("MASTER sampler output is not ordered by datetime and instrument")

    if args.train or args.backtest:
        checkpoint_dir = ROOT / "artifacts" / "smoke_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs = dict(config["task"]["model"]["kwargs"])
        model_kwargs.update(
            n_epochs=1,
            seed=0,
            save_path=f"{checkpoint_dir}/",
            save_prefix="smoke_",
        )
        trainer = MASTERTrainer(**model_kwargs)
        trainer.fit(dataset)
        prediction = trainer.predict(dataset)
        print(f"prediction_shape={prediction.shape}")
        print(f"prediction_index_names={prediction.index.names}")
        if prediction.index.names != ["datetime", "instrument"]:
            raise AssertionError("prediction index is not Qlib-compatible")

        if args.backtest:
            analysis = config["port_analysis_config"]
            strategy = copy.deepcopy(analysis["strategy"])
            strategy["kwargs"]["signal"] = prediction
            backtest_kwargs = copy.deepcopy(analysis["backtest"])
            backtest_kwargs.update(start_time="2023-05-01", end_time="2023-05-31")
            portfolio, _ = backtest(
                strategy=strategy,
                executor={
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
                },
                **backtest_kwargs,
            )
            report, _ = portfolio["1day"]
            print(f"backtest_days={len(report)}")
            print(f"backtest_columns={list(report.columns)}")
            if report.empty:
                raise AssertionError("Qlib backtest produced no portfolio report")


if __name__ == "__main__":
    main()
