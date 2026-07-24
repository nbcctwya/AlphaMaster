"""MASTER baseline runner — CSI300 (CN) & SP500 (US), 5 seeds, Qlib backtest.

Migrated from
``qlib-MASTER/examples/benchmarks/MASTER/paper_baseline/run_baseline.py``.
It uses the pyqlib installation from the active Conda environment and never
imports Qlib from the source checkout.

Adapted from ``examples/benchmarks/MASTER/main.py`` with these changes:
  - ``--market {csi300,sp500}`` selects the config; ``--seeds N`` (default 5);
    ``--smoke`` runs 1 epoch / seed 0 only (fast pipeline check).
  - ``qlib.init`` reads provider/region from the yaml's ``qlib_init`` (no hardcoded CN).
  - No ``sed`` mutation of the yaml (market/universe are fixed per-config-file).
  - Aggregates IC / Rank IC / annualized_return / information_ratio (with & without cost)
    across seeds and prints mean ± std.

Usage (from AlphaMaster with ``PYTHONPATH=src``, using the rvqlab env):
    python -m alphamaster.run_baseline --market csi300
    python -m alphamaster.run_baseline --market sp500 --seeds 5
    python -m alphamaster.run_baseline --market csi300 --smoke
    python -m alphamaster.run_baseline --market sp500 --only_backtest
"""
import inspect
import argparse
import gc
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import pprint as pp

DIRNAME = Path(__file__).absolute().resolve().parent
PROJECT_ROOT = DIRNAME.parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from alphamaster.model import MASTERTrainer

_MARKET_CONFIG = {
    "csi300": "master_csi300.yaml",
    "sp500": "master_sp500.yaml",
}

METRIC_KEYS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
]


def parse_args():
    p = argparse.ArgumentParser(description="MASTER baseline runner (CSI300 / SP500)")
    p.add_argument("--market", choices=list(_MARKET_CONFIG), default="csi300")
    p.add_argument("--config", default=None, help="override config path")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (range: seed_start..seed_start+seeds-1)")
    p.add_argument("--seed-start", type=int, default=0, help="first seed (inclusive); use 1 to skip an already-run seed 0")
    p.add_argument("--only_backtest", action="store_true", help="load saved model, skip training")
    p.add_argument("--smoke", action="store_true", help="1 epoch, seed 0 only (pipeline check)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve() if args.config else CONFIG_DIR / _MARKET_CONFIG[args.market]
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    qlib_path = Path(inspect.getfile(qlib)).resolve()
    forbidden_source = (PROJECT_ROOT.parent / "qlib-MASTER").resolve()
    if qlib_path.is_relative_to(forbidden_source):
        raise RuntimeError(f"Refusing to use Qlib from source checkout: {qlib_path}")
    print(f"pyqlib source: {qlib_path}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    qlib_init = dict(config["qlib_init"])
    qlib_init["exp_manager"] = {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            # MLflow >= 3.12 rejects a run directory if any parent directory is
            # literally named "artifacts".  Keep the tracking store at the
            # project root while model/data artifacts remain under artifacts/.
            "uri": f"file:{PROJECT_ROOT / 'mlruns'}",
            "default_exp_name": "Experiment",
        },
    }
    qlib.init(**qlib_init)
    print(
        f"[{args.market}] config={cfg_path.name}  "
        f"provider={qlib_init['provider_uri']}  region={qlib_init['region']}"
    )

    if args.smoke:
        config["task"]["model"]["kwargs"]["n_epochs"] = 1
        seeds = [0]
        print("[SMOKE] n_epochs=1, seeds=[0]  (pipeline check only)")
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        print(f"seeds: {seeds}")

    # Cache the preprocessed handler so all seeds share it. The filename is derived from the
    # segment dates via .strftime(), so the yaml dates MUST stay unquoted (datetime.date).
    seg_kwargs = config["task"]["dataset"]["kwargs"]
    h_conf = seg_kwargs["handler"]
    handler_dir = ARTIFACTS_DIR / "handlers"
    handler_dir.mkdir(parents=True, exist_ok=True)
    h_path = (
        handler_dir
        / f'handler_{args.market}_{seg_kwargs["segments"]["train"][0].strftime("%Y%m%d")}'
        f'_{seg_kwargs["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    )
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print("Save preprocessed data to", h_path)
    seg_kwargs["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config["task"]["dataset"])

    # dump_all handlers contain raw, infer, and learn copies.  This experiment
    # uses the stock learn copy for train/valid, the stock infer copy for test,
    # and only the market infer copy.  Releasing the unused in-memory copies
    # saves roughly 2.2 GB for SP500 without changing any prepared segment.
    dataset.handler._data = None
    dataset.market_dataset.handler._data = None
    dataset.market_dataset.handler._learn = None
    gc.collect()

    checkpoint_dir = ARTIFACTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {k: [] for k in METRIC_KEYS}
    for seed in seeds:
        print("------------------------")
        print(f"[{args.market}] seed: {seed}")

        config["task"]["model"]["kwargs"]["seed"] = seed
        model_kwargs = dict(config["task"]["model"]["kwargs"])
        model_kwargs["save_path"] = f"{checkpoint_dir}/"
        model = MASTERTrainer(**model_kwargs)

        if not args.only_backtest:
            model.fit(dataset=dataset)
        else:
            model.load_model(checkpoint_dir / f"{config['market']}master_{seed}.pkl")

        with R.start(experiment_name=f"{config['market']}_MASTER_seed{seed}"):
            recorder = R.get_recorder()
            SignalRecord(model, dataset, recorder).generate()
            SigAnaRecord(recorder).generate()
            PortAnaRecord(recorder, config["port_analysis_config"], "day").generate()

            metrics = recorder.list_metrics()
            print(metrics)
            result_path = RESULTS_DIR / f"{args.market}_seed{seed}_backtest.csv"
            result_row = {
                "market": args.market,
                "seed": seed,
                "experiment_name": f"{config['market']}_MASTER_seed{seed}",
                "recorder_id": recorder.id,
                **metrics,
            }
            pd.DataFrame([result_row]).to_csv(result_path, index=False)
            print("Backtest metrics saved to", result_path)
            for k in all_metrics:
                if k in metrics:
                    all_metrics[k].append(metrics[k])
            pp.pprint(all_metrics)

    print(f"\n================ Summary [{args.market}] ================")
    for k in all_metrics:
        if all_metrics[k]:
            vals = np.array(all_metrics[k], dtype=float)
            print(f"{k}: {vals.mean():.6f} ± {vals.std():.6f}  (n={len(vals)})")
        else:
            print(f"{k}: (not recorded)")


if __name__ == "__main__":
    main()
