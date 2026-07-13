# AlphaMaster

AlphaMaster is an independent MASTER research project. It uses the Qlib data,
processing, recording, and backtest APIs installed in the `rvqlab` Conda
environment; it does not import or install the sibling `qlib-MASTER` source
checkout.

The implementation was migrated from these existing files and then adapted in
place:

- `qlib/contrib/model/pytorch_master_ts.py`
- `qlib/contrib/data/dataset.py` (`marketDataHandler` and `MASTERTSDatasetH`)
- `examples/benchmarks/MASTER/paper_baseline/run_baseline.py`
- `examples/benchmarks/MASTER/paper_baseline/us_handlers.py`
- the CSI300 and SP500 paper-baseline YAML files

No historical `qlib-MASTER/results` artifacts are included.

## Design

- `MASTER` is a normal `torch.nn.Module`.
- `MASTERTrainer` is a plain Python trainer and does not inherit a Qlib model.
- Qlib 0.9.7 remains responsible for data access, Alpha158 processing,
  time-series sampling, experiment records, signal analysis, and backtesting.
- The original 158 stock features, 63 market features, eight-day window,
  daily cross-sectional batches, model parameters, splits, and trading
  configuration are retained.

## Run with rvqlab

No package installation is required. The wrapper exposes `src/` on
`PYTHONPATH` and invokes the existing Conda environment directly:

```bash
cd /home/nbcctwya/FactorMaster/AlphaMaster
bash scripts/run_baseline.sh smoke
```

Run one market directly:

```bash
PYTHONPATH=src conda run -n rvqlab --no-capture-output \
  python -m alphamaster.run_baseline --market csi300 --smoke
```

Full five-seed runs are deliberately launched as separate processes:

```bash
bash scripts/run_baseline.sh full
```

Checkpoints, preprocessed handlers, and experiment records are runtime
artifacts and are excluded from version control.

## Verify

```bash
cd /home/nbcctwya/FactorMaster/AlphaMaster
PYTHONPATH=src conda run -n rvqlab python -m unittest discover -s tests -v
```
