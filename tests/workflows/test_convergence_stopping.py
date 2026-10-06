"""Convergence-based early stopping (`--convergence_*_threshold`).

Training stops once the mean and/or std of a validation metric over the last
`--convergence_window` evaluations drops below the thresholds, and the normal
end-of-training outputs (model, results) are still written.
"""

import ase.io
import numpy as np
import pytest

from macefork.tools import ConvergenceMonitor
from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train


def feed(monitor, values, metric="rmse_e_per_atom"):
    for v in values:
        monitor.update({metric: v})
    return monitor


def test_monitor_waits_for_full_window():
    monitor = ConvergenceMonitor(window=3, mean_threshold=1.0)
    assert not feed(monitor, [0.1, 0.1]).converged()
    assert feed(monitor, [0.1]).converged()


def test_monitor_uses_only_last_window():
    monitor = feed(ConvergenceMonitor(window=2, mean_threshold=1.0), [100.0, 0.1])
    assert not monitor.converged()
    assert feed(monitor, [0.1]).converged()


def test_monitor_requires_both_thresholds():
    values = [0.5, 0.7, 0.9]
    monitor = ConvergenceMonitor(window=3, mean_threshold=1.0, std_threshold=0.01)
    assert not feed(monitor, values).converged()
    monitor = ConvergenceMonitor(window=3, mean_threshold=1.0, std_threshold=1.0)
    assert feed(monitor, values).converged()
    assert np.isclose(np.std(monitor.history), np.std(values))


def test_monitor_stage_two_thresholds_are_independent():
    monitor = ConvergenceMonitor(
        window=3, mean_threshold=0.001, stage_two_std_threshold=0.01
    )
    feed(monitor, [0.05, 0.051, 0.052])  # plateaued, but not accurate enough
    assert monitor.stage_two_ready()
    assert not monitor.converged()

    only_stage_two = feed(
        ConvergenceMonitor(window=1, stage_two_mean_threshold=1.0), [0.1]
    )
    assert only_stage_two.stage_two_ready()
    assert not only_stage_two.converged()


def test_monitor_reset():
    monitor = feed(ConvergenceMonitor(window=2, std_threshold=1.0), [0.1])
    monitor.reset()
    assert not feed(monitor, [0.1]).converged()


def test_monitor_rejects_missing_metric_and_thresholds():
    with pytest.raises(ValueError):
        ConvergenceMonitor(window=3)
    monitor = ConvergenceMonitor(metric="rmse_stress", window=3, mean_threshold=1.0)
    with pytest.raises(ValueError):
        monitor.update({"rmse_e_per_atom": 0.1})


def train(tmp_path, name, **extra):
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    params = base_mace_params()
    params.update(
        {
            "name": name,
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 50,
            "eval_interval": 1,
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.update(extra)
    run_mace_train(params)
    log = "".join(p.read_text() for p in (tmp_path / "logs").glob("*.log"))
    return log


def trained_epochs(log):
    """Epochs with a validation line in the log."""
    return sorted(
        {
            int(line.split("Epoch ")[1].split(":")[0])
            for line in log.splitlines()
            if "Epoch " in line and ": head:" in line
        }
    )


def test_run_stops_on_convergence_and_writes_outputs(tmp_path):
    # thresholds that any run satisfies: stops as soon as the window is full
    log = train(
        tmp_path,
        "conv",
        convergence_window=3,
        convergence_mean_threshold=1e6,
        convergence_std_threshold=1e6,
    )

    assert "Stopping optimization: rmse_e_per_atom converged" in log
    assert trained_epochs(log) == [0, 1, 2]
    assert (tmp_path / "model" / "conv.model").exists()
    assert "===========RESULTS===========" in log


def test_convergence_in_stage_one_jumps_to_stage_two(tmp_path):
    log = train(
        tmp_path,
        "conv_swa",
        swa=None,
        start_swa=40,
        convergence_metric="rmse_f",
        convergence_window=2,
        convergence_std_threshold=1e6,
    )

    assert "rmse_f converged, starting Stage Two" in log
    assert "Stopping optimization: rmse_f converged" in log
    # epochs 0,1 in Stage One, then a fresh window of 2 in Stage Two, numbered on
    assert trained_epochs(log) == [0, 1, 2, 3]
    assert (tmp_path / "model" / "conv_swa_stagetwo.model").exists()


def test_stage_two_threshold_then_fixed_stage_two_length(tmp_path):
    # Stage Two starts on its own threshold; the stop threshold is unreachable,
    # so the run ends after stage_two_epochs of Stage Two
    log = train(
        tmp_path,
        "conv_s2",
        swa=None,
        start_swa=40,
        convergence_window=2,
        convergence_mean_threshold=1e-12,
        stage_two_convergence_std_threshold=1e6,
        stage_two_epochs=3,
    )

    assert "rmse_e_per_atom Stage Two criterion met, starting Stage Two" in log
    assert "Stopping optimization after 3 epochs of Stage Two" in log
    assert trained_epochs(log) == [0, 1, 2, 3, 4]
    assert (tmp_path / "model" / "conv_s2_stagetwo.model").exists()


def test_stage_two_epochs_waits_for_an_evaluation(tmp_path):
    log = train(
        tmp_path,
        "s2_eval",
        swa=None,
        start_swa=3,
        stage_two_epochs=2,
        eval_interval=3,
    )

    # Stage Two from epoch 3; 2 epochs would end at 4, the next evaluation is 6
    assert "Stopping optimization after 2 epochs of Stage Two" in log
    assert trained_epochs(log) == [0, 3, 6]
    assert (tmp_path / "model" / "s2_eval_stagetwo.model").exists()
