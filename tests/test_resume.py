"""Resuming a run has to restore more than weights.

The failure this guards against is silent: a resume that reloads the state dict
but not the optimiser moments or the schedule position looks exactly like a
working resume in the log, and produces a run whose LR curve nobody can
reconstruct afterwards. So these assert on the moments and the LR, not on
whether the call raised.
"""
from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from trailer.train import _checkpoint, _resume

TOTAL_STEPS = 20


def _rig(lr: float = 1e-2):
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    opt = torch.optim.AdamW([{"params": net[0].parameters(), "lr": lr * 0.1},
                             {"params": net[2].parameters(), "lr": lr}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=TOTAL_STEPS, pct_start=0.15)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return net, opt, sched, scaler


def _train_a_bit(net, opt, sched, steps: int) -> None:
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        net(torch.randn(8, 4)).square().mean().backward()
        opt.step()
        sched.step()


def _meta(epoch: int) -> dict:
    return {"arch": "unet", "encoder": "resnet34", "variants": ["dem1"],
            "epoch": epoch}


def test_round_trip_restores_weights_moments_and_schedule(tmp_path):
    net, opt, sched, scaler = _rig()
    _train_a_bit(net, opt, sched, 7)
    lr_before = [g["lr"] for g in opt.param_groups]
    moment = opt.state[net[0].weight]["exp_avg"].clone()

    ckpt = tmp_path / "last.pt"
    _checkpoint(ckpt, net, opt, sched, scaler, _meta(3), 0.42,
                [{"epoch": 0}, {"epoch": 1}])

    fresh_net, fresh_opt, fresh_sched, fresh_scaler = _rig()
    start, best, history = _resume(ckpt, fresh_net, fresh_opt, fresh_sched,
                                   fresh_scaler, torch.device("cpu"),
                                   TOTAL_STEPS, epochs=4)

    assert (start, best) == (4, 0.42)
    assert [h["epoch"] for h in history] == [0, 1]
    torch.testing.assert_close(fresh_net[0].weight, net[0].weight)
    # The moments are the part a weights-only "resume" loses, and losing them
    # costs the adaptive step sizes that took 7 steps to accumulate.
    torch.testing.assert_close(
        fresh_opt.state[fresh_net[0].weight]["exp_avg"], moment)
    # Position in the cycle, not merely a valid LR: OneCycle is non-monotonic,
    # so a wrong position can still look like a plausible learning rate.
    assert fresh_sched.last_epoch == sched.last_epoch
    assert [g["lr"] for g in fresh_opt.param_groups] == pytest.approx(lr_before)


def test_weights_only_file_is_a_warm_start_and_says_so(tmp_path, caplog):
    net, opt, sched, _ = _rig()
    _train_a_bit(net, opt, sched, 5)

    # Exactly what model.save wrote before _checkpoint existed.
    old = tmp_path / "best.pt"
    torch.save({"state_dict": net.state_dict(), "meta": _meta(5)}, old)

    fresh_net, fresh_opt, fresh_sched, fresh_scaler = _rig()
    with caplog.at_level(logging.WARNING):
        start, best, history = _resume(old, fresh_net, fresh_opt, fresh_sched,
                                       fresh_scaler, torch.device("cpu"),
                                       TOTAL_STEPS, epochs=34)

    # Weights arrive; everything else restarts. Epoch 5 in the file does NOT
    # become epoch 6 of a schedule that no longer exists.
    torch.testing.assert_close(fresh_net[0].weight, net[0].weight)
    assert (start, best, history) == (0, -1.0, [])
    assert fresh_sched.last_epoch == 0
    assert "warm start" in caplog.text.lower()
    assert "34 epochs" in caplog.text


def test_mismatched_schedule_length_warns(tmp_path, caplog):
    net, opt, sched, scaler = _rig()
    _train_a_bit(net, opt, sched, 3)
    ckpt = tmp_path / "last.pt"
    _checkpoint(ckpt, net, opt, sched, scaler, _meta(1), 0.1, [])

    fresh = _rig()
    with caplog.at_level(logging.WARNING):
        # Resuming with --epochs changed: the cycle would silently not line up.
        _resume(ckpt, *fresh[:3], fresh[3], torch.device("cpu"),
                TOTAL_STEPS * 2, epochs=8)
    assert "will not line up" in caplog.text


def test_checkpoint_stays_readable_by_the_plain_loader(tmp_path):
    """model.load reads ckpt["state_dict"] and ckpt["meta"] and nothing else.

    Export, parity and inference all go through it, so the resume extras have
    to ride alongside those two keys rather than nesting them.
    """
    net, opt, sched, scaler = _rig()
    ckpt = tmp_path / "last.pt"
    _checkpoint(ckpt, net, opt, sched, scaler, _meta(2), 0.3, [])

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert raw["meta"]["epoch"] == 2
    assert raw["state_dict"].keys() == net.state_dict().keys()
    for k in ("opt", "sched", "scaler", "best", "history"):
        assert k in raw
