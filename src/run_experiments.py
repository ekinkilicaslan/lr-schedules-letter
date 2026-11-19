"""Deney gruplarını çalıştırır ve sonuçları results/<grup>.json dosyasına yazar.

Kullanım:
    python src/run_experiments.py main        # 11 program x 5 seed (ana karşılaştırma)
    python src/run_experiments.py sweep       # tepe LR duyarlılığı
    python src/run_experiments.py ablation    # ısınma süresi, CLR yarım döngü, 1cycle pct_start, optimizör
    python src/run_experiments.py lrrange     # Smith LR aralık testi
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from common import (Config, SCHEDULES, SEEDS, RESULTS_DIR, load_splits, train_run,
                    get_device, set_seed, MLP)


def _save(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _log(msg):
    print(msg, flush=True)


def run_main(splits):
    runs = []
    for s in SCHEDULES:
        for seed in SEEDS:
            r = train_run(Config(schedule=s, seed=seed), splits)
            fin = r["final"]
            _log(f"[main] {s:16s} seed={seed} test_acc={fin['test_acc']:.4f} "
                 f"gap={fin['gap_acc']:.4f} ({fin['time_s']:.1f}s)")
            if seed != SEEDS[0]:
                r.pop("lr_trace", None)
            runs.append(r)
        _save("main", runs)


def run_sweep(splits):
    runs = []
    etas = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4]
    scheds = ["constant", "cosine", "clr_triangular2", "warmup_cosine", "one_cycle"]
    for s in scheds:
        for eta in etas:
            for seed in SEEDS[:3]:
                r = train_run(Config(schedule=s, eta=eta, seed=seed), splits,
                              record_lr=False, keep_preds=False)
                r.pop("history")
                _log(f"[sweep] {s:16s} eta={eta} seed={seed} test_acc={r['final']['test_acc']:.4f}")
                runs.append(r)
        _save("sweep", runs)


def run_ablation(splits):
    runs = []
    plans = []
    for w in [0, 1, 3, 5, 10, 20]:
        plans.append(("warmup_epochs", Config(schedule="warmup_cosine", sched_kw={"warmup_epochs": w})))
    for w in [0, 1, 3, 5, 10]:       # yüksek tepe LR altında ısınmanın stabilize edici etkisi
        plans.append(("warmup_epochs_eta0.4", Config(schedule="warmup_cosine", eta=0.4,
                                                     sched_kw={"warmup_epochs": w})))
    for st in [1, 2.5, 5, 10, 15]:
        plans.append(("clr_step_epochs", Config(schedule="clr_triangular2", sched_kw={"step_epochs": st})))
    for bf in [0.01, 0.1, 0.3]:
        plans.append(("clr_base_frac", Config(schedule="clr_triangular2", sched_kw={"base_frac": bf})))
    for p in [0.1, 0.2, 0.3, 0.45]:
        plans.append(("onecycle_pct_start", Config(schedule="one_cycle", sched_kw={"pct_start": p})))
    for s in ["constant", "cosine", "warmup_cosine", "one_cycle"]:
        plans.append(("adamw", Config(schedule=s, optimizer="adamw", eta=3e-3, weight_decay=1e-2)))
    for s in ["constant", "cosine", "clr_triangular2", "one_cycle"]:
        plans.append(("epochs20", Config(schedule=s, epochs=20)))
    for group, cfg in plans:
        for seed in SEEDS[:3]:
            r = train_run(replace(cfg, seed=seed), splits, record_lr=False, keep_preds=False)
            r["group"] = group
            r["history"] = {k: r["history"][k] for k in ("epoch", "train_loss", "val_acc", "test_acc")}
            _log(f"[ablation] {group:22s} {cfg.schedule:16s} {cfg.sched_kw} seed={seed} "
                 f"test_acc={r['final']['test_acc']:.4f}")
            runs.append(r)
        _save("ablation", runs)


def run_lrrange(splits):
    """Smith (2017) LR aralık testi: LR, 1e-4'ten 10'a üstel artırılırken eğitim kaybı izlenir."""
    device = get_device()
    out = []
    for seed in SEEDS[:3]:
        set_seed(seed)
        X = torch.as_tensor(splits["train"][0], device=device)
        y = torch.as_tensor(splits["train"][1], device=device)
        model = MLP().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        lo, hi, steps = 1e-4, 10.0, 600
        gen = torch.Generator().manual_seed(seed)
        lrs, losses, smooth = [], [], None
        for k in range(steps):
            lr = lo * (hi / lo) ** (k / (steps - 1))
            for g in opt.param_groups:
                g["lr"] = lr
            idx = torch.randint(0, len(X), (128,), generator=gen).to(device)
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = loss.item()
            smooth = v if smooth is None else 0.95 * smooth + 0.05 * v
            lrs.append(lr)
            losses.append(smooth)
            if not math.isfinite(v) or smooth > 4 * min(losses):
                break
        out.append({"seed": seed, "lr": lrs, "loss": losses})
        _log(f"[lrrange] seed={seed} steps={len(lrs)} min_loss_lr={lrs[int(np.argmin(losses))]:.4g}")
    _save("lrrange", out)


if __name__ == "__main__":
    groups = sys.argv[1:] or ["lrrange", "main", "sweep", "ablation"]
    splits = load_splits()
    for g in groups:
        {"main": run_main, "sweep": run_sweep, "ablation": run_ablation, "lrrange": run_lrrange}[g](splits)
    _log("TAMAMLANDI")
