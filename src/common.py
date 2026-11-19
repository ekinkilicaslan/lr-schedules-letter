"""Ortak bileşenler: veri yükleme, model, öğrenme oranı programları, eğitim döngüsü.

Tüm öğrenme oranı (LR) programları iterasyon düzeyinde, açık formüllerle
uygulanmıştır; böylece her programın matematiksel tanımı koddan doğrudan
okunabilir ve kitap bölümündeki denklemlerle birebir eşleşir.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "letter_recognition" / "letter-recognition.data"
RESULTS_DIR = ROOT / "results"
FEATURES = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
            "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
SEEDS = [42, 43, 44, 45, 46]          # Tüm deneylerde aynı sabit seed listesi
SPLIT_SEED = 0                        # Eğitim/doğrulama ayrımı tüm koşularda aynı


# --------------------------------------------------------------------------- #
# Tekrarlanabilirlik
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #
def load_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, header=None, names=["letter"] + FEATURES)


def load_splits():
    """Kanonik bölünme: ilk 16.000 örnek eğitim+doğrulama, son 4.000 örnek test
    (Frey ve Slate, 1991). İlk 16.000 örnekten tabakalı 2.000 örnek doğrulama
    için ayrılır. Standartlaştırma yalnızca eğitim kümesinden öğrenilir."""
    from sklearn.model_selection import train_test_split

    df = load_dataframe()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df["letter"].map(ord) - ord("A")).to_numpy(dtype=np.int64)
    X_dev, y_dev, X_te, y_te = X[:16000], y[:16000], X[16000:], y[16000:]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev, y_dev, test_size=2000, stratify=y_dev, random_state=SPLIT_SEED)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    norm = lambda a: (a - mu) / sd
    return {"train": (norm(X_tr), y_tr), "val": (norm(X_va), y_va),
            "test": (norm(X_te), y_te), "mu": mu, "sd": sd}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, d_in=16, hidden=(256, 256), n_cls=26, dropout=0.0):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Öğrenme oranı programları: lr(t), t = 0..T-1 (iterasyon), T = toplam iterasyon
# --------------------------------------------------------------------------- #
def make_schedule(name: str, eta: float, T: int, it_per_epoch: int, **kw):
    """Verilen program adı için t -> öğrenme oranı fonksiyonu döndürür."""
    E = T / it_per_epoch

    if name == "constant":
        return lambda t: eta

    if name == "step":                       # 0.5T ve 0.75T'de 10 kat azalma
        m = [int(0.5 * T), int(0.75 * T)]
        return lambda t: eta * (0.1 ** sum(t >= mi for mi in m))

    if name == "exponential":                # son değer eta * 0.01
        g = 0.01 ** (1.0 / T)
        return lambda t: eta * g ** t

    if name == "cosine":                     # Loshchilov ve Hutter (SGDR, yeniden başlatmasız)
        return lambda t: 0.5 * eta * (1 + math.cos(math.pi * t / T))

    if name in ("clr_triangular", "clr_triangular2", "clr_exp_range"):  # Smith (2017)
        base = kw.get("base_frac", 0.1) * eta
        s = int(kw.get("step_epochs", 5) * it_per_epoch)   # yarım döngü uzunluğu
        gamma = 0.1 ** (1.0 / T)

        def f(t):
            cyc = math.floor(1 + t / (2 * s))
            x = abs(t / s - 2 * cyc + 1)
            amp = (eta - base) * max(0.0, 1 - x)
            if name == "clr_triangular2":
                amp /= 2 ** (cyc - 1)
            elif name == "clr_exp_range":
                amp *= gamma ** t
            return base + amp
        return f

    if name == "sgdr":                       # Sıcak yeniden başlatmalı kosinüs, T0=4, Tmult=2
        T0 = kw.get("T0_epochs", 4) * it_per_epoch
        mult = kw.get("Tmult", 2)

        def f(t):
            Ti, tc = T0, t
            while tc >= Ti:
                tc -= Ti
                Ti *= mult
            return 0.5 * eta * (1 + math.cos(math.pi * tc / Ti))
        return f

    if name in ("warmup_cosine", "warmup_constant", "warmup_step"):   # Goyal vd. (2017)
        W = int(kw.get("warmup_epochs", 5) * it_per_epoch)

        def f(t):
            if W > 0 and t < W:
                return eta * (t + 1) / W
            if name == "warmup_constant":
                return eta
            if name == "warmup_step":
                return eta * (0.1 ** sum(t >= mi for mi in (int(0.5 * T), int(0.75 * T))))
            return 0.5 * eta * (1 + math.cos(math.pi * (t - W) / max(1, T - W)))
        return f

    if name == "one_cycle":                  # Smith ve Topin (2019)
        pct = kw.get("pct_start", 0.3)
        div, fdiv = 25.0, 1e4
        up = int(pct * T)
        lo, hi, end = eta / div, eta, eta / (div * fdiv)
        cosf = lambda a, b, p: b + 0.5 * (a - b) * (1 + math.cos(math.pi * p))

        def f(t):
            if t < up:
                return cosf(lo, hi, t / max(1, up))
            return cosf(hi, end, (t - up) / max(1, T - up))
        return f

    raise ValueError(f"Bilinmeyen program: {name}")


SCHEDULES = ["constant", "step", "exponential", "cosine", "clr_triangular",
             "clr_triangular2", "clr_exp_range", "sgdr", "warmup_constant",
             "warmup_cosine", "one_cycle"]

SCHEDULE_LABELS = {
    "constant": "Sabit", "step": "Basamaklı", "exponential": "Üstel",
    "cosine": "Kosinüs", "clr_triangular": "CLR-üçgen", "clr_triangular2": "CLR-üçgen2",
    "clr_exp_range": "CLR-üstel aralık", "sgdr": "SGDR", "warmup_constant": "Isınma+sabit",
    "warmup_cosine": "Isınma+kosinüs", "warmup_step": "Isınma+basamaklı", "one_cycle": "Tek döngü (1cycle)",
}


# --------------------------------------------------------------------------- #
# Eğitim ve değerlendirme
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    schedule: str = "constant"
    eta: float = 0.1
    epochs: int = 60
    batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 5e-4
    hidden: tuple = (256, 256)
    dropout: float = 0.0
    optimizer: str = "sgd"
    seed: int = 42
    sched_kw: dict = field(default_factory=dict)


@torch.no_grad()
def evaluate(model, X, y, bs=4096):
    model.eval()
    loss_sum, correct, logits_all = 0.0, 0, []
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
        logits_all.append(out)
    return loss_sum / len(X), correct / len(X), torch.cat(logits_all)


def expected_calibration_error(logits, y, n_bins=15):
    p = F.softmax(logits, 1)
    conf, pred = p.max(1)
    acc = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return ece.item()


@torch.no_grad()
def sharpness(model, X, y, alpha=0.05, n_draws=10, seed=0):
    """Ağırlıklara göreli Gauss gürültüsü eklendiğinde eğitim kaybındaki ortalama
    artış (düzlük/keskinlik göstergesi; Keskar vd., 2017 yaklaşımından esinlenmiştir)."""
    base, _, _ = evaluate(model, X, y)
    params = [p for p in model.parameters()]
    orig = [p.detach().clone() for p in params]
    g = torch.Generator(device=X.device).manual_seed(seed)
    inc = []
    for _ in range(n_draws):
        for p, o in zip(params, orig):
            p.copy_(o + alpha * o.abs() * torch.randn(o.shape, generator=g, device=o.device))
        l, _, _ = evaluate(model, X, y)
        inc.append(l - base)
    for p, o in zip(params, orig):
        p.copy_(o)
    return float(np.mean(inc))


def macro_f1(y_true, y_pred, n_cls=26):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_cls))))


def train_run(cfg: Config, splits=None, device=None, record_lr=True, keep_preds=True):
    device = device or get_device()
    splits = splits or load_splits()
    set_seed(cfg.seed)
    to_t = lambda a, dt: torch.as_tensor(np.array(a), dtype=dt, device=device)
    Xtr, ytr = to_t(splits["train"][0], torch.float32), to_t(splits["train"][1], torch.long)
    Xva, yva = to_t(splits["val"][0], torch.float32), to_t(splits["val"][1], torch.long)
    Xte, yte = to_t(splits["test"][0], torch.float32), to_t(splits["test"][1], torch.long)

    model = MLP(hidden=tuple(cfg.hidden), dropout=cfg.dropout).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.eta, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.eta, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(cfg.optimizer)

    n = len(Xtr)
    ipe = math.ceil(n / cfg.batch_size)
    T = ipe * cfg.epochs
    lr_fn = make_schedule(cfg.schedule, cfg.eta, T, ipe, **cfg.sched_kw)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    hist = {k: [] for k in ["epoch", "lr", "train_loss", "train_acc", "val_loss",
                            "val_acc", "test_loss", "test_acc", "batch_loss"]}
    lr_trace = []
    best = {"val_acc": -1.0}
    t = 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        run_loss = torch.zeros((), device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            lr = lr_fn(t)
            for gpar in opt.param_groups:
                gpar["lr"] = lr
            if record_lr:
                lr_trace.append(lr)
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.detach() * len(idx)
            t += 1
        trl, tra, _ = evaluate(model, Xtr, ytr)
        val, vaa, _ = evaluate(model, Xva, yva)
        tel, tea, _ = evaluate(model, Xte, yte)
        for k, v in zip(hist.keys(), [ep, lr, trl, tra, val, vaa, tel, tea, run_loss.item() / n]):
            hist[k].append(v)
        if vaa > best["val_acc"]:
            best = {"epoch": ep, "val_acc": vaa, "test_acc": tea, "test_loss": tel}
        if not math.isfinite(trl):
            break

    tel, tea, logits = evaluate(model, Xte, yte)
    preds = logits.argmax(1).cpu().numpy()
    final = {
        "train_loss": hist["train_loss"][-1], "train_acc": hist["train_acc"][-1],
        "val_acc": hist["val_acc"][-1], "test_loss": tel, "test_acc": tea,
        "test_macro_f1": macro_f1(splits["test"][1], preds),
        "gap_acc": hist["train_acc"][-1] - tea, "gap_loss": tel - hist["train_loss"][-1],
        "ece": expected_calibration_error(logits, yte),
        "sharpness": sharpness(model, Xtr, ytr),
        "epochs_to_97val": next((e for e, a in zip(hist["epoch"], hist["val_acc"]) if a >= 0.97), None),
        "best_val": best, "time_s": time.time() - t0,
    }
    out = {"config": asdict(cfg), "history": hist, "final": final}
    if record_lr:
        out["lr_trace"] = lr_trace[::max(1, ipe // 10)]   # seyreltilmiş iz
    if keep_preds:
        out["test_preds"] = preds.tolist()
    return out
