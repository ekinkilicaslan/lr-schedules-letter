"""Tüm şekilleri (figures/*.png, 300 DPI) ve tabloları (tables/*.csv + tables/tables.md) üretir.

Renk kuralları: kategorik seriler sabit sırada (en fazla 4 seri/panel, çizgi stili ile
ikincil kodlama); büyüklük için tek tonlu ardışık ölçek; korelasyon için iki tonlu
ıraksak ölçek ve nötr orta nokta. Sabit LR her panelde gri referans çizgisidir.
"""
from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.metrics import confusion_matrix

from common import (FEATURES, RESULTS_DIR, ROOT, SCHEDULE_LABELS, SCHEDULES, SEEDS,
                    load_dataframe, load_splits, make_schedule)

FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

# Doğrulanmış kategorik palet (ilk 4 yuva) + nötr referans
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REF = "#8a8986"
INK, INK2 = "#0b0b0b", "#52514e"
LS = ["-", "--", "-.", ":"]

import locale
try:
    locale.setlocale(locale.LC_NUMERIC, "Turkish_Turkey.1254")
except locale.Error:
    pass
trn = lambda s: s.replace(".", ",")
plt.rcParams.update({
    "axes.formatter.use_locale": True,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.dpi": 110,
})

FAMILIES = {
    "Azalan (tavlamalı) programlar": ["step", "exponential", "cosine"],
    "Döngüsel programlar": ["clr_triangular", "clr_triangular2", "clr_exp_range", "sgdr"],
    "Isınmalı programlar": ["warmup_constant", "warmup_cosine", "one_cycle"],
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("şekil:", name)


def fmt(m, s, d=2):
    return f"{m:.{d}f} ± {s:.{d}f}"


# --------------------------------------------------------------------------- #
md = []   # tables.md içeriği


def add_table(name, title, df, note=None):
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig")
    md.append(f"### {title}\n\n" + df.to_markdown(index=False) + ("\n\n" + note if note else "") + "\n")
    print("tablo:", name)


# =========================================================================== #
# Veri seti: G1, G2, T1, T2, T3
# =========================================================================== #
df = load_dataframe()
sp = load_splits()

# G1a — öznitelik histogramları
fig, axes = plt.subplots(4, 4, figsize=(10, 8.2))
for ax, f in zip(axes.flat, FEATURES):
    ax.hist(df[f], bins=np.arange(-0.5, 16.5, 1), color=C[0], edgecolor="white", linewidth=0.8)
    ax.set_title(f, fontsize=10)
    ax.set_xlim(-0.5, 15.5)
    ax.tick_params(labelsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Frekans")
for ax in axes[-1, :]:
    ax.set_xlabel("Tam sayı değer (0–15)")
fig.suptitle("16 özniteliğin dağılımı (n = 20.000)", y=1.0)
fig.tight_layout()
save(fig, "G1a_oznitelik_histogramlari")

# G1b — sınıf dağılımı + standartlaştırılmış kutu grafikleri
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
cnt = df["letter"].value_counts().sort_index()
a1.bar(cnt.index, cnt.values, color=C[0], width=0.75)
a1.axhline(cnt.mean(), color=REF, ls="--", lw=1)
a1.text(25.4, cnt.mean() + 4, f"ort. {cnt.mean():.0f}", ha="right", va="bottom", fontsize=8, color=INK2)
a1.set_ylim(650, 840)
a1.set_xlabel("Sınıf (harf)")
a1.set_ylabel("Örnek sayısı")
a1.set_title("(a) Sınıf dağılımı")
a1.grid(axis="x", visible=False)
z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
bp = a2.boxplot([z[f] for f in FEATURES], tick_labels=FEATURES, widths=0.55, patch_artist=True,
                flierprops=dict(marker=".", markersize=2, markeredgecolor=REF, alpha=0.4),
                medianprops=dict(color=INK, lw=1.2))
for b in bp["boxes"]:
    b.set(facecolor="#cfe0f6", edgecolor=C[0])
a2.set_ylabel("z-puanı")
a2.set_title("(b) Standartlaştırılmış öznitelik kutu grafikleri")
a2.tick_params(axis="x", rotation=60, labelsize=8)
fig.tight_layout()
save(fig, "G1b_sinif_dagilimi_kutu")

# G2 — korelasyon ısı haritası
corr = df[FEATURES].corr(method="pearson")
fig, ax = plt.subplots(figsize=(7.4, 6.2))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=corr.map(lambda v: trn(f"{v:.2f}")), fmt="",
            annot_kws={"size": 6.5}, linewidths=0.6, linecolor="white", square=True,
            cbar_kws={"label": "Pearson r", "shrink": 0.8}, ax=ax)
ax.set_title("Öznitelikler arası Pearson korelasyon matrisi")
ax.grid(False)
save(fig, "G2_korelasyon_isi_haritasi")

# T1 — genel istatistik
t1 = pd.DataFrame([
    ["Toplam örnek", "20.000"], ["Öznitelik sayısı", "16 (tam sayı, 0–15)"], ["Sınıf sayısı", "26 (A–Z)"],
    ["Eksik değer", str(int(df.isna().sum().sum()))],
    ["Yinelenen satır", str(int(df.duplicated().sum()))],
    ["En küçük / en büyük sınıf", f"{cnt.idxmin()} ({cnt.min()}) / {cnt.idxmax()} ({cnt.max()})"],
    ["Sınıf dengesizlik oranı (maks/min)", f"{cnt.max() / cnt.min():.3f}"],
    ["Eğitim / doğrulama / test", "14.000 / 2.000 / 4.000"],
    ["Bölünme kuralı", "İlk 16.000 → eğitim+doğrulama (tabakalı 2.000 doğrulama), son 4.000 → test [Frey ve Slate]"],
    ["Mutlak |r| > 0,7 olan öznitelik çifti", str(int(((corr.abs() > 0.7) & ~np.eye(16, dtype=bool)).values.sum() // 2))],
], columns=["Özellik", "Değer"])
add_table("T1_veri_seti_ozeti", "Tablo T1. Veri seti genel istatistik özeti", t1)

desc = {
    "x-box": "Sınırlayıcı kutunun yatay konumu", "y-box": "Sınırlayıcı kutunun dikey konumu",
    "width": "Kutu genişliği", "high": "Kutu yüksekliği", "onpix": "Toplam açık (siyah) piksel sayısı",
    "x-bar": "Açık piksellerin ortalama x konumu", "y-bar": "Açık piksellerin ortalama y konumu",
    "x2bar": "x'in ortalama varyansı", "y2bar": "y'nin ortalama varyansı", "xybar": "Ortalama x·y korelasyonu",
    "x2ybr": "x·x·y ortalaması", "xy2br": "x·y·y ortalaması", "x-ege": "Soldan sağa ortalama kenar sayısı",
    "xegvy": "x-ege ile y korelasyonu", "y-ege": "Aşağıdan yukarı ortalama kenar sayısı",
    "yegvx": "y-ege ile x korelasyonu",
}
t2 = pd.DataFrame([[i + 1, f, desc[f], f"{df[f].min()}–{df[f].max()}", f"{df[f].mean():.2f}", f"{df[f].std():.2f}",
                    f"{df[f].skew():.2f}"] for i, f in enumerate(FEATURES)],
                  columns=["#", "Öznitelik", "Tanım", "Aralık", "Ortalama", "Std. sapma", "Çarpıklık"])
add_table("T2_oznitelik_tanimlari", "Tablo T2. Öznitelik tanımlama tablosu", t2)

t3 = pd.DataFrame([
    [1, "Okuma ve tip dönüşümü", "CSV → float32 öznitelik, harf → 0–25 tamsayı etiket", "Tüm veri"],
    [2, "Kalite denetimi", "Eksik değer (0) ve aralık (0–15) kontrolü", "Tüm veri"],
    [3, "Kanonik test ayrımı", "Son 4.000 örnek test olarak ayrıldı (Frey ve Slate protokolü)", "Tüm veri"],
    [4, "Doğrulama ayrımı", "İlk 16.000 örnekten tabakalı 2.000 örnek (random_state=0)", "Eğitim+doğrulama"],
    [5, "Standartlaştırma", "z = (x − μ)/σ; μ ve σ yalnızca 14.000 eğitim örneğinden", "Tüm kümeler"],
    [6, "Tensör ve cihaz", "PyTorch tensörü, GPU belleğinde; mini-yığın 128, her epok yeniden karıştırma", "Eğitim"],
], columns=["Adım", "İşlem", "Ayrıntı", "Uygulandığı küme"])
add_table("T3_on_isleme_adimlari", "Tablo T3. Veri ön işleme adımları", t3)

# =========================================================================== #
# G3 — mimari şeması ve G0 — LR program şekilleri
# =========================================================================== #
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.axis("off")
boxes = [("Girdi\n16 öznitelik\n(z-puanı)", "#e8eef8"), ("Tam bağlı\n16→256\n+ ReLU", "#cfe0f6"),
         ("Tam bağlı\n256→256\n+ ReLU", "#cfe0f6"), ("Tam bağlı\n256→26\n(logit)", "#cfe0f6"),
         ("Softmax +\nçapraz entropi\n26 sınıf", "#e8eef8")]
xs = np.linspace(0.06, 0.94, len(boxes))
for x, (t, fc) in zip(xs, boxes):
    ax.add_patch(FancyBboxPatch((x - 0.075, 0.28), 0.15, 0.44, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=C[0], lw=1.2, transform=ax.transAxes))
    ax.text(x, 0.5, t, ha="center", va="center", fontsize=9.5, transform=ax.transAxes)
for a, b in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(b - 0.078, 0.5), xytext=(a + 0.078, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
ax.text(0.5, 0.12, "Parametre: 16·256+256 + 256·256+256 + 256·26+26 = 76.826   |   "
        "SGD (momentum 0,9, ağırlık azaltma 5·10⁻⁴), mini-yığın 128, 60 epok, ηₜ = programa göre",
        ha="center", fontsize=9, color=INK2, transform=ax.transAxes)
ax.text(0.5, 0.9, "Çok katmanlı algılayıcı (MLP) mimarisi", ha="center", fontsize=11, transform=ax.transAxes)
save(fig, "G3_model_mimarisi")

ipe, E = 110, 60
T = ipe * E
tt = np.arange(T)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, (fam, names) in zip(axes, FAMILIES.items()):
    f0 = make_schedule("constant", 0.1, T, ipe)
    ax.plot(tt / ipe, [f0(t) for t in tt], color=REF, ls="-", lw=1.2, label="Sabit")
    for i, n in enumerate(names):
        f = make_schedule(n, 0.1, T, ipe)
        ax.plot(tt / ipe, [f(t) for t in tt], color=C[i], ls=LS[i], label=SCHEDULE_LABELS[n])
    ax.set_title(fam)
    ax.set_xlabel("Epok")
    ax.legend(loc="upper right", frameon=False)
axes[0].set_ylabel("Öğrenme oranı η")
fig.suptitle("İncelenen öğrenme oranı programları (η_maks = 0,1; 60 epok × 110 iterasyon)", y=1.02)
fig.tight_layout()
save(fig, "G0_lr_program_sekilleri")

# =========================================================================== #
# Ana deney: G4, G5, G6, G7, T6, T10
# =========================================================================== #
main = json.load(open(RESULTS_DIR / "main.json", encoding="utf-8"))
by = defaultdict(list)
for r in main:
    by[r["config"]["schedule"]].append(r)


def curve(name, key):
    a = np.array([r["history"][key] for r in by[name]])
    return a.mean(0), a.std(0, ddof=1)


ep = np.arange(1, 61)
for gname, keys, ylab, title, ylim in [
    ("G4_kayip_egrileri", ("train_loss", "val_loss"), "Çapraz entropi kaybı",
     "Eğitim (ince) ve doğrulama (kalın) kayıp eğrileri, 5 seed ortalaması", (0, 0.45)),
    ("G5_dogruluk_egrileri", ("train_acc", "val_acc"), "Doğruluk",
     "Eğitim (ince) ve doğrulama (kalın) doğruluk eğrileri, 5 seed ortalaması", (0.88, 1.0)),
]:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, (fam, names) in zip(axes, FAMILIES.items()):
        for j, n in enumerate(["constant"] + names):
            col = REF if n == "constant" else C[j - 1]
            ls = "-" if n == "constant" else LS[j - 1]
            mtr, _ = curve(n, keys[0])
            mva, sva = curve(n, keys[1])
            ax.plot(ep, mtr, color=col, ls=ls, lw=0.9, alpha=0.75)
            ax.plot(ep, mva, color=col, ls=ls, lw=2.0, label=SCHEDULE_LABELS[n])
            ax.fill_between(ep, mva - sva, mva + sva, color=col, alpha=0.12, lw=0)
        ax.set_title(fam)
        ax.set_xlabel("Epok")
        ax.set_ylim(*ylim)
        ax.legend(loc="lower right" if "dogruluk" in gname else "upper right", frameon=False)
    axes[0].set_ylabel(ylab)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, gname)

# T6 — ana metrikler + eşleştirilmiş t testi (aynı seed = aynı başlatma ve karıştırma) + Holm düzeltmesi
rows, pvals = [], []
base = np.array([r["final"]["test_acc"] for r in sorted(by["constant"], key=lambda r: r["config"]["seed"])])
for n in SCHEDULES:
    rs = sorted(by[n], key=lambda r: r["config"]["seed"])
    g = lambda k: np.array([r["final"][k] for r in rs])
    acc = g("test_acc") * 100
    p = stats.ttest_rel(acc, base * 100).pvalue if n != "constant" else np.nan
    pvals.append(p)
    rows.append({
        "Program": SCHEDULE_LABELS[n], "Test doğruluğu (%)": fmt(acc.mean(), acc.std(ddof=1)),
        "Makro-F1 (%)": f"{g('test_macro_f1').mean() * 100:.2f}",
        "Eğitim doğruluğu (%)": f"{g('train_acc').mean() * 100:.2f}",
        "Genelleme aralığı (puan)": fmt(g("gap_acc").mean() * 100, g("gap_acc").std(ddof=1) * 100),
        "Test kaybı": f"{g('test_loss').mean():.3f}", "ECE (%)": f"{g('ece').mean() * 100:.2f}",
        "Keskinlik (×10⁻³)": f"{g('sharpness').mean() * 1e3:.2f}",
        "Δ sabit (puan)": "—" if n == "constant" else f"{acc.mean() - base.mean() * 100:+.2f}",
        "_acc": acc.mean(), "_n": n,
    })
# Holm
idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
order = sorted(idx, key=lambda i: pvals[i])
m = len(order)
adj = {}
running = 0
for k, i in enumerate(order):
    running = max(running, min(1.0, (m - k) * pvals[i]))
    adj[i] = running
for i, r in enumerate(rows):
    r["p (Holm)"] = "—" if i not in adj else (f"{adj[i]:.4f}" if adj[i] >= 1e-4 else "<0,0001")
t6 = pd.DataFrame(rows)
best_name = t6.sort_values("_acc", ascending=False).iloc[0]["_n"]
t6 = t6.drop(columns=["_acc", "_n"])
add_table("T6_ana_performans", "Tablo T6. Ana performans metrikleri (5 seed, ortalama ± std; test kümesi n = 4.000)", t6,
          "p: sabit LR'ye karşı eşleştirilmiş t testi, Holm–Bonferroni düzeltmeli (10 karşılaştırma).")

# G6a — test doğruluğu nokta grafiği (ort ± std), G6b — tepe LR duyarlılığı
sweep = json.load(open(RESULTS_DIR / "sweep.json", encoding="utf-8"))
sw = defaultdict(list)
for r in sweep:
    sw[(r["config"]["schedule"], r["config"]["eta"])].append(r["final"]["test_acc"] * 100)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
order_n = sorted(SCHEDULES, key=lambda n: np.mean([r["final"]["test_acc"] for r in by[n]]))
fam_of = {n: i for i, ns in enumerate(FAMILIES.values()) for n in ns}
for y, n in enumerate(order_n):
    acc = np.array([r["final"]["test_acc"] for r in by[n]]) * 100
    col = REF if n == "constant" else C[fam_of[n]]
    a1.errorbar(acc.mean(), y, xerr=acc.std(ddof=1), fmt="o", ms=7, color=col, ecolor=col, capsize=3, lw=1.4)
    a1.scatter(acc, np.full_like(acc, y) + 0.22, s=9, color=col, alpha=0.6, lw=0)
    a1.text(acc.mean(), y - 0.36, trn(f"{acc.mean():.2f}"), ha="center", fontsize=7.5, color=INK2)
a1.set_yticks(range(len(order_n)), [SCHEDULE_LABELS[n] for n in order_n])
a1.set_xlabel("Test doğruluğu (%)")
a1.set_title("(a) Programlara göre test doğruluğu (ort. ± std, noktalar: seed)")
a1.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
a1.legend(handles=[Line2D([], [], marker="o", ls="", color=REF, label="Referans (sabit)")] +
          [Line2D([], [], marker="o", ls="", color=C[i], label=f) for i, f in enumerate(FAMILIES)],
          loc="lower right", frameon=False, fontsize=7.5)
etas = sorted({k[1] for k in sw})
mk = ["o", "s", "^", "D", "v"]
for i, n in enumerate(["constant", "cosine", "clr_triangular2", "warmup_cosine", "one_cycle"]):
    mu = [np.mean(sw[(n, e)]) for e in etas]
    sd = [np.std(sw[(n, e)], ddof=1) for e in etas]
    col = REF if n == "constant" else C[i - 1]
    a2.errorbar(etas, mu, yerr=sd, marker=mk[i], ms=6, color=col, ls=LS[(i - 1) % 4] if i else "-",
                capsize=2.5, label=SCHEDULE_LABELS[n])
a2.set_xscale("log")
a2.set_xticks(etas, [trn(str(e)) for e in etas])
a2.set_xlabel("Tepe öğrenme oranı η_maks (log ölçek)")
a2.set_ylabel("Test doğruluğu (%)")
a2.set_title("(b) Tepe öğrenme oranına duyarlılık (3 seed)")
a2.legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "G6_karsilastirmali_analiz")

# Duyarlılık tablosu (ek)
sw_rows = []
for n in ["constant", "cosine", "clr_triangular2", "warmup_cosine", "one_cycle"]:
    row = {"Program": SCHEDULE_LABELS[n]}
    for e in etas:
        row[f"η={e}"] = fmt(np.mean(sw[(n, e)]), np.std(sw[(n, e)], ddof=1))
    va_ = {e: np.mean([r["final"]["val_acc"] for r in sweep if r["config"]["schedule"] == n and r["config"]["eta"] == e]) for e in etas}
    e_star = max(etas, key=lambda e: va_[e])
    row["Doğrulamayla seçilen η*"] = str(e_star)
    row["η*'da test (%)"] = fmt(np.mean(sw[(n, e_star)]), np.std(sw[(n, e_star)], ddof=1))
    sw_rows.append(row)
add_table("T6b_lr_duyarlilik", "Tablo T6b. Tepe öğrenme oranı duyarlılığı — test doğruluğu (%), 3 seed", pd.DataFrame(sw_rows), "η*: son epok doğrulama doğruluğu ortalamasına göre seçilen tepe değer (test kümesi seçimde kullanılmamıştır).")

# G7 — karışıklık matrisi (en iyi program, 5 seed toplamı, satır normalize)
yte = sp["test"][1]
letters = [chr(65 + i) for i in range(26)]
cm_best = sum(confusion_matrix(yte, r["test_preds"], labels=range(26)) for r in by[best_name])
cm_const = sum(confusion_matrix(yte, r["test_preds"], labels=range(26)) for r in by["constant"])
fig, ax = plt.subplots(figsize=(8.2, 7))
cmn = cm_best / cm_best.sum(1, keepdims=True)
sns.heatmap(cmn * 100, cmap="Blues", vmin=0, vmax=100, square=True, xticklabels=letters, yticklabels=letters,
            linewidths=0.3, linecolor="white", cbar_kws={"label": "Satır yüzdesi (%)", "shrink": 0.8}, ax=ax)
off = cmn.copy()
np.fill_diagonal(off, 0)
for (i, j) in zip(*np.where(off >= 0.02)):
    ax.text(j + 0.5, i + 0.5, f"{off[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
ax.set_xlabel("Tahmin edilen sınıf")
ax.set_ylabel("Gerçek sınıf")
ax.set_title(f"Karışıklık matrisi — {SCHEDULE_LABELS[best_name]} (5 seed toplamı, test)\n"
             "köşegen dışı ≥%2 hücreler etiketli")
ax.grid(False)
save(fig, "G7_karisiklik_matrisi")

# T10 — hata analizi
pairs = []
for i in range(26):
    for j in range(26):
        if i != j:
            pairs.append((cm_const[i, j] / 5, cm_best[i, j] / 5, letters[i], letters[j]))
pairs.sort(key=lambda x: -x[1])
t10a = pd.DataFrame([[f"{a} → {b}", f"{c:.1f}", f"{bb:.1f}", f"{bb - c:+.1f}"] for c, bb, a, b in pairs[:10]],
                    columns=["Gerçek → tahmin", "Sabit (ort. hata/seed)", f"{SCHEDULE_LABELS[best_name]} (ort. hata/seed)", "Fark"])
pc_best = np.diag(cm_best) / cm_best.sum(1) * 100
pc_const = np.diag(cm_const) / cm_const.sum(1) * 100
worst = np.argsort(pc_best)[:6]
t10b = pd.DataFrame([[letters[k], int(cm_best[k].sum() / 5), f"{pc_const[k]:.2f}", f"{pc_best[k]:.2f}",
                      f"{pc_best[k] - pc_const[k]:+.2f}"] for k in worst],
                    columns=["Sınıf", "Test örneği", "Sabit doğruluk (%)", f"{SCHEDULE_LABELS[best_name]} doğruluk (%)", "Fark (puan)"])
add_table("T10a_karistirilan_ciftler", "Tablo T10a. En sık karıştırılan sınıf çiftleri (test, seed başına ortalama hata sayısı)", t10a)
add_table("T10b_en_zor_siniflar", "Tablo T10b. En düşük sınıf doğruluklu 6 harf", t10b)

# Ek şekil — genelleme aralığı vs test doğruluğu ve keskinlik
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for n in SCHEDULES:
    col = REF if n == "constant" else C[fam_of[n]]
    gx = np.mean([r["final"]["gap_acc"] for r in by[n]]) * 100
    ty = np.mean([r["final"]["test_acc"] for r in by[n]]) * 100
    sh = np.mean([r["final"]["sharpness"] for r in by[n]]) * 1e3
    a1.scatter(gx, ty, s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
    a1.annotate(SCHEDULE_LABELS[n], (gx, ty), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    a2.scatter(sh, ty, s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
    a2.annotate(SCHEDULE_LABELS[n], (sh, ty), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
a1.set_xlabel("Genelleme aralığı: eğitim − test doğruluğu (puan)")
a1.set_ylabel("Test doğruluğu (%)")
a1.set_title("(a) Genelleme aralığı ve test doğruluğu")
a2.set_xlabel("Keskinlik: göreli ağırlık gürültüsünde kayıp artışı (×10⁻³)")
a2.set_title("(b) Minimum keskinliği ve test doğruluğu")
a2.legend(handles=[Line2D([], [], marker="o", ls="", color=REF, label="Sabit")] +
          [Line2D([], [], marker="o", ls="", color=C[i], label=f) for i, f in enumerate(FAMILIES)],
          frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
save(fig, "G8_genelleme_keskinlik")

# =========================================================================== #
# Ablasyon: T7 + G9 (ısınma η=0,4) ; LR aralık testi: G10
# =========================================================================== #
abl = json.load(open(RESULTS_DIR / "ablation.json", encoding="utf-8"))
ag = defaultdict(list)
for r in abl:
    c = r["config"]
    key = (r["group"], c["schedule"], json.dumps(c["sched_kw"], sort_keys=True))
    ag[key].append(r)
group_names = {
    "warmup_epochs": "Isınma süresi (Isınma+kosinüs, η=0,1)",
    "warmup_epochs_eta0.4": "Isınma süresi (Isınma+kosinüs, η=0,4)",
    "clr_step_epochs": "CLR yarım döngü (CLR-üçgen2)",
    "clr_base_frac": "CLR alt sınır η_min/η_maks (CLR-üçgen2)",
    "onecycle_pct_start": "1cycle yükselme oranı pct_start",
    "adamw": "Optimizör: AdamW (η=3·10⁻³)",
    "epochs20": "Kısa bütçe: 20 epok",
}
t7 = []
for (g, s, kw), rs in ag.items():
    acc = np.array([r["final"]["test_acc"] for r in rs]) * 100
    gap = np.array([r["final"]["gap_acc"] for r in rs]) * 100
    kwd = json.loads(kw)
    setting = ", ".join(f"{k}={v}" for k, v in kwd.items()) if kwd else SCHEDULE_LABELS[s]
    if g in ("adamw", "epochs20"):
        setting = SCHEDULE_LABELS[s]
    div = sum(not np.isfinite(r["final"]["train_loss"]) or r["final"]["train_acc"] < 0.2 for r in rs)
    t7.append([group_names[g], setting, fmt(acc.mean(), acc.std(ddof=1)), fmt(gap.mean(), gap.std(ddof=1)),
               f"{div}/{len(rs)}"])
add_table("T7_ablasyon", "Tablo T7. Ablasyon çalışması (3 seed; test doğruluğu ve genelleme aralığı)",
          pd.DataFrame(t7, columns=["Ablasyon grubu", "Ayar", "Test doğruluğu (%)", "Genelleme aralığı (puan)", "Iraksayan koşu"]))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
for i, g in enumerate(["warmup_epochs", "warmup_epochs_eta0.4"]):
    items = sorted([(json.loads(k[2])["warmup_epochs"], v) for k, v in ag.items() if k[0] == g])
    xs_ = [w for w, _ in items]
    mu = [np.mean([r["final"]["test_acc"] for r in v]) * 100 for _, v in items]
    sd = [np.std([r["final"]["test_acc"] for r in v], ddof=1) * 100 for _, v in items]
    a1.errorbar(xs_, mu, yerr=sd, marker="os"[i], ms=6, color=C[i], ls=LS[i], capsize=3,
                label="η_maks = 0,1" if i == 0 else "η_maks = 0,4")
    # erken epok eğitim kaybı (ilk 5 epok) — kararsızlık göstergesi
    for w, v in items:
        if w in (0, 5) and g == "warmup_epochs_eta0.4":
            h = np.mean([r["history"]["train_loss"] for r in v], 0)
            a2.plot(np.arange(1, len(h) + 1), h, color=C[0] if w == 0 else C[1], ls="-" if w == 0 else "--",
                    label=f"Isınma {w} epok (η_maks = 0,4)")
a1.set_xlabel("Isınma süresi (epok)")
a1.set_ylabel("Test doğruluğu (%)")
a1.set_title("(a) Isınma süresinin etkisi (Isınma+kosinüs)")
a1.legend(frameon=False)
a2.set_yscale("log")
a2.set_xlabel("Epok")
a2.set_ylabel("Eğitim kaybı (log)")
a2.set_title("(b) Yüksek tepe LR'de ısınmalı ve ısınmasız eğitim kaybı")
a2.legend(frameon=False)
fig.tight_layout()
save(fig, "G9_isinma_ablasyonu")

lr = json.load(open(RESULTS_DIR / "lrrange.json", encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, r in enumerate(lr):
    ax.plot(r["lr"], r["loss"], color=C[i], ls=LS[i], label=f"seed {r['seed']}")
ax.set_xscale("log")
ax.axvline(0.1, color=REF, ls="--", lw=1)
ax.text(0.105, ax.get_ylim()[1] * 0.92, "seçilen η_maks = 0,1", fontsize=8, color=INK2)
ax.set_xlabel("Öğrenme oranı (log ölçek)")
ax.set_ylabel("Düzleştirilmiş eğitim kaybı")
ax.set_title("LR aralık testi (Smith, 2017): üstel artan LR ile kayıp")
ax.legend(frameon=False)
save(fig, "G10_lr_aralik_testi")
lr_min = [r["lr"][int(np.argmin(r["loss"]))] for r in lr]

# =========================================================================== #
# T4, T5, T8
# =========================================================================== #
t4 = pd.DataFrame([
    ["Mimari", "16-256-256-26, ReLU, He (Kaiming) normal başlatma, bias = 0"], ["Parametre sayısı", "76.826"],
    ["Kayıp fonksiyonu", "Çapraz entropi (softmax)"], ["Optimizör", "SGD, momentum β = 0,9 (Nesterov değil)"],
    ["Ağırlık azaltma (L2)", "5·10⁻⁴"], ["Mini-yığın", "128 (110 iterasyon/epok)"], ["Epok", "60 (T = 6.600 iterasyon)"],
    ["Tepe öğrenme oranı η_maks", "0,1 (tüm programlarda ortak; LR aralık testiyle seçildi)"],
    ["Basamaklı", "0,5T ve 0,75T'de ×0,1"], ["Üstel", "γ = 0,01^(1/T) (son LR = 0,01·η_maks)"],
    ["Kosinüs", "η_min = 0, yeniden başlatma yok"], ["CLR (üçgen/üçgen2/üstel aralık)", "η_min = 0,1·η_maks, yarım döngü 5 epok; üstel aralıkta γ = 0,1^(1/T)"],
    ["SGDR", "T₀ = 4 epok, T_mult = 2 (4+8+16+32 = 60 epok)"], ["Isınma", "Doğrusal, 5 epok (0 → η_maks)"],
    ["Tek döngü (1cycle)", "pct_start = 0,3; başlangıç η_maks/25; son η_maks/(25·10⁴); kosinüs geçiş; momentum sabit"],
    ["Model seçimi", "Son epok modeli raporlanır (en iyi doğrulama epoğu ek olarak kaydedilir)"],
    ["Seed listesi", f"{SEEDS} (ana deney); ilk 3 seed (duyarlılık, ablasyon)"],
], columns=["Hiperparametre", "Değer"])
add_table("T4_hiperparametreler", "Tablo T4. Model hiperparametreleri (tam konfigürasyon)", t4)

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"
import matplotlib as _m, scipy as _sc
t5 = pd.DataFrame([
    ["İşlemci (CPU)", "AMD Ryzen 7 9800X3D (8 çekirdek)"], ["Bellek (RAM)", "61,6 GB"], ["GPU", gpu],
    ["CUDA (PyTorch derlemesi)", torch.version.cuda], ["İşletim sistemi", f"{platform.system()} {platform.release()} ({platform.version()})"],
    ["Python", sys.version.split()[0]], ["PyTorch", torch.__version__], ["NumPy", np.__version__],
    ["pandas", pd.__version__], ["scikit-learn", sklearn.__version__], ["SciPy", _sc.__version__],
    ["Matplotlib", _m.__version__], ["Seaborn", sns.__version__],
    ["Determinizm", "torch.use_deterministic_algorithms(True), cuDNN deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8"],
    ["Ortalama koşu süresi (60 epok)", f"{np.mean([r['final']['time_s'] for r in main]):.1f} s"],
], columns=["Bileşen", "Değer"])
add_table("T5_hesaplama_ortami", "Tablo T5. Hesaplama ortamı", t5)

bl = json.load(open(RESULTS_DIR / "baselines.json", encoding="utf-8"))
best_acc = np.array([r["final"]["test_acc"] for r in by[best_name]]) * 100
const_acc = np.array([r["final"]["test_acc"] for r in by["constant"]]) * 100
t8 = pd.DataFrame([
    ["Holland tipi uyarlanır sınıflandırıcı", "Frey ve Slate (1991) [Fogarty, 1992]", "16.000/4.000", "82,7"],
    ["Backprop MLP (StatLog)", "Michie vd. (1994)", "15.000/5.000", "67,3"],
    ["RBF ağı (StatLog)", "Michie vd. (1994)", "15.000/5.000", "76,7"],
    ["LVQ (StatLog)", "Michie vd. (1994)", "15.000/5.000", "92,1"],
    ["Tek YSA, 40 gizli birim", "Opitz ve Maclin (1999)", "10-katlı ÇD", "82,0"],
    ["AdaBoost YSA topluluğu", "Opitz ve Maclin (1999)", "10-katlı ÇD", "95,4"],
    ["1-NN", "Fogarty (1992)", "16.000/4.000", "95,7"],
    ["ELM (sigmoid)", "Zhou (2013)", "13.333/6.667", "93,51"],
    ["MLP 16-70-50-26", "Zhou ve Feng (2017)", "16.000/4.000", "95,70"],
    ["YSA topluluğu", "Frosst ve Hinton (2017)", "16.000/4.000*", "95,9"],
    ["sNDF (sinirsel karar ormanı)", "Kontschieder vd. (2015)", "16.000/4.000", "97,08"],
    ["SNN (SELU)", "Klambauer vd. (2017)", "—", "97,26"],
    ["ELM (Gauss çekirdek)", "Zhou (2013)", "13.333/6.667", "97,41"],
    ["Bire-bir SVM", "Hsu ve Lin (2002)", "15.000/5.000", "97,98"],
    ["AdaBoost MLP", "Schwenk ve Bengio (2000)", "önceden tanımlı", "98,5"],
    ["Bu çalışma — 1-NN (z-puanı)", "—", "16.000/4.000", f"{bl['1-NN']['acc_mean'] * 100:.2f}"],
    ["Bu çalışma — Rastgele orman (500)", "—", "16.000/4.000", fmt(bl['Rastgele orman (500 ağaç)']['acc_mean'] * 100, bl['Rastgele orman (500 ağaç)']['acc_std'] * 100)],
    ["Bu çalışma — SVM (RBF)", "—", "16.000/4.000", f"{bl['SVM (RBF, C=10)']['acc_mean'] * 100:.2f}"],
    ["Bu çalışma — MLP, sabit LR", "—", "14.000/2.000/4.000", fmt(const_acc.mean(), const_acc.std(ddof=1))],
    [f"Bu çalışma — MLP, {SCHEDULE_LABELS[best_name]}", "—", "14.000/2.000/4.000", fmt(best_acc.mean(), best_acc.std(ddof=1))],
], columns=["Yöntem", "Kaynak", "Eğitim/test bölünmesi", "Test doğruluğu (%)"])
add_table("T8_literatur_karsilastirma", "Tablo T8. Literatürle karşılaştırma", t8,
          "Hata oranı raporlayan kaynaklarda doğruluk = 100 − hata. Bölünmeler farklı olduğundan karşılaştırma yönelimseldir. "
          "*Frosst ve Hinton bölünmeyi açıkça belirtmemiştir.")

(TAB / "tables.md").write_text("# Tablolar (otomatik üretildi: src/make_figures_tables.py)\n\n" + "\n".join(md), encoding="utf-8")
summary = {"best_schedule": best_name, "lr_range_min_loss_lr": lr_min,
           "best_acc": [best_acc.mean(), best_acc.std(ddof=1)], "const_acc": [const_acc.mean(), const_acc.std(ddof=1)]}
json.dump(summary, open(RESULTS_DIR / "summary.json", "w", encoding="utf-8"), indent=1)
print(summary)
