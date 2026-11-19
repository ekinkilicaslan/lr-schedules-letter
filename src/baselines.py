"""Aynı kanonik bölünmede klasik karşılaştırma yöntemleri (T8'deki 'bu çalışma' satırları için)."""
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from common import RESULTS_DIR, SEEDS, load_splits

sp = load_splits()
Xtr = np.vstack([sp["train"][0], sp["val"][0]])      # 16.000 örnek (kanonik)
ytr = np.concatenate([sp["train"][1], sp["val"][1]])
Xte, yte = sp["test"]

models = {
    "1-NN": lambda s: KNeighborsClassifier(n_neighbors=1),
    "3-NN": lambda s: KNeighborsClassifier(n_neighbors=3),
    "Lojistik regresyon": lambda s: LogisticRegression(max_iter=2000),
    "SVM (RBF, C=10)": lambda s: SVC(C=10, gamma="scale"),
    "Rastgele orman (500 ağaç)": lambda s: RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1),
}
out = {}
for name, mk in models.items():
    seeds = SEEDS[:3] if "orman" in name else [SEEDS[0]]
    accs = []
    for s in seeds:
        t = time.time()
        m = mk(s).fit(Xtr, ytr)
        accs.append(float((m.predict(Xte) == yte).mean()))
    out[name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                 "n_seeds": len(seeds)}
    print(name, out[name], flush=True)
json.dump(out, open(RESULTS_DIR / "baselines.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
