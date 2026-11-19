# Öğrenme Oranı Programlarının Model Genellemesine Etkisi: Döngüsel ve Isınma Stratejileri

Manisa Celal Bayar Üniversitesi, Yapay Sinir Ağları Dersi (2026–2027 Güz)
**Yazarlar:** Prof. Dr. Enver ATİK, Ekin Kılıçaslan
**Veri seti:** [UCI Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) (20.000 örnek, 16 öznitelik, 26 sınıf; doi: 10.24432/C5ZP40)

On bir öğrenme oranı programı aynı koşullarda karşılaştırılmıştır: sabit, basamaklı, üstel, kosinüs, CLR-üçgen, CLR-üçgen2, CLR-üstel aralık, SGDR, ısınma+sabit, ısınma+kosinüs ve 1cycle. Tüm programlarda şu ayarlar ortaktır: mimari (16-256-256-26 MLP), optimizör (SGD, momentum 0,9, ağırlık azaltma 5·10⁻⁴), mini-yığın (128), epok sayısı (60) ve tepe öğrenme oranı (0,1).

## Klasör yapısı

```
data/raw/letter_recognition/   UCI veri dosyaları (letter-recognition.data, .names)
src/common.py                  veri yükleme, MLP, 11 öğrenme oranı programı (açık formüller), eğitim döngüsü, ölçütler
src/run_experiments.py         deney grupları: lrrange | main | sweep | ablation
src/baselines.py               1-NN, 3-NN, lojistik regresyon, SVM, rastgele orman (aynı bölünme)
src/make_figures_tables.py     figures/*.png (300 DPI) ve tables/*.csv üretimi
notebooks/lr_schedules_letter.ipynb   adım adım analiz (çalıştırılmış çıktılarla)
results/                       tüm koşuların ham sonuçları (JSON) ve çalıştırma kayıtları
figures/, tables/              çalışmadaki şekiller ve tablolar
```

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
cd src

# 1) Deneyler (tek GPU'da yaklaşık 30 dk; CPU'da daha uzun)
python run_experiments.py lrrange main sweep ablation   # LR aralık testi, 11 program × 5 seed, duyarlılık, ablasyon
python baselines.py                                     # klasik karşılaştırma yöntemleri

# 2) Şekiller ve tablolar (results/ klasöründen)
python make_figures_tables.py

# 3) Not defteri
jupyter notebook ../notebooks/lr_schedules_letter.ipynb
```

Deney grupları ayrı ayrı da çalıştırılabilir, örneğin `python run_experiments.py main`. Her grup sonucunu `results/<grup>.json` dosyasına yazar.

## Yeniden üretilebilirlik

- **Sabit seed listesi:** Tüm deneylerde `SEEDS = [42, 43, 44, 45, 46]` kullanılır (`src/common.py`). Duyarlılık ve ablasyon deneyleri ilk üç seed'i kullanır.
- **Sabit bölünme:** Eğitim/doğrulama ayrımı `SPLIT_SEED = 0` ile sabittir.
- **Determinizm:** `torch.use_deterministic_algorithms(True)`, cuDNN deterministik ve `CUBLAS_WORKSPACE_CONFIG=:4096:8` ayarları etkindir.
- **Bölünme protokolü:** İlk 16.000 örnek eğitim (14.000) ve tabakalı doğrulama (2.000) için ayrılır. Son 4.000 örnek test kümesidir (Frey ve Slate protokolü).
- **Ön işleme:** Standartlaştırma yalnızca eğitim kümesinden öğrenilir.
- **Test kümesinin kullanımı:** Test kümesi hiperparametre seçiminde kullanılmaz; seçimler doğrulama kümesine göre yapılır.

## Ana sonuçlar (test doğruluğu, 5 seed, ortalama ± std)

| Program | Test doğruluğu (%) |
|---|---|
| Sabit | 96,20 ± 0,24 |
| SGDR | 97,06 ± 0,13 |
| Basamaklı | 97,04 ± 0,18 |
| Isınma+kosinüs | 96,99 ± 0,14 |

Ayrıntılı sonuçlar `tables/` ve `results/` klasörlerindedir.

## Atıf

Veri seti: D. Slate, "Letter Recognition," UCI Machine Learning Repository, 1991, doi: 10.24432/C5ZP40.
