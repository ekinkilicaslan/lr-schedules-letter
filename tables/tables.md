# Tablolar (otomatik üretildi: src/make_figures_tables.py)

### Tablo T1. Veri seti genel istatistik özeti

| Özellik                               | Değer                                                                                      |
|:--------------------------------------|:-------------------------------------------------------------------------------------------|
| Toplam örnek                          | 20.000                                                                                     |
| Öznitelik sayısı                      | 16 (tam sayı, 0–15)                                                                        |
| Sınıf sayısı                          | 26 (A–Z)                                                                                   |
| Eksik değer                           | 0                                                                                          |
| Yinelenen satır                       | 1332                                                                                       |
| En küçük / en büyük sınıf             | H (734) / U (813)                                                                          |
| Sınıf dengesizlik oranı (maks/min)    | 1.108                                                                                      |
| Eğitim / doğrulama / test             | 14.000 / 2.000 / 4.000                                                                     |
| Bölünme kuralı                        | İlk 16.000 → eğitim+doğrulama (tabakalı 2.000 doğrulama), son 4.000 → test [Frey ve Slate] |
| Mutlak |r| > 0,7 olan öznitelik çifti | 4                                                                                          |

### Tablo T2. Öznitelik tanımlama tablosu

|   # | Öznitelik   | Tanım                                 | Aralık   |   Ortalama |   Std. sapma |   Çarpıklık |
|----:|:------------|:--------------------------------------|:---------|-----------:|-------------:|------------:|
|   1 | x-box       | Sınırlayıcı kutunun yatay konumu      | 0–15     |       4.02 |         1.91 |        0.72 |
|   2 | y-box       | Sınırlayıcı kutunun dikey konumu      | 0–15     |       7.04 |         3.3  |       -0.23 |
|   3 | width       | Kutu genişliği                        | 0–15     |       5.12 |         2.01 |        0.23 |
|   4 | high        | Kutu yüksekliği                       | 0–15     |       5.37 |         2.26 |       -0.28 |
|   5 | onpix       | Toplam açık (siyah) piksel sayısı     | 0–15     |       3.51 |         2.19 |        0.94 |
|   6 | x-bar       | Açık piksellerin ortalama x konumu    | 0–15     |       6.9  |         2.03 |       -0.31 |
|   7 | y-bar       | Açık piksellerin ortalama y konumu    | 0–15     |       7.5  |         2.33 |        0.09 |
|   8 | x2bar       | x'in ortalama varyansı                | 0–15     |       4.63 |         2.7  |        1.16 |
|   9 | y2bar       | y'nin ortalama varyansı               | 0–15     |       5.18 |         2.38 |        0.29 |
|  10 | xybar       | Ortalama x·y korelasyonu              | 0–15     |       8.28 |         2.49 |        0.09 |
|  11 | x2ybr       | x·x·y ortalaması                      | 0–15     |       6.45 |         2.63 |        0.29 |
|  12 | xy2br       | x·y·y ortalaması                      | 0–15     |       7.93 |         2.08 |        0.27 |
|  13 | x-ege       | Soldan sağa ortalama kenar sayısı     | 0–15     |       3.05 |         2.33 |        1.06 |
|  14 | xegvy       | x-ege ile y korelasyonu               | 0–15     |       8.34 |         1.55 |        0.05 |
|  15 | y-ege       | Aşağıdan yukarı ortalama kenar sayısı | 0–15     |       3.69 |         2.57 |        0.47 |
|  16 | yegvx       | y-ege ile x korelasyonu               | 0–15     |       7.8  |         1.62 |       -0.2  |

### Tablo T3. Veri ön işleme adımları

|   Adım | İşlem                 | Ayrıntı                                                                      | Uygulandığı küme   |
|-------:|:----------------------|:-----------------------------------------------------------------------------|:-------------------|
|      1 | Okuma ve tip dönüşümü | CSV → float32 öznitelik, harf → 0–25 tamsayı etiket                          | Tüm veri           |
|      2 | Kalite denetimi       | Eksik değer (0) ve aralık (0–15) kontrolü                                    | Tüm veri           |
|      3 | Kanonik test ayrımı   | Son 4.000 örnek test olarak ayrıldı (Frey ve Slate protokolü)                | Tüm veri           |
|      4 | Doğrulama ayrımı      | İlk 16.000 örnekten tabakalı 2.000 örnek (random_state=0)                    | Eğitim+doğrulama   |
|      5 | Standartlaştırma      | z = (x − μ)/σ; μ ve σ yalnızca 14.000 eğitim örneğinden                      | Tüm kümeler        |
|      6 | Tensör ve cihaz       | PyTorch tensörü, GPU belleğinde; mini-yığın 128, her epok yeniden karıştırma | Eğitim             |

### Tablo T6. Ana performans metrikleri (5 seed, ortalama ± std; test kümesi n = 4.000)

| Program            | Test doğruluğu (%)   |   Makro-F1 (%) |   Eğitim doğruluğu (%) | Genelleme aralığı (puan)   |   Test kaybı |   ECE (%) |   Keskinlik (×10⁻³) | Δ sabit (puan)   | p (Holm)   |
|:-------------------|:---------------------|---------------:|-----------------------:|:---------------------------|-------------:|----------:|--------------------:|:-----------------|:-----------|
| Sabit              | 96.20 ± 0.24         |          96.19 |                  98.92 | 2.72 ± 0.33                |        0.13  |      2.21 |                2.94 | —                | —          |
| Basamaklı          | 97.04 ± 0.18         |          97.04 |                  99.79 | 2.74 ± 0.18                |        0.105 |      2.59 |                2.15 | +0.85            | 0.0344     |
| Üstel              | 96.51 ± 0.09         |          96.51 |                  99.62 | 3.10 ± 0.09                |        0.121 |      2.71 |                2.92 | +0.32            | 0.0966     |
| Kosinüs            | 96.96 ± 0.12         |          96.95 |                  99.82 | 2.86 ± 0.12                |        0.109 |      2.7  |                2.48 | +0.76            | 0.0344     |
| CLR-üçgen          | 96.95 ± 0.18         |          96.94 |                  99.8  | 2.85 ± 0.16                |        0.106 |      2.37 |                2.02 | +0.75            | 0.0344     |
| CLR-üçgen2         | 96.74 ± 0.07         |          96.73 |                  99.71 | 2.97 ± 0.06                |        0.119 |      2.7  |                2.74 | +0.54            | 0.0344     |
| CLR-üstel aralık   | 96.71 ± 0.08         |          96.7  |                  99.74 | 3.03 ± 0.06                |        0.118 |      2.72 |                2.76 | +0.51            | 0.0434     |
| SGDR               | 97.06 ± 0.13         |          97.05 |                  99.8  | 2.74 ± 0.14                |        0.108 |      2.74 |                2.34 | +0.87            | 0.0220     |
| Isınma+sabit       | 96.21 ± 0.32         |          96.21 |                  98.99 | 2.78 ± 0.23                |        0.128 |      2.16 |                2.59 | +0.01            | 0.8674     |
| Isınma+kosinüs     | 96.99 ± 0.14         |          96.99 |                  99.82 | 2.83 ± 0.15                |        0.109 |      2.66 |                2.27 | +0.80            | 0.0273     |
| Tek döngü (1cycle) | 96.96 ± 0.12         |          96.95 |                  99.83 | 2.87 ± 0.11                |        0.108 |      2.57 |                2.21 | +0.77            | 0.0344     |

p: sabit LR'ye karşı eşleştirilmiş t testi, Holm–Bonferroni düzeltmeli (10 karşılaştırma).

### Tablo T6b. Tepe öğrenme oranı duyarlılığı — test doğruluğu (%), 3 seed

| Program            | η=0.01       | η=0.025      | η=0.05       | η=0.1        | η=0.2        | η=0.4        |   Doğrulamayla seçilen η* | η*'da test (%)   |
|:-------------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|--------------------------:|:-----------------|
| Sabit              | 95.69 ± 0.08 | 96.39 ± 0.18 | 96.50 ± 0.09 | 96.14 ± 0.27 | 95.71 ± 0.72 | 91.55 ± 0.54 |                      0.05 | 96.50 ± 0.09     |
| Kosinüs            | 94.93 ± 0.24 | 96.02 ± 0.09 | 96.62 ± 0.14 | 96.93 ± 0.14 | 97.14 ± 0.08 | 97.26 ± 0.06 |                      0.4  | 97.26 ± 0.06     |
| CLR-üçgen2         | 93.05 ± 0.29 | 95.31 ± 0.04 | 96.08 ± 0.01 | 96.76 ± 0.08 | 96.88 ± 0.14 | 97.23 ± 0.10 |                      0.4  | 97.23 ± 0.10     |
| Isınma+kosinüs     | 94.95 ± 0.09 | 96.16 ± 0.06 | 96.66 ± 0.05 | 96.98 ± 0.18 | 97.08 ± 0.17 | 97.46 ± 0.10 |                      0.4  | 97.46 ± 0.10     |
| Tek döngü (1cycle) | 94.96 ± 0.08 | 96.13 ± 0.07 | 96.63 ± 0.13 | 96.94 ± 0.15 | 97.17 ± 0.07 | 97.38 ± 0.09 |                      0.4  | 97.38 ± 0.09     |

η*: son epok doğrulama doğruluğu ortalamasına göre seçilen tepe değer (test kümesi seçimde kullanılmamıştır).

### Tablo T10a. En sık karıştırılan sınıf çiftleri (test, seed başına ortalama hata sayısı)

| Gerçek → tahmin   |   Sabit (ort. hata/seed) |   SGDR (ort. hata/seed) |   Fark |
|:------------------|-------------------------:|------------------------:|-------:|
| I → J             |                      5.6 |                     4.4 |   -1.2 |
| J → I             |                      5   |                     3.4 |   -1.6 |
| P → F             |                      2.6 |                     3.2 |    0.6 |
| L → X             |                      2   |                     2.4 |    0.4 |
| Q → A             |                      1.8 |                     2.4 |    0.6 |
| V → B             |                      2   |                     2.4 |    0.4 |
| Y → P             |                      1.2 |                     2.4 |    1.2 |
| A → Y             |                      1.6 |                     2   |    0.4 |
| E → C             |                      3   |                     2   |   -1   |
| H → B             |                      1   |                     2   |    1   |

### Tablo T10b. En düşük sınıf doğruluklu 6 harf

| Sınıf   |   Test örneği |   Sabit doğruluk (%) |   SGDR doğruluk (%) |   Fark (puan) |
|:--------|--------------:|---------------------:|--------------------:|--------------:|
| H       |           151 |                90.07 |               92.58 |          2.52 |
| O       |           139 |                93.67 |               94.1  |          0.43 |
| Y       |           145 |                93.38 |               94.48 |          1.1  |
| L       |           157 |                94.14 |               94.78 |          0.64 |
| I       |           165 |                94.67 |               95.39 |          0.73 |
| P       |           168 |                95.48 |               95.48 |          0    |

### Tablo T7. Ablasyon çalışması (3 seed; test doğruluğu ve genelleme aralığı)

| Ablasyon grubu                          | Ayar               | Test doğruluğu (%)   | Genelleme aralığı (puan)   | Iraksayan koşu   |
|:----------------------------------------|:-------------------|:---------------------|:---------------------------|:-----------------|
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=0    | 96.93 ± 0.14         | 2.89 ± 0.14                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=1    | 96.88 ± 0.10         | 2.94 ± 0.11                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=3    | 97.05 ± 0.08         | 2.78 ± 0.07                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=5    | 96.98 ± 0.18         | 2.85 ± 0.18                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=10   | 97.03 ± 0.08         | 2.80 ± 0.05                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,1)   | warmup_epochs=20   | 96.88 ± 0.06         | 2.95 ± 0.06                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,4)   | warmup_epochs=0    | 97.26 ± 0.06         | 2.52 ± 0.08                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,4)   | warmup_epochs=1    | 97.38 ± 0.13         | 2.46 ± 0.12                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,4)   | warmup_epochs=3    | 97.26 ± 0.05         | 2.55 ± 0.02                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,4)   | warmup_epochs=5    | 97.46 ± 0.10         | 2.39 ± 0.12                | 0/3              |
| Isınma süresi (Isınma+kosinüs, η=0,4)   | warmup_epochs=10   | 97.38 ± 0.07         | 2.47 ± 0.10                | 0/3              |
| CLR yarım döngü (CLR-üçgen2)            | step_epochs=1      | 96.17 ± 0.10         | 3.21 ± 0.15                | 0/3              |
| CLR yarım döngü (CLR-üçgen2)            | step_epochs=2.5    | 96.40 ± 0.09         | 3.18 ± 0.10                | 0/3              |
| CLR yarım döngü (CLR-üçgen2)            | step_epochs=5      | 96.76 ± 0.08         | 2.97 ± 0.08                | 0/3              |
| CLR yarım döngü (CLR-üçgen2)            | step_epochs=10     | 96.86 ± 0.17         | 2.91 ± 0.14                | 0/3              |
| CLR yarım döngü (CLR-üçgen2)            | step_epochs=15     | 96.91 ± 0.10         | 2.89 ± 0.10                | 0/3              |
| CLR alt sınır η_min/η_maks (CLR-üçgen2) | base_frac=0.01     | 96.48 ± 0.10         | 3.15 ± 0.14                | 0/3              |
| CLR alt sınır η_min/η_maks (CLR-üçgen2) | base_frac=0.1      | 96.76 ± 0.08         | 2.97 ± 0.08                | 0/3              |
| CLR alt sınır η_min/η_maks (CLR-üçgen2) | base_frac=0.3      | 96.81 ± 0.04         | 2.87 ± 0.10                | 0/3              |
| 1cycle yükselme oranı pct_start         | pct_start=0.1      | 97.02 ± 0.13         | 2.81 ± 0.13                | 0/3              |
| 1cycle yükselme oranı pct_start         | pct_start=0.2      | 96.92 ± 0.14         | 2.90 ± 0.14                | 0/3              |
| 1cycle yükselme oranı pct_start         | pct_start=0.3      | 96.94 ± 0.15         | 2.89 ± 0.15                | 0/3              |
| 1cycle yükselme oranı pct_start         | pct_start=0.45     | 96.98 ± 0.20         | 2.86 ± 0.20                | 0/3              |
| Optimizör: AdamW (η=3·10⁻³)             | Sabit              | 96.62 ± 1.25         | 2.74 ± 0.29                | 0/3              |
| Optimizör: AdamW (η=3·10⁻³)             | Kosinüs            | 97.08 ± 0.06         | 2.92 ± 0.06                | 0/3              |
| Optimizör: AdamW (η=3·10⁻³)             | Isınma+kosinüs     | 97.04 ± 0.09         | 2.96 ± 0.09                | 0/3              |
| Optimizör: AdamW (η=3·10⁻³)             | Tek döngü (1cycle) | 96.90 ± 0.10         | 3.10 ± 0.10                | 0/3              |
| Kısa bütçe: 20 epok                     | Sabit              | 95.91 ± 0.40         | 2.88 ± 0.21                | 0/3              |
| Kısa bütçe: 20 epok                     | Kosinüs            | 96.49 ± 0.08         | 2.98 ± 0.09                | 0/3              |
| Kısa bütçe: 20 epok                     | CLR-üçgen2         | 96.46 ± 0.19         | 2.99 ± 0.21                | 0/3              |
| Kısa bütçe: 20 epok                     | Tek döngü (1cycle) | 96.62 ± 0.20         | 2.95 ± 0.17                | 0/3              |

### Tablo T4. Model hiperparametreleri (tam konfigürasyon)

| Hiperparametre                  | Değer                                                                                    |
|:--------------------------------|:-----------------------------------------------------------------------------------------|
| Mimari                          | 16-256-256-26, ReLU, He (Kaiming) normal başlatma, bias = 0                              |
| Parametre sayısı                | 76.826                                                                                   |
| Kayıp fonksiyonu                | Çapraz entropi (softmax)                                                                 |
| Optimizör                       | SGD, momentum β = 0,9 (Nesterov değil)                                                   |
| Ağırlık azaltma (L2)            | 5·10⁻⁴                                                                                   |
| Mini-yığın                      | 128 (110 iterasyon/epok)                                                                 |
| Epok                            | 60 (T = 6.600 iterasyon)                                                                 |
| Tepe öğrenme oranı η_maks       | 0,1 (tüm programlarda ortak; LR aralık testiyle seçildi)                                 |
| Basamaklı                       | 0,5T ve 0,75T'de ×0,1                                                                    |
| Üstel                           | γ = 0,01^(1/T) (son LR = 0,01·η_maks)                                                    |
| Kosinüs                         | η_min = 0, yeniden başlatma yok                                                          |
| CLR (üçgen/üçgen2/üstel aralık) | η_min = 0,1·η_maks, yarım döngü 5 epok; üstel aralıkta γ = 0,1^(1/T)                     |
| SGDR                            | T₀ = 4 epok, T_mult = 2 (4+8+16+32 = 60 epok)                                            |
| Isınma                          | Doğrusal, 5 epok (0 → η_maks)                                                            |
| Tek döngü (1cycle)              | pct_start = 0,3; başlangıç η_maks/25; son η_maks/(25·10⁴); kosinüs geçiş; momentum sabit |
| Model seçimi                    | Son epok modeli raporlanır (en iyi doğrulama epoğu ek olarak kaydedilir)                 |
| Seed listesi                    | [42, 43, 44, 45, 46] (ana deney); ilk 3 seed (duyarlılık, ablasyon)                      |

### Tablo T5. Hesaplama ortamı

| Bileşen                        | Değer                                                                                          |
|:-------------------------------|:-----------------------------------------------------------------------------------------------|
| İşlemci (CPU)                  | AMD Ryzen 7 9800X3D (8 çekirdek)                                                               |
| Bellek (RAM)                   | 61,6 GB                                                                                        |
| GPU                            | NVIDIA GeForce RTX 4070 Ti SUPER                                                               |
| CUDA (PyTorch derlemesi)       | 12.6                                                                                           |
| İşletim sistemi                | Windows 10 (10.0.19045)                                                                        |
| Python                         | 3.13.17                                                                                        |
| PyTorch                        | 2.14.1+cu126                                                                                   |
| NumPy                          | 2.5.3                                                                                          |
| pandas                         | 3.0.6                                                                                          |
| scikit-learn                   | 1.9.1                                                                                          |
| SciPy                          | 1.18.1                                                                                         |
| Matplotlib                     | 3.11.2                                                                                         |
| Seaborn                        | 0.13.2                                                                                         |
| Determinizm                    | torch.use_deterministic_algorithms(True), cuDNN deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8 |
| Ortalama koşu süresi (60 epok) | 5.3 s                                                                                          |

### Tablo T8. Literatürle karşılaştırma

| Yöntem                                | Kaynak                               | Eğitim/test bölünmesi   | Test doğruluğu (%)   |
|:--------------------------------------|:-------------------------------------|:------------------------|:---------------------|
| Holland tipi uyarlanır sınıflandırıcı | Frey ve Slate (1991) [Fogarty, 1992] | 16.000/4.000            | 82,7                 |
| Backprop MLP (StatLog)                | Michie vd. (1994)                    | 15.000/5.000            | 67,3                 |
| RBF ağı (StatLog)                     | Michie vd. (1994)                    | 15.000/5.000            | 76,7                 |
| LVQ (StatLog)                         | Michie vd. (1994)                    | 15.000/5.000            | 92,1                 |
| Tek YSA, 40 gizli birim               | Opitz ve Maclin (1999)               | 10-katlı ÇD             | 82,0                 |
| AdaBoost YSA topluluğu                | Opitz ve Maclin (1999)               | 10-katlı ÇD             | 95,4                 |
| 1-NN                                  | Fogarty (1992)                       | 16.000/4.000            | 95,7                 |
| ELM (sigmoid)                         | Zhou (2013)                          | 13.333/6.667            | 93,51                |
| MLP 16-70-50-26                       | Zhou ve Feng (2017)                  | 16.000/4.000            | 95,70                |
| YSA topluluğu                         | Frosst ve Hinton (2017)              | 16.000/4.000*           | 95,9                 |
| sNDF (sinirsel karar ormanı)          | Kontschieder vd. (2015)              | 16.000/4.000            | 97,08                |
| SNN (SELU)                            | Klambauer vd. (2017)                 | —                       | 97,26                |
| ELM (Gauss çekirdek)                  | Zhou (2013)                          | 13.333/6.667            | 97,41                |
| Bire-bir SVM                          | Hsu ve Lin (2002)                    | 15.000/5.000            | 97,98                |
| AdaBoost MLP                          | Schwenk ve Bengio (2000)             | önceden tanımlı         | 98,5                 |
| Bu çalışma — 1-NN (z-puanı)           | —                                    | 16.000/4.000            | 95.20                |
| Bu çalışma — Rastgele orman (500)     | —                                    | 16.000/4.000            | 96.56 ± 0.12         |
| Bu çalışma — SVM (RBF)                | —                                    | 16.000/4.000            | 97.00                |
| Bu çalışma — MLP, sabit LR            | —                                    | 14.000/2.000/4.000      | 96.20 ± 0.24         |
| Bu çalışma — MLP, SGDR                | —                                    | 14.000/2.000/4.000      | 97.06 ± 0.13         |

Hata oranı raporlayan kaynaklarda doğruluk = 100 − hata. Bölünmeler farklı olduğundan karşılaştırma yönelimseldir. *Frosst ve Hinton bölünmeyi açıkça belirtmemiştir.
