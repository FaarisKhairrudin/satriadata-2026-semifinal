# DROWCULA-MODELLING — Unsupervised E-Waste Image Clustering

Pengelompokan 3.961 citra e-waste tanpa label menjadi **17 cluster** memakai
fusi **DINOv3 + KEC** dalam kerangka **DROWCULA** (UMAP + pencarian-K + K-Means),
dilanjut grounding ke taksonomi UNU-KEYs-v.2 dan pemetaan nilai-bahaya (EVI/HI).

## Hasil utama

- **K=17**, silhouette **0,6828**, Davies-Bouldin **0,4370** — tanpa label pelatihan.
- Voting SigLIP2 sepakat dengan label akhir pada **15 dari 17** cluster; 2 sisanya
  keputusan inspeksi manusia.
- 17 cluster → 10 kode UNU-KEYs-v.2 + 4 aliran GAP (baterai, PCB, scrap, outlet).
- Kuadran nilai-bahaya: Q1×8, Q2×1 (CRT), Q3×2, Q4×6.

Angka lengkap: [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) ·
Bukti run + sha256: [`docs/RUN_EVIDENCE.md`](docs/RUN_EVIDENCE.md)

## Metodologi

![Alur metodologi](docs/figures/metodologi_pipeline.png)

Input awal adalah citra e-waste dari babak penyisihan (subset `1_Electronic`,
3.961 citra). Tahap lanjutan setelah clustering memperoleh **kondisi barang**
(data mining condition): caption per citra → klasifikasi kondisi fisik →
rekomendasi penanganan per cluster, dijalankan lewat notebook
`code/notebook/05-07` (lihat [`code/notebook/README.md`](code/notebook/README.md)).

## Implementasi

Satu skrip = satu tujuan, dijalankan berurutan dari repo root:

| Skrip | Kerja singkat |
| --- | --- |
| `00_prepare_embeddings.py` | Folder citra → manifest + embedding DINOv3 |
| `01_dinov3_kmeans_baseline.py` | Baseline K-Means murni, K=5..25 |
| `02_drowcula_dinov3.py` | DROWCULA resep resmi di DINOv3-only |
| `03_kec_drowcula_dinov3.py` | KEC + fusi 1536-D + DROWCULA (metode usulan) |
| `04_evaluate_clustering_spaces.py` | Evaluasi post-hoc antar label |
| `05_kec_drowcula_cluster_figures.py` | Grid top-9/far-9 per cluster |
| `06_unukey_grounding_siglip2.py` | Voting zero-shot → nama produk per cluster |
| `07_cluster_risk_value_mapping.py` | EVI/HI + kuadran keputusan |
| `08_cluster_insight_figures.py` | Figur insight untuk naskah |
| `run_full_pipeline.py` | Folder citra → hasil dalam satu perintah |

Notebook [`03_modeling_full_locked.ipynb`](code/notebook/03_modeling_full_locked.ipynb)
dan [`04_labeling_insight_locked.ipynb`](code/notebook/04_labeling_insight_locked.ipynb)
adalah versi locked dari pipeline di atas dengan output tersimpan — bisa dibaca
tanpa run ulang.

Lanjutan pipeline: alur mendapatkan kondisi barang (data mining condition),
di `code/notebook/05-07`:

| Notebook | Kerja singkat | Input | Output |
| --- | --- | --- | --- |
| [`05_caption_internvl3_runpod.ipynb`](code/notebook/05_caption_internvl3_runpod.ipynb) | Captioning InternVL3-8B di RunPod RTX 4090 | citra train | `captions.csv` per citra |
| [`06_condition_jev_api.ipynb`](code/notebook/06_condition_jev_api.ipynb) | Klasifikasi kondisi fisik (Intact / Disassembled / Damaged) via API | `captions.csv` | `results.csv`, `summary.csv`, interpretasi |
| [`07_recommendation_qwen38.ipynb`](code/notebook/07_recommendation_qwen38.ipynb) | Rekomendasi penanganan per cluster (≤80 kata) | kondisi + `cluster_risk_value.csv` | `recommendations.csv` |

Alurnya berurutan: 05 → 06 → 07; kredensial API diminta via `getpass` saat
run, tidak tersimpan di file.

Detail tiap skrip: [`code/scripts/README.md`](code/scripts/README.md).

## Setup

```bash
python3.12 --version                                          # 3.12, tanpa virtual env
python3.12 -m pip install --user -r requirements.txt          # dependensi
python3.12 -c "import nltk; nltk.download('wordnet')"         # untuk regenerasi KEC
```

Model DINOv3 dan SigLIP2 terunduh otomatis dari Hugging Face Hub saat run
pertama. Kunci API (`OPENAI_API_KEY`, templat di `.env.example`) hanya
diperlukan bila meregenerasi pengetahuan KEC — run normal memakai cache.

## Cara pakai

```bash
# Reproduksi hasil final (berurutan, memakai cache)
python3.12 code/scripts/01_dinov3_kmeans_baseline.py
python3.12 code/scripts/02_drowcula_dinov3.py
python3.12 code/scripts/03_kec_drowcula_dinov3.py
python3.12 code/scripts/04_evaluate_clustering_spaces.py
python3.12 code/scripts/05_kec_drowcula_cluster_figures.py
python3.12 code/scripts/06_unukey_grounding_siglip2.py
python3.12 code/scripts/07_cluster_risk_value_mapping.py
python3.12 code/scripts/08_cluster_insight_figures.py
```

```bash
# Dataset citra baru → hasil di folder output pilihanmu
python3.12 code/scripts/run_full_pipeline.py \
  --input-dir <folder_citra> --workdir <folder_output>
# varian: --mode full | --k-min 2 --k-max 25 | --stages 00,02 | --dry-run
```

Setiap tahap menulis hasilnya ke foldernya sendiri di bawah `--output-dir`/
`--workdir` masing-masing, lengkap dengan metrik dan figur. Label akhir per
cluster divalidasi manusia (`final_cluster_labels.csv`) sebelum masuk tahap
pemetaan nilai-bahaya.

## Struktur

```text
code/
  scripts/       # pipeline final 00–08 + run_full_pipeline.py
  notebook/      # core locked 03–04 + kondisi barang 05–07
data/
  raw/           # data mentah (tidak disertakan; subset 1_Electronic dari penyisihan)
  processed/     # embedding, label, metrik, figures
docs/
  Laporan/       # naskah karya ilmiah
requirements.txt
```

Artefak final ada di `data/processed/electronic/` — label produk
(`final_cluster_labels.csv`), pemetaan nilai-bahaya (`cluster_risk_value.csv`),
dan seluruh figur di `figures/`.
