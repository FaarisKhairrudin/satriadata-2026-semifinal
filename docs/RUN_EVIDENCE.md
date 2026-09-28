# Bukti Hasil Run — Final Pipeline (K=17)

Satu halaman pengikat: tiap tahap → perintahnya → artefak buktinya → sidik
sha256-nya. Semua artefak di bawah ter-commit di repo (commit awal repo final ini); juri bisa verifikasi ulang tanpa run berat.

Lingkungan: `python3.12.14`, dependensi persis di `requirements.txt`
(torch 2.6+cu124, transformers 5.13.1, timm 1.0.27, umap-learn 0.5.12,
scikit-learn 1.9.0). Seed global 42. `data/raw/` read-only.

## Ringkasan hasil

- Data: 3.961 citra e-waste tanpa label.
- **K=17**, silhouette **0,6828**, Davies-Bouldin **0,4370** (fusi DINOv3+KEC).
- Baseline DINOv3-only: sil 0,6786 / DB 0,4392 (K sama, selisih +0,0042 / −0,0022).
- Grounding: 15/17 sepakat vote SigLIP2, 2 koreksi manusia (C10, C16).
- Kuadran EVI/HI: Q1×8, Q2×1 (CRT), Q3×2, Q4×6.

## Tabel bukti per tahap

| Tahap | Perintah | Artefak bukti (ter-commit) | Status |
| --- | --- | --- | --- |
| 01 baseline | `python3.12 code/scripts/01_dinov3_kmeans_baseline.py` | `data/processed/electronic/dinov3_kmeans_baseline/` (sweep K=5..25, K=16) | cache, rerun tanpa `--overwrite` = skip |
| 02 DROWCULA | `python3.12 code/scripts/02_drowcula_dinov3.py` | `drowcula_dinov3/final_metrics.json` (K=17, sil 0,6786), `k_search_scores.csv`, `labels.csv`, `reduction_config.json` | terkunci |
| 03 KEC+DROWCULA | `python3.12 code/scripts/03_kec_drowcula_dinov3.py` | `kec_drowcula_dinov3/final_metrics.json` (K=17, sil 0,6828), `config.json` (DINOv3 ViT-B/16, SigLIP2 base-patch16-256, GPT-4o temp 0.1, UMAP 10/0.1, K-Means n_init 200), `k_search_scores.csv`, `labels.csv` | terkunci; regenerasi KEC butuh `OPENAI_API_KEY` + WordNet |
| 04 evaluasi | `python3.12 code/scripts/04_evaluate_clustering_spaces.py` | `equivalent_space_evaluation/equivalent_space_metrics.csv` + report `docs/progress/2026-09-23_EQUIVALENT_SPACE_REPORT.md` | rerun 27 Sep byte-identical |
| 05 grid | `python3.12 code/scripts/05_kec_drowcula_cluster_figures.py` | `kec_drowcula_dinov3/figures/cluster_c*_top9\|far9.png` (34 file) | bukti inspeksi manusia |
| 06 grounding | `python3.12 code/scripts/06_unukey_grounding_siglip2.py` (default `short`) | `cluster_grounding.csv/.json`, `siglip2_grounding_config.json` (catat `prompt_variant` + templates), `final_cluster_labels.csv` (15 agree / 2 correction) | rerun 27 Sep byte-identical (GPU 3050, ±4 mnt) |
| 07 EVI/HI | `python3.12 code/scripts/07_cluster_risk_value_mapping.py` (default `linear` + `log`) | `cluster_risk_value.csv`, `figures/risk_value_matrix.png` | terkunci 2026-09-26 |
| 08 insight | `python3.12 code/scripts/08_cluster_insight_figures.py` | `figures/insight_*.png` (11 file) | untuk draft §3 |
| Notebooks | buka `code/notebook/03_modeling_full_locked.ipynb` (66 cells, 25 ber-output), `04_labeling_insight_locked.ipynb` (51 cells, 21 ber-output) | tabel, kurva K-search, UMAP, grid cluster tersimpan di file | tanpa run ulang |

## Sidik sha256 artefak kunci

```text
30cb2277…c68c468  kec_drowcula_dinov3/cluster_grounding.csv
b47974e6…8bfc5    kec_drowcula_dinov3/cluster_grounding.json
767922a4…43e0f0   kec_drowcula_dinov3/final_cluster_labels.csv
a343e60c…5ce4f2   kec_drowcula_dinov3/cluster_risk_value.csv
08f35306…e2858e   kec_drowcula_dinov3/final_metrics.json
0916e50c…2e4d607  kec_drowcula_dinov3/config.json
f9c768a6…8f41fa   kec_drowcula_dinov3/k_search_scores.csv
5c47e19f…4fc965   kec_drowcula_dinov3/labels.csv
c1b2dbab…45f8dd   equivalent_space_evaluation/equivalent_space_metrics.csv
```

Cek ulang: `sha256sum <file>` lalu bandingkan awalan di atas
(semua hash di file ini).

## Bukti pipeline jalan ujung-ke-ujung (dataset baru, terisolasi)

```bash
python3.12 code/scripts/run_full_pipeline.py \
  --input-dir <folder_citra> --workdir /tmp/uji_ewaste   # default dino-only, tanpa API
```

Hasil uji 2026-09-27 (50 citra acak, `/tmp`, artefak final tidak tersentuh):
K=3 (sil 0,6299, DB 0,5089), grounding short, 6 grid PNG, eval workdir
DROWCULA 0,6299 vs K-Means 0,4676. Detail: `docs/RESULTS_METHOD.md`.

## Verifikasi cepat (tanpa GPU/API, tanpa menimpa apa pun)

```bash
for f in code/scripts/0*.py; do python3.12 -m py_compile "$f" || exit 1; done
python3.12 code/scripts/06_unukey_grounding_siglip2.py --help
python3.12 code/scripts/07_cluster_risk_value_mapping.py --help
sha256sum data/processed/electronic/kec_drowcula_dinov3/final_cluster_labels.csv
```

## Batasan bukti

- Klaim byte-identical 06 berasal dari rerun 27 Sep pagi (bukan dijalankan di
  sini); rerun 04 final diverifikasi byte-identical di sini.
- `per_image_labels.csv` = artefak mati era `official`, abaikan.
- Angka EVI/HI = estimasi komposisi tipikal kategori + snapshot harga
  2026-09-25/26; perlu verifikasi lapangan.
