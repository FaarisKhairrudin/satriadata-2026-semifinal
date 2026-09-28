# `code/scripts/` — pipeline final

Jalankan dari repo root dengan `python3.12`. Semua path relatif terhadap repo root.
Seed global: 42. `data/raw/` read-only; output ke `data/processed/`.

| # | Skrip | Tujuan (1 kalimat) | Input utama | Output utama |
| --- | --- | --- | --- | --- |
| 00 | `00_prepare_embeddings.py` | Folder citra baru → manifest + embedding DINOv3 (cache-aware). | `--input-dir` folder citra | `--workdir`: `manifest_train.csv`, `embedding_ids.csv`, `embeddings_dinov3.npy` |
| 01 | `01_dinov3_kmeans_baseline.py` | Baseline visual murni DINOv3 + K-Means K=5..25. | `embeddings_dinov3.npy` | `dinov3_kmeans_baseline/` |
| 02 | `02_drowcula_dinov3.py` | Resep DROWCULA resmi di DINOv3 (L2 → UMAP 3-D → K-search 2..25). | embedding DINOv3 | `drowcula_dinov3/` (K=17) |
| 03 | `03_kec_drowcula_dinov3.py` | Metode usulan: fusi DINOv3+KEC (SigLIP2 + GPT-4o) → DROWCULA. | embedding DINOv3 (+ API bila regenerasi) | `kec_drowcula_dinov3/` (K=17, sil 0.6828) |
| 04 | `04_evaluate_clustering_spaces.py` | Evaluasi post-hoc tanpa ubah label (common-space & native). | artefak 01/02/03 | `equivalent_space_evaluation/` |
| 05 | `05_kec_drowcula_cluster_figures.py` | Grid verifikasi top-9/far-9 per cluster (bukti inspeksi manusia). | `labels.csv` + citra | `kec_drowcula_dinov3/figures/` |
| 06 | `06_unukey_grounding_siglip2.py` | Grounding pasca-clustering ke 57 UNU-KEYs-v.2 + 3 GAP (default `--prompt-variant short`, tercatat di config). | label terkunci | `cluster_grounding.csv/.json`, `siglip2_grounding_config.json` |
| 07 | `07_cluster_risk_value_mapping.py` | Indeks EVI/HI + kuadran (default `linear` + `log`). | `final_cluster_labels.csv` (verifikasi manusia) | `cluster_risk_value.csv`, `figures/risk_value_*.png` |
| 08 | `08_cluster_insight_figures.py` | Figur insight K=17. | artefak 03/06/07 | `figures/insight_*.png` |

`run_full_pipeline.py` mengorkestrasi 00,01,02,04,05,06 (mode `dino-only`,
tanpa API) atau +03 (`--mode full`): folder citra → hasil, semua output di
`--workdir`. Setelah grounding otomatis, tulis `final_cluster_labels.csv`
hasil verifikasi manusia, lalu jalankan `07 --outdir <workdir>`.

## Aturan aman

- Rerun yang menimpa artefak final wajib `--overwrite` eksplisit.
- Eksperimen varian 07 gunakan `--outdir /tmp/...`; default tanpa flag =
  `linear` + `log` (final terkunci).
- `03` butuh `OPENAI_API_KEY` + korpus WordNet hanya saat regenerasi KEC.
- 06 rerun (`--prompt-variant short --overwrite`, GPU ±4 mnt) wajib
  byte-identical untuk `cluster_grounding.csv/.json`; diff wajar hanya di
  `siglip2_grounding_config.json`.
