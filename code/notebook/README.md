# `code/notebook/`

Urutan baca = urutan pengerjaan.

## Core clustering

| # | Notebook | Isi |
| --- | --- | --- |
| 03 | `03_modeling_full_locked.ipynb` | DINOv3 + KEC + DROWCULA → 17 cluster (baca cache, visualisasi). |
| 04 | `04_labeling_insight_locked.ipynb` | Grounding UNU-KEYs + EVI/HI (baca cache). |

Keduanya versi locked dengan output tersimpan — bisa dibaca tanpa run ulang.

## Kondisi barang (data mining condition)

| # | Notebook | Isi | Bergantung pada |
| --- | --- | --- | --- |
| 05 | `05_caption_internvl3_runpod.ipynb` | Captioning InternVL3-8B di RunPod RTX 4090 → `captions.csv`. | Citra train |
| 06 | `06_condition_jev_api.ipynb` | Klasifikasi kondisi fisik (Intact / Disassembled / Damaged) via API. | 05 (`captions.csv`) |
| 07 | `07_recommendation_qwen38.ipynb` | Rekomendasi penanganan per cluster (≤80 kata). | 06 + `cluster_risk_value.csv` |

Kredensial API notebook 05–07 diminta via `getpass` saat run, tidak tersimpan
di file.
