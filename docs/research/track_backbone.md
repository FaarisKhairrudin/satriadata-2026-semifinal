# Research: Backbone Visual SOTA Pengganti (DINOv3 / SigLIP 2 / Franca)

Konteks: draft solusi E-Waste Classification & Discovery memakai DINOv2 ViT-L/14 + CLIP ViT-B/32 + consensus K-Means/Agglomerative/BIRCH + mapping taksonomi UNITAR UNU-KEYs-v.2 (54 kategori).

## Summary

DINOv3 (Meta, arXiv 2508.10104) adalah pengganti DINOv2 yang paling direct: keluarga ViT-S/S+/B/L/H+/7B + ConvNeXt-T/S/B/L terdistilasi dari teacher ViT-7B, dengan teknik Gram anchoring yang memperbaiki degradasi fitur dense — relevan langsung untuk clustering citra e-waste. SigLIP 2 (Google, arXiv 2502.14786) adalah pengganti CLIP yang lebih kuat untuk zero-shot similarity ke 54 kategori taksonomi (multilingual, fitur dense lebih baik, varian NaFlex tahan perubahan aspek rasio). Franca (Valeo.ai, CVPR 2026) secara teknis kompetitif dengan DINOv2 tetapi berlisensi Research-Only RAIL — hanya cocok sebagai baseline/ablasi, bukan backbone utama lomba yang berpotensi dikomersialkan. Rekomendasi: DINOv3 ViT-L/16 sebagai backbone clustering + SigLIP 2 SO400M sebagai scorer taksonomi.

## Findings

### (1) DINOv3

1. **Claim:** DINOv3 adalah keluarga model Meta yang dilatih SSL pada LVD-1689M (1,689 miliar gambar hasil kurasi dari ~17 miliar gambar publik) dengan teacher ViT-7B (6,7B param) dan 5 varian ViT + 4 varian ConvNeXt hasil distilasi. **Sources:** [MODEL_CARD.md facebookresearch/dinov3](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md), [Meta blog DINOv3](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/). **Support:** direct evidence. **Confidence:** high.

2. **Claim:** Dimensi embedding / spesifikasi tiap varian ViT (patch 16, 4 register tokens): ViT-S 384 (21M), ViT-S+ 384 (29M, SwiGLU), ViT-B 768 (86M), ViT-L 1024 (300M), ViT-H+ 1280 (840M, SwiGLU), ViT-7B 4096 (6716M, SwiGLU); ConvNeXt: Tiny 29M / Small 50M / Base 89M / Large 198M. **Sources:** [MODEL_CARD.md](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md). **Support:** direct evidence. **Confidence:** high. (Peneliti: ViT-L/16 dim 1024 = drop-in dimensi untuk pipeline DINOv2 ViT-L/14 yang juga 1024-dim; tetapi ukuran patch 16 vs 14 mengubah jumlah token per resolusi — sesuaikan preprocessing.)

3. **Claim:** Cara akses: (a) `torch.hub.load('facebookresearch/dinov3', model='dinov3_vitl16', ...)` dengan checkpoint yang diminta lewat link di repo GitHub (permintaan akses, URL unduhan dikirim via email); (b) mirror komunitas/transformers di HuggingFace `facebook/dinov3-vitl16-pretrain-lvd1689m` dkk. **Sources:** [MODEL_CARD.md](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md), [HF facebook/dinov3-vits16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). **Support:** direct evidence untuk (a); interpretation untuk status ketersediaan HF (repo `facebook/` ada, tetapi distribusi resmi menekankan akses via GitHub). **Confidence:** medium-high.

4. **Claim:** Lisensi adalah "DINOv3 License" (Meta, custom, bukan OSI open-source): royalty-free, worldwide, non-transferable untuk use/reproduce/distribute/derivatives; mewajibkan kepatuhan hukum dagang (larangan penggunaan militer/nuklir/senjata), atribusi publikasi, dan Meta dapat menghentikan lisensi bila dilanggar. Meta blog menyebutnya "commercial license". Komunitas sempat memprotes (GitHub issue #31 meminta Apache-2.0). **Sources:** [LICENSE.md DINOv3](https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/LICENSE.md), [Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [issue #31](https://github.com/facebookresearch/dinov3/issues/31). **Support:** direct evidence. **Confidence:** high. (Peneliti/inference: boleh dipakai untuk lomba/komersial sepanjang patuh ketentuan, tetapi klausul terminasi sepihak + larangan transfer berarti tim harus menyimpan salinan lisensi dan cek aturan lomba soal lisensi model.)

5. **Claim:** Gram anchoring = memakai checkpoint awal training ("Gram teacher", saat kualitas fitur lokal masih tinggi) untuk menjangkar agar kemiripan antar-fitur-lokal dalam satu citra tidak berubah selama training lanjut; hanya diterapkan di fase akhir training (efisien) dan menaikkan mIoU segmentasi Pascal VOC +3 s.d. +5 poin; dilengkapi high-resolution Gram anchoring + RoPE + fine-tune resolusi tinggi (±512–768px, peta fitur stabil >4k). Ini menjawab trade-off DINOv2 (fitur global naik, fitur lokal turun seiring training). **Sources:** [Lightly deep-dive DINOv3, merujuk Fig. 4–6 paper 2508.10104](https://www.lightly.ai/blog/dinov3), [MODEL_CARD.md — Gram anchoring tercantum sebagai komponen training objective](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md). **Support:** interpretation dari sumber sekunder yang merujuk paper (isi paper tidak di-fetch penuh). **Confidence:** medium-high.

6. **Claim:** Bukti keunggulan vs DINOv2/CLIP: +6 mIoU ADE20K vs DINOv2, +6,7 J&F-Mean video tracking (DAVIS), +10,9 GAP instance retrieval (Oxford-Hard); backbone frozen SOTA untuk deteksi/segmentasi/depth tanpa fine-tuning; pada klasifikasi menyamai/melampaui model CLIP-based (SigLIP 2, Perception Encoder). Angka model card: ViT-L/16 IN-ReaL 90,2 / ADE20k 54,9 / DAVIS 79,9; ViT-S/16 ADE20k 47,0. **Sources:** [Lightly deep-dive](https://www.lightly.ai/blog/dinov3), [Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [MODEL_CARD.md tabel evaluasi](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md). **Support:** direct evidence (tabel model card) + interpretation (angka delta dari Lightly). **Confidence:** medium-high. (Peneliti/inference: belum ada bukti publik khusus untuk clustering e-waste; tetapi kombinasi fitur dense berkualitas + retrieval +10,9 GAP adalah sinyal proksi terkuat bahwa embedding DINOv3 mengelompok lebih baik untuk K-Means/Agglomerative/BIRCH.)

### (2) SigLIP 2 (pengganti CLIP untuk mapping 54 kategori UNU-KEYs)

1. **Claim:** SigLIP 2 (Google, arXiv 2502.14786) memperluas sigmoid-loss SigLIP dengan: (a) decoder teks (caption global, caption region, prediksi bounding box), (b) self-distillation global-local loss + masked-prediction loss (diaktifkan setelah ~80% training), (c) adaptasi resolusi — varian FixRes dan NaFlex (resolusi dinamis, menjaga aspek rasio). SigLIP 2 mengungguli SigLIP di semua skala pada zero-shot classification, image-text retrieval, dan transfer ke VLM; multilingual. **Sources:** [HF Blog SigLIP 2](https://huggingface.co/blog/siglip2), [paper 2502.14786](https://arxiv.org/abs/2502.14786). **Support:** direct evidence (blog) untuk arsitektur dan klaim unggul atas SigLIP; klaim "vs CLIP" adalah researcher inference dari rantai SigLIP>CLIP (tidak ada tabel CLIP head-to-head yang diverifikasi di sini). **Confidence:** medium-high.

2. **Claim:** Varian & akses HuggingFace: ukuran Base (86M), Large (303M), SO400M, Giant (1B); ID pola `google/siglip2-{base,large,so400m,giant-opt}-patch{32,16,14}-{224,256,384,512,naflex}` (mis. `google/siglip2-so400m-patch14-384`); varian `-naflex` memakai kelas `Siglip2Model` (fix-res kompatibel `SiglipModel`); ada pipeline `zero-shot-image-classification`. Dimensi embedding citra terverifikasi langsung hanya untuk so400m-patch14-384 = 1152 (dari contoh kode blog). **Sources:** [HF Blog SigLIP 2 — daftar model + contoh kode](https://huggingface.co/blog/siglip2), [koleksi HF google/siglip2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107). **Support:** direct evidence. **Confidence:** high untuk ID dan 1152-dim; dimensi varian lain tidak diverifikasi (lihat Missing evidence).

3. **Claim (relevansi untuk tugas):** Untuk similarity ke 54 label taksonomi, SigLIP 2 lebih cocok daripada CLIP ViT-B/32 karena: sigmoid loss lebih baik untuk banyak kandidat label simultan; encoder multilingual (prompt Bahasa Indonesia ikut terbantu); fitur dense/lokal lebih baik untuk membedakan perangkat e-waste yang mirip visual; NaFlex menghindari distorsi aspek rasio foto produk. **Sources:** fitur dari [HF Blog SigLIP 2](https://huggingface.co/blog/siglip2); aplikasinya ke taksonomi adalah researcher inference. **Support:** inference. **Confidence:** medium.

### (3) Franca (alternatif fully open-source)

 1. **Claim:** Franca (Valeo.ai, arXiv 2507.14137, CVPR 2026) mengklaim "fully open-source" dalam arti data + kode + bobot transparan (ImageNet-21K, LAION-600M) dengan kontribusi: Nested Matryoshka Clustering, RASA (penghilang bias posisional), CyclicMask. Varian: ViT-B/14 (86M, 768-dim), ViT-L/14 (300M, 1024-dim), ViT-g/14 (1,1B, 1536-dim), patch 14; akses `torch.hub.load('valeoai/Franca:v1.1.0', ...)` + checkpoint GitHub releases. **Sources:** [Franca README](https://github.com/valeoai/Franca/blob/main/README.md), [Franca model_card.md](https://github.com/valeoai/Franca/blob/main/model_card.md). **Support:** direct evidence. **Confidence:** high.

 2. **Claim:** Angka Franca: ViT-B/14 @518+RASA — IN-1K k-NN 79,6% vs baseline DINOv2-B 76,8%; ADE20K in-context 35,0 vs 32,4; DAVIS 70,6 vs 69,2. ViT-L/14 @518+RASA — IN linear 85,2; ADE20K 39,6; DAVIS 70,0. Artinya setara/melampaui DINOv2 pada fitur global maupun dense. **Sources:** [Franca README — tabel evaluasi](https://github.com/valeoai/Franca/blob/main/README.md). **Support:** direct evidence. **Confidence:** high.

 3. **Claim (peringatan lisensi):** Model Franca berlisensi Research-Only RAIL License — BUKAN lisensi komersial/fs-open-source, sehingga tidak aman sebagai backbone utama bila solusi lomba berpotensi dikomersialisasi/dipublikasikan di luar riset. Klaim "fully open-source" hanya benar untuk transparansi artefak, bukan kebebasan lisensi. **Sources:** [Franca model_card.md — "License: Research-Only RAIL License"](https://github.com/valeoai/Franca/blob/main/model_card.md). **Support:** direct evidence untuk jenis lisensi; sisanya researcher inference. **Confidence:** high untuk fakta lisensi, medium untuk implikasi lomba (tergantung aturan Satria Data — belum diverifikasi).

## Contradictions

- DINOv3 disebut "open sourced backbones under a commercial license" (Meta blog) vs protes komunitas bahwa lisensinya bukan open-source standar (issue #31). Bukan kontradiksi faktual — melainkan ketegangan istilah "open" vs "source-available dengan izin komersial". Disarankan menyebut "source-available, royalty-free commercial-permitted" bukan "open-source".
- Franca "first fully open-source vision foundation model" (README) vs lisensi Research-Only RAIL (model card). Kontradiksi semu yang sama: terbuka artefaknya, tertutup penggunaannya. Untuk lomba, yang mengikat adalah teks lisensi (RAIL), bukan slogan.
- Klaim raksasa ViT-7B (6716M param) vs nama "7B" — konsisten (6,7B ≈ 7B); bukan kontradiksi.

## Missing evidence

- Dimensi embedding SigLIP 2 varian Base/Large/Giant (hanya SO400M@384 = 1152 yang terverifikasi langsung). Perlu cek config HF per-model sebelum menetapkan reduksi dimensi (PCA/UMAP) pipeline.
- Angka clustering langsung (NMI/ARI/Cluster Accuracy) DINOv3 vs DINOv2 pada dataset e-waste/natural-image — paper hanya lapor klasifikasi/retrieval/segmentasi. Butuh eksperimen internal kecil (k-NN retrieval + K-Means pada subset) sebagai proksi.
- `source_check` terdaftar gagal mengembalikan kutipan (status missing-evidence, confidence 0,20) untuk dua klaim kunci; validasi di brief ini bersandar pada fetch langsung sumber primer (model card, blog resmi, README). Keterbatasan ini dicatat eksplisit.
- Status ketersediaan checkpoint DINOv3 gated (perlu email approval — latensi tidak diketahui) dan aturan lisensi lomba Satria Data 2026 soal model eksternal — belum diverifikasi.
- Lisensi distribusi HF untuk SigLIP 2 (umumnya Apache-2.0 untuk rilis Google/BigVision, tetapi tidak di-fetch di sesi ini) — verifikasi sebelum klaim "fully permissive".

## Sources

- Kept: MODEL_CARD.md facebookresearch/dinov3 (<https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md>) — spesifikasi varian, dimensi, objektif training, tabel evaluasi; sumber primer.
- Kept: Meta blog DINOv3 (<https://ai.meta.com/blog/dinov3-self-supervised-vision-model/>) — klaim SOTA frozen backbone, lisensi komersial, adopsi WRI/NASA.
- Kept: LICENSE.md DINOv3 (<https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/LICENSE.md>) — teks lisensi definitif.
- Kept: Lightly deep-dive DINOv3 (<https://www.lightly.ai/blog/dinov3>) — mekanisme Gram anchoring + angka delta vs DINOv2 (merujuk paper).
- Kept: HF Blog SigLIP 2 (<https://huggingface.co/blog/siglip2>) — objektif training, daftar ID model HF, contoh zero-shot, dimensi 1152 so400m.
- Kept: Koleksi HF google/siglip2 (<https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107>) — inventaris varian.
- Kept: Franca README (<https://github.com/valeoai/Franca/blob/main/README.md>) — klaim, metode, tabel hasil, cara load.
- Kept: Franca model_card.md (<https://github.com/valeoai/Franca/blob/main/model_card.md>) — dimensi embedding, lisensi RAIL.
- Kept: arXiv abs 2508.10104 / 2502.14786 / 2507.14137 — identitas paper (hanya abstrak yang ter-fetch).
- Rejected/deprioritized: BigGo News + Medium/Elisowski (lisensi DINOv3) — sekunder, diganti teks lisensi primer; EmergentMind/1kpapers/AI Paper Summary — ringkasan SEO tanpa data baru.

## Next steps

1. Ajukan akses checkpoint DINOv3 (ViT-L/16 + ViT-B/16 cadangan) segera karena distribusi gated; paralel unduh mirror HF `facebook/dinov3-*` sebagai fallback.
2. Uji proksi cepat: K-Means + k-NN retrieval DINOv3 ViT-L/16 vs DINOv2 ViT-L/14 pada subset e-waste; dan zero-shot SigLIP 2 so400m-patch14-384 vs CLIP ViT-B/32 pada 54 label UNU-KEYs (ID + EN).
3. Verifikasi aturan Satria Data 2026 tentang model eksternal/lisensi; putuskan peran Franca (ablasi saja bila RAIL bermasalah).
