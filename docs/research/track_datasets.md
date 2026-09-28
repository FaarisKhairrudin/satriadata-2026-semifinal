# Research: Dataset Publik Pendukung E-Waste Classification & Discovery (Satria Data 2026 Semifinal)

> Aturan main: dataset penyisihan = data utama. Data publik hanya suplemen (kalibrasi prompt taksonomi, seen-set discovery, evaluasi deteksi/segmentasi). Setiap citra publik yang dipakai wajib dicantumkan sumber/link + dipatuhi lisensinya.

## Ringkasan 5 dataset wajib + temuan tambahan

| # | Dataset | Akses | Citra | Kelas / anotasi | Lisensi | Modul draft |
| --- | --------- | ------- | ------- | ----------------- | --------- | ------------- |
| 1 | E-Waste Vision Dataset (EWasteNet, arXiv 2311.12823) | <https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification> | ~1.053–1.058 (paper: 1.053; README repo: 1.058; sebagian citra dilaporkan hilang) | 8 kelas klasifikasi: camera, keyboard, laptop, microwave, mobile, mouse, smartwatch, TV | Tidak dinyatakan eksplisit ("open-source" menurut paper); WAJIB minta izin/konfirmasi penulis sebelum pakai di karya lomba | Representasi (kalibrasi prompt taksonomi) + Discovery seen-set |
| 2 | E-Waste Classification, East West University (Roboflow Universe) | <https://universe.roboflow.com/east-west-university-yzug3/e-waste-classification> | ~1.906 | Klasifikasi, 6 kelas (nama kelas per halaman Roboflow; verifikasi setelah login/unduh) | CC BY 4.0 (tertera di halaman Roboflow) | Representasi + Discovery seen-set |
| 3 | TACO — Trash Annotations in Context | <http://tacodataset.org/> ; <https://github.com/pedropro/TACO/> ; DOI Zenodo: <https://doi.org/10.5281/zenodo.3354286> | 1.500 citra inti, 4.784 anotasi (dataset terus tumbuh via crowdsourcing) | Instance segmentation format COCO; taksonomi hierarkis 60 kelas / 28 super-kategori | Anotasi CC BY 4.0; tiap citra punya lisensi publik masing-masing (tercantum di file anotasi; default CC BY 4.0 bila kosong) | Kuantifikasi (evaluasi YOLO-World/SAM Count, protokol segmentasi) + Validitas (uji generalisasi out-of-domain) |
| 4 | ZeroWaste (CVPR 2022) | Repo: <https://github.com/dbash/zerowaste> ; data: <https://doi.org/10.5281/zenodo.4899926> ; ringkasan: <https://ai.bu.edu/zerowaste/> | ZeroWaste-f 4.661 frame teranotasi penuh; -s 6.212 tak berlabel; -w 1.202+1.208 frame; -v2 7.720 frame; + ZeroWasteAug | Segmentasi semantik/instans 4 kelas material: cardboard, soft plastic, rigid plastic, metal | CC BY-NC 4.0 (non-komersial!) — dinyatakan eksplisit di README repo; halaman Zenodo kadang menampilkan label generik CC-BY, yang berlaku adalah CC BY-NC 4.0 penulis | Kuantifikasi (benchmark segmentasi clutter) + Validitas (catatan batasan lisensi NC di laporan) |
| 5 | TrashNet (Thung & Yang, Stanford CS229) | <https://github.com/garythung/trashnet/> ; mirror HF: <https://huggingface.co/datasets/garythung/trashnet> | 2.527 (glass 501, paper 594, cardboard 403, plastic 482, metal 410, trash 137); 512×384; latar posterboard putih | Klasifikasi 6 kelas, tanpa bbox/mask | Repo berlisensi MIT (file LICENSE); penulis meminta sitasi repo bila dataset dipakai | Representasi (baseline klasifikasi cepat) + Validitas (pembanding domain bersih vs lapangan) |

**Dataset e-waste publik tambahan yang ditemukan (opsional, pilih 1–2 saja):**

- **E-Waste Image Classification Dataset (18 kelas, Kaggle, harshadsgore)** — <https://www.kaggle.com/datasets/harshadsgore/e-waste-image-classification-dataset-18-classes> — 18 kelas (Air-Conditioner, Battery, Heat-sink, Keyboard, Laptop, Light bulbs, Microchip-IC, Microwave, Mobile, Mouse, Passive-Component, PCB, Printer, Refrigerator, Resistor, Television, Transistor, Washing Machine), split train/val/test; cocok untuk kalibrasi prompt taksonomi granular (PCB/komponen). Lisensi = ketentuan Kaggle per dataset (cek tab lisensi sebelum unduh).
- **E-Waste YOLO Classification Dataset (Hugging Face, akhil2808/YoloDataset)** — <https://huggingface.co/datasets/akhil2808/YoloDataset> — format YOLO; kandidat evaluasi deteksi bila butuh bbox e-waste langsung (bukan sampah umum seperti TACO).
- **PCB DSLR (748 citra, 9.313 chip)** — <https://zenodo.org/records/3886553> ; **V-PCB (747 citra, 8 kelas komponen)** ; **ElectroCom61 (2.121 citra, 61 kelas komponen)** — ketiganya untuk modul kuantifikasi level komponen/PCB bila taksonomi menyentuh PCB.
- **MJU-Waste (2.475 pasang RGBD, MIT)** — <https://github.com/realwecan/mju-waste> — segmentasi sampah umum format VOC/COCO; alternatif TACO bila butuh pasangan depth.

## Detail per dataset

### 1. E-Waste Vision Dataset (EWasteNet, arXiv:2311.12823)

- **Klaim:** ~1.053 citra, 8 kelas (camera, keyboard, laptop, microwave, mobile, mouse, smartwatch, TV). **Sumber:** paper HTML (<https://ar5iv.labs.arxiv.org/html/2311.12823>) dan README repo (menyatakan 1.058 citra). **Dukungan:** direct evidence (angka berbeda 1.053 vs 1.058 = inkonsistensi kecil, catat apa adanya). **Kepercayaan:** medium-tinggi untuk kelas; medium untuk jumlah pasti.
- **Anotasi:** label kelas per folder (klasifikasi saja, tanpa bbox/mask) — interpretasi dari struktur folder repo (`dataset/Keyboards/`, `dataset/Mobile/`, ...). **Kepercayaan:** tinggi.
- **Lisensi:** tidak ada file LICENSE/lisensi eksplisit di repo; paper menyebut "open-source" tanpa nama lisensi. **Ini celah kepatuhan** — researcher inference: jangan klaim CC/MIT; tulis "lisensi tidak dinyatakan;await konfirmasi penulis" di draft. **Kepercayaan:** tinggi bahwa lisensi tidak ditemukan.
- **Catatan mutu:** README repo mencatat "Some of the images are missing" — risiko reproduksibilitas; unduh + hash/file-list sebelum dipakai kalibrasi prompt.
- **Sitasi:** Islam et al., "EWasteNet: A Two-Stream Data Efficient Image Transformer Approach for E-Waste Classification", arXiv:2311.12823 (2023), IEEE ICSECS 2023 (DOI 10.1109/icsecs58457.2023.10256321).

### 2. E-Waste Classification — East West University (Roboflow Universe, ~1.906 citra)

- **Klaim:** 1,9k citra, proyek klasifikasi, 6 kelas, lisensi CC BY 4.0. **Sumber:** halaman Roboflow (<https://universe.roboflow.com/east-west-university-yzug3/e-waste-classification>) via ringkasan pencarian (halaman 403 saat fetch langsung — verifikasi via browser login). **Dukungan:** interpretation (belum fetch langsung). **Kepercayaan:** medium; wajib konfirmasi nama 6 kelas + versi/split setelah unduh.
- **Kepatuhan:** CC BY 4.0 → boleh dipakai termasuk komersial selama atribusi; cantumkan pemilik (East West University), link, lisensi, dan tandai modifikasi di lampiran draft.
- **Sitasi:** East West University, "E-waste classification", Roboflow Universe, diakses [tanggal], URL + CC BY 4.0.

### 3. TACO — Trash Annotations in Context (1.500 citra, segmentasi)

- **Klaim:** 1.500 citra, 4.784 anotasi, 60 kelas / 28 super-kategori, format COCO. **Sumber:** <http://tacodataset.org/> (fetch langsung) + paper arXiv:2003.06975 (<https://doi.org/10.48550/arxiv.2003.06975>) + repo <https://github.com/pedropro/TACO/>. **Dukungan:** direct evidence. **Kepercayaan:** tinggi (untuk snapshot awal; dataset crowdsourced sehingga versi terbaru > 1.500 — selalu catat commit/DOI Zenodo yang diunduh).
- **Lisensi:** anotasi CC BY 4.0; citra masing-masing berlisensi publik sendiri (URL+lisensi per citra di file anotasi). **Sumber:** fetch langsung tacodataset.org. **Kepercayaan:** tinggi.
- **Sitasi:** Proença & Simões, "TACO: Trash Annotations in Context for Litter Detection", arXiv:2003.06975 (2020); dataset DOI <https://doi.org/10.5281/zenodo.3354286>.

### 4. ZeroWaste (CVPR 2022, CC BY-NC)

- **Klaim:** ZeroWaste-f 4.661 frame anotasi penuh; -s 6.212 tanpa label; -w 1.202 before + 1.208 after; -v2 7.720 frame; 4 kelas (cardboard, soft plastic, rigid plastic, metal). **Sumber:** paper CVPR (<https://openaccess.thecvf.com/content/CVPR2022/papers/Bashkirova_ZeroWaste_Dataset_Towards_Deformable_Object_Segmentation_in_Cluttered_Scenes_CVPR_2022_paper.pdf>) + repo README (fetch langsung). **Dukungan:** direct evidence. **Kepercayaan:** tinggi.
- **Lisensi:** CC BY-NC 4.0 eksplisit di README repo ("Our ZeroWaste dataset distributed under CC BY-NC 4.0 ... can be found here [Zenodo]"). **Sumber:** fetch langsung <https://github.com/dbash/zerowaste>. **Kepercayaan:** tinggi. Konsekuensi: hanya untuk evaluasi/penelitian non-komersial; tulis batasan ini di draft; jangan jadikan bagian pipeline komersial.
- **Sitasi (BibTeX resmi dari repo):** Bashkirova et al., "ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes", Proc. CVPR 2022. Data DOI <https://doi.org/10.5281/zenodo.4899926>.

### 5. TrashNet (2.527 citra, 6 kelas)

- **Klaim:** 2.527 citra (501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal, 137 trash), 512×384, diambil iPhone di atas posterboard putih. **Sumber:** fetch langsung README <https://github.com/garythung/trashnet/>. **Dukungan:** direct evidence. **Kepercayaan:** tinggi.
- **Lisensi:** file LICENSE repo = MIT (untuk kode; dataset diminta disitasi). Lisensi dataset tidak dipisahkan eksplisit → researcher inference: perlakukan sebagai "MIT repo + atribusi wajib", dan cantumkan sitasi dataset di draft. **Kepercayaan:** medium-tinggi.
- **Sitasi:** Thung & Yang, "TrashNet" (Stanford CS229 2016–2017), <https://github.com/garythung/trashnet/> + laporan <https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf>.

## Pemetaan ke modul draft

- **Representasi (kalibrasi prompt taksonomi):** EWasteNet-8 + Roboflow-EWU-6 sebagai kosakata inti e-waste; Kaggle-18 untuk granularitas komponen (PCB/resistor/transistor) bila taksonomi butuh. Hanya pakai split terpisah dari data penyisihan agar tidak bocor.
- **Kuantifikasi (YOLO-World / SAM Count):** TACO (bbox+mask COCO, in-the-wild) sebagai protokol evaluasi utama; ZeroWaste-f sebagai uji clutter industri; TrashNet TIDAK untuk deteksi (tanpa bbox) — hanya baseline klasifikasi.
- **Discovery (seen-set):** subset kecil berlabel dari EWasteNet/Roboflow-EWU sebagai "seen" e-waste; TACO/ZeroWaste sebagai out-of-distribution negatif (sampah umum) untuk menguji kebaruan temuan.
- **Validitas:** TrashNet (domain bersih) vs TACO (domain liar) untuk uji ketahanan domain; catat batasan lisensi (khususnya ZeroWaste NC dan EWasteNet tak-berlisensi) di bagian keterbatasan.

## Catatan sitasi untuk karya ilmiah (template)

- Setiap dataset publik: nama, penulis/pemilik, tahun, jumlah citra & kelas yang dipakai, versi/commit/DOI, URL akses, lisensi, tanggal akses, dan pernyataan kepatuhan ("digunakan hanya sebagai data suplemen; data utama = dataset penyisihan").
- Contoh baris lampiran: "E-Waste Vision Dataset (Islam et al. 2023, 8 kelas, ~1.053 citra, <https://github.com/NifulIslam/>..., lisensi tidak dinyatakan — konfirmasi penulis [tanggal]); dipakai N=... citra untuk kalibrasi prompt, terpisah dari data penyisihan."
- Untuk ZeroWaste wajib tambahkan: "CC BY-NC 4.0 — penggunaan non-komersial untuk evaluasi penelitian."

## Kontradiksi

- Jumlah citra EWasteNet: paper menyebut 1.053, README repo menyebut 1.058, plus catatan citra hilang — belum diverifikasi; catat rentang, bukan angka tunggal.
- Lisensi ZeroWaste di halaman Zenodo kadang tampil sebagai CC-BY generik, tetapi README resmi penulis menyatakan CC BY-NC 4.0 — yang diikuti adalah CC BY-NC 4.0.
- Kelas Roboflow-EWU: 6 kelas + CC BY 4.0 menurut ringkasan pencarian; halaman tidak bisa di-fetch langsung (403) — nama kelas menunggu verifikasi unduhan.

## Bukti yang hilang / belum terverifikasi

- Nama 6 kelas + split/train-val-test Roboflow-EWU (butuh buka/unduh via akun Roboflow).
- File LICENSE / pernyataan lisensi resmi EWasteNet (butuh kontak penulis atau cek revisi repo terbaru).
- Total citra pasti Kaggle-18 (halaman Kaggle JS-rendered; butuh buka manual) + lisensi dataset Kaggle tersebut.
- Versi/commit TACO & ZeroWaste yang akan dibekukan untuk draft (putuskan saat unduh; catat DOI/commit hash).

## Sumber

- Kept: tacodataset.org (lisensi + crowdsourcing) — <http://tacodataset.org/> ; garythung/trashnet README (jumlah+kelas+format) — <https://github.com/garythung/trashnet/> ; EWasteNet repo README (1.058 + missing images) — <https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification> ; dbash/zerowaste README (lisensi NC + DOI) — <https://github.com/dbash/zerowaste> ; ar5iv EWasteNet (kelas+1.053) — <https://ar5iv.labs.arxiv.org/html/2311.12823> ; ai.bu.edu/zerowaste — <https://ai.bu.edu/zerowaste/>.
- Rejected/deprioritized: mirror Kaggle TrashNet generik (duplikat, otoritas lebih rendah dari repo asli); klaim agregator SEO tanpa angka primer; halaman Roboflow fetch-403 (tetap dipakai sebagai penunjuk, bukan bukti final).

## Langkah lanjut

1. Unduh + bekukan versi: TACO (catat commit/DOI), ZeroWaste-f subset, TrashNet, EWasteNet, Roboflow-EWU, Kaggle-18 (opsional); simpan file-list + hash di `data/processed/`.
2. Verifikasi manual: 6 kelas Roboflow-EWU, lisensi Kaggle-18, dan izin tertulis EWasteNet.
3. Susun tabel atribusi final (sumber/link/lisensi/jumlah citra dipakai) untuk lampiran draft semifinal.
