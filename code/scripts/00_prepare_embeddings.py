"""00_prepare_embeddings.py — siapkan dataset citra baru menjadi input pipeline.

Tahap 0 pipeline (sebelumnya hanya ada di notebook superseded/01): pindai
folder citra apa pun -> tulis manifest + embedding DINOv3 frozen ke workdir
terisolasi (mirror struktur electronic/, BUKAN ke data/processed/electronic/).

    --input-dir FOLDER_CITRA  ->  --workdir/
        manifest_train.csv        (filepath relatif workdir, split, source_class)
        embedding_ids.csv         (absolute path citra sumber)
        embeddings_dinov3.npy     (N, 768 float32)

Pengaturan embedding identik dengan notebook lama: timm
vit_base_patch16_dinov3.lvd1689m, 256px resize+centercrop, normalisasi
ImageNet, batch 64, seed 42. Tanpa --overwrite dan cache ada -> skip
(cache-aware, tidak recompute).

Tidak menyentuh artefak final: default workdir di /tmp, tidak pernah default
ke data/processed/electronic/.

Contoh:
    python3.12 code/scripts/00_prepare_embeddings.py \\
        --input-dir /tmp/uji_citra --workdir /tmp/ewaste_test_2026-09-27
    python3.12 code/scripts/00_prepare_embeddings.py \\
        --input-dir /tmp/uji_citra --workdir /tmp/ewaste_test --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "seed": 42,
    "model_name": "vit_base_patch16_dinov3.lvd1689m",
    "img_size": 256,
    "batch_size": 64,
    "num_workers": 4,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
    "embedding_dimensions": 768,
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("00_prepare_embeddings")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_images(input_dir: Path) -> list[Path]:
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder citra sumber (dipindai rekursif).")
    parser.add_argument("--workdir", type=Path, required=True,
                        help="Direktori output terisolasi (dibuat bila belum ada).")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Tulis ulang cache walau sudah ada.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input-dir tidak ditemukan: {args.input_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    files = scan_images(args.input_dir)
    if not files:
        raise SystemExit(f"tidak ada citra didukung di {args.input_dir} ({sorted(EXTENSIONS)})")
    log.info("ditemukan %d citra di %s", len(files), args.input_dir)

    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.workdir / "manifest_train.csv"
    ids_path = args.workdir / "embedding_ids.csv"
    emb_path = args.workdir / "embeddings_dinov3.npy"

    if not args.overwrite and manifest_path.exists() and ids_path.exists() and emb_path.exists():
        cached = np.load(emb_path)
        if cached.shape[0] == len(files):
            log.info("cache ada (%s), skip (pakai --overwrite untuk tulis ulang)", cached.shape)
            return
        log.info("cache N=%d != citra N=%d, tulis ulang", cached.shape[0], len(files))

    # Kedua CSV memakai absolute path yang sama persis agar guard urutan
    # downstream (canonical_path di 01/02/03/06) selalu lolos untuk workdir baru.
    abs_paths = [str(p.resolve()) for p in files]
    rows = [
        {"filepath": abs_p, "split": "train", "source_class": args.input_dir.name}
        for abs_p in abs_paths
    ]
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame({"filepath": abs_paths}).to_csv(ids_path, index=False)
    log.info("manifest + embedding_ids ditulis (%d baris)", len(rows))

    try:
        from PIL import Image  # type: ignore[import-not-found]  # noqa: I001
        import timm  # type: ignore[import-not-found]  # noqa: I001
        import torch  # type: ignore[import-not-found]  # noqa: I001
        from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]  # noqa: I001
        import torchvision.transforms as T  # type: ignore[import-not-found]  # noqa: I001
    except ImportError as exc:
        raise SystemExit(f"butuh timm/torch/torchvision/pillow: {exc}") from exc

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = CONFIG["img_size"]
    tfm = T.Compose([
        T.Resize(size), T.CenterCrop(size), T.ToTensor(),
        T.Normalize(CONFIG["imagenet_mean"], CONFIG["imagenet_std"]),
    ])

    class ImageList(Dataset):
        def __init__(self, paths: list[Path]) -> None:
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            try:
                return tfm(Image.open(self.paths[index]).convert("RGB"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"gagal baca {self.paths[index]}: {exc}") from exc

    log.info("memuat %s di %s", CONFIG["model_name"], device)
    try:
        model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=0)
        model.eval().to(device)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"gagal muat model DINOv3: {exc}") from exc

    from tqdm import tqdm  # type: ignore[import-not-found]
    loader = DataLoader(ImageList(files), batch_size=args.batch_size,
                        num_workers=CONFIG["num_workers"], pin_memory=(device == "cuda"))
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DINOv3 embedding"):
            try:
                feats.append(model(batch.to(device)).cpu().numpy())
            except RuntimeError as exc:
                raise SystemExit(f"embedding gagal: {exc}") from exc
    features = np.concatenate(feats).astype(np.float32)
    if features.shape != (len(files), CONFIG["embedding_dimensions"]):
        raise SystemExit(f"bentuk tak terduga: {features.shape}")
    if not np.isfinite(features).all():
        raise SystemExit("embedding mengandung non-finite")
    np.save(emb_path, features)
    log.info("tersimpan %s sha=%s", features.shape, sha256_file(emb_path)[:16])


if __name__ == "__main__":
    main()
