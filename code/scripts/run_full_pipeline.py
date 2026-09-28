"""run_full_pipeline.py — satu perintah input folder citra -> hasil clustering.

Orkestrasi smoke-test / dataset baru (BUKAN rerun final K=17):

    --input-dir FOLDER_CITRA --workdir /tmp/uji_... [--mode dino-only|full]

Tahap (tiap tahap = subprocess skrip bernomor, log + checksum per tahap):

    00  siapkan manifest + embedding DINOv3 di workdir (selalu jalan)
    01  baseline K-Means K=5..25 (ringan, CPU)
    02  DROWCULA DINOv3-only (UMAP + K-search; K default 2..5 untuk smoke test,
        bisa --k-min/--k-max penuh 2..25)
    04  evaluasi post-hoc di workdir (K apa pun; default final tetap K=16/17)
    05  grid top-9/far-9 (fallback normalized_embeddings bila tanpa fusi)
    06  grounding otomatis SigLIP2 (default short; butuh GPU/CPU + transformers)

Mode:

    dino-only (default)  00,01,02,04,05,06 tanpa API, tanpa KEC.
                         Cocok untuk bukti jalan / /tmp.
    full                 tambah 03 KEC+DROWCULA (butuh GPU + OPENAI_API_KEY +
                         korpus WordNet; cache miss = mahal/lama).

Gate manusia: 07 (EVI/HI) TIDAK dijalankan otomatis — butuh
final_cluster_labels.csv hasil verifikasi manusia. 04/08 dilewati di workdir
kecil (hardcoded path final + butuh K=16/17). Pesan gate dicetak jelas.

Tidak pernah menulis ke data/processed/electronic/: semua output di workdir.
Artefak final K=17 tidak tersentuh (verifikasi: git status data/ tetap bersih
kecuali file yang memang belum diputuskan).

Contoh:
    python3.12 code/scripts/run_full_pipeline.py \\
        --input-dir /tmp/smoke_ewaste_input --workdir /tmp/smoke_ewaste_work
    python3.12 code/scripts/run_full_pipeline.py \\
        --input-dir /tmp/smoke_ewaste_input --workdir /tmp/smoke_full \\
        --mode full --k-min 2 --k-max 25
    python3.12 code/scripts/run_full_pipeline.py \\
        --input-dir /tmp/smoke_ewaste_input --workdir /tmp/x --dry-run
    python3.12 code/scripts/run_full_pipeline.py \\
        --input-dir /tmp/smoke_ewaste_input --workdir /tmp/x --stages 00,02,04
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code" / "scripts"
PY = "python3.12"

STAGES_DINO = ("00", "01", "02", "04", "05", "06")
STAGES_FULL = ("00", "01", "02", "03", "04", "05", "06")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_full_pipeline")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(name: str, cmd: list[str], dry_run: bool) -> None:
    log.info("[%s] $ %s", name, " ".join(str(c) for c in cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"tahap {name} gagal (exit {exc.returncode}): {' '.join(str(c) for c in cmd)}") from exc
    except OSError as exc:
        raise SystemExit(f"tahap {name} tidak bisa dijalankan: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dino-only", "full"), default="dino-only")
    parser.add_argument("--stages", default=None,
                        help="Subset tahap, mis. 00,02 (default: semua sesuai mode).")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5,
                        help="Default 2..5 untuk smoke test cepat; 2..25 untuk penuh.")
    parser.add_argument("--prompt-variant", choices=("official", "short"), default="short")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validasi input + cetak perintah, tanpa eksekusi berat.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input-dir tidak ditemukan: {args.input_dir}")
    images = [p for p in args.input_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not images:
        raise SystemExit(f"tidak ada citra di {args.input_dir}")
    log.info("input: %d citra di %s", len(images), args.input_dir)

    stages = tuple(s.strip() for s in args.stages.split(",")) if args.stages else (
        STAGES_FULL if args.mode == "full" else STAGES_DINO)
    log.info("mode=%s stages=%s workdir=%s", args.mode, ",".join(stages), args.workdir)
    if not args.dry_run:
        args.workdir.mkdir(parents=True, exist_ok=True)

    workdir = args.workdir
    manifest = workdir / "manifest_train.csv"
    ids = workdir / "embedding_ids.csv"
    emb = workdir / "embeddings_dinov3.npy"
    drow = workdir / "drowcula_test"
    fused = workdir / "kec_test"

    if "00" in stages:
        run_step("00", [PY, str(SCRIPTS / "00_prepare_embeddings.py"),
                        "--input-dir", str(args.input_dir), "--workdir", str(workdir)], args.dry_run)
    if "01" in stages:
        run_step("01", [PY, str(SCRIPTS / "01_dinov3_kmeans_baseline.py"),
                        "--input", str(emb), "--embedding-ids", str(ids),
                        "--manifest", str(manifest),
                        "--output-dir", str(workdir / "kmeans_test"),
                        "--k-min", str(args.k_min), "--k-max", str(args.k_max)], args.dry_run)
    if "02" in stages:
        run_step("02", [PY, str(SCRIPTS / "02_drowcula_dinov3.py"),
                        "--input", str(emb), "--manifest", str(manifest),
                        "--embedding-ids", str(ids), "--output-dir", str(drow),
                        "--k-min", str(args.k_min), "--k-max", str(args.k_max)], args.dry_run)
    if "03" in stages:
        if args.mode != "full":
            log.warning("tahap 03 dilewati (mode dino-only); pakai --mode full untuk KEC")
        else:
            run_step("03", [PY, str(SCRIPTS / "03_kec_drowcula_dinov3.py"),
                            "--dino", str(emb), "--embedding-ids", str(ids),
                            "--manifest", str(manifest), "--output-dir", str(fused),
                            "--k-min", str(args.k_min), "--k-max", str(args.k_max)], args.dry_run)
    result_dir = fused if ("03" in stages and args.mode == "full") else drow
    if "04" in stages:
        run_step("04", [PY, str(SCRIPTS / "04_evaluate_clustering_spaces.py"),
                        "--proc-dir", str(workdir)], args.dry_run)
    if "05" in stages:
        run_step("05", [PY, str(SCRIPTS / "05_kec_drowcula_cluster_figures.py"),
                        "--result-dir", str(result_dir), "--embedding-ids", str(ids)], args.dry_run)
    if "06" in stages:
        run_step("06", [PY, str(SCRIPTS / "06_unukey_grounding_siglip2.py"),
                        "--result-dir", str(result_dir), "--embedding-ids", str(ids),
                        "--manifest", str(manifest),
                        "--prompt-variant", args.prompt_variant], args.dry_run)

    print()
    print("GATE MANUSIA — tahap berikutnya TIDAK otomatis:")
    print(f"  1. inspeksi grid: {result_dir}/figures/cluster_c*_top9|far9.png")
    print(f"  2. tulis verifikasi: {result_dir}/final_cluster_labels.csv")
    print("     (contoh skema: data/processed/electronic/kec_drowcula_dinov3/final_cluster_labels.csv)")
    print(f"  3. baru jalankan: python3.12 code/scripts/07_cluster_risk_value_mapping.py "
          f"--outdir {result_dir}   # butuh final_cluster_labels.csv; tanpa itu 07 menolak")
    print("  04/08 dilewati di workdir kecil (04 butuh label K=16/17 final; 08 butuh artefak K=17 + verifikasi).")

    if not args.dry_run:
        print()
        print(f"Ringkasan workdir {workdir} ({datetime.datetime.now().strftime('%F %T')}):")
        for path in sorted(workdir.rglob("*")):
            if path.is_file() and path.stat().st_size < 50 * 1024 * 1024:
                try:
                    print(f"  {path.relative_to(workdir)}  sha={sha256_file(path)[:12]}  {path.stat().st_size}B")
                except OSError:
                    print(f"  {path.relative_to(workdir)}  (unreadable)")


if __name__ == "__main__":
    sys.exit(main())
