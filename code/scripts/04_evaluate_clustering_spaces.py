"""Post-hoc equivalent-space evaluation of existing clustering labels.

This script never reruns K-Means, UMAP, or DROWCULA and never changes any
existing assignments. It evaluates the three saved label sets in the exact
spaces requested:

Table 1: all labels in the same saved DINOv3 -> UMAP 3-D space.
Table 2: DINOv3-only and fused methods in their native saved UMAP spaces.

The pure K-Means labels are evaluated in DROWCULA's UMAP space only for metric
comparability. DROWCULA is not treated as a different final clustering
operator: its final assignments are K-Means assignments after UMAP.

Run from repository root:
    python3.12 code/scripts/04_evaluate_clustering_spaces.py
    python3.12 code/scripts/04_evaluate_clustering_spaces.py \\
        --proc-dir /tmp/workdir   # workdir kecil: K apa pun, output di workdir

Path flag aman: default tanpa flag = artefak final (kontrak K=16/17 tidak
berubah). --proc-dir menunjuk workdir lain tidak pernah menulis ke folder
final maupun ke docs/progress/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-not-found]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
DINO_DROW = PROC / "drowcula_dinov3"
FUSED_DROW = PROC / "kec_drowcula_dinov3"
BASELINE = PROC / "dinov3_kmeans_baseline"
OUTPUT = PROC / "equivalent_space_evaluation"
REPORT = ROOT / "docs" / "progress" / "2026-09-23_EQUIVALENT_SPACE_REPORT.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_clustering_spaces")

def save_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not write {path}: {exc}") from exc

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SystemExit(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()

def canonical_path(path: str) -> str:
    value = path.replace("\\", "/")
    marker = "/electronic/"
    if marker in value:
        return "electronic/" + value.split(marker, 1)[1]
    return value

def load_array(path: Path, expected_ndim: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"required evaluation artifact is missing: {path}")
    try:
        value = np.load(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not load {path}: {exc}") from exc
    if value.ndim != expected_ndim or not np.isfinite(value).all():
        raise SystemExit(f"invalid array in {path}: shape={value.shape}")
    dtype = np.int32 if expected_ndim == 1 else np.float32
    return np.asarray(value, dtype=dtype)

def load_embedding_ids(path: Path | None = None) -> list[str]:
    path = path or PROC / "embedding_ids.csv"
    try:
        frame = pd.read_csv(path)
        values = [canonical_path(str(value)) for value in frame["filepath"]]
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load embedding IDs: {exc}") from exc
    if len(set(values)) != len(values):
        raise SystemExit("embedding_ids.csv contains duplicate paths")
    return values

def check_guard(npy_path: Path, labels: np.ndarray, embedding_ids: list[str]) -> None:
    guard_path = npy_path.with_suffix(".csv")
    if not guard_path.exists():
        raise SystemExit(f"label ordering guard is missing: {guard_path}")
    try:
        frame = pd.read_csv(guard_path)
        label_paths = [canonical_path(str(value)) for value in frame["filepath"]]
        csv_labels = frame["cluster"].to_numpy(dtype=np.int32)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load label guard {guard_path}: {exc}") from exc
    if label_paths != embedding_ids:
        raise SystemExit(f"label ordering mismatch between {guard_path} and embedding_ids.csv")
    if not np.array_equal(labels, csv_labels):
        raise SystemExit(f"labels differ between {npy_path} and {guard_path}")

def load_labels(
    path: Path, n: int, expected_k: int, embedding_ids: list[str]
) -> np.ndarray:
    labels = load_array(path, expected_ndim=1)
    if len(labels) != n:
        raise SystemExit(f"label length mismatch in {path}: {len(labels)} vs {n}")
    if len(np.unique(labels)) != expected_k:
        raise SystemExit(
            f"{path} has {len(np.unique(labels))} unique labels; expected K={expected_k}"
        )
    check_guard(path, labels, embedding_ids)
    return labels.astype(np.int32, copy=False)

def load_labels_any(npy_path: Path, n: int, embedding_ids: list[str]) -> tuple[np.ndarray, int]:
    """Workdir variant: K apa pun yang tersimpan (tanpa kontrak K=16/17)."""
    labels = load_array(npy_path, expected_ndim=1)
    if len(labels) != n:
        raise SystemExit(f"label length mismatch in {npy_path}: {len(labels)} vs {n}")
    check_guard(npy_path, labels, embedding_ids)
    return labels.astype(np.int32, copy=False), int(len(np.unique(labels)))

def metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    try:
        return {
            "silhouette": float(silhouette_score(features, labels)),
            "davies_bouldin": float(davies_bouldin_score(features, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        }
    except (ValueError, FloatingPointError) as exc:
        raise SystemExit(f"metric calculation failed: {exc}") from exc

def add_result(
    rows: list[dict[str, Any]],
    method: str,
    representation: str,
    metric_space: str,
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> None:
    rows.append(
        {
            "method": method,
            "representation": representation,
            "metric_space": metric_space,
            "k": k,
            **metrics(features, labels),
        }
    )

def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        encoded = frame.to_json(orient="records")
        if encoded is None:
            raise ValueError("pandas returned no JSON")
        value = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"could not serialize evaluation records: {exc}") from exc
    if not isinstance(value, list):
        raise SystemExit("evaluation records are not a list")
    return [item for item in value if isinstance(item, dict)]

def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Representation | K | Silhouette | DBI | CHI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        try:
            method = str(row["method"])
            representation = str(row["representation"])
            k = int(row["k"])
            silhouette = float(row["silhouette"])
            dbi = float(row["davies_bouldin"])
            chi = float(row["calinski_harabasz"])
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid report row: {row}") from exc
        lines.append(
            f"| {method} | {representation} | {k} | {silhouette:.6f} | "
            f"{dbi:.6f} | {chi:.3f} |"
        )
    return "\n".join(lines)

def report_text(table1: list[dict[str, Any]], table2: list[dict[str, Any]]) -> str:
    return f"""# Equivalent-space post-hoc evaluation

## Policy

This report evaluates existing labels only. It does not rerun K-Means, UMAP,
or DROWCULA, and it does not modify K=16, K=17, or any cluster assignment.
Image ordering was verified against `embedding_ids.csv` for all three label
sets.

Pure K-Means labels are evaluated in the saved DINOv3-only DROWCULA UMAP space
only to make the metrics numerically comparable. This does not imply that
K-Means and DROWCULA are fundamentally different final clustering operators:
DROWCULA's final clustering step is K-Means after dimensionality reduction.

## Table 1 — Common DINOv3→UMAP evaluation

All three existing label sets are evaluated in the exact saved
`DINOv3→UMAP 3-D` representation used by DINOv3-only DROWCULA.

{markdown_table(table1)}

## Table 2 — Native proposed-method diagnostic

Each DROWCULA result is evaluated in its native saved UMAP space.

{markdown_table(table2)}

## Interpretation

Table 1 is the valid direct comparison between the fixed K=16 pure DINOv3
K-Means labels and the fixed K=17 DINOv3-only DROWCULA labels. The KEC labels
are also shown in this common DINOv3 UMAP space for comparison against the
DINOv3-only pipeline.

Table 2 preserves the native proposed-method diagnostic for the fused
DINOv3+KEC representation. Its values must not be numerically compared with
Table 1 unless the metric space is explicitly the same.

This is post-clustering evaluation only. No ground-truth labels were used.
"""

def write_report(table1: list[dict[str, Any]], table2: list[dict[str, Any]], path: Path | None = None) -> None:
    try:
        (path or REPORT).write_text(report_text(table1, table2), encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not write report {path or REPORT}: {exc}") from exc

def run_final() -> None:
    """Jalur default: kontrak K=16/17, output ke folder final + REPORT. Tidak berubah."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    embedding_ids = load_embedding_ids()

    # These are already-saved representations. No UMAP fitting occurs here.
    dino_umap = load_array(DINO_DROW / "reduced_embeddings.npy", 2)
    fused_umap = load_array(FUSED_DROW / "reduced_fused_embeddings.npy", 2)
    n = len(dino_umap)
    if len(embedding_ids) != n or len(fused_umap) != n:
        raise SystemExit("representation row count does not match embedding IDs")

    baseline_labels = load_labels(BASELINE / "labels_k16.npy", n, 16, embedding_ids)
    dino_drow_labels = load_labels(DINO_DROW / "labels.npy", n, 17, embedding_ids)
    fused_drow_labels = load_labels(FUSED_DROW / "labels.npy", n, 17, embedding_ids)

    rows: list[dict[str, Any]] = []
    # Table 1: same DINOv3-only DROWCULA UMAP for every existing assignment.
    add_result(rows, "DINOv3 + K-Means", "DINOv3", "common_dino_umap_3d", dino_umap, baseline_labels, 16)
    add_result(rows, "DINOv3 + DROWCULA", "DINOv3", "common_dino_umap_3d", dino_umap, dino_drow_labels, 17)
    add_result(rows, "KEC + DINOv3 + DROWCULA", "DINOv3+KEC", "common_dino_umap_3d", dino_umap, fused_drow_labels, 17)

    # Table 2: native saved UMAP space for each DROWCULA representation.
    add_result(rows, "DINOv3 + DROWCULA", "DINOv3", "native_dino_umap_3d", dino_umap, dino_drow_labels, 17)
    add_result(rows, "KEC + DINOv3 + DROWCULA", "DINOv3+KEC", "native_fused_umap_3d", fused_umap, fused_drow_labels, 17)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(OUTPUT / "equivalent_space_metrics.csv", index=False)
    table1 = frame_records(result_frame.loc[result_frame["metric_space"].eq("common_dino_umap_3d")].copy())
    table2 = frame_records(result_frame.loc[result_frame["metric_space"].isin(["native_dino_umap_3d", "native_fused_umap_3d"])].copy())
    save_json({"table_1_common_dino_umap": table1, "table_2_native_diagnostics": table2, "post_hoc_only": True, "clustering_rerun": False}, OUTPUT / "equivalent_space_metrics.json")
    save_json(
        {
            "post_hoc_only": True,
            "clustering_rerun": False,
            "assignments_changed": False,
            "metric_policy": {
                "table_1": "all existing labels evaluated in saved DINOv3-only UMAP 3-D",
                "table_2": "DROWCULA labels evaluated in their native saved UMAP spaces",
                "kmeans_note": "K-Means labels are evaluated in DROWCULA UMAP space only for comparability",
            },
            "embedding_ids_sha256": sha256_file(PROC / "embedding_ids.csv"),
            "dino_umap_sha256": sha256_file(DINO_DROW / "reduced_embeddings.npy"),
            "fused_umap_sha256": sha256_file(FUSED_DROW / "reduced_fused_embeddings.npy"),
            "n_samples": n,
            "selected_k_preserved": {"pure_kmeans": 16, "dino_drowcula": 17, "fused_drowcula": 17},
        },
        OUTPUT / "config.json",
    )
    write_report(table1, table2)
    log.info("saved post-hoc evaluation to %s and %s", OUTPUT, REPORT)
    print(result_frame.to_string(index=False))

def run_workdir(proc_dir: Path) -> None:
    """Jalur workdir kecil: K apa pun yang tersimpan, output di workdir saja."""
    dino_drow = proc_dir / "drowcula_test"
    kmeans_dir = proc_dir / "kmeans_test"
    if not (dino_drow / "reduced_embeddings.npy").exists():
        raise SystemExit(f"workdir belum ada hasil 02: {dino_drow}")
    output_dir = proc_dir / "equivalent_space_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_ids = load_embedding_ids(proc_dir / "embedding_ids.csv")

    dino_umap = load_array(dino_drow / "reduced_embeddings.npy", 2)
    n = len(dino_umap)
    if len(embedding_ids) != n:
        raise SystemExit("representation row count does not match embedding IDs")

    dino_labels, dino_k = load_labels_any(dino_drow / "labels.npy", n, embedding_ids)
    rows: list[dict[str, Any]] = []
    add_result(rows, "DINOv3 + DROWCULA", "DINOv3", "native_dino_umap_3d", dino_umap, dino_labels, dino_k)

    # K-Means workdir: pakai K terpilih dari selection_summary.json bila ada.
    selected_k: int | None = None
    summary_path = kmeans_dir / "selection_summary.json"
    if summary_path.exists():
        try:
            selected_k = int(json.loads(summary_path.read_text(encoding="utf-8"))["selected_k_by_silhouette"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            selected_k = None
    if selected_k is not None:
        km_npy = kmeans_dir / f"labels_k{selected_k:02d}.npy"
        if km_npy.exists():
            km_labels, km_k = load_labels_any(km_npy, n, embedding_ids)
            add_result(rows, "DINOv3 + K-Means", "DINOv3", "common_dino_umap_3d", dino_umap, km_labels, km_k)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "equivalent_space_metrics.csv", index=False)
    save_json({"rows": frame_records(result_frame.copy()), "post_hoc_only": True,
               "clustering_rerun": False, "n_samples": n,
               "proc_dir": str(proc_dir)}, output_dir / "equivalent_space_metrics.json")
    save_json({"post_hoc_only": True, "clustering_rerun": False, "n_samples": n,
               "proc_dir": str(proc_dir),
               "embedding_ids_sha256": sha256_file(proc_dir / "embedding_ids.csv"),
               "dino_umap_sha256": sha256_file(dino_drow / "reduced_embeddings.npy")},
              output_dir / "config.json")
    report_path = output_dir / "EVAL.md"
    write_report(frame_records(result_frame.copy()), [], report_path)
    log.info("saved workdir evaluation to %s and %s", output_dir, report_path)
    print(result_frame.to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROC,
                        help="Workdir dataset baru (default: folder electronic final).")
    args = parser.parse_args()
    if args.proc_dir.resolve() != PROC.resolve():
        run_workdir(args.proc_dir)
    else:
        run_final()

if __name__ == "__main__":
    main()
