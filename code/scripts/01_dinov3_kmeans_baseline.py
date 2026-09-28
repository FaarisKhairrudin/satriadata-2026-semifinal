"""Pure frozen DINOv3 ViT-B/16 K-Means baseline for e-waste clustering.

Pipeline:

    existing DINOv3 ViT-B/16 embeddings (768-D)
        -> row-wise L2 normalization
        -> K-Means, K=5..25

This script deliberately contains no KEC, WordNet, LLM, UNU-KEY, TURTLE,
UMAP, or DROWCULA logic. It reuses the precomputed DINOv3 embeddings and
never reads image pixels or trains a backbone.

Run from the repository root:
    python3.12 code/scripts/01_dinov3_kmeans_baseline.py

Outputs are written to:
    data/processed/electronic/dinov3_kmeans_baseline/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]
from sklearn.cluster import KMeans  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
DEFAULT_INPUT = PROC / "embeddings_dinov3.npy"
DEFAULT_EMBEDDING_IDS = PROC / "embedding_ids.csv"
DEFAULT_MANIFEST = PROC / "manifest_train.csv"
DEFAULT_OUTPUT = PROC / "dinov3_kmeans_baseline"

CONFIG: dict[str, Any] = {
    "seed": 42,
    "k_min": 5,
    "k_max": 25,
    "n_init": 20,
    "max_iter": 300,
    "algorithm": "lloyd",
    "backbone": "DINOv3 ViT-B/16",
    "embedding_dimensions": 768,
    "normalization": "row-wise L2",
    "evaluation_space": "row-wise L2-normalized DINOv3 embeddings",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dinov3_kmeans_baseline")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    """Compare old absolute paths with current repository-relative paths."""
    value = path.replace("\\", "/")
    marker = "/electronic/"
    if marker in value:
        return "electronic/" + value.split(marker, 1)[1]
    return value


def load_inputs(
    input_path: Path, embedding_ids_path: Path, manifest_path: Path
) -> tuple[np.ndarray, pd.DataFrame, str]:
    if not input_path.exists():
        raise SystemExit(f"DINOv3 embeddings not found: {input_path}")
    for path, description in (
        (embedding_ids_path, "embedding IDs"),
        (manifest_path, "training manifest"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    try:
        features = np.load(input_path)
        embedding_ids = pd.read_csv(embedding_ids_path)
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load baseline inputs: {exc}") from exc

    if features.ndim != 2 or features.shape[1] != CONFIG["embedding_dimensions"]:
        raise SystemExit(
            f"expected DINOv3 ViT-B/16 shape (N, 768), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise SystemExit("DINOv3 embeddings contain non-finite values")
    for name, frame in (("embedding_ids", embedding_ids), ("manifest", manifest)):
        if "filepath" not in frame.columns:
            raise SystemExit(f"{name} must contain a filepath column")
    if len(features) != len(embedding_ids) or len(features) != len(manifest):
        raise SystemExit(
            "embedding and manifest lengths differ: "
            f"features={len(features)}, embedding_ids={len(embedding_ids)}, "
            f"manifest={len(manifest)}"
        )

    ids_order = [canonical_path(str(value)) for value in embedding_ids["filepath"]]
    manifest_order = [canonical_path(str(value)) for value in manifest["filepath"]]
    if ids_order != manifest_order:
        raise SystemExit("embedding_ids.csv and manifest_train.csv row order differs")
    log.info("validated DINOv3 embeddings and manifest: %s", features.shape)
    return np.asarray(features, dtype=np.float32), manifest, sha256_file(input_path)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    try:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        invalid = ~np.isfinite(norms) | (norms <= 0.0)
        if bool(np.any(invalid)):
            raise ValueError(f"{int(np.count_nonzero(invalid))} invalid feature rows")
        return (features / norms).astype(np.float32, copy=False)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"DINOv3 L2 normalization failed: {exc}") from exc


def clustering_accuracy(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Permutation-invariant ACC for optional post-clustering evaluation."""
    try:
        matrix = contingency_matrix(true_labels, cluster_labels)
        rows, columns = linear_sum_assignment(-matrix)
        total = matrix.sum()
        if total <= 0:
            raise ValueError("contingency matrix is empty")
        return float(matrix[rows, columns].sum() / total)
    except (ValueError, TypeError, FloatingPointError) as exc:
        raise SystemExit(f"clustering ACC calculation failed: {exc}") from exc


def find_ground_truth(manifest: pd.DataFrame) -> tuple[str | None, np.ndarray | None, str]:
    """Use a valid multi-class manifest label only for post-clustering metrics."""
    candidates = ("ground_truth", "label", "target", "class", "source_class")
    for column in candidates:
        if column not in manifest.columns:
            continue
        values = manifest[column]
        if bool(values.isna().any()):
            continue
        labels = values.astype(str).to_numpy()
        unique = np.unique(labels)
        if len(unique) < 2:
            return None, None, f"{column} has only one unique value; evaluation is not meaningful"
        return column, labels, "valid multi-class labels found"
    return None, None, "no ground-truth label column found in manifest"


def evaluate_optional(
    true_labels: np.ndarray | None, cluster_labels: np.ndarray, label_column: str | None
) -> dict[str, Any]:
    if true_labels is None or label_column is None:
        return {
            "status": "not_available",
            "reason": "no valid multi-class ground-truth labels; clustering was unsupervised",
        }
    try:
        return {
            "status": "computed_post_clustering_only",
            "label_column": label_column,
            "ari": float(adjusted_rand_score(true_labels, cluster_labels)),
            "nmi": float(normalized_mutual_info_score(true_labels, cluster_labels)),
            "acc": clustering_accuracy(true_labels, cluster_labels),
        }
    except (ValueError, TypeError, FloatingPointError) as exc:
        raise SystemExit(f"optional post-clustering evaluation failed: {exc}") from exc


def run_sweep(
    normalized: np.ndarray,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    output_dir: Path,
    label_column: str | None,
    true_labels: np.ndarray | None,
    label_status: str,
) -> None:
    scores: list[dict[str, Any]] = []
    for k in tqdm(range(args.k_min, args.k_max + 1), desc="K-Means sweep"):
        model = KMeans(
            n_clusters=k,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.seed,
            algorithm=args.algorithm,
        )
        labels = model.fit_predict(normalized)
        try:
            silhouette = float(silhouette_score(normalized, labels, metric="euclidean"))
            dbi = float(davies_bouldin_score(normalized, labels))
            chi = float(calinski_harabasz_score(normalized, labels))
        except (ValueError, FloatingPointError) as exc:
            raise SystemExit(f"metric calculation failed for K={k}: {exc}") from exc
        sizes = np.bincount(labels, minlength=k)
        try:
            scores.append(
                {
                    "k": k,
                    "inertia": float(model.inertia_),
                    "silhouette": silhouette,
                    "davies_bouldin": dbi,
                    "calinski_harabasz": chi,
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                }
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"could not record K={k} metrics: {exc}") from exc

    scores_frame = pd.DataFrame(scores).sort_values("k")
    scores_frame.to_csv(output_dir / "k_search_scores.csv", index=False)

    for record in scores:
        try:
            k = int(record["k"])
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid K in sweep record: {record}") from exc
        model = KMeans(
            n_clusters=k,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.seed,
            algorithm=args.algorithm,
        )
        labels = model.fit_predict(normalized).astype(np.int32)
        assignments = manifest.copy()
        assignments["cluster"] = labels
        assignments.to_csv(output_dir / f"labels_k{k:02d}.csv", index=False)
        np.save(output_dir / f"labels_k{k:02d}.npy", labels)
        pd.DataFrame(
            {"cluster": np.arange(k, dtype=np.int32), "size": np.bincount(labels, minlength=k)}
        ).to_csv(output_dir / f"cluster_sizes_k{k:02d}.csv", index=False)

        evaluation = evaluate_optional(true_labels, labels, label_column)
        record["evaluation"] = evaluation
        save_json(evaluation, output_dir / f"evaluation_k{k:02d}.json")

    # The baseline reports the full sweep and identifies the best silhouette K.
    try:
        best = max(
            scores,
            key=lambda record: (float(record["silhouette"]), -int(record["k"])),
        )
        best_k = int(best["k"])
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"could not select best K: {exc}") from exc
    save_json(
        {
            "selected_k_by_silhouette": best_k,
            "selection_metric": "silhouette",
            "selection_space": CONFIG["evaluation_space"],
            "k_search_range": [args.k_min, args.k_max],
            "label_column_considered": label_column,
            "ground_truth_status": label_status,
            "post_clustering_evaluation_only": True,
        },
        output_dir / "selection_summary.json",
    )
    log.info("highest silhouette is K=%d (%.6f)", best_k, best["silhouette"])


def save_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not write {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embedding-ids", type=Path, default=DEFAULT_EMBEDDING_IDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--k-min", type=int, default=CONFIG["k_min"])
    parser.add_argument("--k-max", type=int, default=CONFIG["k_max"])
    parser.add_argument("--n-init", type=int, default=CONFIG["n_init"])
    parser.add_argument("--max-iter", type=int, default=CONFIG["max_iter"])
    parser.add_argument("--algorithm", choices=("lloyd", "elkan"), default=CONFIG["algorithm"])
    parser.add_argument("--overwrite", action="store_true", help="replace existing output artifacts")
    args = parser.parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        parser.error("require 2 <= k-min <= k-max")
    if args.n_init < 1 or args.max_iter < 1:
        parser.error("n-init and max-iter must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.input = args.input.resolve()
    args.embedding_ids = args.embedding_ids.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, manifest, input_fingerprint = load_inputs(
        args.input, args.embedding_ids, args.manifest
    )
    normalized = l2_normalize(features)
    np.save(args.output_dir / "normalized_embeddings.npy", normalized)
    label_column, true_labels, label_status = find_ground_truth(manifest)
    log.info("ground-truth status: %s", label_status)
    run_sweep(
        normalized,
        args,
        manifest,
        args.output_dir,
        label_column,
        true_labels,
        label_status,
    )
    config = {
        **CONFIG,
        "seed": args.seed,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "input_embeddings": str(args.input),
        "input_sha256": input_fingerprint,
        "embedding_ids": str(args.embedding_ids),
        "manifest": str(args.manifest),
        "n_samples": len(normalized),
        "ground_truth_label_column": label_column,
        "ground_truth_status": label_status,
        "uses_kec": False,
        "uses_wordnet": False,
        "uses_llm": False,
        "uses_unu_keys": False,
        "uses_turtle": False,
        "uses_drowcula": False,
        "backbone_frozen": True,
        "output_dir": str(args.output_dir),
    }
    save_json(config, args.output_dir / "config.json")
    log.info("baseline complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
