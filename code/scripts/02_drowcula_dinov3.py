"""DROWCULA clustering adapted to existing DINOv3 ViT-B/16 embeddings.

This script implements the official DROWCULA unknown-K recipe:

    DINOv3 embeddings -> row-wise L2 normalization -> UMAP ->
    exhaustive silhouette K search -> K-Means final clusters

The original DROWCULA paper uses DINOv2. This experiment substitutes the
existing DINOv3 ViT-B/16 768-D embeddings; it does not claim that DINOv3 was
used in the original paper.

UMAP is fitted once and cached. All K candidates are evaluated on the same
cached reduced representation, so repeated runs do not redo dimensionality
reduction. K-Means settings follow the official UMAP implementation in the
DROWCULA repository: n_init=200 and max_iter=10000.

Usage from the repository root:
    python3.12 code/scripts/02_drowcula_dinov3.py

Install the one additional dependency if needed:
    python3.12 -m pip install --user umap-learn
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
from sklearn.cluster import KMeans  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tqdm import tqdm

# Official DROWCULA UMAP defaults from the released DROWCULA-UMAP notebook.
CONFIG: dict[str, Any] = {
    "seed": 42,
    "k_min": 2,
    "k_max": 25,
    "umap_n_components": 3,
    "umap_n_neighbors": 10,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",
    "kmeans_n_init": 200,
    "kmeans_max_iter": 10000,
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "electronic" / "embeddings_dinov3.npy"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "electronic" / "manifest_train.csv"
DEFAULT_EMBEDDING_IDS = ROOT / "data" / "processed" / "electronic" / "embedding_ids.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "electronic" / "drowcula_dinov3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drowcula_dinov3")


def set_seed(seed: int) -> None:
    """Set the random seeds used by NumPy, Python, and UMAP/K-Means callers."""
    random.seed(seed)
    np.random.seed(seed)


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    """Return float32 row-wise L2-normalized features."""
    try:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        invalid = ~np.isfinite(norms) | (norms <= 0.0)
        if bool(np.any(invalid)):
            bad = int(np.count_nonzero(invalid))
            raise ValueError(f"input contains {bad} zero or non-finite feature rows")
        return (features / norms).astype(np.float32, copy=False)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ValueError(f"could not L2-normalize embeddings: {exc}") from exc


def relative_manifest_path(path: str, root: Path) -> str:
    """Normalize equivalent absolute/relative manifest paths for comparison."""
    normalized = path.replace("\\", "/")
    marker = "/electronic/"
    if marker in normalized:
        return "electronic/" + normalized.split(marker, 1)[1]
    candidate = Path(normalized)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return normalized


def embedding_fingerprint(input_path: Path) -> str:
    """Hash the embedding bytes so a reduced cache cannot silently go stale."""
    digest = hashlib.sha256()
    try:
        with input_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SystemExit(f"could not hash embeddings {input_path}: {exc}") from exc
    return digest.hexdigest()


def load_inputs(
    input_path: Path, manifest_path: Path, embedding_ids_path: Path | None
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Load and validate DINOv3 features plus their row-order manifests."""
    if not input_path.exists():
        raise SystemExit(f"DINOv3 embedding file not found: {input_path}")
    if not manifest_path.exists():
        raise SystemExit(f"embedding manifest not found: {manifest_path}")

    try:
        features = np.load(input_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not load embeddings {input_path}: {exc}") from exc
    if features.ndim != 2 or features.shape[1] != 768:
        raise SystemExit(
            f"expected existing DINOv3 ViT-B/16 embeddings with shape (N, 768), "
            f"got {features.shape}"
        )
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise SystemExit("DINOv3 embeddings contain non-finite values")

    try:
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load manifest {manifest_path}: {exc}") from exc
    if "filepath" not in manifest.columns:
        raise SystemExit(f"manifest must contain a filepath column: {manifest_path}")
    if len(manifest) != len(features):
        raise SystemExit(
            f"embedding/manifest row mismatch: {len(features)} vs {len(manifest)}"
        )
    if bool(manifest["filepath"].isna().any()):
        raise SystemExit("manifest filepath column contains missing values")

    if embedding_ids_path is not None and embedding_ids_path.exists():
        try:
            embedding_ids = pd.read_csv(embedding_ids_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise SystemExit(
                f"could not load embedding ID manifest {embedding_ids_path}: {exc}"
            ) from exc
        if "filepath" not in embedding_ids.columns or len(embedding_ids) != len(manifest):
            raise SystemExit(
                "embedding_ids.csv must have the same row count and a filepath column"
            )
        manifest_keys = [relative_manifest_path(str(p), ROOT) for p in manifest["filepath"]]
        id_keys = [relative_manifest_path(str(p), ROOT) for p in embedding_ids["filepath"]]
        if manifest_keys != id_keys:
            raise SystemExit(
                "manifest row order does not match embedding_ids.csv; refusing "
                "to assign labels to potentially mismatched images"
            )
        log.info("validated embedding row order against %s", embedding_ids_path)

    fingerprint = embedding_fingerprint(input_path)
    log.info("loaded DINOv3 embeddings: shape=%s dtype=%s", features.shape, features.dtype)
    log.info("loaded manifest: %d rows", len(manifest))
    return features, manifest, fingerprint


def reduction_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return only values that determine the cached UMAP representation."""
    return {
        "seed": args.seed,
        "umap_n_components": args.umap_n_components,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "normalization": "row-wise L2",
    }


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_or_fit_umap(
    normalized: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    input_fingerprint: str,
) -> np.ndarray:
    """Load a compatible reduced cache or fit UMAP exactly once."""
    reduced_path = output_dir / "reduced_embeddings.npy"
    metadata_path = output_dir / "reduction_config.json"
    expected = {
        **reduction_config(args),
        "input_fingerprint_sha256": input_fingerprint,
        "input_shape": list(normalized.shape),
    }
    metadata = read_json(metadata_path)

    if not args.overwrite and reduced_path.exists() and metadata == expected:
        try:
            reduced = np.load(reduced_path)
            compatible_shape = reduced.shape == (len(normalized), args.umap_n_components)
            if compatible_shape and bool(np.isfinite(reduced).all()):
                log.info("loaded cached UMAP representation: %s", reduced.shape)
                return np.asarray(reduced, dtype=np.float32)
        except (OSError, ValueError):
            log.warning("cached UMAP representation is unreadable; recomputing")

    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "UMAP is required by the official DROWCULA pipeline. Install it with "
            "`python3.12 -m pip install --user umap-learn`."
        ) from exc

    log.info(
        "fitting UMAP: components=%d neighbors=%d min_dist=%s metric=%s",
        args.umap_n_components,
        args.umap_n_neighbors,
        args.umap_min_dist,
        args.umap_metric,
    )
    reducer = umap.UMAP(
        n_components=args.umap_n_components,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        transform_seed=args.seed,
    )
    reduced = np.asarray(reducer.fit_transform(normalized), dtype=np.float32)
    if not np.isfinite(reduced).all():
        raise SystemExit("UMAP produced non-finite reduced embeddings")
    np.save(reduced_path, reduced)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
    log.info("saved UMAP cache: %s", reduced_path)
    return reduced


def evaluate_k_search(
    reduced: np.ndarray, args: argparse.Namespace, output_dir: Path
) -> tuple[int, np.ndarray, pd.DataFrame]:
    """Run the official exhaustive silhouette-based unknown-K search."""
    records: list[dict[str, Any]] = []
    best_k: int | None = None
    best_silhouette = -np.inf
    best_labels: np.ndarray | None = None

    candidates = range(args.k_min, args.k_max + 1)
    for k in tqdm(candidates, desc="DROWCULA K search"):
        model = KMeans(
            n_clusters=k,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            random_state=args.seed,
            algorithm="lloyd",
        )
        labels = model.fit_predict(reduced)
        try:
            silhouette = float(silhouette_score(reduced, labels, metric="euclidean"))
            dbi = float(davies_bouldin_score(reduced, labels))
            chi = float(calinski_harabasz_score(reduced, labels))
        except (ValueError, FloatingPointError) as exc:
            raise SystemExit(f"metric calculation failed for K={k}: {exc}") from exc
        sizes = np.bincount(labels, minlength=k)
        try:
            record = {
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": dbi,
                "calinski_harabasz": chi,
                "inertia": float(model.inertia_),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            }
        except (TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"could not record metrics for K={k}: {exc}") from exc
        records.append(record)
        log.info(
            "K=%d silhouette=%.6f DBI=%.6f CHI=%.3f sizes=[%d,%d]",
            k,
            silhouette,
            dbi,
            chi,
            sizes.min(),
            sizes.max(),
        )
        # The official algorithm keeps the first maximizer on ties.
        if silhouette > best_silhouette:
            best_k = k
            best_silhouette = silhouette
            best_labels = labels.copy()

    if best_k is None or best_labels is None:
        raise SystemExit("K search produced no valid candidate")
    scores = pd.DataFrame(records)
    scores.to_csv(output_dir / "k_search_scores.csv", index=False)
    log.info("selected K=%d by maximum reduced-space silhouette", best_k)
    return best_k, best_labels, scores


def save_outputs(
    normalized: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
    best_k: int,
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    input_path: Path,
    manifest_path: Path,
    input_fingerprint: str,
    output_dir: Path,
) -> None:
    """Write all reproducibility artifacts for the selected clustering."""
    try:
        sizes = np.bincount(labels, minlength=best_k)
        best_rows = scores.loc[scores["k"] == best_k]
        if len(best_rows) != 1:
            raise ValueError(f"expected exactly one score row for K={best_k}")
        best_row = best_rows.iloc[0]
        final_metrics = {
            "selected_k": best_k,
            "selection_metric": "silhouette",
            "selection_space": "UMAP-reduced L2-normalized DINOv3 embeddings",
            "silhouette": float(best_row["silhouette"]),
            "davies_bouldin": float(best_row["davies_bouldin"]),
            "calinski_harabasz": float(best_row["calinski_harabasz"]),
            "inertia": float(best_row["inertia"]),
            "n_samples": int(len(labels)),
            "reduced_dimensions": int(reduced.shape[1]),
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SystemExit(f"could not assemble final metrics: {exc}") from exc
    config = {
        **CONFIG,
        "seed": args.seed,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "umap_n_components": args.umap_n_components,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "backbone": "DINOv3 ViT-B/16",
        "embedding_dimensions": 768,
        "method_adaptation": "DROWCULA adapted from DINOv2 to existing DINOv3 embeddings",
        "input_embeddings": str(input_path),
        "input_manifest": str(manifest_path),
        "input_fingerprint_sha256": input_fingerprint,
        "output_directory": str(output_dir),
    }

    np.save(output_dir / "normalized_embeddings.npy", normalized)
    np.save(output_dir / "labels.npy", labels.astype(np.int32))
    with (output_dir / "selected_k.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{best_k}\n")
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    cluster_sizes = pd.DataFrame(
        {"cluster": np.arange(best_k, dtype=np.int32), "size": sizes.astype(np.int64)}
    )
    cluster_sizes.to_csv(output_dir / "cluster_sizes.csv", index=False)

    assignments = manifest.copy()
    assignments.insert(len(assignments.columns), "cluster", labels.astype(np.int32))
    assignments.to_csv(output_dir / "labels.csv", index=False)

    log.info("saved final metrics: %s", output_dir / "final_metrics.json")
    log.info("cluster sizes: %s", sizes.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--embedding-ids",
        type=Path,
        default=DEFAULT_EMBEDDING_IDS,
        help="optional existing embedding_ids.csv used as a row-order guard",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--k-min", type=int, default=CONFIG["k_min"])
    parser.add_argument("--k-max", type=int, default=CONFIG["k_max"])
    parser.add_argument(
        "--umap-n-components", type=int, default=CONFIG["umap_n_components"]
    )
    parser.add_argument(
        "--umap-n-neighbors", type=int, default=CONFIG["umap_n_neighbors"]
    )
    parser.add_argument("--umap-min-dist", type=float, default=CONFIG["umap_min_dist"])
    parser.add_argument("--umap-metric", default=CONFIG["umap_metric"])
    parser.add_argument(
        "--kmeans-n-init", type=int, default=CONFIG["kmeans_n_init"]
    )
    parser.add_argument(
        "--kmeans-max-iter", type=int, default=CONFIG["kmeans_max_iter"]
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="refit UMAP and redo all outputs instead of using compatible caches",
    )
    args = parser.parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        parser.error("require 2 <= k-min <= k-max")
    if args.umap_n_components < 2 or args.umap_n_neighbors < 2:
        parser.error("UMAP components and neighbors must be at least 2")
    if args.umap_min_dist < 0:
        parser.error("UMAP min-dist must be non-negative")
    if args.kmeans_n_init < 1 or args.kmeans_max_iter < 1:
        parser.error("K-Means n-init and max-iter must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.input = args.input.resolve()
    args.manifest = args.manifest.resolve()
    args.embedding_ids = args.embedding_ids.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, manifest, input_fingerprint = load_inputs(
        args.input, args.manifest, args.embedding_ids
    )
    normalized = l2_normalize_rows(features)
    reduced = load_or_fit_umap(
        normalized, args, args.output_dir, input_fingerprint
    )
    best_k, labels, scores = evaluate_k_search(reduced, args, args.output_dir)
    save_outputs(
        normalized,
        reduced,
        labels,
        best_k,
        scores,
        manifest,
        args,
        args.input,
        args.manifest,
        input_fingerprint,
        args.output_dir,
    )
    log.info("DROWCULA pipeline complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
