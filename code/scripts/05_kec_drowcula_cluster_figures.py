"""KEC + DROWCULA cluster figures: top-9/far-9 image grids per cluster.

For each selected K=17 cluster, this script saves two grids:

- top-9: nearest samples to the fused-space centroid
- far-9: farthest samples from the fused-space centroid

Images are read through existing absolute embedding paths; assignments are
joined to them using repository-relative manifest paths. No clustering method
is changed here.

Run from repository root:
    python3.12 code/scripts/05_kec_drowcula_cluster_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
RESULT = PROC / "kec_drowcula_dinov3"
FIG_DIR = RESULT / "figures"

CONFIG = {
    "seed": 42,
    "topn": 9,
    "figsize": (12, 12),
    "thumb": 256,
    "dpi": 100,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kec_drowcula_figures")


def canonical_path(path: str) -> str:
    value = path.replace("\\", "/")
    marker = "/electronic/"
    if marker in value:
        return "electronic/" + value.split(marker, 1)[1]
    return value


def resolve_image_path(relative_path: str, embedding_ids: pd.DataFrame) -> Path:
    wanted = canonical_path(relative_path)
    for raw in embedding_ids["filepath"]:
        if canonical_path(str(raw)) == wanted:
            return Path(str(raw))
    raise SystemExit(f"no readable image found for manifest path: {relative_path}")


def load_thumb(path: Path) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((CONFIG["thumb"], CONFIG["thumb"]))
        return image
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, str(exc)[:160])
        return Image.new("RGB", (CONFIG["thumb"], CONFIG["thumb"]), (200, 200, 200))


def save_grid(paths: list[Path], title: str, output: Path) -> None:
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=CONFIG["figsize"])
    axes = np.atleast_2d(axes)
    # strict=False: smoke test / cluster kecil boleh < 9 citra (grid diisi yang ada).
    for axis, path in zip(axes.flat, paths, strict=False):
        axis.imshow(load_thumb(path))
        axis.axis("off")
        axis.set_title(path.name[:30], fontsize=8)
    for axis in axes.flat[len(paths) :]:
        axis.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(output, dpi=CONFIG["dpi"])
    except OSError as exc:
        raise SystemExit(f"could not write {output}: {exc}") from exc
    finally:
        plt.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT,
                        help="Folder label + fused features + output figures (default: folder KEC final).")
    parser.add_argument("--embedding-ids", type=Path, default=PROC / "embedding_ids.csv")
    args = parser.parse_args()
    result_dir: Path = args.result_dir
    fig_dir: Path = result_dir / "figures"
    try:
        best_k = int((result_dir / "selected_k.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"selected K is missing or invalid: {exc}") from exc
    try:
        labels_frame = pd.read_csv(result_dir / "labels.csv")
        embedding_ids = pd.read_csv(args.embedding_ids)
        # DINO-only smoke test tidak punya fused_features.npy: pakai normalized_embeddings.npy.
        fused_path = result_dir / "fused_features.npy"
        if not fused_path.exists():
            fused_path = result_dir / "normalized_embeddings.npy"
        fused = np.load(fused_path).astype(np.float32)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load fused clustering inputs: {exc}") from exc
    if len(fused) != len(labels_frame):
        raise SystemExit("fused features and labels have different lengths")

    saved_labels = labels_frame["cluster"].to_numpy()
    centroids = np.zeros((best_k, fused.shape[1]), dtype=np.float32)
    for index in range(best_k):
        members = fused[saved_labels == index]
        if len(members) == 0:
            raise SystemExit(f"cluster {index} is empty; refusing to visualize")
        centroids[index] = members.mean(axis=0)
    distances = np.linalg.norm(fused - centroids[saved_labels], axis=1)
    assignment = labels_frame.assign(distance=distances).sort_values(["cluster", "distance"])

    out_fig_dir = fig_dir
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    # Kompatibel mundur: run final lama memakai FIG_DIR modul; run workdir memakai --result-dir.
    for cluster in tqdm(sorted(assignment["cluster"].unique()), desc="cluster figures"):
        try:
            cluster_id = int(cluster)
            group = assignment[assignment["cluster"] == cluster]
            top_rows = group.head(CONFIG["topn"])
            far_rows = group.tail(CONFIG["topn"])
            top_paths = [resolve_image_path(path, embedding_ids) for path in top_rows["filepath"]]
            far_paths = [resolve_image_path(path, embedding_ids) for path in far_rows["filepath"]]
            top_title = f"KEC+DROWCULA c{cluster_id:02d} nearest-{CONFIG['topn']} (n={len(group)})"
            far_title = f"KEC+DROWCULA c{cluster_id:02d} farthest-{CONFIG['topn']} (n={len(group)})"
            top_output = out_fig_dir / f"cluster_c{cluster_id:02d}_top9.png"
            far_output = out_fig_dir / f"cluster_c{cluster_id:02d}_far9.png"
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise SystemExit(f"could not prepare figure for cluster {cluster}: {exc}") from exc
        save_grid(top_paths, top_title, top_output)
        save_grid(far_paths, far_title, far_output)
    log.info("saved %d figures to %s", 2 * best_k, out_fig_dir)


if __name__ == "__main__":
    main()
