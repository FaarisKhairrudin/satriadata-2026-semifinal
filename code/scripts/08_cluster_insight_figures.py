"""Insight figures for fixed KEC+DROWCULA clusters (K=17).

Reads existing outputs only. No clustering, no UMAP refit, no UNU use in features.
UNU-KEYs/policy tables from docs/00_UNUKEY.md are used only as plot annotations.

Inputs (all in data/processed/electronic/kec_drowcula_dinov3/):
  k_search_scores.csv, labels.csv, cluster_sizes.csv,
  cluster_grounding.csv (auto SigLIP2 scores), final_cluster_labels.csv (human-verified),
  reduced_fused_embeddings.npy (N,3), final_metrics.json, selected_k.txt

Outputs (figures/):
  insight_01_k_search.png, insight_02_umap_pairwise.png, insight_03_sizes.png,
  insight_04_cosine_margin.png, insight_05_vote_share.png, insight_06_eu_composition.png,
  insight_07_hg_flag.png, insight_08_weee_targets.png, insight_09_summary.png,
  insight_10_umap_3d.png, insight_11_umap_fused_vs_dino.png

Run from repo root:
  python3.12 code/scripts/08_cluster_insight_figures.py [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "data" / "processed" / "electronic" / "kec_drowcula_dinov3"
DINO_RES = ROOT / "data" / "processed" / "electronic" / "drowcula_dinov3"
FIG = RESULT / "figures"

CONFIG = {
    "seed": 42,
    "dpi": 150,
    # Hg-bearing UNU-KEYs (Minamata, docs/00_UNUKEY.md): kept as plot flags only.
    "hg_keys": {"0302", "0303", "0306", "0309", "0408", "0502", "0503", "0504"},
    "hazardous_gap": {"GAP-BATT", "GAP-PCB"},
    "view_elev": 18,
    "view_azim": -60,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("insight_figs")

# Clean style scoped to insight_03 only (via plt.rc_context, other figures untouched).
STYLE_03 = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#D7DFE5",
    "text.color": "#243746",
    "axes.labelcolor": "#526675",
    "xtick.color": "#526675",
    "ytick.color": "#243746",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# Emoji per verified cluster id, reused from 07_cluster_risk_value_mapping.py
# (same ids, same human-verified streams). Rendered with Noto Emoji; falls back
# to the default font when the file is missing (may show tofu boxes then).
_EMOJI_PATH = Path("/usr/share/fonts/google-noto-emoji-fonts/NotoEmoji-Regular.ttf")
EMOJI_FONT = FontProperties(fname=str(_EMOJI_PATH)) if _EMOJI_PATH.exists() else None
EMOJI_MAP = {
    0: "🖨", 1: "📱", 2: "⚙", 3: "⌨", 4: "🧺", 5: "🖱", 6: "📺",
    7: "♨", 8: "💻", 9: "🔋", 10: "📻", 11: "📺", 12: "⚡", 13: "🧺",
    14: "📻", 15: "🖥", 16: "🔌",
}


def emoji_kwargs() -> dict:
    return {"fontproperties": EMOJI_FONT} if EMOJI_FONT is not None else {}


def radial_label_pos(centroids: np.ndarray, offset: float = 3.0) -> np.ndarray:
    """Push label anchors radially outward from the data center (deterministic)."""
    center = centroids.mean(axis=0)
    direction = centroids - center
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    return centroids + direction / norms * offset

OUTPUTS = [
    "insight_01_k_search.png",
    "insight_02_umap_pairwise.png",
    "insight_03_sizes.png",
    "insight_04_cosine_margin.png",
    "insight_05_vote_share.png",
    "insight_06_eu_composition.png",
    "insight_07_hg_flag.png",
    "insight_08_weee_targets.png",
    "insight_09_summary.png",
    "insight_10_umap_3d.png",
    "insight_11_umap_fused_vs_dino.png",
]


def short_eu(value: str) -> str:
    mapping = {
        "Small IT": "Small IT",
        "Small equipment": "Small equip.",
        "Large equipment (excl. PV)": "Large equip.",
        "Temperature exchange equipment": "Temp. exch.",
        "Screens and monitors": "Screens",
    }
    if value in mapping:
        return mapping[value]
    if "A1181" in value:
        return "Basel A1181 (haz.)"
    if "Y49" in value:
        return "Basel Y49 (non-haz.)"
    return "HS/other"


def weee_group(eu: str) -> str:
    if eu in ("Small IT", "Small equipment"):
        return "5&6 Small"
    if eu in ("Large equipment (excl. PV)", "Temperature exchange equipment"):
        return "1&4 Large+Temp"
    if eu == "Screens and monitors":
        return "2 Screens"
    return "non-UNU"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    np.random.seed(CONFIG["seed"])
    FIG.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and all((FIG / name).exists() for name in OUTPUTS):
        log.info("all insight figures exist, skip (use --overwrite)")
        return

    scores = pd.read_csv(RESULT / "k_search_scores.csv")
    labels = pd.read_csv(RESULT / "labels.csv")
    ground = pd.read_csv(RESULT / "cluster_grounding.csv")
    verified = pd.read_csv(RESULT / "final_cluster_labels.csv")
    with open(RESULT / "final_metrics.json") as handle:
        metrics = json.load(handle)
    best_k = int((RESULT / "selected_k.txt").read_text(encoding="utf-8").strip())
    umap3 = np.load(RESULT / "reduced_fused_embeddings.npy").astype(np.float32)
    clusters = labels["cluster"].to_numpy()
    assert len(umap3) == len(labels) == int(metrics["n_samples"])

    # Join auto grounding with human-verified names on cluster id.
    merged = ground.merge(
        verified[["cluster", "final_label", "unu_key_code", "eu_category", "vote_agreement"]],
        on="cluster",
        suffixes=("_auto", "_final"),
        how="left",
    )
    n_corrections = int((merged["vote_agreement"] == "human correction").sum())

    # 1. K search: silhouette + DBI select K; inertia + min-size diagnose cost.
    _, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(scores["k"], scores["silhouette"], marker="o", label="silhouette")
    ax[0].axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax[0].set_ylabel("silhouette")
    ax[0].legend(fontsize=8)
    ax2 = ax[0].twinx()
    ax2.plot(scores["k"], scores["davies_bouldin"], marker="s", color="orange", label="DBI (lower=better)")
    ax2.set_ylabel("Davies-Bouldin")
    ax[1].plot(scores["k"], scores["inertia"], marker="o", label="inertia")
    ax[1].set_ylabel("inertia")
    ax[1].set_xlabel("K")
    axb = ax[1].twinx()
    axb.plot(scores["k"], scores["min_cluster_size"], marker="s", color="green", label="min size")
    axb.set_ylabel("min cluster size")
    plt.suptitle(f"DROWCULA K search (silhouette max at K={best_k}, N={len(labels)})")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[0], dpi=CONFIG["dpi"])
    plt.close()

    # 2. UMAP pairwise scatter: structure of the space K was selected in.
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) if best_k <= 20 else cmap(i / best_k) for i in range(best_k)]
    _, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (d1, d2) in zip(axes, [(0, 1), (0, 2), (1, 2)], strict=True):
        for c in range(best_k):
            pts = umap3[clusters == c]
            axis.scatter(pts[:, d1], pts[:, d2], s=4, alpha=0.5, color=colors[c], label=f"C{c:02d}" if d1 == 0 and d2 == 1 else None)
        axis.set_xlabel(f"UMAP {d1}")
        axis.set_ylabel(f"UMAP {d2}")
    axes[0].legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    plt.suptitle("Fused DINOv3+KEC UMAP (3-D) pairwise view, colored by final K=17 labels")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[1], dpi=CONFIG["dpi"])
    plt.close()

    # 10. UMAP 3-D static view (same 3 dims, fixed angle for reproducibility).
    # Star = fused-space centroid per cluster, emoji + Cxx tag on top of it.
    static_fig = plt.figure(figsize=(9, 7))
    static_ax = static_fig.add_subplot(111, projection="3d")
    for c in range(best_k):
        pts = umap3[clusters == c]
        static_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.5,
                          color=colors[c], label=f"C{c:02d}")
    centroids = np.stack([umap3[clusters == c].mean(axis=0) for c in range(best_k)])
    static_ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=140,
                      marker="*", color="black", edgecolors="white", linewidths=0.8, zorder=5)
    label_pos = radial_label_pos(centroids)
    for c in range(best_k):
        static_ax.plot([centroids[c, 0], label_pos[c, 0]],
                       [centroids[c, 1], label_pos[c, 1]],
                       [centroids[c, 2], label_pos[c, 2]],
                       color="0.35", lw=0.8, alpha=0.8, zorder=5)
        static_ax.text(label_pos[c, 0], label_pos[c, 1], label_pos[c, 2],
                       EMOJI_MAP.get(c, ""), fontsize=14, va="bottom", ha="center", zorder=6,
                       **emoji_kwargs())
        static_ax.text(label_pos[c, 0], label_pos[c, 1], label_pos[c, 2],
                       f"C{c:02d}", fontsize=8, va="top", ha="center", zorder=6)
    static_ax.view_init(elev=CONFIG["view_elev"], azim=CONFIG["view_azim"])
    static_ax.set_xlabel("UMAP 0")
    static_ax.set_ylabel("UMAP 1")
    static_ax.set_zlabel("UMAP 2")
    static_ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    plt.title(f"Fused UMAP 3-D view (elev={CONFIG['view_elev']}, azim={CONFIG['view_azim']}, N={len(labels)})")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[9], dpi=CONFIG["dpi"])
    plt.close()

    # 11. Fused vs DINOv3-only UMAP 3-D side-by-side. Separate UMAP fits, so
    # orientation is arbitrary: compare structure + metrics, not coordinates.
    # Cluster ids do NOT correspond between the two runs (same palette reuse).
    dino_umap = np.load(DINO_RES / "reduced_embeddings.npy").astype(np.float32)
    dino_labels = pd.read_csv(DINO_RES / "labels.csv")["cluster"].to_numpy()
    with open(DINO_RES / "final_metrics.json") as handle:
        dino_metrics = json.load(handle)
    assert len(dino_umap) == len(dino_labels) == len(labels)
    cmp_fig = plt.figure(figsize=(15, 6))
    left = cmp_fig.add_subplot(121, projection="3d")
    for c in range(best_k):
        pts = dino_umap[dino_labels == c]
        left.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.5, color=colors[c])
    # Neutral Dxx tags only: left ids do NOT match verified labels, so no emoji here.
    dino_centroids = np.stack([dino_umap[dino_labels == c].mean(axis=0) for c in range(best_k)])
    left.scatter(dino_centroids[:, 0], dino_centroids[:, 1], dino_centroids[:, 2], s=140,
                 marker="*", color="black", edgecolors="white", linewidths=0.8, zorder=5)
    dino_pos = radial_label_pos(dino_centroids)
    for c in range(best_k):
        left.plot([dino_centroids[c, 0], dino_pos[c, 0]],
                  [dino_centroids[c, 1], dino_pos[c, 1]],
                  [dino_centroids[c, 2], dino_pos[c, 2]],
                  color="0.35", lw=0.8, alpha=0.8, zorder=5)
        left.text(dino_pos[c, 0], dino_pos[c, 1], dino_pos[c, 2],
                  f"D{c:02d}", fontsize=8, va="center", ha="center", zorder=6)
    left.view_init(elev=CONFIG["view_elev"], azim=CONFIG["view_azim"])
    left.set_xlabel("UMAP 0")
    left.set_ylabel("UMAP 1")
    left.set_zlabel("UMAP 2")
    left.set_title(f"DINOv3-only (768-D, sil {dino_metrics['silhouette']:.4f})")
    right = cmp_fig.add_subplot(122, projection="3d")
    for c in range(best_k):
        pts = umap3[clusters == c]
        right.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.5,
                      color=colors[c], label=f"C{c:02d}")
    right.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=140,
                  marker="*", color="black", edgecolors="white", linewidths=0.8, zorder=5)
    for c in range(best_k):
        right.plot([centroids[c, 0], label_pos[c, 0]],
                   [centroids[c, 1], label_pos[c, 1]],
                   [centroids[c, 2], label_pos[c, 2]],
                   color="0.35", lw=0.8, alpha=0.8, zorder=5)
        right.text(label_pos[c, 0], label_pos[c, 1], label_pos[c, 2],
                   EMOJI_MAP.get(c, ""), fontsize=14, va="bottom", ha="center", zorder=6,
                   **emoji_kwargs())
        right.text(label_pos[c, 0], label_pos[c, 1], label_pos[c, 2],
                   f"C{c:02d}", fontsize=8, va="top", ha="center", zorder=6)
    right.view_init(elev=CONFIG["view_elev"], azim=CONFIG["view_azim"])
    right.set_xlabel("UMAP 0")
    right.set_ylabel("UMAP 1")
    right.set_zlabel("UMAP 2")
    right.set_title(f"Fused DINOv3+KEC (1536-D, sil {metrics['silhouette']:.4f})")
    right.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    d_sil = metrics["silhouette"] - dino_metrics["silhouette"]
    d_dbi = metrics["davies_bouldin"] - dino_metrics["davies_bouldin"]
    plt.suptitle(f"Same UMAP recipe (3-D, n10, d0.1, seed 42), separate fits. "
                 f"Δsil {d_sil:+.4f}, ΔDBI {d_dbi:+.4f} (DBI lower=better).\n"
                 f"Left Dxx = DINO-only ids (no verified mapping); emoji only on fused panel.",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[10], dpi=CONFIG["dpi"])
    plt.close()

    # 3. Cluster sizes with verified labels (horizontal, biggest on top).
    # STYLE_03 scoped here only; single-hue gradient by size (varied, not rainbow).
    ordered = verified.sort_values("n_samples", ascending=True).reset_index(drop=True)
    ordered["tag"] = ordered.apply(lambda r: f"C{int(r['cluster']):02d} · {r['final_label']}", axis=1)
    pos = np.arange(len(ordered))
    mean_n = float(ordered["n_samples"].mean())
    shades = (ordered["n_samples"] - ordered["n_samples"].min()) / max(ordered["n_samples"].max() - ordered["n_samples"].min(), 1)
    with plt.rc_context(STYLE_03):
        _, ax3 = plt.subplots(figsize=(11, 7))
        bars = ax3.barh(pos, ordered["n_samples"],
                        color=[plt.get_cmap("Blues")(0.35 + 0.65 * s) for s in shades],
                        edgecolor="white", height=0.68, zorder=3)
        mean_line = ax3.axvline(mean_n, color="#243746", linestyle="--", linewidth=1.4,
                                label=f"mean {mean_n:.0f}")
        ax3.set_yticks(pos)
        ax3.set_yticklabels(ordered["tag"], fontsize=9)
        ax3.set_xlabel("n_samples")
        ax3.set_title(f"Cluster distribution by verified label (N={ordered['n_samples'].sum()}, K={len(ordered)})",
                      fontsize=12)
        ax3.xaxis.grid(True, color="#D7DFE5", linewidth=0.6, alpha=0.8, zorder=0)
        ax3.set_axisbelow(True)
        ax3.legend(handles=[mean_line], fontsize=9, frameon=False, loc="lower right")
        for bar, value in zip(bars, ordered["n_samples"], strict=True):
            ax3.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                     str(int(value)), fontsize=9, va="center")
        plt.tight_layout()
        plt.savefig(FIG / OUTPUTS[2], dpi=CONFIG["dpi"])
        plt.close()

    # 4. Cosine vs margin: auto grounding is weak (all cosine <0.16, no margin >=0.04).
    conf_color = {"MODERATE": "blue", "LOW": "grey", "HIGH (COVERAGE_GAP:BATT)": "red",
                  "HIGH (COVERAGE_GAP:PCB)": "darkred", "HIGH (COVERAGE_GAP)": "red"}
    plt.figure(figsize=(8, 5))
    for _, row in merged.iterrows():
        plt.scatter(row["cosine_score"], row["margin_vs_second"], s=row["n_samples"] / 3,
                    color=conf_color.get(str(row["confidence"]), "black"), alpha=0.7)
        plt.text(row["cosine_score"], row["margin_vs_second"], f"C{int(row['cluster']):02d}", fontsize=7)
    plt.axhline(0.04, color="red", linestyle="--", label="HIGH margin 0.04 (none pass)")
    plt.xlabel("top-1 cosine (SigLIP2 centroid vs taxonomy)")
    plt.ylabel("margin top1-top2")
    plt.title("Auto grounding confidence is weak: low cosine, tiny margins")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[3], dpi=CONFIG["dpi"])
    plt.close()

    # 5. Vote share: which clusters are heterogeneous and needed human correction.
    vote = merged.sort_values("vote_winner_share").reset_index(drop=True)
    vote_colors = {"MAJORITY": "green", "MAJORITY (COVERAGE_GAP)": "darkgreen",
                   "PLURALITY": "orange", "HETEROGENEOUS": "red",
                   "HETEROGENEOUS (COVERAGE_GAP)": "red"}
    plt.figure(figsize=(10, 5))
    pos = np.arange(len(vote))
    bars = plt.barh(pos, vote["vote_winner_share"],
                    color=[vote_colors.get(str(v), "grey") for v in vote["vote_confidence"]])
    plt.axvline(0.5, color="black", linestyle="--", label="majority 0.5")
    plt.axvline(0.35, color="grey", linestyle=":", label="plurality 0.35")
    plt.yticks(pos, vote["cluster"].astype(str), fontsize=8)
    plt.xlabel("winner vote share (per-image SigLIP2 votes)")
    plt.ylabel("cluster")
    plt.title(f"Vote heterogeneity ({n_corrections}/17 human-corrected; red = needs review)")
    plt.legend(fontsize=8)
    for bar, (_, row) in zip(bars, vote.iterrows(), strict=True):
        plt.text(float(row["vote_winner_share"]) + 0.01, bar.get_y() + bar.get_height() / 2,
                 str(row["vote_code"])[:18], fontsize=7, va="center")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[4], dpi=CONFIG["dpi"])
    plt.close()

    # 6. EU breakdown: which verified clusters sit inside each EU group + sizes.
    comp = verified.copy()
    comp["eu_short"] = comp["eu_category"].map(short_eu)
    eu_order = comp.groupby("eu_short")["n_samples"].sum().sort_values(ascending=True).index.tolist()
    plt.figure(figsize=(13, 6.5))
    pos = np.arange(len(eu_order))
    for yi, eu in enumerate(eu_order):
        group = comp[comp["eu_short"] == eu].sort_values("n_samples", ascending=True)
        left = 0
        for _, row in group.iterrows():
            cid = int(row["cluster"])
            width = int(row["n_samples"])
            plt.barh(yi, width, left=left, color=colors[cid], edgecolor="white")
            emo = EMOJI_MAP.get(cid, "")
            cx = left + width / 2
            plt.text(cx, yi + 0.14, emo, fontsize=10, ha="center", va="center",
                     **emoji_kwargs())
            plt.text(cx, yi - 0.18, f"C{cid:02d} ({width})" if width > 110 else f"C{cid:02d}",
                     fontsize=8, ha="center", va="center")
            left += width
        plt.text(left + 15, yi, f"total {left}", fontsize=8, va="center")
    plt.yticks(pos, [f"{eu} ({int(comp[comp['eu_short'] == eu]['n_samples'].sum())})" for eu in eu_order], fontsize=8)
    plt.xlabel("n_samples (segments = clusters)")
    plt.title("Verified EU groups and their cluster breakdown (same Cxx colors as other plots; names in insight_03)")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[5], dpi=CONFIG["dpi"])
    plt.close()

    # 7. Hazard flag from docs/00_UNUKEY.md Hg list + Basel A1181 GAPs.
    merged["hazard"] = merged["unu_key_code_final"].apply(
        lambda code: 2 if code in CONFIG["hazardous_gap"]
        else (1 if str(code) in CONFIG["hg_keys"] else 0)
    )
    hz = merged.sort_values(["hazard", "n_samples"], ascending=[False, False]).reset_index(drop=True)
    hz_colors = {2: "darkred", 1: "orange", 0: "lightgrey"}
    plt.figure(figsize=(10, 5))
    pos = np.arange(len(hz))
    bars = plt.barh(pos, hz["n_samples"],
                    color=[hz_colors[v] for v in hz["hazard"]])
    plt.yticks(pos, hz["cluster"].astype(str), fontsize=8)
    plt.xlabel("n_samples")
    plt.ylabel("cluster")
    plt.title("Hazard triage flag (red=Basel A1181 GAP-BATT/PCB, orange=Hg-list UNU, grey=other)")
    for bar, (_, row) in zip(bars, hz.iterrows(), strict=True):
        plt.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                 f"{row['unu_key_code_final']} {str(row['final_label'])[:28]}", fontsize=7, va="center")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[6], dpi=CONFIG["dpi"])
    plt.close()

    # 8. WEEE recycle/recovery targets as policy reference (not cluster performance).
    weee = {"5&6 Small": (55, 75), "2 Screens": (70, 80), "1&4 Large+Temp": (80, 85)}
    comp["weee"] = comp["eu_category"].map(weee_group)
    share = comp.groupby("weee")["n_samples"].sum()
    groups = list(weee)
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(groups))
    plt.bar(x - 0.2, [weee[g][0] for g in groups], width=0.4, label="recycle target %")
    plt.bar(x + 0.2, [weee[g][1] for g in groups], width=0.4, label="recovery target %")
    for i, group in enumerate(groups):
        plt.text(i, 88, f"n={int(share.get(group, 0))}", ha="center", fontsize=8)
    plt.xticks(x, groups, fontsize=8)
    plt.ylim(0, 100)
    plt.ylabel("% mass (EU WEEE Art.11)")
    plt.title("WEEE targets per group with sample counts (targets are reference, not scores)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[7], dpi=CONFIG["dpi"])
    plt.close()

    # 9. One-panel method summary for the paper.
    n_gap = int(merged["confidence"].str.contains("COVERAGE_GAP").sum())
    lines = [
        f"N=3961, K={best_k} (silhouette {metrics['silhouette']:.3f}, DBI {metrics['davies_bouldin']:.3f})",
        f"Sizes: max {verified['n_samples'].max()} (C01), min {verified['n_samples'].min()} (C16)",
        f"Auto grounding: cosine 0.085-0.154, 0/17 pass margin 0.04, {n_gap} GAP",
        f"Human verification: {n_corrections}/17 corrected (e.g. C04/C13->0104, C06->0408, C08->0303)",
        "Policy: Screens 4 clusters incl. CRT 0407; A1181 C09 GAP-BATT + C12 GAP-PCB isolated",
        "Read: top-9/far-9 grids in figures/ + docs/progress/2026-09-23_CLUSTER_GROUNDING_DINOV3.md",
    ]
    plt.figure(figsize=(10, 3.5))
    plt.axis("off")
    plt.title("KEC+DINOv3+DROWCULA (K=17) + SigLIP2 grounding: summary", fontsize=11)
    for i, line in enumerate(lines):
        plt.text(0.02, 0.82 - i * 0.13, "- " + line, fontsize=9, transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(FIG / OUTPUTS[8], dpi=CONFIG["dpi"])
    plt.close()

    log.info("saved %d figures to %s", len(OUTPUTS), FIG)


if __name__ == "__main__":
    main()
