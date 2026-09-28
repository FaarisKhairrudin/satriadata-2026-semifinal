"""Map fixed KEC+DROWCULA clusters to UNU-KEYs-v.2 with SigLIP2.

This is a post-clustering interpretation step only. It does not rerun K-Means,
DROWCULA, UMAP, K selection, or alter any existing labels.

Pipeline:

    fixed KEC+DINOv3+DROWCULA labels (K=17)
        -> SigLIP2 image embeddings for each image
        -> SigLIP2-space centroid for each existing cluster
        -> SigLIP2 text embeddings for 57 UNU-KEYs-v.2 categories (+ 3 GAP streams);
           prompt wording controlled by --prompt-variant (default: short)
        -> cosine ranking and confidence/margin

UNU-KEYs are used only to interpret already-created clusters. They are not
used during KEC, DINOv3, UMAP, K selection, or K-Means.

Run from repository root:
    python3.12 code/scripts/06_unukey_grounding_siglip2.py --overwrite

Required packages:
    python3.12 -m pip install --user torch transformers pillow tqdm

Default model:
    google/siglip2-base-patch16-256

Outputs:
    data/processed/electronic/kec_drowcula_dinov3/cluster_grounding.csv
    data/processed/electronic/kec_drowcula_dinov3/cluster_grounding.json
    data/processed/electronic/kec_drowcula_dinov3/siglip2_image_embeddings.npy
    data/processed/electronic/kec_drowcula_dinov3/siglip2_taxonomy_embeddings.npy
    data/processed/electronic/kec_drowcula_dinov3/siglip2_grounding_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
DEFAULT_RESULT = PROC / "kec_drowcula_dinov3"
DEFAULT_EMBEDDING_IDS = PROC / "embedding_ids.csv"
DEFAULT_MANIFEST = PROC / "manifest_train.csv"

CONFIG: dict[str, Any] = {
    "seed": 42,
    "model_id": "google/siglip2-base-patch16-256",
    "image_batch_size": 32,
    "text_batch_size": 64,
    "high_confidence_margin": 0.04,
    "moderate_confidence_margin": 0.015,
    "top_n": 5,
    "majority_threshold": 0.5,
    "plurality_threshold": 0.35,
    "image_feature_normalization": "row-wise L2 before centroid averaging",
    "taxonomy_feature_normalization": "row-wise L2 after template averaging",
    "label_rule": "per-image majority vote; multi-candidate below majority threshold",
}

TEMPLATES = (
    "a photo of {}, an electronic device or e-waste.",
    "a photo of {}.",
    "e-waste: {}.",
)

# Coverage-gap categories for waste streams that are e-waste under Basel/HS but
# are not finished products in UNU-KEYs-v.2 (batteries, PCBs, mixed scrap).
SUPPLEMENTARY_EWASTE = [
    ("BATT", "Portable batteries, dry cells, accumulator cells, AA AAA lithium-ion batteries"),
    ("PCB", "Printed circuit boards, microcircuits, motherboards and IC scrap"),
    ("SCRAP", "Dismantled mixed e-waste scrap, shredded residues and casing fragments"),
]

# UNU-KEYs-v.2 categories from the supplied E-waste Statistics Guidelines.
UNU_KEYS = [
    ("0001", "Central Heating", "Large equipment (excl. PV)"),
    ("0002", "Photovoltaic Panels", "Photovoltaic panels"),
    ("0101", "Professional Heating and Ventilation", "Large equipment (excl. PV)"),
    ("0102", "Dishwashers", "Large equipment (excl. PV)"),
    ("0103", "Kitchen equipment, large furnaces and ovens", "Large equipment (excl. PV)"),
    ("0104", "Washing Machines and combined dryers", "Large equipment (excl. PV)"),
    ("0105", "Dryers and centrifuges", "Large equipment (excl. PV)"),
    ("0106", "Household Heating and Ventilation", "Large equipment (excl. PV)"),
    ("0108", "Refrigerators and combi-refrigerators", "Temperature exchange equipment"),
    ("0109", "Freezers", "Temperature exchange equipment"),
    ("0111", "Air Conditioners, household and portable", "Temperature exchange equipment"),
    ("0112", "Other Cooling equipment, dehumidifiers", "Temperature exchange equipment"),
    ("0113", "Professional Cooling equipment", "Temperature exchange equipment"),
    ("0114", "Microwaves", "Small equipment"),
    ("0201", "Other small household equipment, irons and fans", "Small equipment"),
    ("0202", "Equipment for food preparation, toasters and grills", "Small equipment"),
    ("0203", "Small equipment for hot water, coffee machines and kettles", "Small equipment"),
    ("0204", "Vacuum Cleaners", "Small equipment"),
    ("0205", "Personal Care equipment, hair dryers and razors", "Small equipment"),
    ("0301", "Small IT equipment, routers, mice and keyboards", "Small IT"),
    ("0302", "Desktop PCs", "Small IT"),
    ("0303", "Laptops and tablets", "Screens and monitors"),
    ("0304", "Printers and scanners, copiers", "Small IT"),
    ("0305", "Telecommunications equipment excluding mobile phones", "Small IT"),
    ("0306", "Mobile Phones and smartphones", "Small IT"),
    ("0307", "Other IT equipment, data storage and servers", "Small IT"),
    ("0308", "Cathode Ray Tube Monitors", "Screens and monitors"),
    ("0309", "Flat-Panel Display Monitors, LCD and LED", "Screens and monitors"),
    ("0310", "Servers and data center units", "Small IT"),
    ("0401", "Small Consumer Electronics, headphones, cameras and remote controls", "Small equipment"),
    ("0402", "Portable Audio and Video, MP3 players and e-readers", "Small equipment"),
    ("0403", "Music Instruments, Radio and Hi-Fi equipment", "Small equipment"),
    ("0404", "Video equipment, DVD players and projectors", "Small equipment"),
    ("0405", "Speakers", "Small equipment"),
    ("0407", "Cathode Ray Tube Televisions", "Screens and monitors"),
    ("0408", "Flat-Panel Display TVs, LCD and LED", "Screens and monitors"),
    ("0409", "E-cigarettes and vaporising devices", "Small equipment"),
    ("0501", "Small lighting equipment excluding LED", "Small equipment"),
    ("0502", "Compact Fluorescent Lamps", "Lamps"),
    ("0503", "Straight Tube Fluorescent Lamps", "Lamps"),
    ("0504", "Special Lamps, professional mercury and sodium", "Lamps"),
    ("0505", "LED Lamps", "Lamps"),
    ("0506", "Household Luminaires", "Small equipment"),
    ("0507", "Professional Luminaires", "Small equipment"),
    ("0601", "Household Tools, drills and saws", "Small equipment"),
    ("0602", "Professional Tools, welding and milling", "Large equipment (excl. PV)"),
    ("0701", "Toys, racing sets and drones", "Small equipment"),
    ("0702", "Game Consoles", "Small IT"),
    ("0703", "Leisure equipment", "Large equipment (excl. PV)"),
    ("0704", "E-bikes", "Large equipment (excl. PV)"),
    ("0705", "Charging stations", "Large equipment (excl. PV)"),
    ("0801", "Household medical equipment", "Small equipment"),
    ("0802", "Professional Medical equipment", "Large equipment (excl. PV)"),
    ("0901", "Household Monitoring and Control, smoke detectors and thermostats", "Small equipment"),
    ("0902", "Professional Monitoring and Control", "Large equipment (excl. PV)"),
    ("1001", "Non-cooled Dispensers, vending machines", "Large equipment (excl. PV)"),
    ("1002", "Cooled Dispensers", "Temperature exchange equipment"),
]

# Canonical single-concept prompt names (adopted 2026-09-27 as the official
# grounding configuration): replacing official compound descriptions with plain
# object names raised per-cluster vote agreement with the human final labels
# from 7/17 to 15/17 on the same checkpoint and cached image embeddings.
# History: post-hoc diagnostic audit first (variant "short" vs "official");
# adoption made "short" the default because compound UNU descriptions caused
# systematic prompt-naming bias (dishwashers vs washing machines, speakers vs
# flat TVs). The legacy official-descriptions variant is kept for audit via
# --prompt-variant official. See
# docs/progress/2026-09-27_SIGLIP2_PROMPT_AUDIT.md.
SHORT_NAMES: dict[str, str] = {
    "0001": "central heating unit", "0002": "solar panel",
    "0101": "industrial heating unit", "0102": "dishwasher",
    "0103": "large kitchen oven", "0104": "washing machine",
    "0105": "clothes dryer", "0106": "household heater",
    "0108": "refrigerator", "0109": "freezer",
    "0111": "air conditioner", "0112": "dehumidifier",
    "0113": "industrial cooler", "0114": "microwave oven",
    "0201": "small household appliance", "0202": "toaster",
    "0203": "coffee machine", "0204": "vacuum cleaner",
    "0205": "hair dryer", "0301": "computer keyboard",
    "0302": "desktop computer", "0303": "laptop",
    "0304": "printer", "0305": "router", "0306": "smartphone",
    "0307": "hard drive", "0308": "crt computer monitor",
    "0309": "computer monitor", "0310": "server rack",
    "0401": "camera", "0402": "mp3 player", "0403": "radio",
    "0404": "dvd player", "0405": "loudspeaker",
    "0407": "crt television", "0408": "flat screen tv",
    "0409": "e-cigarette", "0501": "desk lamp",
    "0502": "compact fluorescent lamp", "0503": "fluorescent tube lamp",
    "0504": "industrial lamp", "0505": "led lamp",
    "0506": "ceiling light fixture", "0507": "industrial light fixture",
    "0601": "power drill", "0602": "welding machine",
    "0701": "electric toy", "0702": "game console",
    "0703": "electric leisure equipment", "0704": "electric bicycle",
    "0705": "ev charging station", "0801": "home medical device",
    "0802": "hospital medical equipment", "0901": "smoke detector",
    "0902": "industrial control panel", "1001": "vending machine",
    "1002": "refrigerated vending machine",
    "BATT": "battery", "PCB": "circuit board",
    "SCRAP": "pile of electronic scrap",
}


def variant_entries(entries: Sequence[tuple[str, ...]], variant: str) -> list[tuple[str, ...]]:
    """Apply SHORT_NAMES to (code, name, ...) tuples when variant == "short"."""
    if variant == "official":
        return list(entries)
    return [
        (entry[0], SHORT_NAMES.get(entry[0], entry[1])) + tuple(entry[2:])
        for entry in entries
    ]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("07_siglip2_grounding")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
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


def load_fixed_assignments(
    result_dir: Path, embedding_ids_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Path], str]:
    labels_path = result_dir / "labels.csv"
    try:
        labels = pd.read_csv(labels_path)
        ids = pd.read_csv(embedding_ids_path)
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not load grounding inputs: {exc}") from exc
    for name, frame in (("labels", labels), ("embedding_ids", ids), ("manifest", manifest)):
        if "filepath" not in frame.columns:
            raise SystemExit(f"{name} must contain filepath")
    if "cluster" not in labels.columns:
        raise SystemExit("labels.csv must contain cluster")
    if len(labels) != len(ids) or len(labels) != len(manifest):
        raise SystemExit("labels, embedding IDs, and manifest have different lengths")

    id_order = [canonical_path(str(value)) for value in ids["filepath"]]
    manifest_order = [canonical_path(str(value)) for value in manifest["filepath"]]
    label_order = [canonical_path(str(value)) for value in labels["filepath"]]
    if id_order != manifest_order or id_order != label_order:
        raise SystemExit("image ordering mismatch; refusing to map clusters")

    paths: list[Path] = []
    for raw in ids["filepath"]:
        path = Path(str(raw))
        if not path.exists():
            raise SystemExit(f"image path from embedding_ids.csv is missing: {raw}")
        paths.append(path)
    clusters = labels["cluster"].to_numpy(dtype=np.int32)
    unique = np.unique(clusters)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int32)):
        raise SystemExit(f"cluster IDs must be contiguous from 0: {unique.tolist()}")
    log.info("validated fixed labels: N=%d K=%d", len(labels), len(unique))
    return labels, paths, sha256_file(labels_path)


def l2_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    invalid = ~np.isfinite(norms) | (norms <= 0.0)
    try:
        bad_count = int(np.count_nonzero(invalid))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"could not count invalid feature rows: {exc}") from exc
    if bool(np.any(invalid)):
        raise SystemExit(f"invalid feature rows: {bad_count}")
    return (features / norms).astype(np.float32, copy=False)


def load_siglip2(model_id: str):
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModel,
            AutoProcessor,
        )
    except ImportError as exc:
        raise SystemExit("install torch and transformers before SigLIP2 grounding") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info("loading SigLIP2 model %s on %s", model_id, device)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device, torch


def load_cached_or_none(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32)
    except (OSError, ValueError) as exc:
        log.warning("cached file %s is unreadable: %s", path, str(exc)[:160])
        return None


def siglip2_output(output: Any, preferred: str, torch: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    for name in (preferred, "image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is not None:
            if name == "last_hidden_state":
                return value.mean(dim=1)
            return value
    raise RuntimeError("SigLIP2 output has no usable embedding")


def encode_images(model: Any, processor: Any, device: str, torch: Any,
                  paths: list[Path], batch_size: int) -> np.ndarray:
    from PIL import Image  # type: ignore[import-not-found]

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="SigLIP2 images"):
            images: list[Image.Image] = []
            try:
                images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
                inputs = processor(images=images, return_tensors="pt").to(device)
                encoded = model.get_image_features(**inputs)
                outputs.append(siglip2_output(encoded, "image_embeds", torch).float().cpu().numpy())
            except (OSError, RuntimeError, ValueError) as exc:
                raise SystemExit(f"SigLIP2 image encoding failed at row {start}: {exc}") from exc
            finally:
                for image in images:
                    image.close()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def encode_texts(model: Any, processor: Any, device: str, torch: Any,
                 texts: list[str], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="SigLIP2 taxonomy"):
            try:
                inputs = processor(
                    text=texts[start : start + batch_size],
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                encoded = model.get_text_features(**inputs)
                outputs.append(siglip2_output(encoded, "text_embeds", torch).float().cpu().numpy())
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(f"SigLIP2 text encoding failed at row {start}: {exc}") from exc
    return np.concatenate(outputs, axis=0).astype(np.float32)


def taxonomy_embeddings(model: Any, processor: Any, device: str, torch: Any,
                        batch_size: int, entries: Sequence[tuple[str, ...]]) -> np.ndarray:
    result: list[np.ndarray] = []
    for entry in entries:
        name = entry[1]
        prompts = [template.format(name) for template in TEMPLATES]
        encoded = l2_rows(encode_texts(model, processor, device, torch, prompts, batch_size))
        result.append(encoded.mean(axis=0))
    return l2_rows(np.stack(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=CONFIG["model_id"])
    parser.add_argument("--image-batch-size", type=int, default=CONFIG["image_batch_size"])
    parser.add_argument("--text-batch-size", type=int, default=CONFIG["text_batch_size"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=["official", "short"],
        default="short",
        help="short = canonical single-concept names (official pipeline); official = UNU-KEYs descriptions (legacy audit variant)",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT,
                        help="Folder label terkunci + output grounding (default: folder KEC final).")
    parser.add_argument("--embedding-ids", type=Path, default=DEFAULT_EMBEDDING_IDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.image_batch_size < 1 or args.text_batch_size < 1:
        parser.error("batch sizes must be positive")
    set_seed(CONFIG["seed"])
    RESULT: Path = args.result_dir
    EMBEDDING_IDS: Path = args.embedding_ids
    MANIFEST: Path = args.manifest
    RESULT.mkdir(parents=True, exist_ok=True)

    labels, image_paths, labels_hash = load_fixed_assignments(RESULT, EMBEDDING_IDS, MANIFEST)
    n = len(labels)
    try:
        k = int(np.unique(labels["cluster"].to_numpy(dtype=np.int32)).size)
    except (TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"could not count clusters: {exc}") from exc
    image_cache = RESULT / "siglip2_image_embeddings.npy"
    taxonomy_cache = RESULT / "siglip2_taxonomy_embeddings.npy"
    out_csv = RESULT / "cluster_grounding.csv"
    out_json = RESULT / "cluster_grounding.json"

    if not args.overwrite and image_cache.exists() and taxonomy_cache.exists() and out_csv.exists() and out_json.exists():
        log.info("grounding outputs exist; use --overwrite to regenerate")
        return

    log.info("prompt variant: %s", args.prompt_variant)
    model, processor, device, torch = load_siglip2(args.model_id)
    if not image_cache.exists() or args.overwrite:
        image_features = l2_rows(encode_images(model, processor, device, torch, image_paths, args.image_batch_size))
        np.save(image_cache, image_features)
    else:
        image_features = np.load(image_cache).astype(np.float32)
    if image_features.shape[0] != n:
        raise SystemExit(f"SigLIP2 image feature count mismatch: {image_features.shape} vs {n}")

    if not taxonomy_cache.exists() or args.overwrite:
        taxonomy_features = taxonomy_embeddings(
            model, processor, device, torch, args.text_batch_size,
            variant_entries(UNU_KEYS, args.prompt_variant),
        )
        np.save(taxonomy_cache, taxonomy_features)
    else:
        taxonomy_features = np.load(taxonomy_cache).astype(np.float32)
    if taxonomy_features.shape != (len(UNU_KEYS), image_features.shape[1]):
        raise SystemExit(f"SigLIP2 taxonomy feature shape mismatch: {taxonomy_features.shape}")
    supplementary_features = taxonomy_embeddings(
        model, processor, device, torch, args.text_batch_size,
        variant_entries(SUPPLEMENTARY_EWASTE, args.prompt_variant),
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Existing assignments are fixed. Only centroids and semantic rankings are computed.
    centroids: list[np.ndarray] = []
    cluster_sizes: list[int] = []
    for cluster in range(k):
        member_features = image_features[labels["cluster"].to_numpy() == cluster]
        if len(member_features) == 0:
            raise SystemExit(f"cluster {cluster} is empty")
        cluster_sizes.append(len(member_features))
        centroids.append(l2_rows(member_features.mean(axis=0, keepdims=True))[0])
    centroid_matrix = np.stack(centroids)
    similarities = centroid_matrix @ taxonomy_features.T
    supplementary_similarities = centroid_matrix @ supplementary_features.T

    # Per-image zero-shot classification over UNU-KEYs + gap categories, then
    # majority aggregation inside each fixed cluster. Assignments never change.
    all_taxonomy = np.concatenate([taxonomy_features, supplementary_features], axis=0)
    try:
        image_scores = image_features @ all_taxonomy.T
        image_votes = np.argmax(image_scores, axis=1)
    except (ValueError, FloatingPointError) as exc:
        raise SystemExit(f"per-image voting failed: {exc}") from exc
    category_count = all_taxonomy.shape[0]

    def category_label(index: int) -> tuple[str, str, str]:
        if index < len(UNU_KEYS):
            code, name, eu = UNU_KEYS[index]
            return code, name, eu
        gap_code, gap_name = SUPPLEMENTARY_EWASTE[index - len(UNU_KEYS)]
        return f"GAP-{gap_code}", gap_name, "Basel/HS non-UNU e-waste stream"

    def vote_confidence(share: float) -> str:
        try:
            majority = float(CONFIG["majority_threshold"])
            plurality = float(CONFIG["plurality_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid vote thresholds in CONFIG: {exc}") from exc
        if share >= majority:
            return "MAJORITY"
        if share >= plurality:
            return "PLURALITY"
        return "HETEROGENEOUS"

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for cluster in range(k):
        try:
            ranking = np.argsort(-similarities[cluster])
            top = int(ranking[0])
            second = int(ranking[1])
            margin = float(similarities[cluster, top] - similarities[cluster, second])
            raw_score = float(similarities[cluster, top])
        except (IndexError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid similarity ranking for cluster {cluster}: {exc}") from exc
        code, name, eu_category = UNU_KEYS[top]

        # Coverage-gap check: a supplementary category clearly beating the best
        # UNU-KEY indicates a battery/PCB/scrap stream, not a finished product.
        try:
            supplementary_ranking = np.argsort(-supplementary_similarities[cluster])
            gap_index = int(supplementary_ranking[0])
            gap_code, gap_name = SUPPLEMENTARY_EWASTE[gap_index]
            gap_score = float(supplementary_similarities[cluster, gap_index])
        except (IndexError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid supplementary ranking for cluster {cluster}: {exc}") from exc
        is_gap = gap_score > raw_score + 0.005

        if margin >= CONFIG["high_confidence_margin"]:
            confidence = "HIGH"
        elif margin >= CONFIG["moderate_confidence_margin"]:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        if is_gap:
            confidence = f"HIGH (COVERAGE_GAP:{gap_code})"
            code, name = f"GAP-{gap_code}", gap_name
            eu_category = "Basel/HS non-UNU e-waste stream"
            raw_score = gap_score
        rows.append({
            "cluster": cluster,
            "n_samples": cluster_sizes[cluster],
            "unu_key_code": code,
            "primary_category": name,
            "eu_category": eu_category,
            "cosine_score": raw_score,
            "margin_vs_second": margin,
            "confidence": confidence,
            "mapping_model": args.model_id,
        })
        try:
            top_matches = [
                {
                    "rank": rank + 1,
                    "code": UNU_KEYS[int(index)][0],
                    "name": UNU_KEYS[int(index)][1],
                    "eu_category": UNU_KEYS[int(index)][2],
                    "cosine": float(similarities[cluster, int(index)]),
                }
                for rank, index in enumerate(ranking[: CONFIG["top_n"]])
            ]
        except (IndexError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid top matches for cluster {cluster}: {exc}") from exc
        details[f"cluster_{cluster:02d}"] = {
            "cluster_id": cluster,
            "n_samples": cluster_sizes[cluster],
            "assigned_unu_key": {"code": code, "name": name, "eu_category": eu_category},
            "confidence": confidence,
            "coverage_gap": {
                "is_gap": is_gap,
                "code": gap_code,
                "name": gap_name,
                "cosine": gap_score,
            },
            "top_matches": top_matches,
        }

        # ---- Per-image majority vote (primary labeling evidence) ----
        member_rows = np.flatnonzero(labels["cluster"].to_numpy() == cluster)
        member_votes = image_votes[member_rows]
        counts = np.bincount(member_votes, minlength=category_count)
        vote_order = np.argsort(-counts)
        vote_rows: list[dict[str, Any]] = []
        try:
            for rank, index in enumerate(vote_order[: CONFIG["top_n"]]):
                idx = int(index)
                v_code, v_name, v_eu = category_label(idx)
                share = float(counts[idx]) / max(1, len(member_votes))
                vote_rows.append(
                    {
                        "rank": rank + 1,
                        "code": v_code,
                        "name": v_name,
                        "eu_category": v_eu,
                        "votes": int(counts[idx]),
                        "share": share,
                    }
                )
            winner_code, winner_name, winner_eu = category_label(int(vote_order[0]))
            winner_share = float(counts[int(vote_order[0])]) / max(1, len(member_votes))
        except (IndexError, TypeError, ValueError, FloatingPointError, KeyError) as exc:
            raise SystemExit(f"invalid vote aggregation for cluster {cluster}: {exc}") from exc

        try:
            gap_vote_count = sum(
                int(counts[len(UNU_KEYS) + j]) for j in range(len(SUPPLEMENTARY_EWASTE))
            )
            gap_vote_share = float(gap_vote_count) / max(1, len(member_votes))
        except (IndexError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid gap vote tally for cluster {cluster}: {exc}") from exc

        if winner_code.startswith("GAP-"):
            final_code, final_name = winner_code, winner_name
            final_eu = winner_eu
            vote_conf = f"{vote_confidence(winner_share)} (COVERAGE_GAP)"
        else:
            try:
                majority_threshold = float(CONFIG["majority_threshold"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"invalid majority threshold: {exc}") from exc
            if winner_share >= majority_threshold:
                final_code, final_name, final_eu = winner_code, winner_name, winner_eu
                vote_conf = "MAJORITY"
            else:
                # Heterogeneous cluster: report multi-candidate instead of forcing one.
                candidates: list[str] = []
                for index in vote_order:
                    try:
                        candidate_code = category_label(int(index))[0]
                    except (IndexError, TypeError, ValueError, KeyError) as exc:
                        raise SystemExit(
                            f"invalid vote candidate for cluster {cluster}: {exc}"
                        ) from exc
                    if not candidate_code.startswith("GAP-"):
                        candidates.append(candidate_code)
                    if len(candidates) >= 3:
                        break
                final_code = "/".join(candidates) + "?"
                final_name = "mixed cluster (multi-candidate)"
                final_eu = "-"
                vote_conf = vote_confidence(winner_share)

        details[f"cluster_{cluster:02d}"]["majority_vote"] = {
            "winner": {"code": winner_code, "name": winner_name, "share": winner_share},
            "vote_confidence": vote_conf,
            "gap_vote_share": gap_vote_share,
            "final_label": {"code": final_code, "name": final_name, "eu_category": final_eu},
            "distribution": vote_rows,
        }
        rows[-1].update(
            {
                "vote_code": final_code,
                "vote_name": final_name,
                "vote_winner_share": winner_share,
                "vote_confidence": vote_conf,
                "gap_vote_share": gap_vote_share,
                "top3_votes": "; ".join(
                    f"{r['code']}:{r['share']:.0%}" for r in vote_rows[:3]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    save_json(details, out_json)
    save_json(
        {
            **CONFIG,
            "prompt_variant": args.prompt_variant,
            "prompt_templates": list(TEMPLATES),
            "model_id": args.model_id,
            "image_batch_size": args.image_batch_size,
            "text_batch_size": args.text_batch_size,
            "labels_file": str(RESULT / "labels.csv"),
            "labels_sha256": labels_hash,
            "embedding_ids_file": str(EMBEDDING_IDS),
            "embedding_ids_sha256": sha256_file(EMBEDDING_IDS),
            "n_samples": n,
            "n_clusters": k,
            "n_taxonomy_categories": len(UNU_KEYS),
            "n_supplementary_categories": len(SUPPLEMENTARY_EWASTE),
            "gap_rule": "supplementary category beats best UNU-KEY by more than 0.005 cosine",
            "label_rule": CONFIG["label_rule"],
            "majority_threshold": CONFIG["majority_threshold"],
            "plurality_threshold": CONFIG["plurality_threshold"],
            "post_clustering_interpretation_only": True,
            "clustering_rerun": False,
            "k_selection_changed": False,
            "labels_changed": False,
            "unu_keys_used_during_clustering": False,
            "siglip2_image_embeddings": str(image_cache),
            "siglip2_taxonomy_embeddings": str(taxonomy_cache),
        },
        RESULT / "siglip2_grounding_config.json",
    )
    log.info("saved %d fixed-cluster UNU-KEY mappings to %s", k, out_csv)
    for row in rows:
        log.info(
            "cluster=%02d n=%d | centroid: %s (%.3f) | vote: %s share=%.0f%% [%s]",
            row["cluster"], row["n_samples"], row["unu_key_code"],
            row["cosine_score"], row["vote_code"], 100 * row["vote_winner_share"],
            row["vote_confidence"],
        )


if __name__ == "__main__":
    main()
