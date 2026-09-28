"""KEC + DROWCULA adapted to existing DINOv3 ViT-B/16 embeddings.

Final pipeline:

    e-waste image
      ├─ SigLIP2 -> KEC knowledge grounding -> kappa (768-D)
      └─ existing DINOv3 ViT-B/16 visual embedding (768-D)
            -> independently row-wise L2-normalize both blocks
            -> concatenate [DINOv3 || kappa] (1536-D)
            -> UMAP
            -> exhaustive unknown-K search (K=2..25)
            -> K-Means final clusters

This is a novel integration. The KEC paper does not propose DROWCULA, and the
DROWCULA paper does not propose KEC, SigLIP2, or this fused representation. The
DROWCULA paper uses DINOv2; this script adapts its procedure to DINOv3 and KEC.
The e-waste statistics reference is used only for documentation/context. UNU-
KEYs are never loaded or used in feature construction, K selection, or labels.

The script regenerates KEC with the official KEC construction recipe and GPT-4o
when its cached KEC artifacts are missing or --overwrite is supplied. The
OPENAI_API_KEY environment variable is required for regeneration. No key is
written to any output.

Run from repository root:
    python3.12 code/scripts/03_kec_drowcula_dinov3.py

Optional dependencies:
    python3.12 -m pip install --user torch transformers openai nltk umap-learn
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
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
from tqdm import tqdm  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
DEFAULT_DINO = PROC / "embeddings_dinov3.npy"
DEFAULT_EMBEDDING_IDS = PROC / "embedding_ids.csv"
DEFAULT_MANIFEST = PROC / "manifest_train.csv"
DEFAULT_OUTPUT = PROC / "kec_drowcula_dinov3"

CONFIG: dict[str, Any] = {
    "seed": 42,
    "siglip2_model": "google/siglip2-base-patch16-256",
    "gpt_model": "gpt-4o",
    "gpt_temperature": 0.1,
    "gpt_max_concurrency": 20,
    "image_batch_size": 8,
    "text_batch_size": 128,
    "kec_overcluster_divisor": 300,
    "kec_top_nouns": 5,
    "kec_alpha": 0.8,
    "kec_beta": 0.8,
    "kec_lambda1": 2,
    "kec_lambda2": 1,
    "kec_kmeans_n_init": 20,
    "kec_kmeans_max_iter": 300,
    "drowcula_k_min": 2,
    "drowcula_k_max": 25,
    "umap_n_components": 3,
    "umap_n_neighbors": 10,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",
    "drowcula_kmeans_n_init": 200,
    "drowcula_kmeans_max_iter": 10000,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kec_drowcula_dinov3")

P_CONCEPT = (
    "Combine the following nouns into one representative concept of 2-4 words "
    "and give one short description focused only on visible appearance. The "
    "concept must encompass the nouns without adding unrelated categories. "
    'Return JSON only with keys "concept" and "description". Nouns: {nouns}'
)
P_UNI = (
    "List exactly {k} representative and distinctive visual attributes for the "
    "concept below. Describe visible appearance only, not object function. "
    'Return JSON only with key "attributes" containing an array of strings. '
    "Concept: {concept}. Description: {description}"
)
P_BI = (
    "Give exactly {k} visual attribute(s) that distinguish these two concepts. "
    "Describe visible differences only, not function. "
    'Return JSON only with key "attributes" containing an array of strings. '
    "Concept A: {a}. Concept B: {b}."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not write {path}: {exc}") from exc


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from exc


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
    """Make relative and old absolute paths comparable without reading labels."""
    value = path.replace("\\", "/")
    marker = "/electronic/"
    if marker in value:
        return "electronic/" + value.split(marker, 1)[1]
    return value


def load_manifests(
    embedding_ids_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Path]]:
    """Validate embedding order and resolve image paths from embedding_ids.csv."""
    try:
        ids = pd.read_csv(embedding_ids_path)
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"could not read manifests: {exc}") from exc
    for name, frame in (("embedding_ids", ids), ("manifest", manifest)):
        if "filepath" not in frame.columns:
            raise SystemExit(f"{name} must contain filepath")
    if len(ids) != len(manifest):
        raise SystemExit(f"manifest row mismatch: {len(ids)} vs {len(manifest)}")
    id_keys = [canonical_path(str(value)) for value in ids["filepath"]]
    manifest_keys = [canonical_path(str(value)) for value in manifest["filepath"]]
    if id_keys != manifest_keys:
        raise SystemExit("embedding_ids.csv and manifest_train.csv row order differs")

    image_paths: list[Path] = []
    for raw in ids["filepath"]:
        candidate = Path(str(raw))
        if not candidate.exists():
            candidate = ROOT / str(raw)
        if not candidate.exists():
            raise SystemExit(f"image referenced by embedding_ids.csv is missing: {raw}")
        image_paths.append(candidate)
    log.info("validated %d image paths and manifest rows", len(image_paths))
    return manifest, image_paths


def l2_rows(features: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        invalid = ~np.isfinite(norms) | (norms <= 0.0)
        if bool(np.any(invalid)):
            raise ValueError(f"{int(np.count_nonzero(invalid))} invalid feature rows")
        return (values / norms).astype(np.float32, copy=False)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"L2 normalization failed: {exc}") from exc


def load_dino(path: Path, n: int) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise SystemExit(f"DINOv3 embedding file is missing: {path}")
    try:
        features = np.load(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not load DINOv3 embeddings: {exc}") from exc
    if features.shape != (n, 768):
        raise SystemExit(f"expected DINOv3 shape ({n}, 768), got {features.shape}")
    if not np.isfinite(features).all():
        raise SystemExit("DINOv3 embeddings contain non-finite values")
    return np.asarray(features, dtype=np.float32), sha256_file(path)


def load_siglip_model(model_id: str):
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModel,
            AutoProcessor,
        )
    except ImportError as exc:
        raise SystemExit("install torch and transformers before regenerating KEC") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info("loading %s on %s", model_id, device)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device, torch


def output_tensor(output: Any, preferred: str, torch: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    for name in (preferred, "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is not None:
            if name == "last_hidden_state":
                return value.mean(dim=1)
            return value
    raise RuntimeError(f"model output has no usable {preferred} feature")


def extract_image_features(model: Any, processor: Any, device: str, torch: Any,
                           image_paths: list[Path], batch_size: int) -> np.ndarray:
    from PIL import Image  # type: ignore[import-not-found]

    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="SigLIP2 images"):
            paths = image_paths[start : start + batch_size]
            try:
                images = [Image.open(path).convert("RGB") for path in paths]
                inputs = processor(images=images, return_tensors="pt").to(device)
                output = model.get_image_features(**inputs)
                value = output_tensor(output, "image_embeds", torch)
                features.append(value.float().cpu().numpy())
            except (OSError, RuntimeError, ValueError) as exc:
                raise SystemExit(f"SigLIP2 image encoding failed at row {start}: {exc}") from exc
            finally:
                for image in locals().get("images", []):
                    image.close()
    return np.concatenate(features, axis=0).astype(np.float32)


def extract_text_features(model: Any, processor: Any, device: str, torch: Any,
                          texts: list[str], batch_size: int) -> np.ndarray:
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="SigLIP2 text"):
            chunk = texts[start : start + batch_size]
            try:
                inputs = processor(
                    text=chunk,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                output = model.get_text_features(**inputs)
                value = output_tensor(output, "text_embeds", torch)
                features.append(value.float().cpu().numpy())
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(f"SigLIP2 text encoding failed at row {start}: {exc}") from exc
    return np.concatenate(features, axis=0).astype(np.float32)


def clean_noun(value: str) -> str | None:
    noun = value.lower().replace("_", " ").strip()
    if len(noun) < 3 or len(noun.split()) > 3:
        return None
    if not all(char.isalpha() or char in " -" for char in noun):
        return None
    return noun


def build_wordnet_nouns() -> list[str]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-not-found]
        nouns = {
            cleaned
            for synset in wn.all_synsets(pos="n")
            for lemma in synset.lemma_names()
            if (cleaned := clean_noun(lemma)) is not None
        }
    except LookupError as exc:
        raise SystemExit("download the NLTK wordnet corpus before running KEC") from exc
    result = sorted(nouns)
    if not result:
        raise SystemExit("WordNet noun list is empty")
    log.info("WordNet nouns: %d", len(result))
    return result


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            neighbours = np.flatnonzero(adjacency[current])
            for value in neighbours:
                try:
                    neighbour = int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise SystemExit(f"invalid adjacency index: {value}") from exc
                if neighbour not in seen:
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        value = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"GPT response was not JSON: {text[:300]}") from exc
    if not isinstance(value, dict):
        raise ValueError("GPT response JSON was not an object")
    return value


async def ask_gpt(client: Any, prompt: str, semaphore: asyncio.Semaphore, model: str,
                  temperature: float) -> dict[str, Any]:
    async with semaphore:
        last_error = ""
        for attempt in range(4):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return parse_json_response(content)
            except Exception as exc:  # API failures need bounded retries.
                last_error = str(exc)[:300]
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"GPT request failed after retries: {last_error}")


async def generate_knowledge(concepts_input: list[dict[str, Any]], args: argparse.Namespace,
                            output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("install openai before regenerating KEC") from exc
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        env_path = ROOT / ".env"
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "OPENAI_API_KEY":
                    api_key = value.strip().strip("\\\"'")
                    break
        except OSError:
            pass
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for GPT-4o KEC regeneration")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.gpt_max_concurrency)
    concept_prompts = [
        P_CONCEPT.format(nouns=", ".join(item["nouns"])) for item in concepts_input
    ]
    concept_results = await asyncio.gather(*[
        ask_gpt(client, prompt, semaphore, args.gpt_model, args.gpt_temperature)
        for prompt in concept_prompts
    ])
    concepts: list[dict[str, Any]] = []
    for item, result in zip(concepts_input, concept_results, strict=True):
        concept = str(result.get("concept", "")).strip()
        description = str(result.get("description", "")).strip()
        if not concept or not description:
            raise SystemExit(f"GPT returned incomplete concept for members {item['members']}")
        concepts.append({**item, "concept": concept, "description": description, "model": args.gpt_model})
    save_json(
        [{key: value for key, value in item.items() if key != "text_proxy"} for item in concepts],
        output_dir / "concepts.json",
    )

    uni_prompts = [
        P_UNI.format(k=args.kec_lambda1, concept=item["concept"], description=item["description"])
        for item in concepts
    ]
    uni_results = await asyncio.gather(*[
        ask_gpt(client, prompt, semaphore, args.gpt_model, args.gpt_temperature)
        for prompt in uni_prompts
    ])
    attributes: dict[str, Any] = {}
    for index, result in enumerate(uni_results):
        values = [str(value).strip() for value in result.get("attributes", []) if str(value).strip()]
        if len(values) < args.kec_lambda1:
            raise SystemExit(f"GPT returned too few uni-attributes for concept q{index}")
        attributes[f"q{index}"] = {"uni": values[: args.kec_lambda1], "bi": []}
    save_json(attributes, output_dir / "attributes.json")

    concept_vectors = np.asarray(
        [item["text_proxy"] for item in concepts], dtype=np.float32
    )
    concept_vectors = l2_rows(concept_vectors)
    similarity = concept_vectors @ concept_vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours: dict[str, list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index in range(len(concepts)):
        logits = similarity[index]
        finite = np.isfinite(logits)
        probabilities = np.zeros(len(concepts), dtype=np.float32)
        if bool(np.any(finite)):
            shifted = logits[finite] - np.max(logits[finite])
            finite_indices = np.flatnonzero(finite)
            probabilities[finite_indices] = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(-probabilities)
        cumulative = 0.0
        selected: list[int] = []
        for candidate in order:
            try:
                neighbour = int(candidate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SystemExit(f"invalid neighbour index: {candidate}") from exc
            if neighbour == index or not finite[neighbour]:
                continue
            selected.append(neighbour)
            try:
                probability = float(probabilities[neighbour])
            except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
                raise SystemExit(f"invalid neighbour probability at {neighbour}") from exc
            cumulative += probability
            pair: tuple[int, int] = (min(index, neighbour), max(index, neighbour))
            pairs.add(pair)
            if cumulative >= args.kec_beta:
                break
        neighbours[str(index)] = selected
    save_json(neighbours, output_dir / "neighbour_pairs.json")

    pair_list = sorted(pairs)
    bi_prompts = [
        P_BI.format(
            k=args.kec_lambda2,
            a=concepts[left]["concept"],
            b=concepts[right]["concept"],
        )
        for left, right in pair_list
    ]
    bi_results = await asyncio.gather(*[
        ask_gpt(client, prompt, semaphore, args.gpt_model, args.gpt_temperature)
        for prompt in bi_prompts
    ])
    for (left, right), result in zip(pair_list, bi_results, strict=True):
        values = [str(value).strip() for value in result.get("attributes", []) if str(value).strip()]
        if len(values) < args.kec_lambda2:
            raise SystemExit(f"GPT returned too few bi-attributes for pair {left},{right}")
        for current, other in ((left, right), (right, left)):
            attributes[f"q{current}"]["bi"].extend(
                f"vs {concepts[other]['concept']}: {value}" for value in values[: args.kec_lambda2]
            )
    save_json(attributes, output_dir / "attributes.json")
    await client.close()
    return concepts, attributes, neighbours


def ensure_kec(args: argparse.Namespace, output_dir: Path, image_paths: list[Path]) -> tuple[np.ndarray, str]:
    """Regenerate or load KEC artifacts, returning kappa and its fingerprint."""
    kappa_path = output_dir / "kappa.npy"
    required = [
        kappa_path,
        output_dir / "concepts.json",
        output_dir / "attributes.json",
        output_dir / "neighbour_pairs.json",
        output_dir / "over_clusters.npz",
        output_dir / "nouns_per_cluster.json",
        output_dir / "noun_list.json",
        output_dir / "noun_embeddings.npy",
        output_dir / "kec_image_embeddings.npy",
    ]
    if not args.overwrite and all(path.exists() for path in required):
        try:
            kappa = np.load(kappa_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"could not load cached KEC kappa: {exc}") from exc
        if kappa.shape != (len(image_paths), 768):
            raise SystemExit(f"cached kappa has wrong shape: {kappa.shape}")
        log.info("loaded cached KEC kappa: %s", kappa.shape)
        return np.asarray(kappa, dtype=np.float32), sha256_file(kappa_path)

    model, processor, device, torch = load_siglip_model(args.siglip2_model)
    image_features = extract_image_features(
        model, processor, device, torch, image_paths, args.image_batch_size
    )
    if image_features.shape != (len(image_paths), 768):
        raise SystemExit(
            f"SigLIP2 base is expected to produce 768-D image features, got {image_features.shape}"
        )
    image_features = l2_rows(image_features)
    np.save(output_dir / "kec_image_embeddings.npy", image_features)

    nouns = build_wordnet_nouns()
    noun_text = [f"a photo of {noun}." for noun in nouns]
    noun_features = l2_rows(
        extract_text_features(model, processor, device, torch, noun_text, args.text_batch_size)
    )
    if noun_features.shape[1] != 768:
        raise SystemExit(f"SigLIP2 base is expected to produce 768-D text features, got {noun_features.shape}")
    np.save(output_dir / "noun_embeddings.npy", noun_features.astype(np.float16))
    save_json(nouns, output_dir / "noun_list.json")

    n = len(image_features)
    k_over = max(2, round(n / args.kec_overcluster_divisor))
    initial = KMeans(
        n_clusters=k_over,
        n_init=args.kec_kmeans_n_init,
        max_iter=args.kec_kmeans_max_iter,
        random_state=args.seed,
        algorithm="lloyd",
    ).fit(image_features)
    centroids = l2_rows(initial.cluster_centers_)
    np.savez(
        output_dir / "over_clusters.npz",
        labels_over=initial.labels_.astype(np.int32),
        centroids=centroids,
        k_over=k_over,
    )
    similarities = centroids @ noun_features.T
    nouns_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for cluster in range(k_over):
        indices = np.argsort(-similarities[cluster])[: args.kec_top_nouns]
        entries: list[dict[str, Any]] = []
        for raw_index in indices:
            try:
                index = int(raw_index)
                noun = nouns[index]
                score = float(similarities[cluster, index])
            except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
                raise SystemExit(f"invalid noun ranking at cluster {cluster}") from exc
            entries.append({"noun": noun, "score": score})
        nouns_by_cluster[str(cluster)] = entries
    save_json(nouns_by_cluster, output_dir / "nouns_per_cluster.json")

    noun_index = {noun: index for index, noun in enumerate(nouns)}
    textual_centroids = np.stack([
        noun_features[[noun_index[item["noun"]] for item in nouns_by_cluster[str(cluster)]]].mean(axis=0)
        for cluster in range(k_over)
    ])
    textual_centroids = l2_rows(textual_centroids)
    multimodal_similarity = args.kec_alpha * (centroids @ centroids.T) + (1 - args.kec_alpha) * (textual_centroids @ textual_centroids.T)
    components = connected_components(multimodal_similarity > args.kec_beta)
    concepts_input = []
    for members in components:
        member_nouns = sorted({
            item["noun"]
            for member in members
            for item in nouns_by_cluster[str(member)]
        })
        concepts_input.append({"members": members, "nouns": member_nouns})
    log.info("KEC over-clusters=%d, merged concepts=%d", k_over, len(concepts_input))

    # GPT requests use concept proxy vectors only to choose KEC bi-concept neighbours.
    # The actual concept/attribute features are encoded below with SigLIP2.
    for item in concepts_input:
        item["text_proxy"] = textual_centroids[item["members"]].mean(axis=0)
    concepts, attributes, neighbours = asyncio.run(
        generate_knowledge(concepts_input, args, output_dir)
    )

    concept_texts = [item["concept"] for item in concepts]
    description_texts = [item["description"] for item in concepts]
    concept_vectors = l2_rows(extract_text_features(
        model, processor, device, torch, concept_texts, args.text_batch_size
    ))
    description_vectors = l2_rows(extract_text_features(
        model, processor, device, torch, description_texts, args.text_batch_size
    ))
    zeta = l2_rows(concept_vectors + description_vectors)

    attribute_lists: list[list[str]] = []
    for index, item in enumerate(concepts):
        values = list(attributes[f"q{index}"].get("uni", [])) + list(attributes[f"q{index}"].get("bi", []))
        attribute_lists.append(values or [item["concept"]])
    flat_attributes = [f"a photo showing {value}." for values in attribute_lists for value in values]
    attribute_vectors = l2_rows(extract_text_features(
        model, processor, device, torch, flat_attributes, args.text_batch_size
    ))

    omega_logits = image_features @ zeta.T
    omega_logits -= omega_logits.max(axis=1, keepdims=True)
    omega = np.exp(omega_logits)
    omega /= omega.sum(axis=1, keepdims=True)
    kappa = np.zeros_like(image_features, dtype=np.float32)
    offset = 0
    for index, values in enumerate(attribute_lists):
        current = attribute_vectors[offset : offset + len(values)]
        offset += len(values)
        instantiated = image_features[:, None, :] * current[None, :, :]
        attribute_mean = instantiated.mean(axis=1)
        kappa += omega[:, index : index + 1] * (zeta[index][None, :] + attribute_mean)
    np.save(kappa_path, kappa.astype(np.float32))
    save_json(
        {
            "siglip2_model": args.siglip2_model,
            "image_embedding_shape": list(image_features.shape),
            "noun_embedding_shape": list(noun_features.shape),
            "concept_count": len(concepts),
            "k_over": k_over,
            "alpha": args.kec_alpha,
            "beta": args.kec_beta,
            "lambda1": args.kec_lambda1,
            "lambda2": args.kec_lambda2,
            "gpt_model": args.gpt_model,
        },
        output_dir / "kec_config.json",
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return kappa.astype(np.float32), sha256_file(kappa_path)


def fit_umap(fused: np.ndarray, args: argparse.Namespace, output_dir: Path,
             fingerprint: str) -> np.ndarray:
    reduced_path = output_dir / "reduced_fused_embeddings.npy"
    config_path = output_dir / "reduction_config.json"
    expected = {
        "input_fingerprint_sha256": fingerprint,
        "input_shape": list(fused.shape),
        "normalization": "independent row-wise L2 on DINOv3 and kappa blocks",
        "n_components": args.umap_n_components,
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": args.umap_metric,
        "random_state": args.seed,
    }
    try:
        cached_config = load_json(config_path) if config_path.exists() else None
        if not args.overwrite and cached_config == expected and reduced_path.exists():
            reduced = np.load(reduced_path)
            if reduced.shape == (len(fused), args.umap_n_components) and bool(np.isfinite(reduced).all()):
                log.info("loaded cached UMAP fused representation: %s", reduced.shape)
                return np.asarray(reduced, dtype=np.float32)
    except (OSError, ValueError, json.JSONDecodeError):
        log.warning("invalid UMAP cache; recomputing")
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("install umap-learn before running DROWCULA") from exc
    reducer = umap.UMAP(
        n_components=args.umap_n_components,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        transform_seed=args.seed,
    )
    reduced = np.asarray(reducer.fit_transform(fused), dtype=np.float32)
    if not np.isfinite(reduced).all():
        raise SystemExit("UMAP produced non-finite values")
    np.save(reduced_path, reduced)
    save_json(expected, config_path)
    return reduced


def run_drowcula(reduced: np.ndarray, args: argparse.Namespace, output_dir: Path,
                 manifest: pd.DataFrame) -> tuple[int, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_k: int | None = None
    best_labels: np.ndarray | None = None
    best_silhouette = -np.inf
    for k in tqdm(range(args.k_min, args.k_max + 1), desc="DROWCULA K search"):
        model = KMeans(
            n_clusters=k,
            n_init=args.drowcula_kmeans_n_init,
            max_iter=args.drowcula_kmeans_max_iter,
            random_state=args.seed,
            algorithm="lloyd",
        )
        labels = model.fit_predict(reduced)
        try:
            silhouette = float(silhouette_score(reduced, labels))
            dbi = float(davies_bouldin_score(reduced, labels))
            chi = float(calinski_harabasz_score(reduced, labels))
        except (ValueError, FloatingPointError) as exc:
            raise SystemExit(f"DROWCULA metric calculation failed for K={k}: {exc}") from exc
        sizes = np.bincount(labels, minlength=k)
        try:
            row = {
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": dbi,
                "calinski_harabasz": chi,
                "inertia": float(model.inertia_),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            }
        except (TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"could not record DROWCULA metrics for K={k}: {exc}") from exc
        rows.append(row)
        if silhouette > best_silhouette:
            best_k, best_silhouette, best_labels = k, silhouette, labels.copy()
    if best_k is None or best_labels is None:
        raise SystemExit("DROWCULA K search returned no result")
    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "k_search_scores.csv", index=False)
    np.save(output_dir / "labels.npy", best_labels.astype(np.int32))
    assignments = manifest.copy()
    assignments["cluster"] = best_labels.astype(np.int32)
    assignments.to_csv(output_dir / "labels.csv", index=False)
    sizes = np.bincount(best_labels, minlength=best_k)
    pd.DataFrame({"cluster": np.arange(best_k), "size": sizes}).to_csv(
        output_dir / "cluster_sizes.csv", index=False
    )
    try:
        selected_rows = scores.loc[scores["k"] == best_k]
        if len(selected_rows) != 1:
            raise ValueError(f"expected one score row for K={best_k}")
        selected = selected_rows.iloc[0]
        final_metrics = {
            "selected_k": best_k,
            "selection_metric": "silhouette",
            "selection_space": "UMAP-reduced fused DINOv3+KEC features",
            "silhouette": float(selected["silhouette"]),
            "davies_bouldin": float(selected["davies_bouldin"]),
            "calinski_harabasz": float(selected["calinski_harabasz"]),
            "inertia": float(selected["inertia"]),
            "n_samples": len(best_labels),
            "reduced_dimensions": reduced.shape[1],
        }
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
        raise SystemExit(f"could not assemble final metrics: {exc}") from exc
    save_json(final_metrics, output_dir / "final_metrics.json")
    (output_dir / "selected_k.txt").write_text(f"{best_k}\n", encoding="utf-8")
    log.info("DROWCULA selected K=%d, sizes=%s", best_k, sizes.tolist())
    return best_k, best_labels, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--embedding-ids", type=Path, default=DEFAULT_EMBEDDING_IDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siglip2-model", default=CONFIG["siglip2_model"])
    parser.add_argument("--gpt-model", default=CONFIG["gpt_model"])
    parser.add_argument("--gpt-temperature", type=float, default=CONFIG["gpt_temperature"])
    parser.add_argument("--gpt-max-concurrency", type=int, default=CONFIG["gpt_max_concurrency"])
    parser.add_argument("--image-batch-size", type=int, default=CONFIG["image_batch_size"])
    parser.add_argument("--text-batch-size", type=int, default=CONFIG["text_batch_size"])
    parser.add_argument("--kec-overcluster-divisor", type=int, default=CONFIG["kec_overcluster_divisor"])
    parser.add_argument("--kec-top-nouns", type=int, default=CONFIG["kec_top_nouns"])
    parser.add_argument("--kec-alpha", type=float, default=CONFIG["kec_alpha"])
    parser.add_argument("--kec-beta", type=float, default=CONFIG["kec_beta"])
    parser.add_argument("--kec-lambda1", type=int, default=CONFIG["kec_lambda1"])
    parser.add_argument("--kec-lambda2", type=int, default=CONFIG["kec_lambda2"])
    parser.add_argument("--kec-kmeans-n-init", type=int, default=CONFIG["kec_kmeans_n_init"])
    parser.add_argument("--kec-kmeans-max-iter", type=int, default=CONFIG["kec_kmeans_max_iter"])
    parser.add_argument("--umap-n-components", type=int, default=CONFIG["umap_n_components"])
    parser.add_argument("--umap-n-neighbors", type=int, default=CONFIG["umap_n_neighbors"])
    parser.add_argument("--umap-min-dist", type=float, default=CONFIG["umap_min_dist"])
    parser.add_argument("--umap-metric", default=CONFIG["umap_metric"])
    parser.add_argument("--drowcula-kmeans-n-init", type=int, default=CONFIG["drowcula_kmeans_n_init"])
    parser.add_argument("--drowcula-kmeans-max-iter", type=int, default=CONFIG["drowcula_kmeans_max_iter"])
    parser.add_argument("--k-min", type=int, default=CONFIG["drowcula_k_min"])
    parser.add_argument("--k-max", type=int, default=CONFIG["drowcula_k_max"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        parser.error("require 2 <= k-min <= k-max")
    if args.gpt_max_concurrency < 1 or args.image_batch_size < 1 or args.text_batch_size < 1:
        parser.error("batch sizes and GPT concurrency must be positive")
    if args.kec_overcluster_divisor < 1 or args.kec_top_nouns < 1:
        parser.error("KEC divisor and top-nouns must be positive")
    if args.kec_lambda1 < 1 or args.kec_lambda2 < 1:
        parser.error("KEC attribute counts must be positive")
    if args.umap_n_components < 2 or args.umap_n_neighbors < 2:
        parser.error("UMAP components and neighbors must be at least 2")
    if args.drowcula_kmeans_n_init < 1 or args.drowcula_kmeans_max_iter < 1:
        parser.error("DROWCULA K-Means settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    for path_name in ("dino", "embedding_ids", "manifest", "output_dir"):
        setattr(args, path_name, getattr(args, path_name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, image_paths = load_manifests(args.embedding_ids, args.manifest)
    dino, dino_fingerprint = load_dino(args.dino, len(manifest))
    kappa, kappa_fingerprint = ensure_kec(args, args.output_dir, image_paths)
    if kappa.shape != dino.shape:
        raise SystemExit(f"DINOv3/KEC shape mismatch: {dino.shape} vs {kappa.shape}")

    dino_norm = l2_rows(dino)
    kappa_norm = l2_rows(kappa)
    fused = np.concatenate([dino_norm, kappa_norm], axis=1).astype(np.float32)
    np.save(args.output_dir / "fused_features.npy", fused)
    np.save(args.output_dir / "dino_normalized.npy", dino_norm)
    np.save(args.output_dir / "kappa_normalized.npy", kappa_norm)
    fused_fingerprint = sha256_file(args.output_dir / "fused_features.npy")
    reduced = fit_umap(fused, args, args.output_dir, fused_fingerprint)
    best_k, labels, scores = run_drowcula(reduced, args, args.output_dir, manifest)

    config = {
        **CONFIG,
        "seed": args.seed,
        "dino_input": str(args.dino),
        "dino_sha256": dino_fingerprint,
        "embedding_ids": str(args.embedding_ids),
        "manifest": str(args.manifest),
        "siglip2_model": args.siglip2_model,
        "gpt_model": args.gpt_model,
        "gpt_temperature": args.gpt_temperature,
        "gpt_max_concurrency": args.gpt_max_concurrency,
        "image_batch_size": args.image_batch_size,
        "text_batch_size": args.text_batch_size,
        "kec_overcluster_divisor": args.kec_overcluster_divisor,
        "kec_top_nouns": args.kec_top_nouns,
        "kec_alpha": args.kec_alpha,
        "kec_beta": args.kec_beta,
        "kec_lambda1": args.kec_lambda1,
        "kec_lambda2": args.kec_lambda2,
        "kec_kmeans_n_init": args.kec_kmeans_n_init,
        "kec_kmeans_max_iter": args.kec_kmeans_max_iter,
        "umap_n_components": args.umap_n_components,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "drowcula_kmeans_n_init": args.drowcula_kmeans_n_init,
        "drowcula_kmeans_max_iter": args.drowcula_kmeans_max_iter,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "selected_k": best_k,
        "kappa_sha256": kappa_fingerprint,
        "fused_sha256": fused_fingerprint,
        "feature_dimensions": {"dino": 768, "kappa": 768, "fused": 1536},
        "normalization": "independent row-wise L2 for DINOv3 and kappa before concatenation",
        "kec_method": "KEC concepts/attributes/grounding adapted to SigLIP2 and GPT-4o",
        "drowcula_method": "official-style UMAP + exhaustive silhouette K search + K-Means",
        "unu_keys_used_in_clustering": False,
        "ground_truth_used_for_k": False,
        "turtle_used": False,
        "output_dir": str(args.output_dir),
    }
    save_json(config, args.output_dir / "config.json")
    log.info("complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
