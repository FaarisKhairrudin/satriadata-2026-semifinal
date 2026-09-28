# Results Method — Techniques, Models, and Equation Index

Companion to [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) (which holds the
*results*). This document records **how** every result was produced: the
exact models, techniques, hyperparameters, and every equation used, mapped to
the numbered scripts in `code/scripts/`.

The final locked pipeline (one script per stage, in experiment order):

```text
01_dinov3_kmeans_baseline.py        pure visual baseline
02_drowcula_dinov3.py               DINOv3-only DROWCULA
03_kec_drowcula_dinov3.py           KEC + DINOv3 + DROWCULA (proposed method)
04_evaluate_clustering_spaces.py    equivalent-space post-hoc evaluation
05_kec_drowcula_cluster_figures.py  top-9 / far-9 prototype grids
06_unukey_grounding_siglip2.py      zero-shot labeling + majority vote + human check
07_cluster_risk_value_mapping.py    EVI / HI risk-value quadrants
08_cluster_insight_figures.py      9 insight figures (K-search, UMAP, votes, EU, Hg)
```

Deprecated experiments (legacy SigLIP-base KEC, DeepSeek mining, TURTLE,
old K=14 figures, dataset builder) live in `code/scripts/archive/` with an
explanation in `archive/README.md`.

---

## Model index

| Model | Role | Output | Where used |
| --- | --- | --- | --- |
| **DINOv3 ViT-B/16** (frozen) | Visual representation | 768-D global embedding per image | 01, 02, 03, 04 (cached `embeddings_dinov3.npy`) |
| **SigLIP2 base patch16-256** (`google/siglip2-base-patch16-256`) | Image–text alignment for KEC | 768-D image & text embeddings | 03 (KEC image features + all text encoding), 06 (UNU-KEY grounding) |
| **GPT-4o** (`gpt-4o`, temperature 0.1, JSON mode) | Concept & attribute mining (LLM knowledge source) | concept name + description, λ1=2 uni-attributes, λ2=1 bi-attribute | 03 (KEC stage 2/3) |
| **WordNet (NLTK)** | Noun candidate vocabulary | 113,964 cleaned nouns | 03 (KEC image–text mapping) |
| **UMAP** (`umap-learn`) | Manifold dimensionality reduction | 3-D representation | 02, 03 (DROWCULA reduction step) |
| **scikit-learn KMeans** | Final cluster assignment | labels per candidate K | 01, 02, 03 (DROWCULA final step) |

Excluded by design (documented negative result / rule): TURTLE, UNU-KEYs during
clustering, ground-truth labels for K selection.

---

## 01 — Pure DINOv3 K-Means baseline

**Goal:** visual-only reference point, no knowledge injection.

**Technique.** Cached DINOv3 ViT-B/16 embeddings (frozen, no fine-tuning) are
row-wise L2-normalized, then K-Means (Lloyd) is swept over K = 5..25 with
`n_init=20`, `max_iter=300`, `seed=42`. Every K is retained (labels, sizes,
metrics); the summary reports the K with maximum silhouette.

**Equations.**

**1. L2 normalization:** `x̂ = x / ‖x‖₂`

**2. K-Means objective (inertia):** `J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖₂`

**3. Silhouette:** `s(i) = (b(i) − a(i)) / max{a(i), b(i)}`, averaged over samples

**4. Davies–Bouldin:** `DB = (1/K) Σₖ max_{l≠k} (σₖ + σₗ)/d(μₖ, μₗ)`

**5. Calinski–Harabasz:** `CH = [B/(K−1)] / [W/(N−K)]` (B/W = between/within dispersion)

**Selection space:** original normalized 768-D space (this is the honest
baseline space — see 04 for why cross-space numbers must not be compared
directly). Result: best K = 16, silhouette 0.1637.

---

## 02 — DINOv3-only DROWCULA (official recipe)

**Goal:** the Dimensionally-Reduced Open-World Clustering recipe (paper uses
DINOv2; here adapted to existing DINOv3 ViT-B/16 features), with unknown-K
estimation.

**Technique (exactly the official Algorithm 1).**

**Step 1.** Frozen ViT embeddings → **row-wise L2 normalization** (Eq. 1).

**Step 2.** Fit **UMAP once** on the normalized features: `n_components=3`,
`n_neighbors=10`, `min_dist=0.1`, `metric=euclidean`, `random_state=42`
(paper's Table 12 hyperparameters).

**Step 3.** **Exhaustive K search** K = 2..25: K-Means (`n_init=200`,
`max_iter=10000`, matching the official DROWCULA-UMAP notebook) on the reduced
space.

**Step 4.** Score each candidate with **silhouette in the reduced UMAP space**
(Eq. 3) and keep the first maximizer:

```text
K̂ = argmax_{K∈[2,Kmax]} Silhouette( UMAP(Normalize(F)), KMeans_K(...) )
```

**Step 5.** The winning K-Means labels are the final clusters (K̂ = 17).

**Key implementation detail:** UMAP is fitted once and cached
(`reduced_embeddings.npy` + `reduction_config.json` keyed by input SHA-256), so
repeated K evaluations never re-run the manifold learning.

---

## 03 — KEC + DINOv3 + DROWCULA (proposed method)

**Goal:** inject structured textual knowledge (KEC) into the visual embedding,
then apply the same DROWCULA open-world clustering. This is a novel integration:
neither the KEC paper nor the DROWCULA paper proposes this combination.

### Stage A — KEC image–text mapping (paper §3.1)

**6.** SigLIP2 image features for all 3,961 images: `xᵢ` (768-D, L2-normalized
via Eq. 1).

**7.** WordNet nouns cleaned to 113,964 candidates; text-encoded as
`"a photo of {w}."` → `tⱼ`.

**8.** Over-clustering: K-Means on `xᵢ` (Eq. 2) with

```text
K_over = max(2, round(N / 300))          → 13 over-clusters
```

**9.** Top-noun retrieval per over-cluster centroid (paper Eq. 1):

```text
Sₚ = TopK( μₚᵀ tⱼ , 5 )
```

### Stage B — Hierarchical knowledge construction (paper §3.2–3.3, via GPT-4o)

**10.** Multi-modal cluster similarity and merging (paper Eq. 2, α = 0.8):

```text
Rᵢⱼ = α·sim(μᵢ, μⱼ) + (1−α)·sim(νᵢ, νⱼ)
Gᵢⱼ = 1[Rᵢⱼ > β],  β = 0.8 → connected components = merged concepts
```

**11.** LLM prompts (GPT-4o, temp 0.1, JSON mode, ≤20 concurrent) produce one
concept `c_q` + description `d_q` per merged component, λ1 = 2 uni-concept
attributes, and λ2 = 1 bi-concept attribute per neighbor pair (neighbor
selection: softmax over concept-embedding similarities until cumulative mass
≥ β).

### Stage C — Knowledge grounding (paper §3.4, Eq. 4)

**12.** Concept representation `ζ_q = φ_q + ψ_q` (concept + description text
embeddings).

**13.** Attention weight of image i over concepts:

```text
ωᵢ,q = exp(xᵢᵀ ζ_q) / Σ_l exp(xᵢᵀ ζ_l)
```

**14.** Attribute instantiation and averaging: `ξ̄qᵢ = mean_l( xᵢ ⊙ ξq,l )`

**15.** Knowledge-enhanced feature:

```text
κᵢ = cᵢ + aᵢ = Σ_q ωᵢ,q ζ_q + Σ_q ωᵢ,q ξ̄qᵢ
```

### Stage D — Fusion + DROWCULA

**16.** Fusion (KEC training-free pathway, paper §3.5):

```text
fusedᵢ = [ x̂ᵢ(dino) ‖ κ̂ᵢ ]        ∈ ℝ¹⁵³⁶
```

with **independent** row-wise L2 normalization (Eq. 1) of the DINOv3 block and
the kappa block before concatenation.

**17.** Then the **identical DROWCULA recipe as 02** (UMAP 3-D → K-Means
`n_init=200` per candidate → silhouette selection, K = 2..25).
Result: K̂ = 17.

**Caching.** `kappa.npy`, `fused_features.npy`, and the reduced UMAP space are
cached with SHA-256 input fingerprints; reruns without `--overwrite` skip the
SigLIP2/GPT-4o stages entirely.

---

## 04 — Equivalent-space post-hoc evaluation

**Goal:** make the three existing label sets (K=16 pure K-Means, K=17
DINOv3-only DROWCULA, K=17 fused) numerically comparable **without rerunning
any clustering**.

**18.** Reuse the saved 3-D UMAP space of DINOv3-only DROWCULA as the common
evaluation space for all three label sets (Table 1).

**19.** Evaluate each DROWCULA label set in its **native** UMAP space (Table 2).

**20.** Metrics recomputed: Silhouette (Eq. 3), DBI (Eq. 4), CH (Eq. 5).

Ordering is guarded by `embedding_ids.csv` vs each `labels.csv` (canonical
path matching + label equality check). Conclusion of the audit: the earlier
0.16-vs-0.68 "jump" was a metric-space mismatch, not an algorithmic gain.

---

## 05 — Cluster prototype grids

For each verified cluster: rank members by Euclidean distance to the
fused-space centroid (computed from the *saved* labels — no refit), export the
9 nearest (`top9`) and 9 farthest (`far9`) images as 3×3 grids. These are the
human-verification artifacts for stage 3 of the labeling pipeline.

---

## 06 — UNU-KEYs labeling: zero-shot voting + centroid + human check

**Goal:** name each fixed cluster with UNU-KEYs-v.2 codes (57 categories)
without ever letting the taxonomy touch clustering.

**Technique.**

**21.** SigLIP2 text anchors per category, prompt-ensembled (3 e-waste-anchored
templates, `padding="max_length", max_length=64`):

```text
t_u = normalize( mean_m normalize( encode(template_m(category_u)) ) )
```

**22.** **Per-image zero-shot classification** (the primary signal):

```text
ŷᵢ = argmax_u  cos( xᵢ , t_u )       over 57 UNU-KEYs + 3 GAP categories
```

**23.** **Majority aggregation** inside each fixed cluster:

```text
share(u, C) = #{i ∈ C : ŷᵢ = u} / |C|
winner share ≥ 50%  → MAJORITY (single label)
35% ≤ share < 50%   → PLURALITY (single label, weaker)
share < 35%         → HETEROGENEOUS → multi-candidate label "a/b/c?"
```

**24.** **Coverage-gap rule** (streams that are e-waste under Basel/HS but not
finished products in UNU-KEYs-v.2): a supplementary category (GAP-BATT
batteries, GAP-PCB boards, GAP-SCRAP mixed scrap) that outscores the best
UNU-KEY by > 0.005 cosine wins outright.

**25.** **Centroid cosine** (secondary evidence): cluster centroid
`μ_C = normalize(mean of member image features)` ranked against the same
taxonomy.

**26.** **Human verification** (final word): top-9 nearest-centroid grids
inspected visually, corroborated by filename evidence; 7 labels confirmed, 10
corrected. Final labels: `final_cluster_labels.csv`.

Result: 17 clusters → 10 UNU-KEYs-v.2 codes + 4 GAP streams (GAP-BATT,
GAP-PCB, GAP-SCRAP, GAP-OTHER — the last being power outlets/switches, which
have no finished-product UNU-KEY and map to HS 8549.99).

---

## 07 — Risk-value mapping (EVI / HI)

**Goal:** convert verified clusters into strategic quadrants for the paper's
interpretation section. Post-clustering only; no feedback into clustering.

Formula lock decision (linear quotients + log10-standardized scores, 17/17
stable, chosen over the Hakanson-log worktree variant): see
keputusan formula lock HI 2026-09-26.

### Equations

**27. Economic Value Intensity** (USD per kg of waste):

```text
EVI = Au·p_Au + Ag·p_Ag + Pd·p_Pd + Cu·p_Cu + Co·p_Co + Al·p_Al + Fe·p_Fe
```

with prices p in USD/mg (snapshot 2026-09-25/26, LBMA/spot + LME): Au 0.1392,
Ag 0.00207, Pd 0.0409, Cu 0.00001474, Co 0.0000397, Al 0.000003278,
Fe 0.0000004.

**28. Hazard Intensity** (5-term unweighted sum of regulatory quotients;
locked 2026-09-26 after retiring the severity-weight and expert-score variants):

```text
HI = Hg/15 + Pb/1000 + POPs/1000 + Co/8000 + Ni/2000
```

Threshold denominators are regulatory: Hg/15 = Minamata COP-5 waste threshold;
Pb/1000 = RoHS Annex II 1,000 ppm (applied to whole-unit estimates — documented
approximation); POPs has no universal threshold → the /1000 term is a proposed
normalisation (Basel low-POP 50 mg/kg PBDE as reference). Co/8000 and Ni/2000
follow the TTLC toxicity thresholds (California Title 22), the same limits used
in the spent-battery characterisation literature (battery electrodes run
~45× the Co TTLC). No severity weights: every quotient enters HI as-is, so the
weight-scheme sensitivity test is retired — there are no free weights to vary.
Concentration sources: Co from the existing benchmark table; Ni added 2026-09-26
from UNITAR Table 26 per EU-6PV category (screens 2,160 / large 7,780 /
small equipment 5,760 / small IT 4,320 mg/kg), with literature values for the
GAP streams (batteries 75,272 mg/kg electrode mean, PCB ~735 mg/kg PWB) and
two documented team estimates (mixed scrap = small-equipment average, outlets
100 mg/kg brass).

**29. Data sources (verified 2026-09-23 against the guidelines):**

- **Unit weight** `w_unit_kg` — UNITAR Table 25, **2024 column** (monitors 8.20 kg,
  smartphones 0.08 kg, printers 12.13 kg, CRT 33.20 kg, washing machines 74.36 kg)
- **Metal composition** — UNITAR Table 26 by EU-6PV category (0309 and 0104 match
  the table exactly); CRT lead 85,000 mg/kg from CRT-specific literature because
  Table 26's *screens* average (4,560 mg/kg) is LCD-diluted
- **Mercury** — guidelines mercury-risk list only (our 0303, 0306, 0309, 0408);
  0304 and 0407 carry no Hg term
- **POPs** — no per-category mg/kg in the guidelines (only national tonnage) →
  retained as documented estimates
- Unit weight affects **only** the mass/value proxy, never EVI/HI or quadrants

**30. Standardization:** min–max to [0, 100]; both EVI and HI are log10-scaled
first (EVI: PCB USD 64/kg vs mice USD 0.29/kg spans ~200×; HI: raw CRT 87–88
vs mice ~3.5 spans ~25×). HI score uses log10(x + 1) then min–max,
identical treatment to EVI's log10(x + 0.1).

**31. Quadrants** at the median split of each axis:

```text
EVI ≥ med, HI ≥ med → Q1 Critical Urban Mining
EVI <  med, HI ≥ med → Q2 Hazardous Neutralization
EVI ≥ med, HI <  med → Q3 Fast Circular Recovery
else                  → Q4 General / Inert Residue
```

**32. Sensitivity:** retired 2026-09-26. The weight-scheme test (Baseline /
Mercury-Heavy / Equal-Risk) applied to the severity-weighted HI; the locked
5-term formula has no free weights to vary, so the test no longer applies.
Historical result for the weighted variant was 17/17 stable.

### Input validation highlight

The CRT television extreme (HI = 100) was cross-checked against independent
sources: funnel glass contains 22–25% PbO (≈20–23% Pb metal), giving ≈8% Pb
whole-unit (≈2.4 kg per 28 kg set) — consistent with the US EPA average of
"four pounds of lead" per color CRT and with Basel Annex VIII A1181/A2010,
which lists CRT glass as *presumptively hazardous*. Details in
`docs/progress/2026-09-23_RISK_VALUE_MAPPING.md`.

---

## Artifact map

| Path | Content |
| --- | --- |
| `data/processed/electronic/dinov3_kmeans_baseline/` | K=5..25 sweep, labels, config (01) |
| `data/processed/electronic/drowcula_dinov3/` | UMAP cache, K-search, labels (02) |
| `data/processed/electronic/kec_drowcula_dinov3/` | KEC artifacts, kappa, fused/reduced features, labels, grounding, risk-value (03/06/07) |
| `data/processed/electronic/equivalent_space_evaluation/` | Table 1 / Table 2 metrics (04) |
| `docs/RESULTS_SUMMARY.md` | results narrative with tables + matrix figure |
| `docs/progress/2026-09-2*.md` | dated experiment logs |
