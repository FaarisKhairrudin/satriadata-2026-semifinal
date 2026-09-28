# Current Clustering Results Summary

## Current preferred result

The current preferred internal-metric configuration is:

```text
DINOv3 ViT-B/16
    + regenerated KEC kappa from SigLIP2 + GPT-4o
    → independent L2 normalization
    → concatenation
    → UMAP
    → DROWCULA unknown-K search, K=2..25
    → K-Means
```

It selects **K=17**. This is the same K selected by the DINOv3-only
DROWCULA baseline.

## Metric comparison

Both runs use 3,961 images, seed 42, the same UMAP configuration, the same
K-search range, and official-style DROWCULA K-Means settings. Metrics are
internal metrics only; no ground-truth labels or UNU-KEYs were used for K
selection or clustering.

The earlier comparison mixed original-space K-Means metrics with UMAP-space
DROWCULA metrics. The corrected post-hoc evaluation is in
`progress/2026-09-23_EQUIVALENT_SPACE_REPORT.md`.


| Method                | K   | Silhouette ↑  | DBI ↓         | CHI ↑     | Inertia ↓    |
| --------------------- | ---: | -------------: | -------------: | ---------: | ------------: |
| DINOv3-only DROWCULA  | 17  | 0.678621      | 0.439182      | 15648.248 | 2816.492     |
| DINOv3 + KEC DROWCULA | 17  | **0.682785**  | **0.437006**  | 15591.744 | **2757.929** |
| Fused − baseline      | 0   | **+0.004164** | **−0.002176** | −56.504   | −58.563      |
| Relative change       | 0%  | **+0.613%**   | **−0.496%**   | −0.361%   | **−2.080%**  |


## Corrected common-space metrics

### Table 1 — Common DINOv3→UMAP evaluation

All existing label sets are evaluated in the exact saved 3-D UMAP space used by
DINOv3-only DROWCULA. The K-Means assignments are unchanged; they are placed in
this UMAP space only for metric comparability.

This is the **common-space post-hoc comparison available from the existing
fixed assignments**. It is not a fully controlled algorithm ablation: K-Means
uses K=16, DROWCULA uses K=17, and the K values were selected under different
original protocols. The table does not rerun K-Means after UMAP.


| Method            | K   | Silhouette ↑ | DBI ↓        | CHI ↑         |
| ----------------- | ---: | ------------: | ------------: | -------------: |
| DINOv3 + K-Means  | 16  | 0.465532     | 1.145760     | 2108.992      |
| DINOv3 + DROWCULA | 17  | **0.678621** | **0.439182** | **15648.248** |


**Table 1 interpretation:** Under a common DINOv3→UMAP evaluation space, the
existing DINOv3-only DROWCULA labels obtain a higher silhouette score than the
existing pure K-Means labels. This is a post-hoc fixed-label comparison. It
must not be summarized as “DROWCULA is better than K-Means” solely because
0.6786 is greater than 0.4655. The methods use different K values, and the
K-Means labels were not regenerated in UMAP space.

### Table 2 — Native proposed-method diagnostic


| Method                  | Native space              | K   | Silhouette ↑ | DBI ↓        | CHI ↑     |
| ----------------------- | ------------------------- | ---: | ------------: | ------------: | ---------: |
| DINOv3 + DROWCULA       | DINOv3→UMAP 3-D           | 17  | 0.678621     | 0.439182     | 15648.248 |
| KEC + DINOv3 + DROWCULA | fused DINOv3+KEC→UMAP 3-D | 17  | **0.682785** | **0.437006** | 15591.744 |


**Table 2 interpretation:** Each method is evaluated in the native UMAP space
used by its own DROWCULA pipeline. The KEC-enhanced representation improves
silhouette by 0.0042 and reduces DBI by 0.0022 relative to DINOv3-only
DROWCULA. Because the feature spaces differ, these values are native-method
diagnostics rather than representation-independent scores.

## Metric interpretation

### Silhouette

Silhouette is the DROWCULA K-selection metric. Higher is better. The fused
method reaches 0.682785 at K=17, versus 0.678621 for DINOv3-only. This is a
small but positive improvement of 0.004164.

### Davies–Bouldin index

Lower is better. DBI improves from 0.439182 to 0.437006, a small reduction of
0.002176. This agrees with the silhouette improvement.

### Calinski–Harabasz index

Higher is conventionally better, but CHI is sensitive to representation scale
and partition structure. CHI decreases slightly from 15648.248 to 15591.744.
Therefore, the fused representation should not be described as improving every
internal metric.

### Inertia

Fused inertia decreases from 2816.492 to 2757.929. This is supplementary
because inertia depends on the feature representation and should not be treated
as a universal quality measure across different spaces.

## Defensible conclusion

The DINOv3-only DROWCULA labels achieved a silhouette of 0.6786 when
evaluated in the common DINOv3→UMAP space, compared with 0.4655 for the
existing pure K-Means labels. This comparison was performed post hoc without
changing the original assignments. It is a common-space diagnostic, not a claim
of a fully controlled K-Means-versus-DROWCULA algorithm ablation.

In their respective native DROWCULA spaces, the KEC-enhanced representation
produced a modest improvement over DINOv3-only DROWCULA: silhouette increased
from 0.6786 to 0.6828 and DBI decreased from 0.4392 to 0.4370. Because these
native spaces differ, the result should be reported as a native-method
diagnostic rather than representation-independent classification performance.

Recommended paper wording:

> Under a common DINOv3→UMAP evaluation space, the existing DINOv3-only
> DROWCULA labels obtained a higher silhouette score than the existing pure
> K-Means labels. This was a post-hoc fixed-label comparison; the original K
> values and assignments were preserved. In their respective native DROWCULA
> spaces, the KEC-enhanced representation produced a modest improvement over
> DINOv3-only DROWCULA, increasing silhouette from 0.6786 to 0.6828 and
> reducing DBI from 0.4392 to 0.4370. These results are internal clustering
> diagnostics and should not be interpreted as externally validated
> classification accuracy.

No clustering was rerun and no assignments were changed.

The KEC artifacts were regenerated with SigLIP2 and GPT-4o; they are not
bit-identical to the missing earlier KEC artifacts.

## Semantic grounding: from visual clusters to UNU-KEYs-v.2

After clustering was locked (K=17, assignments frozen), each cluster was mapped
to the international e-waste taxonomy through a three-stage interpretation
pipeline. UNU-KEYs were used only here — never during feature construction,
K selection, or clustering.

### Process

1. **Per-image zero-shot voting (SigLIP2).** Every one of the 3,961 images was
 classified individually against 57 UNU-KEYs-v.2 categories plus 3
 coverage-gap categories (batteries, PCBs, mixed scrap) using
 `google/siglip2-base-patch16-256` with e-waste-anchored prompt templates.
2. **Majority aggregation per cluster.** Votes were counted inside each fixed
 cluster. A winner with ≥50% of votes is reported as MAJORITY, 35–50% as
 PLURALITY, and below 35% as HETEROGENEOUS (labeled multi-candidate rather
 than forced into one name).
3. **Human verification.** The top-9 images nearest and furthest each fused-space centroid  
 were visually inspected, corroborated by filename evidence. This step  
 confirmed 7 labels and corrected 10 (e.g. a "Dishwashers" vote was actually  
 washing machines 0104; an "Air Conditioners" vote was actually CRT  
 televisions 0407). Human verification is the final word.

### Final human-verified mapping


| Cluster | n   | Final label                   | UNU-KEY   | Category name (UNU-KEYs-v.2)                      | EU category                | Verification |
| -------: | ---: | ----------------------------- | --------- | ------------------------------------------------- | -------------------------- | ------------ |
| 00      | 281 | Printers and scanners         | 0304      | Printers and scanners, copiers                    | Small IT                   | confirmed    |
| 01      | 524 | Mobile phones                 | 0306      | Mobile Phones and smartphones                     | Small IT                   | confirmed    |
| 02      | 357 | Mixed e-waste scrap pile      | GAP-SCRAP | not a finished product — Basel Y49 / HS 8549.99   | —                          | confirmed    |
| 03      | 290 | Keyboards                     | 0301      | Small IT equipment, routers, mice and keyboards   | Small IT                   | confirmed    |
| 04      | 101 | Washing machines (top-load)   | 0104      | Washing Machines and combined dryers              | Large equipment (excl. PV) | corrected    |
| 05      | 343 | Computer mice                 | 0301      | Small IT equipment, routers, mice and keyboards   | Small IT                   | confirmed    |
| 06      | 141 | Televisions (flat-panel)      | 0408      | Flat-Panel Display TVs, LCD and LED               | Screens and monitors       | corrected    |
| 07      | 242 | Microwaves                    | 0114      | Microwaves (incl. combined)                       | Small equipment            | corrected    |
| 08      | 228 | Laptops and tablets           | 0303      | Laptops and tablets                               | Screens and monitors       | corrected    |
| 09      | 277 | Portable batteries            | GAP-BATT  | not a finished product — Basel A1181 / HS 8549.21 | —                          | confirmed    |
| 10      | 122 | Music player boxes            | 0403      | Music Instruments, Radio and Hi-Fi equipment      | Small equipment            | corrected    |
| 11      | 211 | CRT televisions (convex)      | 0407      | Cathode Ray Tube Televisions                      | Screens and monitors       | corrected    |
| 12      | 356 | Printed circuit boards        | GAP-PCB   | not a finished product — Basel A1181 / HS 8549.31 | —                          | confirmed    |
| 13      | 196 | Washing machines (front-load) | 0104      | Washing Machines and combined dryers              | Large equipment (excl. PV) | corrected    |
| 14      | 105 | Radio and Hi-Fi equipment     | 0403      | Music Instruments, Radio and Hi-Fi equipment      | Small equipment            | corrected    |
| 15      | 150 | Flat-panel display monitors   | 0309      | Flat-Panel Display Monitors, LCD and LED          | Screens and monitors       | corrected    |
| 16      | 37  | Power outlets and switches    | GAP-OTHER | no finished-product UNU-KEY — HS 8549.99          | —                          | corrected    |


### Coverage summary

The 17 unsupervised visual clusters map onto **10 UNU-KEYs-v.2 codes plus 4
documented coverage-gap streams**. The duplicates are meaningful, not noise:
0301 splits into keyboards vs mice, 0104 into top-load vs front-load washing
machines, 0403 into music-player boxes vs radio/hi-fi, and televisions split
into flat-panel (0408) vs CRT (0407). The four GAP streams are waste flows
that the e-waste guidelines recognize (Basel A1181 hazardous, Basel Y49, HS
8549.x) but that are not finished products in UNU-KEYs-v.2 — their detection
is a strength of the unsupervised approach, since a fixed 57-class classifier
could not have isolated them.

### Why this mapping is defensible

- No label leakage: UNU-KEYs entered only after clustering was frozen.
- Every cluster name rests on three independent signals: per-image vote
distribution, centroid cosine, and direct human inspection of prototype
images.
- Where the automated signals disagreed with human inspection, the human
verdict was recorded as final and the disagreement was kept in the audit
trail (`vote_agreement` column in `final_cluster_labels.csv`).

## Risk-value mapping: from verified clusters to decision quadrants

The final step converts each verified cluster into a strategic positioning
using two guideline-grounded indices — an **Economic Value Intensity (EVI)**
for urban-mining potential and a **Hazard Intensity (HI)** for environmental
toxicity (script `07_cluster_risk_value_mapping.py`). This is post-clustering
interpretation only: it ranks the fixed K=17 clusters and never feeds back
into clustering.

### Equations

**Economic Value Intensity** — metal content (mg/kg of waste) priced at market
value (USD per mg), summing seven metals:

```text
EVI = Au·0.138 + Ag·0.0022 + Pd·0.0455 + Cu·0.0000126
    + Co·0.00003 + Al·0.0000025 + Fe·0.00000035     [USD/kg]
```

(prices = snapshot 2026-09-25/26, LBMA/spot + LME; re-verify at submission)

**Hazard Intensity** — toxicant concentrations normalized by regulatory
thresholds (unweighted sum, locked 2026-09-26):

```text
HI = Hg/15 + Pb/1000 + POPs/1000 + Co/8000 + Ni/2000
```

- Hg/15: Minamata Convention COP-5 waste threshold (15 mg/kg)
- Pb/1000: RoHS Annex II limit (1,000 ppm) — applied to whole-unit estimates,
  a documented approximation (RoHS regulates homogeneous materials)
- POPs: no single universal threshold (Stockholm regulates per compound) —
  the /1000 term is a proposed normalisation (Basel low-POP 50 mg/kg PBDE
  as reference)
- Co/8000 and Ni/2000: TTLC toxicity thresholds (California Title 22), the
  same limits used in spent-battery characterisation literature; Co from the
  benchmark table, Ni from UNITAR Table 26 per EU-6PV category + literature
  for the GAP streams
- No severity weights; the weight-scheme sensitivity test is retired because
  there are no free weights to vary

Both indices are min–max standardized to 0–100 on a log10 scale (identical
treatment: EVI log10(x+0.1), HI log10(x+1)). Quadrants are assigned at the
**median split** of each axis.

### Results (17 clusters, baseline scheme)

| Cluster | Stream | EVI (USD/kg) | HI | Quadrant | Stable |
| ---: | --- | ---: | ---: | --- | --- |
| 12 | Printed circuit boards | 64.63 | 30.3 | Q1 Critical Urban Mining | yes |
| 01 | Smartphones | 57.44 | 8.0 | Q1 Critical Urban Mining | yes |
| 08 | Laptops & tablets | 25.53 | 5.8 | Q1 Critical Urban Mining | yes |
| 09 | Portable batteries | 7.22 | 70.8 | Q1 Critical Urban Mining | yes |
| 15 | LCD/LED monitors | 5.20 | 5.0 | Q1 Critical Urban Mining | yes |
| 16 | Power outlets & switches | 3.73 | 6.1 | Q1 Critical Urban Mining | yes |
| 06 | Flat-panel TVs | 2.59 | 5.1 | Q1 Critical Urban Mining | yes |
| 00 | Printers | 2.06 | 4.8 | Q1 Critical Urban Mining | yes |
| 11 | CRT televisions | 0.77 | 88.5 | Q2 Hazardous Neutralization | yes |
| 10 | Music player boxes | 1.11 | 4.5 | Q3 Fast Circular Recovery | yes |
| 14 | Radio / Hi-Fi | 1.11 | 4.5 | Q3 Fast Circular Recovery | yes |
| 07 | Microwaves | 0.97 | 3.5 | Q4 General residue | yes |
| 02 | Mixed scrap | 0.85 | 4.0 | Q4 General residue | yes |
| 04 | Washing machines (top-load) | 0.58 | 4.2 | Q4 General residue | yes |
| 13 | Washing machines (front-load) | 0.58 | 4.2 | Q4 General residue | yes |
| 03 | Keyboards | 0.48 | 3.7 | Q4 General residue | yes |
| 05 | Computer mice | 0.31 | 3.5 | Q4 General residue | yes |

Sensitivity: the weight-scheme test is retired (no free weights in the
5-term formula). Quadrants are identical across all four formula variants
tested (weighted, unweighted, 4-term TTLC, 5-term TTLC): Q1×8, Q2×1,
Q3×2, Q4×6 — the strategic conclusions are formula-invariant.

![Risk-Value Decision Matrix — EVI vs HI for the 17 verified e-waste clusters](../data/processed/electronic/kec_drowcula_dinov3/figures/risk_value_matrix.png)

*Figure: 2D Risk-Value decision matrix. Bubble size ∝ cluster image count;
icons show the verified stream; quadrant splits at the median of each axis.
The sensitivity curves are in `figures/risk_value_sensitivity.png`.*

### CRT televisions: the hazard extreme is valid, not an artifact

CRT's HI=100 is driven 96% by one term: **lead at 85,000 mg/kg (8.5% of unit
weight ≈ 2.38 kg per 28 kg set)**. Cross-checked against independent sources:

- Peer-reviewed CRT studies report funnel glass at **22–25% PbO** (≈20–23% Pb
  metal), with the panel/funnel/neck structure averaging **~8% Pb** for a whole
color CRT — matching our input exactly.
- US EPA: color TVs contain an **average of four pounds of lead**; 21 of 30
  color CRTs failed the TCLP hazardous threshold (avg 18.5–22.2 mg/L vs 5 mg/L
  limit), which is why EPA created a dedicated CRT Rule in 2006.
- Basel Convention technical guidelines: **"glass from cathode ray tubes" is
  always hazardous** (Annex VIII A1181/A2010, category Y31 "Lead; lead
  compounds") — presumptively hazardous, no composition test needed.

The extreme position is therefore the **textbook-expected result**: CRTs carry
kilograms of lead per unit with minimal precious-metal content, which is
exactly why CRT glass has its own international waste code.

### Limitations

- EVI/HI are **relative priority indices for ranking**, not absolute tonnage or
  revenue estimates (mass proxy = image count × typical unit weight).
- Metal prices are a provisional snapshot; record LBMA/LME source + date at
  submission.
- Benchmark compositions are category-typical estimates, not laboratory assays.

### Data provenance check (against the guidelines)

All benchmark inputs were cross-checked against `00_UNUKEY.md` and the source
PDF after the first figure was produced:

- **EU-6PV category mapping:** 10/10 verified labels match the guidelines table.
- **Unit weights:** aligned to the guidelines' **Table 25 (2024 column)** — e.g.
  monitors 8.20 kg, smartphones 0.08 kg, printers 12.13 kg, CRT 33.20 kg,
  washing machines 74.36 kg. This changes only the mass/value proxy
  (41,206 → 43,592 kg; US$54.8k → **US$56.3k**), never the quadrant positions,
  because unit weight does not enter EVI or HI.
- **Mercury:** two values not present in the guidelines' mercury list were
  removed (printers 0304, CRT 0407); their HI moved 19.27 → 17.60 and
  530.20 → 529.20 with **no quadrant change**. All remaining Hg values are on the
  guideline list (0303, 0306, 0309, 0408).
- **Composition:** values for monitors (0309) and washing machines (0104) match
  Table 26 columns exactly; CRT lead (85,000 mg/kg) intentionally uses
  CRT-specific literature rather than the LCD-diluted Table 26 *screens*
  average (4,560 mg/kg), and this is stated in the figure footnote.
- **Quadrant invariance was verified after alignment:** Q1×8, Q2×1, Q3×2, Q4×6 —
  identical to the pre-alignment run.

## Artifacts

- Baseline: `data/processed/electronic/drowcula_dinov3/`
- Fused: `data/processed/electronic/kec_drowcula_dinov3/`
- Detailed comparison: `docs/progress/2026-09-22_DROWCULA_COMPARISON.md`
- Grounding process: `docs/progress/2026-09-23_SIGLIP2_CLUSTER_GROUNDING.md`
- Final verified labels: `final_cluster_labels.csv` / `.json` (fused directory)
- Risk-value table: `cluster_risk_value.csv` (fused directory)
- Risk-value figures: `figures/risk_value_matrix.png`, `figures/risk_value_sensitivity.png`
- Risk-value process: `docs/progress/2026-09-23_RISK_VALUE_MAPPING.md`

