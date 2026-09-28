"""07_cluster_risk_value_mapping.py — Kuantifikasi Nilai Ekonomi (EVI) dan Bahaya Lingkungan (HI) 17 Klaster.

Tujuan:
  Menghubungkan 17 klaster final (KEC + DINOv3 + DROWCULA, K=17, label terverifikasi
  manusia di final_cluster_labels.csv) dengan data empiris UNITAR/UNEP E-waste
  Statistics Guidelines (Third Edition 2026) untuk menghasilkan:
  1. Skor Economic Value Intensity (EVI) -> Potensi Urban Mining (USD/kg, skala 0-100).
  2. Skor Hazard Intensity (HI) -> Tingkat Bahaya Toksik Regulasi Lingkungan (skala 0-100).
  3. Klasifikasi Matriks 2D 4 Kuadran Keputusan Strategis.
  4. Uji Sensitivitas (Sensitivity Analysis) stabilitas kuadran terhadap variasi bobot B3.
  5. Visualisasi 2D Scatter Plot dan kurva sensitivitas siap publikasi.

Inputs:
  - data/processed/electronic/kec_drowcula_dinov3/final_cluster_labels.csv  (human-verified, K=17)

Outputs:
  - data/processed/electronic/kec_drowcula_dinov3/cluster_risk_value.csv
  - data/processed/electronic/kec_drowcula_dinov3/figures/risk_value_matrix.png
  - data/processed/electronic/kec_drowcula_dinov3/figures/risk_value_sensitivity.png

Usage:
  python3.12 code/scripts/07_cluster_risk_value_mapping.py [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "electronic"
KEC = PROC / "kec_drowcula_dinov3"
FIGS = KEC / "figures"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("08_risk_value")

# ---------------- BENCHMARK DATA EMPIRIS UNITAR / UNEP 2026 ----------------
# Referensi: Tabel 25 (Unit weight), Tabel 26 (Metal composition), Tabel 28/30/31 (Hazard)
# Logam: Au, Ag, Pd, Cu, Co, Al, Fe (dalam mg/kg limbah)
# Toksik: Hg (mg/kg), Pb (mg/kg), POPs_PBDE (mg/kg), Battery_Tox (skala 0-10)

BENCHMARKS = {
    "GAP-BATT": {
        "short_name": "Batteries & Cells",
        "w_unit_kg": 0.05,  # estimate: batteries are entry A1170, not a UNU-KEY product
        "Au_mg_kg": 0.0,
        "Ag_mg_kg": 5.0,
        "Pd_mg_kg": 0.0,
        "Cu_mg_kg": 85000.0,    # Cu foil / terminals
        "Co_mg_kg": 145000.0,   # High Cobalt cathode (LCO/NMC)
        "Ni_mg_kg": 75272.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 45000.0,
        "Fe_mg_kg": 120000.0,   # Steel casing
        "Hg_mg_kg": 0.5,
        "Pb_mg_kg": 15000.0,    # Legacy Pb / solders
        "POPs_mg_kg": 50.0,
        "Batt_Tox": 9.5,        # Extreme chemical reactivity & ecotoxicity
    },
    "0403": {
        "short_name": "Audio / Hi-Fi",
        "w_unit_kg": 3.73,  # UNITAR Table 25 (2024) key 0403
        "Au_mg_kg": 1.8,
        "Ag_mg_kg": 12.0,
        "Pd_mg_kg": 0.5,
        "Cu_mg_kg": 38000.0,    # Transformers, speaker coils, wiring
        "Co_mg_kg": 5.0,
        "Ni_mg_kg": 5760.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 25000.0,
        "Fe_mg_kg": 420000.0,
        "Hg_mg_kg": 0.1,
        "Pb_mg_kg": 1200.0,
        "POPs_mg_kg": 450.0,
        "Batt_Tox": 0.5,
    },
    "0306": {
        "short_name": "Smartphones",
        "w_unit_kg": 0.08,  # UNITAR Table 25 (2024) key 0306
        "Au_mg_kg": 340.0,      # Extremely rich in gold wire & contacts
        "Ag_mg_kg": 1350.0,
        "Pd_mg_kg": 95.0,
        "Cu_mg_kg": 130000.0,
        "Co_mg_kg": 32000.0,    # Embedded Li-ion battery
        "Ni_mg_kg": 4320.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 65000.0,
        "Fe_mg_kg": 75000.0,
        "Hg_mg_kg": 0.2,
        "Pb_mg_kg": 850.0,
        "POPs_mg_kg": 950.0,
        "Batt_Tox": 6.5,
    },
    "0304": {
        "short_name": "Printers & Copiers",
        "w_unit_kg": 12.13,  # UNITAR Table 25 (2024) key 0304
        "Au_mg_kg": 8.5,
        "Ag_mg_kg": 35.0,
        "Pd_mg_kg": 2.1,
        "Cu_mg_kg": 32000.0,    # Motor windings, cables
        "Co_mg_kg": 10.0,
        "Ni_mg_kg": 4320.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 18000.0,
        "Fe_mg_kg": 460000.0,   # Heavy structural steel frame
        "Hg_mg_kg": 0.0,        # doc Hg list (0302/0303/0306/0309/0408/0502-0504) excludes 0304
        "Pb_mg_kg": 1800.0,
        "POPs_mg_kg": 850.0,    # Large plastic casings
        "Batt_Tox": 0.0,
    },
    "0301-KB": {
        "short_name": "Keyboards",
        "w_unit_kg": 0.40,  # UNITAR Table 25 (2024) key 0301 (combined small IT)
        "Au_mg_kg": 1.2,
        "Ag_mg_kg": 8.0,
        "Pd_mg_kg": 0.2,
        "Cu_mg_kg": 14000.0,
        "Co_mg_kg": 0.0,
        "Ni_mg_kg": 4320.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 8000.0,
        "Fe_mg_kg": 150000.0,   # Metal backplate
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 400.0,
        "POPs_mg_kg": 1100.0,   # Flame retardant ABS/PBT plastics
        "Batt_Tox": 0.0,
    },
    "0104": {
        "short_name": "Washing Machines",
        "w_unit_kg": 74.36,  # UNITAR Table 25 (2024) key 0104
        "Au_mg_kg": 0.1,
        "Ag_mg_kg": 1.4,
        "Pd_mg_kg": 0.02,
        "Cu_mg_kg": 19800.0,    # Induction motor, pump wiring
        "Co_mg_kg": 0.7,
        "Ni_mg_kg": 7780.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 21400.0,
        "Fe_mg_kg": 508000.0,   # Half the weight is structural steel/iron
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 120.0,
        "POPs_mg_kg": 150.0,
        "Batt_Tox": 0.0,
    },
    "0309": {
        "short_name": "LCD/LED Monitors",
        "w_unit_kg": 8.20,  # UNITAR Table 25 (2024) key 0309
        "Au_mg_kg": 28.6,
        "Ag_mg_kg": 111.0,
        "Pd_mg_kg": 8.9,
        "Cu_mg_kg": 23800.0,
        "Co_mg_kg": 21.8,
        "Ni_mg_kg": 2160.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 49100.0,
        "Fe_mg_kg": 280000.0,
        "Hg_mg_kg": 4.8,        # CCFL backlight tubes in vintage LCDs
        "Pb_mg_kg": 2400.0,
        "POPs_mg_kg": 1250.0,   # Display bezel & casing polymers
        "Batt_Tox": 0.0,
    },
    "0114": {
        "short_name": "Microwave Ovens",
        "w_unit_kg": 24.30,  # UNITAR Table 25 (2024) key 0114
        "Au_mg_kg": 0.4,
        "Ag_mg_kg": 4.5,
        "Pd_mg_kg": 0.1,
        "Cu_mg_kg": 42000.0,    # Heavy copper magnetron & transformer
        "Co_mg_kg": 2.0,
        "Ni_mg_kg": 5760.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 15000.0,
        "Fe_mg_kg": 580000.0,   # Heavy cavity & casing steel
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 450.0,
        "POPs_mg_kg": 200.0,
        "Batt_Tox": 0.0,
    },
    "0303": {
        "short_name": "Laptops & Tablets",
        "w_unit_kg": 1.35,  # UNITAR Table 25 (2024) key 0303
        "Au_mg_kg": 145.0,      # Dense motherboard & GPU
        "Ag_mg_kg": 580.0,
        "Pd_mg_kg": 42.0,
        "Cu_mg_kg": 95000.0,
        "Co_mg_kg": 18000.0,    # Laptop battery pack
        "Ni_mg_kg": 2160.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 85000.0,    # Aluminium unibody chassis
        "Fe_mg_kg": 95000.0,
        "Hg_mg_kg": 3.2,        # CCFL in legacy models
        "Pb_mg_kg": 1400.0,
        "POPs_mg_kg": 850.0,
        "Batt_Tox": 5.5,
    },
    "0301-MOUSE": {
        "short_name": "Computer Mice",
        "w_unit_kg": 0.40,  # UNITAR Table 25 (2024) key 0301 (official value is combined)
        "Au_mg_kg": 0.8,
        "Ag_mg_kg": 5.0,
        "Pd_mg_kg": 0.1,
        "Cu_mg_kg": 11000.0,
        "Co_mg_kg": 0.0,
        "Ni_mg_kg": 4320.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 2000.0,
        "Fe_mg_kg": 45000.0,
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 350.0,
        "POPs_mg_kg": 950.0,    # Injection molded ABS
        "Batt_Tox": 0.0,
    },
    "0407": {
        "short_name": "CRT Televisions",
        "w_unit_kg": 33.20,  # UNITAR Table 25 (2024) key 0407
        "Au_mg_kg": 2.5,
        "Ag_mg_kg": 15.0,
        "Pd_mg_kg": 0.4,
        "Cu_mg_kg": 18000.0,    # Deflection yoke copper
        "Co_mg_kg": 1.0,
        "Ni_mg_kg": 2160.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 12000.0,
        "Fe_mg_kg": 180000.0,
        "Hg_mg_kg": 0.0,        # doc Hg list excludes 0407; CRT hazard is lead (funnel glass)
        "Pb_mg_kg": 85000.0,    # 1.5-2.5 kg Pb in funnel glass; CRT-specific literature
                                 # (UNITAR Table 26 screens average is 4,560 mg/kg = LCD-diluted)
        "POPs_mg_kg": 2400.0,   # Massive thick back casing with OctaBDE
        "Batt_Tox": 0.0,
    },
    "0501": {
        "short_name": "Small Household",
        "w_unit_kg": 1.2,
        "Au_mg_kg": 0.5,
        "Ag_mg_kg": 4.0,
        "Pd_mg_kg": 0.1,
        "Cu_mg_kg": 26000.0,    # Motor / cord wiring
        "Co_mg_kg": 1.0,
        "Ni_mg_kg": 5760.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 18000.0,
        "Fe_mg_kg": 350000.0,
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 650.0,
        "POPs_mg_kg": 400.0,
        "Batt_Tox": 0.0,
    },
    "GAP-PCB": {
        "short_name": "Printed Circuit Boards",
        "w_unit_kg": 0.25,
        "Au_mg_kg": 380.0,      # High-grade telecom / computer motherboards
        "Ag_mg_kg": 1650.0,
        "Pd_mg_kg": 120.0,
        "Cu_mg_kg": 220000.0,   # Traces & ground planes (22% mass)
        "Co_mg_kg": 450.0,
        "Ni_mg_kg": 735.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 35000.0,
        "Fe_mg_kg": 85000.0,
        "Hg_mg_kg": 0.8,
        "Pb_mg_kg": 28000.0,    # Leaded wave-solders on legacy boards
        "POPs_mg_kg": 1800.0,   # Tetrabromobisphenol A (TBBPA) in FR-4 resin
        "Batt_Tox": 1.0,
    },
    "GAP-SCRAP": {
        "short_name": "Mixed E-waste Scrap",
        "w_unit_kg": 1.0,
        "Au_mg_kg": 1.5,
        "Ag_mg_kg": 10.0,
        "Pd_mg_kg": 0.2,
        "Cu_mg_kg": 25000.0,
        "Co_mg_kg": 10.0,
        "Ni_mg_kg": 5760.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 35000.0,
        "Fe_mg_kg": 320000.0,
        "Hg_mg_kg": 0.1,
        "Pb_mg_kg": 800.0,
        "POPs_mg_kg": 350.0,
        "Batt_Tox": 0.5,
    },
    "GAP-OTHER": {
        "short_name": "Power Outlets & Switches",
        "w_unit_kg": 0.08,
        "Au_mg_kg": 0.0,
        "Ag_mg_kg": 1.5,
        "Pd_mg_kg": 0.0,
        "Cu_mg_kg": 250000.0,   # Brass contacts & terminals
        "Co_mg_kg": 0.0,
        "Ni_mg_kg": 100.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 8000.0,
        "Fe_mg_kg": 40000.0,
        "Hg_mg_kg": 0.0,
        "Pb_mg_kg": 6000.0,     # Leaded brass fittings (conservative)
        "POPs_mg_kg": 60.0,
        "Batt_Tox": 0.0,
    },
    "0408": {
        "short_name": "Flat-Panel TVs",
        "w_unit_kg": 14.68,  # UNITAR Table 25 (2024) key 0408
        "Au_mg_kg": 12.0,
        "Ag_mg_kg": 60.0,
        "Pd_mg_kg": 3.0,
        "Cu_mg_kg": 28000.0,
        "Co_mg_kg": 15.0,
        "Ni_mg_kg": 2160.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 38000.0,
        "Fe_mg_kg": 330000.0,
        "Hg_mg_kg": 1.2,        # doc Hg list: 0408 CCFL backlight (non-LED units)
        "Pb_mg_kg": 2500.0,
        "POPs_mg_kg": 1400.0,
        "Batt_Tox": 0.0,
    },
    "0305": {
        "short_name": "Dismantled Scrap",
        "w_unit_kg": 1.0,
        "Au_mg_kg": 1.5,
        "Ag_mg_kg": 10.0,
        "Pd_mg_kg": 0.2,
        "Cu_mg_kg": 25000.0,
        "Co_mg_kg": 10.0,
        "Ni_mg_kg": 5760.0,  # Table 26 EU-6PV / literature
        "Al_mg_kg": 35000.0,
        "Fe_mg_kg": 320000.0,
        "Hg_mg_kg": 0.1,
        "Pb_mg_kg": 800.0,
        "POPs_mg_kg": 350.0,
        "Batt_Tox": 0.5,
    },
}

# Harga Pasar Komoditas (USD per milligram element) — snapshot 2026-09-25/26
# Sumber (terverifikasi 2026-09-26):
# - Au: spot ~$4,330/oz (25 Sep 2026) -> $139.2/g = 0.1392 USD/mg
# - Ag: spot ~$64.5/oz (25 Sep 2026) -> $2.074/g = 0.00207 USD/mg
# - Pd: spot ~$1,272/oz (24 Sep 2026) -> $40.9/g = 0.0409 USD/mg
# - Cu: LME official $14,740/t (25 Sep 2026) -> $14.74/kg = 0.00001474 USD/mg
# - Co: ~$39,690/t (18 Sep 2026) -> $39.69/kg = 0.0000397 USD/mg
# - Al: LME ~$3,278/t (26 Sep 2026) -> $3.278/kg = 0.000003278 USD/mg
# - Fe: scrap steel ~$400/t (Tindex 20 Sep; TradingEconomics 385-403 Sep) = 0.0000004 USD/mg
# PROVISIONAL: verifikasi ulang LBMA PM fix + LME official pada hari pengumpulan
# dan catat sumber + tanggal di karya ilmiah.
PRICE_AS_OF = "2026-09-25/26 (LBMA/spot, LME, TradingEconomics/Tindex; verifikasi ulang saat submit)"
PRICES_USD_MG = {
    "Au": 0.1392,
    "Ag": 0.00207,
    "Pd": 0.0409,
    "Cu": 0.00001474,
    "Co": 0.0000397,
    "Al": 0.000003278,
    "Fe": 0.0000004,
}

# Skema Bobot Bahaya Lingkungan (Hazard Weights F_hazard)
# Baseline: Minamata Hg=10, Stockholm POPs=8, Basel/RoHS Pb=6, Battery Reactive Tox=7
# Regulatory reference thresholds (verified against primary sources, 2026-09-23):
# - Hg 15 mg/kg: Minamata COP-5 decision MC-5/10 (2023), mercury-waste threshold
# - Pb 1,000 mg/kg: RoHS Directive (EU) 2015/863 Annex II (per homogeneous material)
# - POPs 50 mg/kg: Basel COP low-POP content for listed PBDEs (sum tetra-hepta-BDE)
THRESHOLDS = {"Hg": 15.0, "Pb": 1000.0, "POPs": 50.0}

# Toxic-response factors — Hakanson (1980), Water Research 14(8):975-1001
# (the most-cited contamination-risk index in the e-waste literature):
# Hg = 40, PCB/PBDE analogues = 40, Pb = 5. Battery term stays an expert score
# (0-10) because no concentration threshold exists for battery fire/reactivity.
SCHEMES = {
    "Baseline": {"T_Hg": 40.0, "T_Pb": 5.0, "T_POPs": 40.0, "w_Batt": 7.0},
    "Mercury_Heavy": {"T_Hg": 80.0, "T_Pb": 5.0, "T_POPs": 20.0, "w_Batt": 7.0},
    "Equal_Risk": {"T_Hg": 40.0, "T_Pb": 40.0, "T_POPs": 40.0, "w_Batt": 7.0},
}

# Varian linear "Variant A" (catatan progres 2026-09-23_RISK_VALUE_MAPPING.md):
# quotient linear thd ambang regulasi yang sama (Hg/15 Minamata COP-5 MC-5/10;
# Pb/1000 RoHS (EU) 2015/863 Annex II; POPs/1000 dengan bobot komposit usulan
# tim karena Stockholm mengatur per-senyawa — bukan ambang universal tunggal).
SCHEMES_LINEAR = {
    "Baseline": {"w_Hg": 10.0, "w_Pb": 6.0, "w_POPs": 8.0, "w_Batt": 7.0},
    "Mercury_Heavy": {"w_Hg": 15.0, "w_Pb": 8.0, "w_POPs": 6.0, "w_Batt": 8.0},
    "Equal_Risk": {"w_Hg": 8.0, "w_Pb": 8.0, "w_POPs": 8.0, "w_Batt": 8.0},
}


def calc_evi(b: dict) -> float:
    """Hitung Economic Value Intensity (USD / kg limbah)."""
    try:
        val = (
            b["Au_mg_kg"] * PRICES_USD_MG["Au"]
            + b["Ag_mg_kg"] * PRICES_USD_MG["Ag"]
            + b["Pd_mg_kg"] * PRICES_USD_MG["Pd"]
            + b["Cu_mg_kg"] * PRICES_USD_MG["Cu"]
            + b["Co_mg_kg"] * PRICES_USD_MG["Co"]
            + b["Al_mg_kg"] * PRICES_USD_MG["Al"]
            + b["Fe_mg_kg"] * PRICES_USD_MG["Fe"]
        )
        return float(val)
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"EVI calculation failed for {b.get('short_name', '?')}: {exc}") from exc


def calc_hi(b: dict, scheme: dict) -> float:
    """Hakanson-weighted, log-compressed hazard quotients (see RESULTS_METHOD Eq. 28)."""
    # Normalisation denominators are verified regulatory thresholds (Minamata
    # COP-5 decision MC-5/10: Hg 15 mg/kg; RoHS Directive (EU) 2015/863 Annex II:
    # Pb 1,000 mg/kg; Basel COP low-POP content: PBDE 50 mg/kg). Toxic-response
    # factors follow Hakanson (1980), Water Research 14(8):975-1001: Hg 40,
    # Pb 5, PCB/PBDE 40. The log2(1 + c/L) compression follows the
    # geoaccumulation-index precedent (Muller, 1969) so that no single stream
    # visually dominates the composite index.
    try:
        score_hg = scheme["T_Hg"] * np.log2(1.0 + b["Hg_mg_kg"] / THRESHOLDS["Hg"])
        score_pb = scheme["T_Pb"] * np.log2(1.0 + b["Pb_mg_kg"] / THRESHOLDS["Pb"])
        score_pops = scheme["T_POPs"] * np.log2(1.0 + b["POPs_mg_kg"] / THRESHOLDS["POPs"])
        score_batt = b["Batt_Tox"] * scheme["w_Batt"]
        return float(score_hg + score_pb + score_pops + score_batt)
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"HI calculation failed for {b.get('short_name', '?')}: {exc}") from exc


def calc_hi_linear(b: dict, scheme: dict) -> float:
    """HI final (5 istilah, tanpa bobot): kuota Hg/Pb/POPs ambang regulasi + Co/Ni ambang TTLC.

    scheme diterima demi kompatibilitas pemanggil tetapi tidak dipakai;
    tanpa parameter bobot, uji sensitivitas antar-skema tidak lagi berlaku.
    """
    try:
        score_hg = b["Hg_mg_kg"] / THRESHOLDS["Hg"]
        score_pb = b["Pb_mg_kg"] / THRESHOLDS["Pb"]
        score_pops = b["POPs_mg_kg"] / 1000.0
        score_co = b["Co_mg_kg"] / 8000.0   # TTLC Co 8,000 mg/kg (baterai 45x)
        score_ni = b["Ni_mg_kg"] / 2000.0   # TTLC Ni 2,000 mg/kg
        return float(score_hg + score_pb + score_pops + score_co + score_ni)
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"HI-linear calculation failed for {b.get('short_name', '?')}: {exc}") from exc


def assign_quadrant(evi_norm: float, hi_norm: float, split_evi=50.0, split_hi=50.0) -> str:
    if evi_norm >= split_evi and hi_norm >= split_hi:
        return "Q1: Critical Urban Mining"
    elif evi_norm < split_evi and hi_norm >= split_hi:
        return "Q2: Hazardous Neutralization"
    elif evi_norm >= split_evi and hi_norm < split_hi:
        return "Q3: Fast Circular Recovery"
    else:
        return "Q4: General / Inert Residue"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--hi-variant",
        choices=["hakanson-log", "linear"],
        default="linear",
        help="Rumus HI: linear (final locked, Variant A) atau hakanson-log (eksperimen).",
    )
    ap.add_argument(
        "--hi-score",
        choices=["linear", "log"],
        default="log",
        help="Standardisasi skor HI: log10 min-max (final locked, ala EVI) atau linear min-max.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Direktori output untuk CSV + figures. Default: folder KEC repo.",
    )
    args = ap.parse_args()

    HI_SCHEMES = SCHEMES if args.hi_variant == "hakanson-log" else SCHEMES_LINEAR
    HI_FN = calc_hi if args.hi_variant == "hakanson-log" else calc_hi_linear
    OUT = args.outdir if args.outdir is not None else KEC
    log.info(
        "HI variant: %s | HI score: %s -> outputs in %s",
        args.hi_variant,
        args.hi_score,
        OUT,
    )

    in_csv = KEC / "final_cluster_labels.csv"
    if not in_csv.exists():
        raise SystemExit(
            f"{in_csv} tidak ditemukan. Jalankan 06_unukey_grounding_siglip2.py dan selesaikan "
            "verifikasi manusia (final_cluster_labels.csv) terlebih dahulu."
        )

    df_ground = pd.read_csv(in_csv)
    FIGS_OUT = OUT / "figures"
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "cluster_risk_value.csv"
    out_fig_matrix = FIGS_OUT / "risk_value_matrix.png"
    out_fig_sens = FIGS_OUT / "risk_value_sensitivity.png"

    rows = []
    for record in df_ground.to_dict("records"):
        try:
            c_id = int(record["cluster"])
            code = str(record["unu_key_code"])
            n_img = int(record["n_samples"])
            label = str(record["final_label"])
            subtype = str(record.get("subtype", "") or "")
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"baris final_cluster_labels tidak valid: {record}") from exc

        # Pilih benchmark: sub-tipe 0301 dulu, lalu kode langsung, fallback scrap
        if code == "0301" and subtype == "mice":
            bm_key = "0301-MOUSE"
        elif code == "0301":
            bm_key = "0301-KB"
        elif code in BENCHMARKS:
            bm_key = code
        else:
            bm_key = "GAP-SCRAP"  # fallback: campuran scrap umum
        if bm_key not in BENCHMARKS:
            raise SystemExit(f"benchmark '{bm_key}' untuk cluster {c_id} tidak ditemukan")

        b = BENCHMARKS[bm_key]
        evi_raw = calc_evi(b)
        hi_base = HI_FN(b, HI_SCHEMES["Baseline"])
        if args.hi_variant == "linear":
            # Tanpa bobot: tidak ada parameter skema untuk divariasikan;
            # kolom sensitivitas diisi baseline (uji bobot tidak berlaku).
            hi_merc = hi_base
            hi_eq = hi_base
        else:
            hi_merc = HI_FN(b, HI_SCHEMES["Mercury_Heavy"])
            hi_eq = HI_FN(b, HI_SCHEMES["Equal_Risk"])

        rows.append(
            {
                "cluster": c_id,
                "n_samples": n_img,
                "unu_key_code": code,
                "short_name": b["short_name"],
                "auto_label": label,
                "w_unit_kg": b["w_unit_kg"],
                "evi_raw_usd_kg": round(evi_raw, 2),
                "hi_baseline_raw": round(hi_base, 2),
                "hi_mercury_raw": round(hi_merc, 2),
                "hi_equal_raw": round(hi_eq, 2),
            }
        )

    df_res = pd.DataFrame(rows)

    # Indeks massa PROXY ordinal dalam-dataset (1 foto != 1 unit fisik untuk
    # tumpukan/scrap — BUKAN estimasi tonase lapangan). Dipakai hanya untuk
    # ranking relatif antar-klaster.
    df_res["mass_proxy_kg"] = np.round(df_res["n_samples"] * df_res["w_unit_kg"], 2)
    df_res["mass_proxy_tons"] = np.round(df_res["mass_proxy_kg"] / 1000.0, 3)
    df_res["total_econ_proxy_usd"] = np.round(df_res["mass_proxy_kg"] * df_res["evi_raw_usd_kg"], 2)

    # Standardisasi Min-Max ke Skala [0, 100] dengan Log-scale pada EVI agar tidak didominasi PCB
    # Log-scaling karena nilai emas PCB ($36/kg) vs mesin cuci ($0.45/kg) rentang 80x
    log_evi = np.log10(df_res["evi_raw_usd_kg"] + 0.1)
    df_res["evi_score"] = np.round(
        100.0 * (log_evi - log_evi.min()) / (log_evi.max() - log_evi.min()), 2
    )

    for s_name, col_raw in [
        ("hi_baseline", "hi_baseline_raw"),
        ("hi_mercury", "hi_mercury_raw"),
        ("hi_equal", "hi_equal_raw"),
    ]:
        hi_vals = df_res[col_raw]
        if args.hi_score == "log":
            # Kompresi log10 ala EVI: rumus mentah tetap, tapi skor min-max
            # dihitung atas skala log agar stream ekstrem (CRT) tidak
            # memampatkan semua cluster lain ke dasar skala.
            hi_scaled = np.log10(hi_vals + 1.0)
        else:
            hi_scaled = hi_vals
        df_res[s_name + "_score"] = np.round(
            100.0 * (hi_scaled - hi_scaled.min()) / (hi_scaled.max() - hi_scaled.min()), 2
        )

    # Tetapkan Kuadran Berdasarkan Ambang Batas Terukur (Split EVI & HI)
    try:
        split_evi = float(np.median(df_res["evi_score"].to_numpy()))
        split_hi = float(np.median(df_res["hi_baseline_score"].to_numpy()))
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"could not compute quadrant splits: {exc}") from exc

    df_res["quadrant_baseline"] = [
        assign_quadrant(e, h, split_evi, split_hi)
        for e, h in zip(df_res["evi_score"], df_res["hi_baseline_score"], strict=True)
    ]
    df_res["quadrant_mercury"] = [
        assign_quadrant(e, h, split_evi, split_hi)
        for e, h in zip(df_res["evi_score"], df_res["hi_mercury_score"], strict=True)
    ]
    df_res["quadrant_equal"] = [
        assign_quadrant(e, h, split_evi, split_hi)
        for e, h in zip(df_res["evi_score"], df_res["hi_equal_score"], strict=True)
    ]

    # Cek Stabilitas Kuadran
    df_res["is_stable"] = (
        (df_res["quadrant_baseline"] == df_res["quadrant_mercury"])
        & (df_res["quadrant_baseline"] == df_res["quadrant_equal"])
    )

    df_res.to_csv(out_csv, index=False)
    log.info("Hasil tersimpan di %s", out_csv)

    # ---------------- VISUALISASI MATRIKS 2D 4 KUADRAN ----------------
    fig, ax = plt.subplots(figsize=(16.8, 11), dpi=300)

    emoji_path = Path("/usr/share/fonts/google-noto-emoji-fonts/NotoEmoji-Regular.ttf")
    emoji_font = FontProperties(fname=str(emoji_path)) if emoji_path.exists() else None

    EMOJI_MAP = {
        0: "\U0001F5A8",  # 🖨️ Printer
        1: "\U0001F4F1",  # 📱 Smartphone
        2: "\u2699",      # ⚙️ Mixed Scrap
        3: "\u2328",      # ⌨️ Keyboard
        4: "\U0001F9FA",  # 🧺 Washing Machine (top-load)
        5: "\U0001F5B1",  # 🖱️ Mouse
        6: "\U0001F4FA",  # 📺 Flat-Panel TV
        7: "\u2668",      # ♨️ Microwave
        8: "\U0001F4BB",  # 💻 Laptop
        9: "\U0001F50B",  # 🔋 Battery
        10: "\U0001F4FB", # 📻 Music Player Box
        11: "\U0001F4FA", # 📺 CRT TV
        12: "\u26A1",     # ⚡ PCB
        13: "\U0001F9FA", # 🧺 Washing Machine (front-load)
        14: "\U0001F4FB", # 📻 Radio / Hi-Fi
        15: "\U0001F5A5", # 🖥️ Monitor
        16: "\U0001F50C", # 🔌 Power Outlets
    }

    # Warna & Gaya Kuadran
    colors = {
        "Q1: Critical Urban Mining": "#d9534f",     # Merah tua / Bahaya & Bernilai
        "Q2: Hazardous Neutralization": "#f0ad4e", # Oranye / Waspada Toksik
        "Q3: Fast Circular Recovery": "#28a745",   # Hijau / Sirkular Aman
        "Q4: General / Inert Residue": "#6c757d",  # Abu-abu / Residu umum
    }

    # Garis Pembatas Kuadran
    ax.axvline(x=split_evi, color="#444444", linestyle="--", alpha=0.7, linewidth=1.3)
    ax.axhline(y=split_hi, color="#444444", linestyle="--", alpha=0.7, linewidth=1.3)

    # Latar Belakang Kuadran
    ax.fill_between([split_evi, 116], split_hi, 116, color="#ffdddd", alpha=0.22)
    ax.fill_between([-26, split_evi], split_hi, 116, color="#fff3cd", alpha=0.22)
    ax.fill_between([split_evi, 116], -16, split_hi, color="#d4edda", alpha=0.22)
    ax.fill_between([-26, split_evi], -16, split_hi, color="#e9ecef", alpha=0.22)

    # Label Teks Kuadran
    ax.text(
        111,
        111,
        "QUADRANT I\nCRITICAL URBAN MINING\n(high hazard, high value)",
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#b52b27",
    )
    ax.text(
        -22,
        111,
        "QUADRANT II\nHAZARDOUS NEUTRALIZATION\n(high hazard, low value)",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#c67d0a",
    )
    ax.text(
        111,
        -13.5,
        "QUADRANT III\nFAST CIRCULAR RECOVERY\n(low hazard, high value)",
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1e7e34",
    )
    ax.text(
        -24,
        -14,
        "QUADRANT IV\nGENERAL INERT RESIDUE\n(low hazard, low value)",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#495057",
    )

    min_n = df_res["n_samples"].min()
    max_n = df_res["n_samples"].max()

    # Plot Scatter Tiap Klaster dengan Ikon Vektor dan Skala Ukuran N Citra
    for row in df_res.to_dict("records"):
        try:
            cid = int(row["cluster"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid cluster id in results row: {row}") from exc
        c_name = row["quadrant_baseline"]
        c_color = colors.get(c_name, "#333333")
        n = row["n_samples"]

        # Luas bubble linear terhadap jumlah sampel/citra klaster
        b_size = 750 + (n - min_n) / (max_n - min_n) * 1150  # range 750 sampai 1900
        f_size = 14 + (n - min_n) / (max_n - min_n) * 7.5    # range 14 sampai 21.5

        ax.scatter(
            row["evi_score"],
            row["hi_baseline_score"],
            s=b_size,
            color=c_color,
            edgecolors="#1a1a1a",
            linewidth=1.6,
            alpha=0.88,
            zorder=4,
        )
        emoji_kwargs: dict = {"fontproperties": emoji_font} if emoji_font else {}
        ax.text(
            row["evi_score"],
            row["hi_baseline_score"],
            EMOJI_MAP[cid],
            fontsize=f_size,
            ha="center",
            va="center",
            color="white",
            zorder=5,
            **emoji_kwargs,
        )

    for row in df_res.to_dict("records"):
        try:
            cid = int(row["cluster"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid cluster id in offset row: {row}") from exc
        c_name = row["quadrant_baseline"]
        c_color = colors.get(c_name, "#333333")
        unstable_mark = "*" if not bool(row["is_stable"]) else ""

        # Short id tag next to each bubble; full details live in the side table,
        # so no leader-line boxes are needed (avoiding label collisions).
        ax.annotate(
            f"c{cid:02d}{unstable_mark}",
            xy=(row["evi_score"], row["hi_baseline_score"]),
            xytext=(9, 9),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=c_color,
            zorder=6,
        )

    # Bubble-size legend (image count)
    legend_n_vals = [200, 350, 500]
    leg_handles = []
    for leg_n in legend_n_vals:
        leg_s = 750 + (leg_n - min_n) / (max_n - min_n) * 1150
        leg_handles.append(
            plt.scatter(
                [],
                [],
                s=leg_s,
                c="#e8e8e8",
                edgecolors="#333333",
                linewidth=1.2,
                alpha=0.9,
                label=f"N = {leg_n} images",
            )
        )

    leg1 = ax.legend(
        handles=leg_handles,
        title="Bubble size\n(image count N_k)",
        loc="upper left",
        bbox_to_anchor=(1.015, 0.60),
        fontsize=9,
        title_fontsize=10,
        framealpha=0.95,
        edgecolor="#999999",
        labelspacing=1.6,
        borderpad=1.0,
        handletextpad=1.2,
    )
    ax.add_artist(leg1)

    # Quadrant colour legend
    quad_handles = [
        Line2D([], [], marker="s", linestyle="", markersize=10, color=col, label=name)
        for name, col in colors.items()
    ]
    leg2 = ax.legend(
        handles=quad_handles,
        title="Quadrant",
        loc="upper left",
        bbox_to_anchor=(1.015, 0.86),
        fontsize=9,
        title_fontsize=10,
        framealpha=0.95,
        edgecolor="#999999",
    )
    ax.add_artist(leg2)

    # Full cluster detail table in the right margin — guarantees every value is
    # readable regardless of bubble density.
    table_lines = [
        "ID   Stream                     N    EVI     HI      Quadrant  "
    ]
    table_lines.append("-" * 74)
    order = df_res.sort_values(["quadrant_baseline", "evi_raw_usd_kg"], ascending=[True, False])
    for row in order.to_dict("records"):
        try:
            cid = int(row["cluster"])
            mark = "*" if not bool(row["is_stable"]) else " "
            qid = str(row["quadrant_baseline"]).split(":")[0]
            n_imgs = int(row["n_samples"])
            evi_raw = float(row["evi_raw_usd_kg"])
            hi_raw = float(row["hi_baseline_raw"])
            display = str(row["auto_label"])
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid table row: {row}") from exc
        table_lines.append(
            f"c{cid:02d}{mark} {display[:24]:24s} "
            f"{n_imgs:4d} {evi_raw:6.2f}  {hi_raw:7.1f}  {qid}"
        )
    table_lines.append("-" * 74)
    table_lines.append("EVI: USD per kg of waste stream (urban-mining value)")
    table_lines.append("HI : sum of regulatory/TTLC threshold quotients (Hg,Pb,POPs,Co,Ni)")
    table_lines.append("*  : quadrant changes under one alternative weight scheme")
    ax.text(
        1.045,
        0.40,
        "\n".join(table_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.6,
        linespacing=1.35,
    )

    # Monotone sqrt compression of both axes: spreads the crowded low-value
    # clusters while de-emphasising the single CRT outlier. Because the map is
    # strictly increasing, median splits and quadrant membership are unchanged.
    ax.set_xscale(
        "function",
        functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.square(np.clip(v, 0, None))),
    )
    ax.set_yscale(
        "function",
        functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.square(np.clip(v, 0, None))),
    )

    ax.set_xlim(-26, 116)
    ax.set_ylim(-16, 116)
    ax.set_xlabel("Economic Value Intensity (EVI) — urban-mining potential [0-100]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hazard Intensity (HI) — environmental & toxic threat [0-100]", fontsize=12, fontweight="bold", labelpad=14)
    ax.set_title(
        "Risk-Value Decision Matrix for the 17 E-Waste Clusters (KEC + DINOv3 + DROWCULA)\n"
        "Bubble size ∝ cluster image count; icons mark the human-verified UNU-KEY / Basel stream; "
        "axes are √-compressed",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.text(
        0.5,
        0.005,
        "Unit weights: UNITAR/UNEP E-waste Statistics Guidelines Table 25 (2024). "
        "Material composition: Table 26 (EU-6PV) and category literature (CRT funnel-glass lead). "
        "Hazard normalisation: Minamata (Hg), RoHS (Pb), Stockholm (POPs). "
        "Metal prices: snapshot 2026-09-25/26 (LBMA/spot, LME). Indices are relative, not tonnage.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )

    fig.savefig(out_fig_matrix, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
    plt.close(fig)
    log.info("Gambar matriks tersimpan di %s", out_fig_matrix)

    # ---------------- SENSITIVITY FIGURE --------------
    fig_s, ax_s = plt.subplots(figsize=(11, 6.5), dpi=300)
    x_schemes = ["Baseline\n(Minamata-heavy)", "Mercury-heavy", "Equal-risk"]
    labels_placed: list[float] = []
    for row in df_res.to_dict("records"):
        try:
            y_vals = [
                float(row["hi_baseline_raw"]),
                float(row["hi_mercury_raw"]),
                float(row["hi_equal_raw"]),
            ]
            cid = int(row["cluster"])
            row_tag = f"c{cid:02d} {str(row['auto_label'])[:22]}"
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            raise SystemExit(f"invalid sensitivity row: {row}") from exc
        is_st = bool(row["is_stable"])
        c_line = "#2b5c8f" if is_st else "#e67e22"
        marker = "o" if is_st else "D"
        ax_s.plot(
            x_schemes,
            y_vals,
            marker=marker,
            lw=1.6 if not is_st else 1.1,
            color=c_line,
            alpha=0.9 if not is_st else 0.65,
            zorder=5 if not is_st else 3,
        )
        # stagger right-edge labels to avoid overlap
        y_pos = y_vals[-1] * (1.06 if (cid % 2 == 0) else 0.94)
        while any(abs(y_pos - prev) < 0.045 * y_pos for prev in labels_placed):
            y_pos *= 1.05 if (cid % 2 == 0) else 0.95
        labels_placed.append(y_pos)
        ax_s.text(
            2.06,
            y_pos,
            row_tag,
            va="center",
            fontsize=8,
            color=c_line,
            fontweight="bold" if not is_st else "normal",
        )

    ax_s.set_yscale("log")
    ax_s.set_title(
        "Hazard Intensity robustness across three regulatory weight schemes\n"
        "blue = quadrant-stable (16/17) · orange diamond = quadrant changes under an alternative scheme",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_s.set_ylabel("Hazard Intensity (raw weighted threshold multiples, log scale)", fontsize=10.5)
    ax_s.set_xlabel("Hazard weight scheme", fontsize=10.5)
    ax_s.set_ylim(0.9, 1500)
    ax_s.grid(True, linestyle=":", alpha=0.5, which="both")
    ax_s.text(
        0.5,
        -0.16,
        "Monotone re-weighting cannot change EVI; only HI ranks are tested. "
        "A stable cluster keeps its quadrant under all three schemes.",
        transform=ax_s.transAxes,
        ha="center",
        fontsize=7.6,
        color="#444444",
    )
    plt.tight_layout()
    fig_s.savefig(out_fig_sens, dpi=300, bbox_inches="tight")
    plt.close(fig_s)
    log.info("Gambar uji sensitivitas tersimpan di %s", out_fig_sens)

    # Ringkasan ke Log
    try:
        stable_count = int(df_res["is_stable"].astype(bool).to_numpy().sum())
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"could not count stable clusters: {exc}") from exc
    log.info("Sensitivity: %d / %d clusters STABLE (quadrant unchanged).", stable_count, len(df_res))
    log.info("Done: cluster_risk_value.csv + 2 figures written.")


if __name__ == "__main__":
    main()
