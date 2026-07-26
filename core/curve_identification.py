"""
Curve Identification Engine
===========================

Single source of truth for wireline curve mnemonic recognition and metadata.

Architecture overview
---------------------
- CurveInfo: Immutable per-curve recognition record used by the UI/pipeline.
- build_mnemonic_database(): One reconciled mnemonic table with provenance
  (`source` on every entry). No parallel copies.
- CurveIdentificationEngine: Matching pipeline (exact / Levenshtein fuzzy /
  unit+description context / pattern fallback) plus industry metadata APIs
  (wavelet, color, track scale, log-scale, family lookup) and duplicate
  resolution.

Data provenance policy
----------------------
Entries are tagged with a `source` string. Allowed styles:
  - "CWLS reference" / "observed in demo_data/POLISH_DEMO_1.las"
  - "SLB chartbook common mnemonic" (widely published tool mnemonics)
  - "industry common alias" (cross-vendor aliases used in field LAS files)
  - "legacy POLISH library (unverified)" only when retained intentionally

Concatenated multi-tool strings (e.g. RLLD_HRLA_AT90_AHT90) and synthetic
numbered variants (GR_1..GR_100) are intentionally excluded: they appeared in
the initial polish10 commit without an external source.

Backward-compatible aliases
---------------------------
ComprehensiveMnemonicLibrary and ComprehensiveCurveManager both resolve to
CurveIdentificationEngine so existing imports keep working during migration.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from petrophysics.constants import PetrophysicalConstants
except Exception:  # pragma: no cover - optional during isolated unit tests
    PetrophysicalConstants = None  # type: ignore


OHM_M_UNITS = ['OHMM', 'ohm.m', 'OHM-M']


@dataclass
class CurveInfo:
    """Immutable curve information with full recognition data."""

    curve_name: str
    curve_type: str = 'UNKNOWN'
    unit: str = ''
    description: str = ''
    type_confidence: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False

    curve_family: str = 'unknown'
    physics_type: str = ''
    typical_range: Tuple[float, float] = (0.0, 1.0)
    log_scale: bool = False
    industry_color: str = '#000000'
    track_scale: Tuple[float, float] = (0.0, 1.0)
    processing_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.curve_name or not isinstance(self.curve_name, str):
            raise ValueError("Invalid curve name")
        if not 0.0 <= self.type_confidence <= 1.0:
            raise ValueError("Invalid confidence value")

        required_stats = ['count', 'missing', 'missing_percent', 'min', 'max', 'mean', 'std']
        for key in required_stats:
            if key not in self.statistics:
                self.statistics[key] = 0.0


def build_mnemonic_database() -> Dict[str, Dict[str, Any]]:
    """
    Build the single reconciled mnemonic database.

    Each entry includes a `source` field for provenance. Mnemonics are unique
    across curve families (enforced by tests).
    """
    return {
        # === RESISTIVITY FAMILY ===
        'RESISTIVITY_DEEP': {
            # RT observed in demo_data/POLISH_DEMO_1.las; others are SLB/BH common deep tools
            'mnemonics': ['RT', 'ILD', 'LLD', 'RLLD', 'AT90', 'AHT90', 'RT_HRLA', 'RILD', 'RLL3'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 10000],
            'log_scale': True,
            'curve_family': 'resistivity',
            'description': 'Deep investigation resistivity',
            'physics': 'electromagnetic_induction',
            'typical_values': {'shale': [1, 20], 'sand': [10, 1000], 'carbonate': [100, 10000]},
            'industry_color': '#FF0000',
            'track_scale': (0.2, 2000),
            'wavelet_type': 'db8',
            'filter_params': {'bilateral_sigma_s': 10.0, 'bilateral_sigma_r': 0.1},
            'source': 'observed in demo_data/POLISH_DEMO_1.las (RT); SLB chartbook common mnemonic (ILD/LLD/AT90/AHT90)',
        },
        'RESISTIVITY_MEDIUM': {
            'mnemonics': ['ILM', 'LLM', 'RLLM', 'AT60', 'AHT60', 'RT_MRLA', 'RILM', 'RLL2'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 10000],
            'log_scale': True,
            'curve_family': 'resistivity',
            'description': 'Medium investigation resistivity',
            'industry_color': '#FF4444',
            'track_scale': (0.2, 2000),
            'wavelet_type': 'db8',
            'source': 'SLB chartbook common mnemonic (ILM/LLM/AT60/AHT60)',
        },
        'RESISTIVITY_SHALLOW': {
            'mnemonics': ['ILS', 'LLS', 'RLLS', 'AT30', 'AHT30', 'RT_SRLA', 'SFLU', 'MSFL', 'RILS', 'RLL1'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 1000],
            'log_scale': True,
            'curve_family': 'resistivity',
            'description': 'Shallow investigation resistivity',
            'industry_color': '#FF8888',
            'track_scale': (0.2, 1000),
            'wavelet_type': 'db8',
            'source': 'SLB chartbook common mnemonic (ILS/LLS/SFLU/MSFL/AT30)',
        },
        'RESISTIVITY_MICRO': {
            'mnemonics': ['MCFL', 'RXO', 'MCFP', 'MI', 'MIR', 'MIRI', 'MN', 'MNR', 'MNOR'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 1000],
            'log_scale': True,
            'curve_family': 'resistivity',
            'description': 'Micro-resistivity / flushed-zone resistivity',
            'physics': 'electromagnetic_induction',
            'industry_color': '#FFAAAA',
            'track_scale': (0.2, 1000),
            'wavelet_type': 'db8',
            'source': 'SLB chartbook common mnemonic (RXO/MCFL); industry common alias (MI/MN)',
        },
        'RESISTIVITY_LATEROLOG': {
            # Discrete laterolog electrode spacings; LLD/LLM/LLS live in deep/medium/shallow
            'mnemonics': ['LL3', 'LL7', 'LL8', 'LL9'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 10000],
            'log_scale': True,
            'curve_family': 'resistivity',
            'description': 'Laterolog electrode-spacing resistivity',
            'industry_color': '#FF0000',
            'track_scale': (0.2, 2000),
            'wavelet_type': 'db8',
            'source': 'SLB chartbook common mnemonic (LL3/LL7/LL8/LL9)',
        },

        # === GAMMA RAY FAMILY ===
        'GAMMA_RAY_TOTAL': {
            'mnemonics': ['GR', 'GRC', 'GRCX', 'HSGR', 'ECGR', 'SGR', 'TGR'],
            'units': ['GAPI', 'API', 'cps', 'CPS'],
            'range': [0, 500],
            'log_scale': False,
            'curve_family': 'gamma_ray',
            'description': 'Total gamma ray',
            'physics': 'natural_radioactivity',
            'typical_values': {'shale': [80, 200], 'sand': [10, 80], 'carbonate': [5, 50]},
            'industry_color': '#008000',
            'track_scale': (0, 150),
            'wavelet_type': 'db6',
            'filter_params': {'savgol_window': 11, 'savgol_poly': 3},
            'source': 'observed in demo_data/POLISH_DEMO_1.las (GR); SLB chartbook common mnemonic (HSGR/ECGR/SGR)',
        },
        'GAMMA_RAY_SPECTRAL': {
            'mnemonics': ['HCGR'],
            'units': ['GAPI', 'API'],
            'range': [0, 300],
            'curve_family': 'gamma_ray_spectral',
            'description': 'Computed spectral gamma ray (Th+K)',
            'industry_color': '#228B22',
            'track_scale': (0, 150),
            'wavelet_type': 'db6',
            'source': 'SLB chartbook common mnemonic (HCGR)',
        },
        'THORIUM': {
            'mnemonics': ['THOR', 'TH', 'HTHO', 'STHO'],
            'units': ['PPM', 'ppm'],
            'range': [0, 50],
            'curve_family': 'gamma_ray_spectral',
            'description': 'Thorium content',
            'industry_color': '#228B22',
            'track_scale': (0, 50),
            'wavelet_type': 'db6',
            'source': 'SLB chartbook common mnemonic (HTHO/THOR)',
        },
        'URANIUM': {
            'mnemonics': ['URAN', 'U', 'HURA', 'SURA'],
            'units': ['PPM', 'ppm'],
            'range': [0, 20],
            'curve_family': 'gamma_ray_spectral',
            'description': 'Uranium content',
            'industry_color': '#228B22',
            'track_scale': (0, 20),
            'wavelet_type': 'db6',
            'source': 'SLB chartbook common mnemonic (HURA/URAN)',
        },
        'POTASSIUM': {
            'mnemonics': ['POTA', 'K', 'HPOT', 'SPOT'],
            'units': ['%', 'PERCENT'],
            'range': [0, 8],
            'curve_family': 'gamma_ray_spectral',
            'description': 'Potassium content',
            'industry_color': '#228B22',
            'track_scale': (0, 8),
            'wavelet_type': 'db6',
            'source': 'SLB chartbook common mnemonic (HPOT/POTA)',
        },

        # === NEUTRON POROSITY FAMILY ===
        'NEUTRON_POROSITY': {
            'mnemonics': ['NPHI', 'NPOR', 'NEUT', 'TNPH', 'CNL', 'SNPH', 'APLC'],
            'units': ['V/V', 'PU', 'FRAC', '%', 'PERCENT'],
            'range': [-0.15, 0.6],
            'curve_family': 'neutron',
            'description': 'Neutron porosity',
            'physics': 'neutron_hydrogen_interaction',
            'typical_values': {'tight': [0, 0.1], 'reservoir': [0.1, 0.3], 'vuggy': [0.3, 0.6]},
            'industry_color': '#0000FF',
            'track_scale': (0.45, -0.15),
            'wavelet_type': 'coif4',
            'filter_params': {'bilateral_sigma_s': 8.0, 'bilateral_sigma_r': 0.05},
            'source': 'observed in demo_data/POLISH_DEMO_1.las (NPHI); SLB chartbook common mnemonic (TNPH/CNL/NPOR)',
        },
        'NEUTRON_COMPENSATED': {
            'mnemonics': ['CNPOR', 'NPHI_LS', 'NPHI_SS', 'NPHI_DOL'],
            'units': ['V/V', 'PU', 'FRAC'],
            'range': [-0.1, 0.5],
            'curve_family': 'neutron',
            'description': 'Lithology-referenced compensated neutron porosity',
            'industry_color': '#0000FF',
            'track_scale': (0.45, -0.15),
            'wavelet_type': 'coif4',
            'source': 'SLB chartbook common mnemonic (NPHI_LS/NPHI_SS/NPHI_DOL)',
        },
        'NEUTRON_EPITHERMAL': {
            'mnemonics': ['ENPH', 'ETNP', 'ENN'],
            'units': ['V/V', 'PU'],
            'range': [0, 0.6],
            'curve_family': 'neutron',
            'description': 'Epithermal neutron porosity',
            'industry_color': '#0000FF',
            'track_scale': (0.45, -0.15),
            'wavelet_type': 'coif4',
            'source': 'industry common alias (ENPH/ETNP)',
        },
        'NEUTRON_DUAL': {
            'mnemonics': ['DPOR', 'DNPH', 'DNPHI'],
            'units': ['V/V', 'PU', 'FRAC', '%'],
            'range': [-0.15, 0.6],
            'curve_family': 'neutron',
            'description': 'Dual neutron porosity',
            'physics': 'neutron_hydrogen_interaction',
            'industry_color': '#0000FF',
            'track_scale': (0.45, -0.15),
            'wavelet_type': 'coif4',
            'source': 'industry common alias (DPOR/DNPH)',
        },

        # === DENSITY FAMILY ===
        'BULK_DENSITY': {
            'mnemonics': ['RHOB', 'RHOZ', 'DENB', 'DENS', 'ROHB', 'ZDEN', 'BDCN'],
            'units': ['G/C3', 'g/cm3', 'G/CM3', 'G/CC', 'KG/M3'],
            'range': [1.0, 3.5],
            'curve_family': 'density',
            'description': 'Formation bulk density',
            'physics': 'gamma_ray_compton_scattering',
            'typical_values': {'gas': [1.8, 2.2], 'oil': [2.0, 2.4], 'water': [2.2, 2.8]},
            'industry_color': '#FF0000',
            'track_scale': (1.95, 2.95),
            'wavelet_type': 'db4',
            'filter_params': {'median_kernel': 5},
            'source': 'observed in demo_data/POLISH_DEMO_1.las (RHOB); SLB chartbook common mnemonic (RHOZ/ZDEN)',
        },
        'PHOTOELECTRIC_FACTOR': {
            'mnemonics': ['PEF', 'PE', 'PEFZ', 'PEFC', 'ZPE', 'PEFS', 'PEFE', 'PEFR'],
            'units': ['B/E', 'b/e', 'BARNS/ELECTRON', 'BARN/E'],
            'range': [0.0, 20.0],
            'curve_family': 'photoelectric',
            'description': 'Photoelectric absorption factor',
            'physics': 'photoelectric_absorption',
            'typical_values': {'sandstone': [1.6, 1.8], 'limestone': [5.0, 5.1], 'dolomite': [3.1, 3.2]},
            'industry_color': '#FF00FF',
            'track_scale': (0, 20),
            'wavelet_type': 'sym5',
            'source': 'SLB chartbook common mnemonic (PEF/PEFZ); industry common alias (PEFE/PEFR)',
        },
        'DENSITY_CORRECTION': {
            'mnemonics': ['DRHO', 'DRHB', 'DCOR', 'RHOC'],
            'units': ['G/C3', 'g/cm3', 'G/CC'],
            'range': [-0.5, 0.5],
            'curve_family': 'density',
            'description': 'Density correction',
            'industry_color': '#AA0000',
            'track_scale': (-0.25, 0.25),
            'wavelet_type': 'db4',
            'source': 'SLB chartbook common mnemonic (DRHO)',
        },

        # === SONIC FAMILY ===
        'SONIC_COMPRESSIONAL': {
            # DTSM is shear (SLB); kept only under SONIC_SHEAR
            'mnemonics': ['DT', 'DTC', 'DTCO', 'AC', 'DTLN'],
            'units': ['US/F', 'us/ft', 'USEC/FT', 'US/FT'],
            'range': [40, 200],
            'curve_family': 'sonic',
            'description': 'Compressional transit time',
            'physics': 'acoustic_wave_propagation',
            'industry_color': '#800080',
            'track_scale': (140, 40),
            'wavelet_type': 'bior4.4',
            'filter_params': {'bilateral_sigma_s': 12.0, 'bilateral_sigma_r': 0.2},
            'source': 'observed in demo_data/POLISH_DEMO_1.las (DT); SLB chartbook common mnemonic (DTC/DTCO)',
        },
        'SONIC_SHEAR': {
            'mnemonics': ['DTS', 'DTSH', 'DTSM'],
            'units': ['US/F', 'us/ft', 'USEC/FT', 'US/FT'],
            'range': [80, 400],
            'curve_family': 'sonic',
            'description': 'Shear transit time',
            'industry_color': '#9932CC',
            'track_scale': (240, 40),
            'wavelet_type': 'bior4.4',
            'source': 'SLB chartbook common mnemonic (DTS/DTSH/DTSM)',
        },
        'SONIC_STONELEY': {
            'mnemonics': ['DTST', 'DTSTM', 'DTTU'],
            'units': ['US/F', 'us/ft', 'US/FT'],
            'range': [100, 500],
            'curve_family': 'sonic',
            'description': 'Stoneley wave transit time',
            'industry_color': '#8B008B',
            'track_scale': (400, 100),
            'wavelet_type': 'bior4.4',
            'source': 'SLB chartbook common mnemonic (DTST)',
        },

        # === SPONTANEOUS POTENTIAL ===
        'SPONTANEOUS_POTENTIAL': {
            'mnemonics': ['SP', 'SSP', 'PSP', 'SPONT', 'SPC', 'SPCX', 'SPLG'],
            'units': ['MV', 'mV', 'MILLIVOLT'],
            'range': [-200, 200],
            'curve_family': 'spontaneous_potential',
            'description': 'Spontaneous potential',
            'physics': 'electrochemical_potential',
            'typical_values': {'shale': [-20, 0], 'sand': [-100, -20], 'carbonate': [-50, 0]},
            'industry_color': '#FFA500',
            'track_scale': (-200, 100),
            'wavelet_type': 'db6',
            'source': 'industry common alias (SP/SSP/SPC); merged from both legacy databases',
        },

        # === CALIPER FAMILY ===
        'CALIPER': {
            'mnemonics': ['CALI', 'CAL', 'HCAL', 'BS', 'CALS', 'CALX', 'CALL', 'CALM', 'DCAL', 'MCAL', 'MCALI', 'C1', 'C2', 'C3', 'C4'],
            'units': ['IN', 'in', 'INCH', 'MM', 'CM'],
            'range': [4, 24],
            'curve_family': 'caliper',
            'description': 'Borehole caliper',
            'physics': 'mechanical_measurement',
            'typical_values': {'in_gauge': [6, 8.5], 'out_of_gauge': [8.5, 16]},
            'industry_color': '#000000',
            'track_scale': (6, 16),
            'wavelet_type': 'db4',
            'source': 'observed in demo_data/POLISH_DEMO_1.las (CALI); SLB/industry common alias (HCAL/DCAL/CALS)',
        },

        # === DEPTH REFERENCE ===
        'DEPTH': {
            'mnemonics': ['DEPT', 'DEPTH', 'MD', 'MDEPTH', 'TVD', 'TVDEPTH', 'TVDSS', 'KB', 'DF'],
            'units': ['M', 'FT', 'ft', 'FEET', 'METER', 'METERS'],
            'range': [0, 50000],
            'curve_family': 'depth',
            'description': 'Depth measurement',
            'physics': 'depth_reference',
            'industry_color': '#000000',
            'track_scale': (0, 10000),
            'wavelet_type': 'db2',
            'source': 'CWLS reference / observed in demo_data/POLISH_DEMO_1.las (DEPT); industry common alias (TVD/TVDSS/KB/DF)',
        },

        # === ADVANCED LOGGING TOOLS ===
        'NMR_POROSITY': {
            'mnemonics': ['MPHI', 'TCMR', 'CMRP', 'NMR_POR'],
            'units': ['V/V', 'PU', '%'],
            'range': [0, 0.4],
            'curve_family': 'nmr',
            'description': 'NMR total porosity',
            'industry_color': '#4169E1',
            'track_scale': (0.4, 0.0),
            'wavelet_type': 'coif4',
            'source': 'SLB chartbook common mnemonic (TCMR/MPHI)',
        },
        'NMR_PERMEABILITY': {
            'mnemonics': ['MPERM', 'KPERM', 'KINT'],
            'units': ['MD', 'mD', 'MILLIDARCY'],
            'range': [0.001, 10000],
            'log_scale': True,
            'curve_family': 'nmr',
            'description': 'NMR permeability',
            'industry_color': '#4169E1',
            'track_scale': (0.01, 10000),
            'wavelet_type': 'coif4',
            'source': 'industry common alias (MPERM/KPERM)',
        },
        'FORMATION_PRESSURE': {
            'mnemonics': ['PRES', 'FP', 'FPRES', 'PFOR'],
            'units': ['PSI', 'PA', 'BAR', 'KPA'],
            'range': [0, 20000],
            'curve_family': 'pressure',
            'description': 'Formation pressure',
            'industry_color': '#FF4500',
            'track_scale': (0, 20000),
            'wavelet_type': 'db4',
            'source': 'industry common alias (PRES/FP)',
        },
        'FORMATION_TEMPERATURE': {
            'mnemonics': ['TEMP', 'FTEMP', 'TEMF'],
            'units': ['DEGC', 'DEGF', 'F', 'C'],
            'range': [20, 200],
            'curve_family': 'temperature',
            'description': 'Formation temperature',
            'industry_color': '#FF6347',
            'track_scale': (20, 200),
            'wavelet_type': 'db4',
            'source': 'industry common alias (TEMP/FTEMP)',
        },

        # === BOREHOLE GEOMETRY ===
        'BOREHOLE_AZIMUTH': {
            'mnemonics': ['AZIM', 'AZI', 'HAZI'],
            'units': ['DEG', 'DEGREE'],
            'range': [0, 360],
            'curve_family': 'geometry',
            'description': 'Borehole azimuth',
            'industry_color': '#2F4F4F',
            'track_scale': (0, 360),
            'wavelet_type': 'db4',
            'source': 'industry common alias (AZIM/HAZI)',
        },
        'BOREHOLE_DEVIATION': {
            'mnemonics': ['DEVI', 'DEV', 'HDEV'],
            'units': ['DEG', 'DEGREE'],
            'range': [0, 90],
            'curve_family': 'geometry',
            'description': 'Borehole deviation',
            'industry_color': '#2F4F4F',
            'track_scale': (0, 90),
            'wavelet_type': 'db4',
            'source': 'industry common alias (DEVI/HDEV)',
        },

        # === IMAGING ===
        'FORMATION_RESISTIVITY_IMAGING': {
            'mnemonics': ['FMI', 'HRLA', 'OBMI', 'STAR'],
            'units': list(OHM_M_UNITS),
            'range': [0.1, 10000],
            'log_scale': True,
            'curve_family': 'imaging',
            'description': 'Formation micro-resistivity imaging',
            'industry_color': '#2F4F4F',
            'track_scale': (0.2, 2000),
            'wavelet_type': 'db4',
            'source': 'SLB chartbook common mnemonic (FMI/HRLA/OBMI)',
        },
        'ACOUSTIC_IMAGING': {
            'mnemonics': ['BHTV', 'UBI', 'CBIL'],
            'units': ['DB', 'AMP'],
            'range': [0, 100],
            'curve_family': 'imaging',
            'description': 'Acoustic borehole imaging',
            'industry_color': '#2F4F4F',
            'track_scale': (0, 100),
            'wavelet_type': 'db4',
            'source': 'SLB chartbook common mnemonic (UBI/BHTV)',
        },

        # === GEOCHEMICAL ===
        'CARBON_OXYGEN_RATIO': {
            'mnemonics': ['COR', 'C/O', 'CARB'],
            'units': ['RATIO', 'V/V'],
            'range': [0, 2],
            'curve_family': 'geochemical',
            'description': 'Carbon/Oxygen ratio',
            'industry_color': '#8B4513',
            'track_scale': (0, 2),
            'wavelet_type': 'db4',
            'source': 'industry common alias (COR/C/O)',
        },
        'SILICON_CALCIUM_RATIO': {
            'mnemonics': ['SICA', 'SI/CA', 'SILI'],
            'units': ['RATIO'],
            'range': [0, 10],
            'curve_family': 'geochemical',
            'description': 'Silicon/Calcium ratio',
            'industry_color': '#8B4513',
            'track_scale': (0, 10),
            'wavelet_type': 'db4',
            'source': 'industry common alias (SICA/SI/CA)',
        },
    }


class CurveIdentificationEngine:
    """
    Unified curve identification and metadata engine.

    Combines the mnemonic matching pipeline formerly in ComprehensiveMnemonicLibrary
    with the metadata / duplicate-resolution APIs formerly in ComprehensiveCurveManager,
    backed by a single mnemonic database.
    """

    def __init__(self) -> None:
        self.mnemonic_database = build_mnemonic_database()
        self._curve_info: Dict[str, CurveInfo] = {}
        self._lock = threading.RLock()
        self.physical_constants = PetrophysicalConstants() if PetrophysicalConstants else None

    def _build_comprehensive_database(self) -> Dict[str, Dict[str, Any]]:
        """Compatibility shim used by legacy code that called the private builder."""
        return build_mnemonic_database()

    # ------------------------------------------------------------------
    # Matching helpers (from ComprehensiveMnemonicLibrary)
    # ------------------------------------------------------------------

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _fuzzy_match_mnemonic(self, mnemonic: str, known_mnemonic: str, threshold: float = 0.7) -> Tuple[bool, float]:
        mnemonic_clean = mnemonic.upper().strip()
        known_clean = known_mnemonic.upper().strip()

        if mnemonic_clean == known_clean:
            return True, 1.0

        max_len = max(len(mnemonic_clean), len(known_clean))
        if max_len == 0:
            return False, 0.0

        distance = self._levenshtein_distance(mnemonic_clean, known_clean)
        similarity = 1.0 - (distance / max_len)
        return similarity >= threshold, similarity

    def _context_aware_recognition(
        self,
        unit: str,
        value_range: Optional[Tuple[float, float]],
        curve_data: Dict[str, Any],
    ) -> float:
        confidence_boost = 0.0
        unit_clean = unit.upper().strip() if unit else ''

        if unit_clean:
            curve_units = [u.upper() for u in curve_data.get('units', [])]
            if unit_clean in curve_units:
                confidence_boost += 0.15
            else:
                for cu in curve_units:
                    if unit_clean.replace('CM3', 'C3').replace('CM', 'C') in cu or \
                       cu.replace('CM3', 'C3').replace('CM', 'C') in unit_clean:
                        confidence_boost += 0.1
                        break

        if value_range:
            curve_range = curve_data.get('range', [])
            if len(curve_range) == 2:
                min_val, max_val = value_range
                curve_min, curve_max = curve_range
                overlap_min = max(min_val, curve_min)
                overlap_max = min(max_val, curve_max)
                if overlap_max > overlap_min:
                    denom = max(max_val, curve_max) - min(min_val, curve_min)
                    if denom > 0:
                        confidence_boost += ((overlap_max - overlap_min) / denom) * 0.2

        return confidence_boost

    def _pattern_based_identification(
        self,
        data: Optional[np.ndarray],
        curve_data: Dict[str, Any],
    ) -> float:
        if data is None or len(data) < 20:
            return 0.0

        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 10:
            return 0.0

        confidence_boost = 0.0
        curve_range = curve_data.get('range', [])
        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))

        if len(curve_range) == 2:
            curve_min, curve_max = curve_range
            if curve_min <= data_min <= curve_max and curve_min <= data_max <= curve_max:
                confidence_boost += 0.15
            elif curve_min * 0.5 <= data_min <= curve_max * 2.0:
                confidence_boost += 0.05

        if curve_data.get('log_scale', False) and data_min > 0 and data_max > 0:
            if data_max / data_min > 10:
                confidence_boost += 0.1

        return confidence_boost

    def _correlation_analysis(
        self,
        mnemonic: str,
        curve_data: Dict[str, Any],
        auxiliary_curves: Dict[str, np.ndarray],
    ) -> float:
        curve_family = curve_data.get('curve_family', '')
        confidence_boost = 0.0
        family_correlations = {
            'resistivity': ['GR', 'SP', 'RHOB'],
            'density': ['NPHI', 'GR', 'DT'],
            'neutron': ['RHOB', 'GR', 'DT'],
            'gamma_ray': ['SP', 'RHOB'],
            'sonic': ['RHOB', 'NPHI'],
        }
        expected_curves = family_correlations.get(curve_family, [])
        for aux_name in auxiliary_curves:
            if any(exp in aux_name.upper() for exp in expected_curves):
                confidence_boost += 0.05
        return min(0.15, confidence_boost)

    def _resolve_conflict(
        self,
        primary: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        unit: str,
        value_range: Optional[Tuple[float, float]],
        all_curve_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        all_candidates = [primary] + alternatives

        exact_matches = [c for c in all_candidates if c['method'] == 'exact']
        if len(exact_matches) == 1:
            return exact_matches[0]

        if all_curve_names:
            suite_scores: Dict[str, int] = {}
            families = {
                'RESISTIVITY_DEEP': ['RESISTIVITY_MEDIUM', 'RESISTIVITY_SHALLOW'],
                'RESISTIVITY_MEDIUM': ['RESISTIVITY_DEEP', 'RESISTIVITY_SHALLOW'],
                'RESISTIVITY_SHALLOW': ['RESISTIVITY_DEEP', 'RESISTIVITY_MEDIUM'],
                'NEUTRON_POROSITY': ['BULK_DENSITY', 'PHOTOELECTRIC_FACTOR'],
                'BULK_DENSITY': ['NEUTRON_POROSITY', 'PHOTOELECTRIC_FACTOR'],
                'GAMMA_RAY_TOTAL': ['SPONTANEOUS_POTENTIAL'],
                'GAMMA_RAY_SPECTRAL': ['THORIUM', 'URANIUM', 'POTASSIUM'],
            }
            for candidate in all_candidates:
                score = 0
                related_types = families.get(candidate['curve_type'], [])
                for curve_name in all_curve_names:
                    name_upper = curve_name.upper()
                    for related_type in related_types:
                        keywords = related_type.lower().split('_')
                        if any(kw in name_upper.lower() for kw in keywords):
                            score += 1
                suite_scores[candidate['curve_type']] = score
            if suite_scores:
                max_score = max(suite_scores.values())
                if max_score > 0:
                    best_types = [t for t, s in suite_scores.items() if s == max_score]
                    if len(best_types) == 1:
                        return next(c for c in all_candidates if c['curve_type'] == best_types[0])

        unit_clean = unit.upper().strip() if unit else ''
        if unit_clean:
            unit_matches = []
            for candidate in all_candidates:
                curve_units = [u.upper() for u in candidate['curve_data'].get('units', [])]
                if unit_clean in curve_units:
                    unit_matches.append(candidate)
            if len(unit_matches) == 1:
                return unit_matches[0]
            if unit_matches:
                all_candidates = unit_matches

        if value_range:
            best_overlap = 0.0
            best_candidate = primary
            for candidate in all_candidates:
                curve_range = candidate['curve_data'].get('range', [])
                if len(curve_range) == 2:
                    min_val, max_val = value_range
                    curve_min, curve_max = curve_range
                    overlap_min = max(min_val, curve_min)
                    overlap_max = min(max_val, curve_max)
                    if overlap_max > overlap_min:
                        range_span = max(max_val, curve_max) - min(min_val, curve_min)
                        overlap = (overlap_max - overlap_min) / range_span if range_span > 0 else 0
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_candidate = candidate
            if best_overlap > 0.5:
                return best_candidate

        return max(all_candidates, key=lambda x: x['confidence'])

    def identify_curve(
        self,
        mnemonic: str,
        unit: str = '',
        description: str = '',
        data: Optional[np.ndarray] = None,
        value_range: Optional[Tuple[float, float]] = None,
        auxiliary_curves: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Identify curve type with confidence using the unified matching pipeline."""
        mnemonic_clean = mnemonic.upper().strip()
        mnemonic_normalized = (
            mnemonic_clean.replace('.', '').replace('_', '').replace('-', '').replace(' ', '')
        )
        unit_clean = unit.upper().strip() if unit else ''
        desc_clean = description.upper().strip() if description else ''

        candidates: List[Dict[str, Any]] = []

        for curve_type, curve_data in self.mnemonic_database.items():
            confidence = 0.0
            match_method = 'none'

            curve_mnemonics = [m.upper() for m in curve_data.get('mnemonics', [])]
            if mnemonic_clean in curve_mnemonics:
                confidence = 0.95
                match_method = 'exact'
            else:
                curve_normalized = [
                    m.upper().replace('.', '').replace('_', '').replace('-', '').replace(' ', '')
                    for m in curve_data.get('mnemonics', [])
                ]
                if mnemonic_normalized in curve_normalized:
                    confidence = 0.9
                    match_method = 'normalized'
                else:
                    for known_mnemonic in curve_data.get('mnemonics', []):
                        matched, similarity = self._fuzzy_match_mnemonic(
                            mnemonic_clean, known_mnemonic, threshold=0.7
                        )
                        if matched:
                            confidence = max(confidence, 0.75 * similarity)
                            match_method = 'fuzzy'
                            break

            confidence += self._context_aware_recognition(unit_clean, value_range, curve_data)
            if data is not None:
                confidence += self._pattern_based_identification(data, curve_data)

            if desc_clean:
                curve_desc = curve_data.get('description', '').upper()
                matches = sum(1 for word in curve_desc.split() if word in desc_clean)
                confidence += min(0.05, matches * 0.01)

            if auxiliary_curves:
                confidence += self._correlation_analysis(mnemonic_clean, curve_data, auxiliary_curves)

            confidence = min(1.0, confidence)
            if confidence > 0.3:
                candidates.append({
                    'curve_type': curve_type,
                    'confidence': confidence,
                    'method': match_method,
                    'curve_data': curve_data.copy(),
                })

        if not candidates or max(c['confidence'] for c in candidates) < 0.5:
            for curve_type, curve_data in self.mnemonic_database.items():
                for known_mnemonic in curve_data.get('mnemonics', []):
                    known_clean = known_mnemonic.upper().strip()
                    known_normalized = (
                        known_clean.replace('.', '').replace('_', '').replace('-', '').replace(' ', '')
                    )
                    confidence = 0.0
                    if mnemonic_clean == known_clean:
                        confidence = 0.85
                    elif mnemonic_normalized == known_normalized:
                        confidence = 0.8
                    elif mnemonic_clean in known_clean and len(mnemonic_clean) >= 3:
                        confidence = 0.7
                    elif known_clean in mnemonic_clean and len(known_clean) >= 3:
                        confidence = 0.7
                    elif mnemonic_normalized in known_normalized and len(mnemonic_normalized) >= 3:
                        confidence = 0.65
                    elif known_normalized in mnemonic_normalized and len(known_normalized) >= 3:
                        confidence = 0.65
                    else:
                        continue

                    confidence += self._context_aware_recognition(unit_clean, value_range, curve_data)
                    confidence = min(1.0, confidence)
                    candidates.append({
                        'curve_type': curve_type,
                        'confidence': confidence,
                        'method': 'partial',
                        'curve_data': curve_data.copy(),
                    })
                    break

        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        if len(candidates) > 1 and candidates[0]['confidence'] > 0.7:
            top_confidence = candidates[0]['confidence']
            alternatives = [c for c in candidates[1:] if c['confidence'] >= top_confidence * 0.9]
            if alternatives:
                all_curve_names = list(auxiliary_curves.keys()) if auxiliary_curves else []
                resolved = self._resolve_conflict(
                    candidates[0], alternatives, unit_clean, value_range, all_curve_names=all_curve_names
                )
                return resolved['curve_type'], resolved['confidence'], resolved['curve_data']

        if candidates:
            best = candidates[0]
            return best['curve_type'], best['confidence'], best['curve_data']

        return 'UNKNOWN', 0.0, {}

    def validate_curve_identification(
        self, curve_name: str, identified_type: str, confidence: float
    ) -> Dict[str, Any]:
        if identified_type in self.mnemonic_database:
            return {'valid': True, 'confidence_level': 'GOOD'}
        return {'valid': False, 'confidence_level': 'LOW'}

    def get_curve_processing_parameters(self, curve_type: str) -> Dict[str, Any]:
        entry = self.mnemonic_database.get(curve_type, {})
        return {
            'gap_filling_threshold': 100,
            'denoising_method': 'auto',
            'wavelet_type': entry.get('wavelet_type', 'db4'),
            'filter_params': entry.get('filter_params', {}),
        }

    # ------------------------------------------------------------------
    # Metadata / duplicate APIs (from ComprehensiveCurveManager)
    # ------------------------------------------------------------------

    def detect_and_resolve_duplicates(self, identified_curves: Dict[str, Any]) -> Dict[str, Any]:
        type_mapping: Dict[str, List[Dict[str, Any]]] = {}
        for curve_name, curve_info in identified_curves.items():
            if hasattr(curve_info, 'curve_type'):
                curve_type = curve_info.curve_type
                confidence = curve_info.type_confidence
                stats = curve_info.statistics if hasattr(curve_info, 'statistics') else {}
                unit = curve_info.unit if hasattr(curve_info, 'unit') else ''
            else:
                curve_type = curve_info.get('curve_type', 'UNKNOWN')
                confidence = curve_info.get('type_confidence', 0.0)
                stats = curve_info.get('statistics', {})
                unit = curve_info.get('unit', '')

            if curve_type == 'UNKNOWN':
                continue

            type_mapping.setdefault(curve_type, []).append({
                'name': curve_name,
                'confidence': confidence,
                'missing_pct': stats.get('missing_percent', 100) if isinstance(stats, dict) else 100,
                'unit': unit,
            })

        duplicates = {
            curve_type: curves
            for curve_type, curves in type_mapping.items()
            if len(curves) > 1
        }
        if not duplicates:
            return {'duplicates_found': {}, 'resolution_needed': [], 'auto_resolved': {}}

        auto_resolved: Dict[str, str] = {}
        needs_user_input: List[str] = []

        for curve_type, candidates in duplicates.items():
            if len(candidates) == 2:
                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: (x['confidence'], -x['missing_pct']),
                    reverse=True,
                )
                best, second = sorted_candidates[0], sorted_candidates[1]
                if (
                    best['confidence'] - second['confidence'] > 0.15
                    or second['missing_pct'] - best['missing_pct'] > 30
                    or (best['missing_pct'] < 10 and second['missing_pct'] > 40)
                ):
                    auto_resolved[curve_type] = best['name']
                else:
                    needs_user_input.append(curve_type)
            else:
                needs_user_input.append(curve_type)

        return {
            'duplicates_found': duplicates,
            'resolution_needed': needs_user_input,
            'auto_resolved': auto_resolved,
        }

    def create_comprehensive_curve_info(
        self, curve_name: str, unit: str = '', description: str = ''
    ) -> CurveInfo:
        with self._lock:
            curve_type, confidence, curve_data = self.identify_curve(curve_name, unit, description)
            typical_range = tuple(curve_data.get('range', [0.0, 1.0]))
            curve_family = curve_data.get('curve_family', 'unknown')
            processing_params = {
                'wavelet_type': curve_data.get('wavelet_type', 'db4'),
                'filter_params': curve_data.get('filter_params', {}),
                'typical_values': curve_data.get('typical_values', {}),
                'gap_fill_params': {
                    'max_gap_size': 100 if curve_family in ['resistivity', 'gamma_ray'] else 50,
                    'confidence_threshold': 0.8,
                    'method_priority': ['gaussian_process', 'cubic_spline', 'linear'],
                },
                'source': curve_data.get('source', ''),
            }
            curve_info = CurveInfo(
                curve_name=curve_name,
                curve_type=curve_type,
                unit=unit or (curve_data.get('units', [''])[0] if curve_data.get('units') else ''),
                description=description or curve_data.get('description', f'Curve {curve_name}'),
                type_confidence=confidence,
                curve_family=curve_family,
                physics_type=curve_data.get('physics', ''),
                typical_range=typical_range,  # type: ignore[arg-type]
                log_scale=curve_data.get('log_scale', False),
                industry_color=curve_data.get('industry_color', '#000000'),
                track_scale=tuple(curve_data.get('track_scale', typical_range)),  # type: ignore[arg-type]
                processing_params=processing_params,
            )
            self._curve_info[curve_name] = curve_info
            return curve_info

    def get_curve_info(self, curve_name: str) -> CurveInfo:
        with self._lock:
            if curve_name not in self._curve_info:
                return self.create_comprehensive_curve_info(curve_name)
            return self._curve_info[curve_name]

    def get_processing_params_for_curve(self, curve_name: str) -> Dict[str, Any]:
        return self.get_curve_info(curve_name).processing_params

    def get_optimal_wavelet_for_curve(self, curve_name: str) -> str:
        return self.get_curve_info(curve_name).processing_params.get('wavelet_type', 'db4')

    def get_industry_color_for_curve(self, curve_name: str) -> str:
        return self.get_curve_info(curve_name).industry_color

    def get_track_scale_for_curve(self, curve_name: str) -> Tuple[float, float]:
        return self.get_curve_info(curve_name).track_scale

    def is_log_scale_curve(self, curve_name: str) -> bool:
        return self.get_curve_info(curve_name).log_scale

    def get_curves_by_family(self, family: str) -> Dict[str, CurveInfo]:
        with self._lock:
            return {
                name: info
                for name, info in self._curve_info.items()
                if info.curve_family == family
            }

    def validate_curve_range(self, curve_name: str, data: np.ndarray) -> Dict[str, Any]:
        curve_info = self.get_curve_info(curve_name)
        min_expected, max_expected = curve_info.typical_range
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return {'valid': False, 'reason': 'no_valid_data'}

        min_actual = float(np.min(valid_data))
        max_actual = float(np.max(valid_data))
        tolerance_factor = 2.0
        range_valid = (min_expected / tolerance_factor) <= min_actual and max_actual <= (
            max_expected * tolerance_factor
        )
        return {
            'valid': range_valid,
            'expected_range': (min_expected, max_expected),
            'actual_range': (min_actual, max_actual),
            'confidence': curve_info.type_confidence,
            'curve_type': curve_info.curve_type,
        }


# Backward-compatible aliases (thin wrappers / same class)
ComprehensiveMnemonicLibrary = CurveIdentificationEngine
ComprehensiveCurveManager = CurveIdentificationEngine
