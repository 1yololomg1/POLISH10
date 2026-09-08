# === CRASH DIAGNOSTICS ===
# Installed before the scientific stack is imported. The failure mode this
# guards against is an exception raised while importing numpy/matplotlib in a
# packaged build, which happens before any application code runs and which a
# windowed executable can otherwise only report as an unreadable dialog.
# The reporter depends on the standard library alone, so it cannot itself be a
# casualty of a broken dependency. It writes locally and never uses the network.
from core import crash_report  # noqa: E402

crash_report.write_startup_record()
crash_report.install_global_handlers()

# === CONSTANTS FOR DUPLICATED LITERALS ===
OHM_M_UNITS = ['OHMM', 'ohm.m', 'OHM-M']

# === ERROR MESSAGE CONSTANTS ===
ERROR_TITLE_MEMORY = "Memory Error"
ERROR_TITLE_DATA = "Data Error"
ERROR_TITLE_FILE = "File Error"
ERROR_TITLE_PROCESSING = "Processing Error"
ERROR_TITLE_SECURITY = "Security Error"
ERROR_TITLE_VISUALIZATION = "Visualization Error"

# === UI CONSTANTS (canonical definitions in ui.constants) ===
from ui.constants import (  # noqa: E402
    FONT_DEFAULT,
    FONT_SIZE_DEFAULT,
    FONT_SIZE_HEADING,
    FONT_SIZE_LARGE,
)

# === UI STRING CONSTANTS ===
LABEL_DEPTH_M = "Depth (m)"
LABEL_PROCESSED_DATA = "Processed Data"
LABEL_ORIGINAL_DATA = "Original Data"
LABEL_MISSING_DATA = "Missing Data"
LABEL_SIGNIFICANT_CHANGES = "Significant Changes"
LABEL_OFFSET_POINTS = "Offset Points"
LABEL_NUMBER_OF_CURVES = "Number of Curves"
LABEL_UPPER_RIGHT = "upper right"
LABEL_UPPER_LEFT = "upper left"

# === PROCESSING THRESHOLD CONSTANTS ===
GAP_THRESHOLD_GEOLOGICAL = 200  # Points for geological gap classification
GAP_THRESHOLD_LARGE = 500  # Points for large gap threshold (default)
GAP_THRESHOLD_MAX = 1000  # Maximum gap size for processing
MEMORY_LIMIT_DEFAULT_MB = 2048  # Default memory limit in MB
QUALITY_THRESHOLD_LOW = 0.5
QUALITY_THRESHOLD_MEDIUM = 0.7
QUALITY_THRESHOLD_HIGH = 0.9

# === UI EVENT / DIALOG CONSTANTS (canonical definitions in ui.constants) ===
from ui.constants import (  # noqa: E402
    EVENT_CONFIGURE,
    EVENT_MOUSEWHEEL,
    EVENT_SHIFT_MOUSEWHEEL,
    DIALOG_SELECT_ALL,
    DIALOG_DESELECT_ALL,
    DIALOG_ROOT_WINDOW,
)
# ===== COPY MODULE IMPORT (auto-inserted to fix NameError) =====
import copy
# ===== PSUTIL MODULE AVAILABILITY CHECK (auto-inserted to fix NameError) =====
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass
# ===== LASIO MODULE AVAILABILITY CHECK (auto-inserted to fix NameError) =====
LASIO_AVAILABLE = False
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    pass
# ===== QUEUE MODULE IMPORT (auto-inserted to fix NameError) =====
import queue
# ===== BETA SYSTEM FLAG (auto-inserted to prevent NameError) =====
BETA_SYSTEM_AVAILABLE = False
"""
Advanced Wireline Data Preprocessing System
Overview: Production-grade preprocessing, gap filling, denoising, QC, visualization, and reporting for wireline data.

Core capabilities implemented in this codebase:
- Gap filling: linear, cubic spline, Gaussian Process, kriging, polynomial, multi-curve correlation
- Denoising: wavelet (when available), bilateral, Savitzky–Golay, median
- Curve identification: mnemonic-driven recognition with curve info/parameters
- QC: range validation, outlier detection (IQR), completeness metrics
- Visualization: depth-based plots, comparison, uncertainty, correlation, multi-curve, industry log display
- Reporting: processing report and LAS previews (original/processed)
- UI/UX: Tkinter app with tabs for Data, Processing, Visualization, and Report

ARCHITECTURE OVERVIEW:

MAIN CLASSES AND RESPONSIBILITIES
- AdvancedPreprocessingApplication: Tk application root; orchestrates UI, data load, processing, visualization, reports.
- PetrophysicalButtons: Centralized UI component factory for consistent styling.
- CurveIdentificationEngine: Unified curve identification, mnemonic database, and metadata.
- AdvancedGapFiller: Gap detection and filling (linear, spline, GP, kriging, polynomial, multi-curve).
- AdvancedSignalProcessor: Denoising and smoothing operations.
- ScaleAwareProcessor: Scale-aware processing for log-normal/bounded/normal/discrete curves.
- DepthValidationManager/ReservoirDepthManager: Depth curve validation and reference standardization.
- GeologicalZoneManager/ZoneAwareGapFiller: Boundary detection and zone-aware gap filling.
- PetrophysicalRelationshipValidator: Post-processing physics/relationship validation.
- EnvironmentalCorrectionsManager: Borehole, temperature, and tool corrections.
- SecureVisualizationManager/ThreadSafeVisualizationManager: Robust figure/canvas lifecycle and thread-safe plotting.
- IndustryUnitStandardizer: Unit standardization UI and conversions; upload-time fractional standardization.
- ProcessingHistoryManager: Undo/redo and operation tracking.
- ArchieEquationCalculator/RelativeRockPropertiesModel: Petrophysical computations and RRP-based gap support.
- BetaFeatureFlags/BetaAnalytics/BetaFeedbackCollector: Optional beta system (gated by BETA_SYSTEM_AVAILABLE).

KEY FUNCTIONS
- main(): Application entry point; constructs and runs AdvancedPreprocessingApplication.
- load_file(): Loads LAS/CSV/Excel; analyze_curves(); then optional standardize_fractional_curves_on_upload(); updates previews/UI.
- analyze_curves(), ensure_curve_statistics(), update_data_display(): Curve identification, stats, and UI refresh.
- start_processing() / process_data_thread(): End-to-end pipeline: depth validation → optional normalization → geological zone detection → environmental corrections → uniformization → gap filling → denoising → relationship validation → final uniformization → previews.
- standardize_fractional_curves_on_upload(): Targeted %→v/v conversion for porosity/saturation/volume/probability families.
- Plotting functions: plot_comparison(), plot_uncertainty(), plot_quality_metrics(), plot_correlation_matrix(), plot_scatter(), plot_3d_visualization(), plot_multi_curve(), plot_log_display().
- Reporting/Export: create_comprehensive_report(), generate_report(), export_data(), previews for original/processed LAS.

DATA FLOW
Startup → Tk root/init → setup_ui() → (optional) startup dialog sets standardize-on-upload → user loads file → analyze_curves() → optional upload standardization (%→v/v) → display/preview update → processing pipeline (depth validation → zone detection → corrections → uniformization → gap/denoise → validation) → processed previews and report.
"""

# At the VERY TOP of your file, before any other matplotlib imports
import matplotlib
matplotlib.use('TkAgg')  # Must come BEFORE importing pyplot

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure seaborn for professional well logging visualizations
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    try:
        # Older versions
        from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk
    except ImportError:
        NavigationToolbar2Tk = None  # Fallback: handle missing toolbar gracefully
from matplotlib.figure import Figure
try:
    from mpl_toolkits.mplot3d import Axes3D  # Optional 3D toolkit
except Exception:
    Axes3D = None
import threading
import time
import os
import gc

import random
import json
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
import sys
import platform

# ===== PLACEHOLDER/BACKSTOP CLASSES FOR OPTIONAL SYSTEMS =====
# These lightweight definitions ensure static type checkers (pyright) and
# runtime both have symbols available even when optional subsystems are
# disabled or not bundled. They introduce no side effects.

class BetaFeatureFlags:
    """Placeholder for beta feature flags when beta system is disabled."""
    def __init__(self):
        pass


class BetaAnalytics:
    """Placeholder analytics sink used when beta analytics is not enabled."""
    def __init__(self, feature_flags: Optional["BetaFeatureFlags"] = None):
        self._feature_flags = feature_flags


class BetaFeedbackCollector:
    """Placeholder feedback collector used when beta feedback is not enabled."""
    def __init__(self, feature_flags: Optional["BetaFeatureFlags"] = None):
        self._feature_flags = feature_flags


class SafeFileHandler:
    """Robust file operations wrapper used by analytics and reporting paths.
    
    Implemented here to avoid optional import failures and undefined symbol
    errors during static analysis. Methods are conservative and avoid raising
    exceptions; they return simple success/failure signals.
    
    Enhanced with security validation functions for path traversal protection
    and file size limits.
    
    Debug logging can be enabled for development/debugging purposes via
    enable_debug_mode() method. When disabled (default), all methods fail
    silently as designed for security-critical operations.
    """

    # Security constants
    MAX_FILE_SIZE_MB = 500.0  # Maximum file size in MB
    ALLOWED_READ_EXTENSIONS = ['.las', '.csv', '.xlsx', '.xls', '.dlis', '.lis']
    ALLOWED_WRITE_EXTENSIONS = ['.las', '.csv', '.xlsx']
    
    # Debug mode flag (disabled by default for production security)
    _DEBUG_MODE = False
    _DEBUG_LOGGER = None  # Optional logger function (e.g., app.log_processing)
    
    @classmethod
    def enable_debug_mode(cls, logger_func=None):
        """Enable debug logging for SafeFileHandler operations.
        
        Args:
            logger_func: Optional logging function (e.g., app.log_processing).
                        If None, uses Python's logging module.
        """
        cls._DEBUG_MODE = True
        cls._DEBUG_LOGGER = logger_func
    
    @classmethod
    def disable_debug_mode(cls):
        """Disable debug logging (default production behavior)."""
        cls._DEBUG_MODE = False
        cls._DEBUG_LOGGER = None
    
    @classmethod
    def _debug_log(cls, message: str, category: str = "SafeFileHandler"):
        """Internal debug logging method.
        
        Only logs when debug mode is enabled. Does not expose sensitive
        path information or security details.
        
        Args:
            message: Log message
            category: Log category/prefix
        """
        if not cls._DEBUG_MODE:
            return
        
        log_msg = f"[{category}] {message}"
        
        # Use provided logger function if available
        if cls._DEBUG_LOGGER:
            try:
                cls._DEBUG_LOGGER(log_msg)
                return
            except Exception:
                # Logger failed - fall through to standard logging
                pass
        
        # Fallback to Python logging module
        try:
            import logging
            logging.debug(log_msg)
        except Exception:
            # Logging not available - silently ignore (fail-safe)
            pass

    @staticmethod
    def validate_file_path(filepath: str, allowed_dir: str = None) -> Optional["Path"]:
        """Safely normalize and validate file paths to prevent path traversal attacks.
        
        Args:
            filepath: Path to validate
            allowed_dir: Optional directory to restrict paths within
            
        Returns:
            Normalized Path object if valid, None if invalid
        """
        try:
            from pathlib import Path
            
            # Normalize path (resolves .., ., symlinks, etc.)
            path = Path(filepath).resolve()
            
            # Check if path exists
            if not path.exists():
                SafeFileHandler._debug_log(
                    f"Path validation failed: path does not exist (sanitized: {SafeFileHandler.sanitize_path_for_display(str(path))})",
                    "PathValidation"
                )
                return None
            
            # If allowed_dir specified, ensure path is within it (for export operations)
            if allowed_dir:
                try:
                    allowed = Path(allowed_dir).resolve()
                    # Check if path is within allowed directory
                    path.relative_to(allowed)
                except ValueError:
                    # Path is outside allowed directory - security violation
                    SafeFileHandler._debug_log(
                        f"Security violation: path outside allowed directory (sanitized: {SafeFileHandler.sanitize_path_for_display(str(path))})",
                        "SecurityViolation"
                    )
                    return None
            
            return path
        except Exception as e:
            # Any exception during path validation is a security concern
            SafeFileHandler._debug_log(
                f"Path validation exception: {type(e).__name__} (sanitized path: {SafeFileHandler.sanitize_path_for_display(str(filepath))})",
                "PathValidationError"
            )
            return None
    
    @staticmethod
    def validate_file_size(filepath: str, max_size_mb: float = None) -> bool:
        """Validate file size before loading to prevent memory exhaustion.
        
        Args:
            filepath: Path to file to check
            max_size_mb: Maximum size in MB (defaults to MAX_FILE_SIZE_MB)
            
        Returns:
            True if file size is acceptable, False otherwise
        """
        try:
            if max_size_mb is None:
                max_size_mb = SafeFileHandler.MAX_FILE_SIZE_MB
            
            if not os.path.exists(filepath):
                SafeFileHandler._debug_log(
                    f"File size validation failed: file does not exist (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                    "FileSizeValidation"
                )
                return False
            
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            is_valid = size_mb <= max_size_mb
            
            if not is_valid:
                SafeFileHandler._debug_log(
                    f"File size validation failed: {size_mb:.2f}MB > {max_size_mb:.2f}MB (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                    "FileSizeValidation"
                )
            
            return is_valid
        except Exception as e:
            # If we can't determine size, be conservative
            SafeFileHandler._debug_log(
                f"File size validation exception: {type(e).__name__} (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                "FileSizeValidationError"
            )
            return False
    
    @staticmethod
    def validate_file_extension(filepath: str, allowed_extensions: list = None, mode: str = 'read') -> bool:
        """Validate file extension against allowed list and check for double extensions.
        
        Args:
            filepath: Path to validate
            allowed_extensions: List of allowed extensions (defaults based on mode)
            mode: 'read' or 'write' to determine default extensions
            
        Returns:
            True if extension is valid, False otherwise
        """
        try:
            from pathlib import Path
            
            path = Path(filepath)
            ext = path.suffix.lower()
            
            # Get default extensions if not provided
            if allowed_extensions is None:
                if mode == 'read':
                    allowed_extensions = SafeFileHandler.ALLOWED_READ_EXTENSIONS
                else:
                    allowed_extensions = SafeFileHandler.ALLOWED_WRITE_EXTENSIONS
            
            # Check exact match
            if ext not in allowed_extensions:
                SafeFileHandler._debug_log(
                    f"Extension validation failed: '{ext}' not in allowed list {allowed_extensions} (mode: {mode}, sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                    "ExtensionValidation"
                )
                return False
            
            # Check for double extensions (security concern: file.txt.las)
            # Get the stem and check if it has another extension
            stem_ext = Path(path.stem).suffix.lower()
            if stem_ext and stem_ext in ['.txt', '.bak', '.tmp', '.old']:
                # Suspicious: has a hidden extension
                SafeFileHandler._debug_log(
                    f"Security violation: suspicious double extension detected '{stem_ext}' + '{ext}' (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                    "SecurityViolation"
                )
                return False
            
            return True
        except Exception as e:
            SafeFileHandler._debug_log(
                f"Extension validation exception: {type(e).__name__} (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                "ExtensionValidationError"
            )
            return False
    
    @staticmethod
    def sanitize_path_for_display(filepath: str) -> str:
        """Sanitize file paths for user display (privacy protection).
        
        Args:
            filepath: Path to sanitize
            
        Returns:
            Sanitized path showing only last two directory levels
        """
        try:
            from pathlib import Path
            path = Path(filepath)
            if len(path.parts) <= 2:
                return str(path)
            # Show only last two levels
            return f".../{path.parent.name}/{path.name}"
        except Exception as e:
            # Path sanitization failure - log in debug mode only
            SafeFileHandler._debug_log(
                f"Path sanitization exception: {type(e).__name__}",
                "PathSanitization"
            )
            return "..."

    @staticmethod
    def safe_write_json(filepath: str, data: Any) -> bool:
        """Safely write JSON data to file.
        
        Args:
            filepath: Path to JSON file
            data: Data to write (must be JSON serializable)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            SafeFileHandler._debug_log(
                f"JSON write failed: {type(e).__name__} - {str(e)} (sanitized: {SafeFileHandler.sanitize_path_for_display(filepath)})",
                "JSONWriteError"
            )
            return False

    @staticmethod
    def safe_read_json(filepath: str) -> Optional[Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(
                f"JSON file reading failed for '{filepath}': {str(e)}. "
                f"This may be due to invalid JSON format, file not found, "
                f"encoding issues, or permissions. Check file format and accessibility.",
                UserWarning
            )
            return None



# Scientific computing libraries with individual checks
SCIPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
PYWT_AVAILABLE = False

try:
    from scipy import signal, interpolate, stats, optimize, spatial
    from scipy.ndimage import gaussian_filter1d, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    pass

ADVANCED_LIBS = SCIPY_AVAILABLE and SKLEARN_AVAILABLE and PYWT_AVAILABLE

warnings.filterwarnings('ignore')
# Use modern seaborn style instead of deprecated seaborn-v0_8
try:
    plt.style.use('seaborn')
except Exception:
    plt.style.use('default')  # Fallback to default style



#=============================================================================
# SCIENTIFIC CONSTANTS AND PHYSICAL PARAMETERS
# Based on industry standards and peer-reviewed literature
#=============================================================================
# NOTE: PetrophysicalConstants has been extracted to petrophysics/constants.py
# Import maintained here for backward compatibility during modularization
from petrophysics.constants import PetrophysicalConstants, PHYSICAL_CONSTANTS, load_basin_parameters, get_basin_names

# Legacy class definition removed - now imported from petrophysics.constants
# Original code preserved in advanced_preprocessing_system10_PRE_PHASE2_BACKUP_*.py

#=============================================================================
# ARCHIE'S EQUATION AND PETROPHYSICAL CALCULATIONS
#=============================================================================

# Legacy class definitions removed - now imported from core.petrophysical_models
# Original code preserved in advanced_preprocessing_system10_PRE_PHASE2_BACKUP_*.py

#=============================================================================
# ARCHIE'S EQUATION AND PETROPHYSICAL CALCULATIONS
#=============================================================================
# NOTE: ArchieEquationCalculator and RelativeRockPropertiesModel have been extracted to core/petrophysical_models.py
# Import maintained here for backward compatibility during modularization
from core.petrophysical_models import ArchieEquationCalculator, RelativeRockPropertiesModel, ARCHIE_CALCULATOR
from core.environmental_corrections import EnvironmentalCorrectionsManager
from core.error_handler import CentralizedErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory

# Legacy class definitions removed - now imported from core.petrophysical_models
# Original code preserved in advanced_preprocessing_system10_PRE_PHASE2_BACKUP_*.py

#=============================================================================
# GAP FILLING AND SIGNAL PROCESSING CLASSES
#=============================================================================
# NOTE: RelativeRockPropertiesModel has been extracted to core/petrophysical_models.py
# Import maintained here for backward compatibility during modularization

#=============================================================================
# CURVE IDENTIFICATION ENGINE (merged mnemonic library + curve manager)
#=============================================================================
# Single source of truth lives in core/curve_identification.py.
# Backward-compatible aliases keep existing attribute names working until
# call sites are migrated to self.curve_identifier.

from core.curve_identification import (
    CurveIdentificationEngine,
    CurveInfo,
    # Backward-compatible aliases for external imports/tests
    ComprehensiveMnemonicLibrary,
    ComprehensiveCurveManager,
    build_mnemonic_database,
)

from core.reporting import StandardizationReporter


# ============================================================================
# ENHANCED SECURE VISUALIZATION AND STATUS MANAGEMENT SYSTEM
# ============================================================================

import weakref
from contextlib import contextmanager
import gc
import threading
import numpy as np
import tkinter as tk
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
try:
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    Axes3D = None
import matplotlib.pyplot as plt

from ui.visualization import SecureVisualizationManager
from ui.status import SecureStatusManager
from ui.log_display_renderer import LogDisplayRenderer
from ui.batch_processing import BatchProcessingManager


#=============================================================================
# ADVANCED GAP FILLING ENGINE - The Stalwart
#=============================================================================
# NOTE: GeologicalContext, GapClassificationResult, GapFillingParameters, and
# AdvancedGapFiller have been extracted to core/gap_filling.py.
# Import maintained here for backward compatibility during modularization.
from core.gap_filling import (
    GeologicalContext,
    GapClassificationResult,
    GapFillingParameters,
    AdvancedGapFiller,
)

@dataclass
class DepthValidationResult:
    """Result of depth curve validation with detailed feedback"""
    is_valid: bool
    failure_reason: str = ""
    failure_details: dict = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)
    
    def get_user_message(self) -> str:
        """Generate user-friendly error message with remediation steps"""
        if self.is_valid:
            return "Depth validation passed"
        
        message = f"Depth Validation Failed: {self.failure_reason}\n\n"
        
        if self.failure_details:
            message += "Details:\n"
            for key, value in self.failure_details.items():
                message += f"  - {key}: {value}\n"
        
        if self.remediation_steps:
            message += "\nRecommended Actions:\n"
            for i, step in enumerate(self.remediation_steps, 1):
                message += f"  {i}. {step}\n"
        
        return message


@dataclass
class LithologyTrackCurves:
    """The GR and SP traces drawn into Track 1 of the industry log display.

    plot_log_display attaches QC indicators to the GR and SP traces only after
    all four tracks have been drawn, but the curve selection and the
    null-sentinel-to-NaN conversion both happen inside _plot_lithology_track.
    Returning the resolved names alongside the converted arrays lets the caller
    reuse exactly what was plotted. Recomputing them at the call site would
    duplicate the selection and conversion logic and allow the QC indicators to
    describe an array that differs from the rendered trace.

    An absent curve is reported as an empty name list with a None array, which
    is the condition the caller tests before adding indicators.
    """

    gr_curves: List[str] = field(default_factory=list)
    gr_data: Optional[np.ndarray] = None
    sp_curves: List[str] = field(default_factory=list)
    sp_data: Optional[np.ndarray] = None


#=============================================================================
# ADVANCED SIGNAL PROCESSING - DENOISING & SMOOTHING
#=============================================================================
# NOTE: AdvancedSignalProcessor has been extracted to core/denoising.py.
# Import maintained here for backward compatibility during modularization.
from core.denoising import AdvancedSignalProcessor

#=============================================================================
# DEPTH VALIDATION MANAGER
#=============================================================================

class DepthValidationManager:
    """Industry-standard depth validation and management"""
    
    def __init__(self, log_processing: Optional[Callable[[str], None]] = None):
        self.required_depth_keywords = ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']
        self.depth_validation_rules = {
            'min_interval': 10.0,      # Minimum 10m interval for reservoir work
            'max_step': 5.0,           # Maximum 5m step size
            'monotonic': True,         # Must be monotonically increasing
            'reasonable_range': (0, 10000)  # 0-10km reasonable depth range
        }
        self.log_processing = log_processing if log_processing is not None else (lambda msg: None)  # No-op if not provided
        self.app = None  # Will be set if app reference is needed

    
    def validate_and_identify_depth(self, data_columns, curve_info, data):
        """Identify and validate depth curve - FAIL if invalid"""
        depth_candidates = []
        
        # Find potential depth curves
        for col in data_columns:
            if any(keyword in col.upper() for keyword in self.required_depth_keywords):
                depth_candidates.append(col)
        
        # If no candidates found in columns, check if DataFrame index contains depth (common with lasio)
        if not depth_candidates and hasattr(data, 'index') and len(data.index) > 0:
            try:
                index_values = data.index.values
                # Ensure numeric and 1D
                if np.issubdtype(np.asarray(index_values).dtype, np.number):
                    index_series = pd.Series(index_values)
                    clean_index = index_series.dropna()
                    if len(clean_index) >= 10 and clean_index.is_monotonic_increasing:
                        total_interval = float(clean_index.max() - clean_index.min())
                        min_range, max_range = self.depth_validation_rules['reasonable_range']
                        if (total_interval >= self.depth_validation_rules['min_interval'] and
                            clean_index.min() >= min_range and clean_index.max() <= max_range):
                            # Insert as a proper depth column at position 0
                            depth_col_name = 'DEPT'
                            if depth_col_name in data.columns:
                                base_name = depth_col_name
                                suffix = 1
                                while f"{base_name}_{suffix}" in data.columns:
                                    suffix += 1
                                depth_col_name = f"{base_name}_{suffix}"
                            data.insert(0, depth_col_name, index_values)
                            depth_candidates.append(depth_col_name)
                            # Log integration-compatible message
                            if hasattr(self, 'log_processing'):
                                try:
                                    self.log_processing(
                                        f"DEPTH FIX: Added depth from DataFrame index as '{depth_col_name}' "
                                        f"(range {clean_index.min():.2f}-{clean_index.max():.2f})"
                                    )
                                except Exception as log_error:
                                    # Logging failed - continue without logging this message
                                    if hasattr(self, 'log_processing'):
                                        try:
                                            self.log_processing(f"Warning: Failed to log depth fix message: {type(log_error).__name__}")
                                        except Exception:
                                            pass  # Can't log logging failure
            except Exception as depth_error:
                # If any issue occurs, use centralized error handler if available
                if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                    self.app.handle_processing_error(
                        depth_error,
                        "Depth validation from index",
                        "Attempting to extract depth from DataFrame index",
                        show_dialog=False
                    )
                elif hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: Depth validation from index failed: {type(depth_error).__name__}: {str(depth_error)}")
                    except Exception:
                        pass  # Can't log logging failure
                # Fall through to original error handling below

        if not depth_candidates:
            raise ValueError(
                "CRITICAL ERROR: No depth curve found in data.\n"
                "Reservoir analysis requires valid depth reference.\n"
                f"Expected curve names: {self.required_depth_keywords}\n"
                "Cannot proceed with fake depth - this would corrupt reservoir analysis."
            )
        
        # Validate each candidate
        valid_depth = None
        validation_failures = {}
        
        for candidate in depth_candidates:
            validation_result = self._validate_depth_curve(data[candidate])
            if validation_result.is_valid:
                valid_depth = candidate
                break
            else:
                validation_failures[candidate] = validation_result
        
        if not valid_depth:
            # Build detailed error message with all validation failures
            error_msg = f"CRITICAL ERROR: Found depth curves {depth_candidates} but none passed validation.\n\n"
            
            for curve_name, result in validation_failures.items():
                error_msg += f"\n{curve_name} VALIDATION FAILURE:\n"
                error_msg += result.get_user_message()
                error_msg += "\n" + "="*60 + "\n"
            
            raise ValueError(error_msg)
        
        return valid_depth
    
    def _validate_depth_curve(self, depth_data):
        """Validate depth curve meets industry standards
        
        Returns:
            DepthValidationResult: Detailed validation result with failure reasons and remediation steps
        """
        clean_depth = depth_data.dropna()
        
        # Check sufficient data points
        if len(clean_depth) < 10:
            return DepthValidationResult(
                is_valid=False,
                failure_reason="Insufficient data points for depth validation",
                failure_details={
                    "Valid points": len(clean_depth),
                    "Minimum required": 10,
                    "Total points": len(depth_data),
                    "Missing points": len(depth_data) - len(clean_depth)
                },
                remediation_steps=[
                    "Check if depth curve is severely corrupted or mostly null values",
                    "Verify correct depth column was selected",
                    "Consider manual data repair if this is the only available depth curve"
                ]
            )
        
        # Check monotonic increasing
        if not clean_depth.is_monotonic_increasing:
            non_monotonic_indices = []
            for i in range(1, len(clean_depth)):
                if clean_depth.iloc[i] <= clean_depth.iloc[i-1]:
                    non_monotonic_indices.append(i)
            
            return DepthValidationResult(
                is_valid=False,
                failure_reason="Depth curve is not monotonically increasing",
                failure_details={
                    "Non-monotonic points": len(non_monotonic_indices),
                    "First violation at index": non_monotonic_indices[0] if non_monotonic_indices else "N/A",
                    "Depth range": f"{clean_depth.min():.2f} to {clean_depth.max():.2f}m"
                },
                remediation_steps=[
                    "Check for depth reversals or duplicated depth values",
                    "Verify depth curve was not corrupted during data transfer",
                    "Consider sorting depth data if order is simply reversed",
                    "Check if multiple logging runs were concatenated incorrectly"
                ]
            )
        
        # Check reasonable interval
        total_interval = clean_depth.max() - clean_depth.min()
        min_interval = self.depth_validation_rules['min_interval']
        if total_interval < min_interval:
            return DepthValidationResult(
                is_valid=False,
                failure_reason="Depth interval too small for reservoir analysis",
                failure_details={
                    "Actual interval": f"{total_interval:.2f}m",
                    "Minimum required": f"{min_interval:.2f}m",
                    "Start depth": f"{clean_depth.min():.2f}m",
                    "End depth": f"{clean_depth.max():.2f}m"
                },
                remediation_steps=[
                    f"Ensure logged interval is at least {min_interval}m for meaningful analysis",
                    "Check if this is core data (different requirements) or incomplete log",
                    "Verify depth units are correct (meters vs feet)",
                    "Consider if this is a test log or calibration run"
                ]
            )
        
        # Check step sizes (warning only, not failure)
        steps = clean_depth.diff().dropna()
        max_step = steps.max()
        max_allowed_step = self.depth_validation_rules['max_step']
        if max_step > max_allowed_step:
            warnings.warn(
                f"Large depth step detected: {max_step:.2f}m (max recommended: {max_allowed_step:.2f}m). "
                f"This may indicate gaps in logging or tool malfunctions.",
                UserWarning
            )
        
        # Check reasonable range
        min_range, max_range = self.depth_validation_rules['reasonable_range']
        if clean_depth.min() < min_range or clean_depth.max() > max_range:
            return DepthValidationResult(
                is_valid=False,
                failure_reason="Depth range outside reasonable limits for wireline logging",
                failure_details={
                    "Actual range": f"{clean_depth.min():.2f} to {clean_depth.max():.2f}m",
                    "Acceptable range": f"{min_range:.2f} to {max_range:.2f}m",
                    "Minimum depth": f"{clean_depth.min():.2f}m (limit: {min_range:.2f}m)",
                    "Maximum depth": f"{clean_depth.max():.2f}m (limit: {max_range:.2f}m)"
                },
                remediation_steps=[
                    "Verify depth units (meters vs feet) - incorrect units are common",
                    "Check if depth reference is correct (KB, GL, MSL, etc.)",
                    "Confirm this is wireline log data, not seismic or other data type",
                    "Review data source for depth datum corrections needed"
                ]
            )
        
        return DepthValidationResult(is_valid=True)

#=============================================================================
# RESERVOIR DEPTH MANAGER
#=============================================================================

class ReservoirDepthManager:
    """Manage depth references for reservoir characterization"""
    
    def __init__(self):
        self.reference_types = {
            'MD': 'Measured Depth',
            'TVD': 'True Vertical Depth', 
            'TVDSS': 'True Vertical Depth Sub-Sea',
            'DEPT': 'Depth (generic)'
        }
    
    def standardize_depth_reference(self, data, curve_info):
        """Establish standard depth reference for reservoir work"""
        
        # Priority order for depth selection (reservoir industry standard)
        priority_order = ['MD', 'DEPT', 'DEPTH', 'TVD', 'TVDSS']
        
        selected_depth = None
        for depth_type in priority_order:
            if depth_type in data.columns:
                selected_depth = depth_type
                break
        
        if not selected_depth:
            raise ValueError("No valid depth reference found")
        
        # Validate and clean depth
        depth_series = data[selected_depth].copy()
        self._clean_depth_series(depth_series)
        
        # Set as primary depth reference
        data['DEPTH_PRIMARY'] = depth_series
        
        # Calculate depth metadata for reservoir work
        depth_metadata = self._calculate_depth_metadata(depth_series)
        
        return selected_depth, depth_metadata
    
    def _clean_depth_series(self, depth_series):
        """Clean depth series for reservoir analysis"""
        # Remove duplicates (keep first occurrence)
        depth_series = depth_series.drop_duplicates(keep='first')
        
        # Fill small gaps in depth (< 1m) with interpolation
        small_gaps = depth_series.isna() & (depth_series.shift(1).notna()) & (depth_series.shift(-1).notna())
        if small_gaps.any():
            depth_series.interpolate(method='linear', inplace=True, limit=2)
        
        return depth_series
    
    def _calculate_depth_metadata(self, depth_series):
        """Calculate depth metadata for reservoir characterization"""
        clean_depth = depth_series.dropna()
        
        return {
            'start_depth': clean_depth.min(),
            'end_depth': clean_depth.max(),
            'total_interval': clean_depth.max() - clean_depth.min(),
            'average_sampling': clean_depth.diff().median(),
            'sample_count': len(clean_depth),
            'depth_type': 'measured_depth',  # Default assumption
            'quality_score': self._assess_depth_quality(clean_depth)
        }
    
    def _assess_depth_quality(self, depth_series):
        """Assess quality of depth series for reservoir work"""
        if len(depth_series) < 10:
            return 0.0
        
        # Check monotonicity
        is_monotonic = depth_series.is_monotonic_increasing
        monotonic_score = 1.0 if is_monotonic else 0.3
        
        # Check sampling consistency
        depth_diffs = depth_series.diff().dropna()
        if len(depth_diffs) > 0:
            sampling_std = depth_diffs.std()
            sampling_mean = depth_diffs.mean()
            consistency_score = max(0.0, 1.0 - (sampling_std / sampling_mean) if sampling_mean > 0 else 0.0)
        else:
            consistency_score = 0.0
        
        # Check reasonable range
        depth_range = depth_series.max() - depth_series.min()
        range_score = min(1.0, depth_range / 100.0)  # Prefer intervals > 100m
        
        # Overall quality score
        quality_score = (monotonic_score * 0.5 + consistency_score * 0.3 + range_score * 0.2)
        
        return min(1.0, max(0.0, quality_score))

#=============================================================================
# GEOLOGICAL ZONE MANAGER
#=============================================================================

class GeologicalZoneManager:
    """Geological zone detection and boundary-aware processing"""
    
    def __init__(self):
        self.zone_detection_params = {
            'gr_threshold_multiplier': 2.5,  # GR changes > 2.5 * std indicate boundaries
            'min_zone_thickness': 3.0,       # Minimum 3m zone thickness
            'boundary_buffer': 1.0           # 1m buffer around boundaries
        }

    
    def detect_geological_boundaries(self, depth, gamma_ray_curve):
        """Detect geological boundaries using gamma ray signature"""
        
        if gamma_ray_curve is None or len(gamma_ray_curve) < 10:
            warnings.warn(
                "Geological boundary detection skipped: Insufficient gamma ray data. "
                "Zone-aware processing will not be available. Minimum 10 valid points required.",
                UserWarning
            )
            return []
        
        # Calculate GR gradient to find sharp changes
        gr_clean = pd.Series(gamma_ray_curve).dropna()
        depth_clean = pd.Series(depth).dropna()
        
        if len(gr_clean) != len(depth_clean):
            # Align depth and GR data
            min_len = min(len(gr_clean), len(depth_clean))
            gr_clean = gr_clean.iloc[:min_len]
            depth_clean = depth_clean.iloc[:min_len]
        
        # Smooth GR to reduce noise before boundary detection
        gr_values = gr_clean.values if hasattr(gr_clean, 'values') else np.asarray(gr_clean)
        if SCIPY_AVAILABLE:
            from scipy.signal import savgol_filter
            gr_smoothed = savgol_filter(gr_values, window_length=5, polyorder=2)
        else:
            kernel = np.ones(5) / 5.0
            gr_smoothed = np.convolve(gr_values, kernel, mode='same')
        
        # Calculate gradient
        gr_gradient = np.gradient(gr_smoothed)
        gradient_threshold = np.std(gr_gradient) * self.zone_detection_params['gr_threshold_multiplier']

        # Find significant gradient changes
        boundary_indices = np.nonzero(np.abs(gr_gradient) > gradient_threshold)[0]
                                                                                
        # Filter boundaries by minimum zone thickness
        filtered_boundaries = self._filter_boundaries_by_thickness(
            boundary_indices, depth_clean, self.zone_detection_params['min_zone_thickness']
        )

        # Convert to depth values
        boundary_depths = [depth_clean.iloc[idx] for idx in filtered_boundaries]

        # Information logging removed
        # System status handled - operation continues
        pass  # f"Detected {len(boundary_depths)} geological boundaries")
        return boundary_depths
    
    def _filter_boundaries_by_thickness(self, boundary_indices, depth_values, min_thickness):
        """Filter out boundaries that create zones thinner than minimum"""
        if len(boundary_indices) < 2:
            return boundary_indices
        
        filtered = [boundary_indices[0]]  # Always keep first boundary
        
        for i in range(1, len(boundary_indices)):
            current_depth = depth_values.iloc[boundary_indices[i]]
            last_kept_depth = depth_values.iloc[filtered[-1]]
            
            if current_depth - last_kept_depth >= min_thickness:
                filtered.append(boundary_indices[i])
        
        return filtered
    
    def create_zone_masks(self, depth, boundary_depths):
        """Create zone masks for boundary-aware processing"""
        zones = []
        depth_array = np.array(depth)
        
        # Create zones between boundaries
        for i in range(len(boundary_depths) + 1):
            if i == 0:
                # First zone: start to first boundary
                start_depth = depth_array[0]
                end_depth = boundary_depths[0] if boundary_depths else depth_array[-1]
            elif i == len(boundary_depths):
                # Last zone: last boundary to end
                start_depth = boundary_depths[-1]
                end_depth = depth_array[-1]
            else:
                # Middle zones: between boundaries
                start_depth = boundary_depths[i-1]
                end_depth = boundary_depths[i]
            
            # Create mask for this zone
            zone_mask = (depth_array >= start_depth) & (depth_array <= end_depth)
            
            zones.append({
                'zone_id': i,
                'start_depth': start_depth,
                'end_depth': end_depth,
                'thickness': end_depth - start_depth,
                'mask': zone_mask,
                'sample_count': np.sum(zone_mask)
            })
        
        return zones

#=============================================================================
# ZONE AWARE GAP FILLER
#=============================================================================

class ZoneAwareGapFiller(AdvancedGapFiller):
    """Gap filling that respects geological boundaries"""
    
    def __init__(self, params, zone_manager, error_callback=None):
        super().__init__(params, error_callback=error_callback)
        self.zone_manager = zone_manager
        
    def fill_gaps_with_zone_awareness(self, data, curve_type, depth, gamma_ray=None, auxiliary_curves=None):
        """Fill gaps while respecting geological boundaries"""
        
        # Detect geological boundaries
        if gamma_ray is not None:
            boundary_depths = self.zone_manager.detect_geological_boundaries(depth, gamma_ray)
            zones = self.zone_manager.create_zone_masks(depth, boundary_depths)
        else:
            warnings.warn(
                "No gamma ray data available for zone detection. "
                "Processing entire dataset as a single zone (no boundary detection).",
                UserWarning
            )
            zones = [{'zone_id': 0, 'mask': np.ones(len(data), dtype=bool)}]
        
        filled_data = data.copy()
        zone_results = []
        
        # Process each zone independently
        for zone in zones:
            zone_mask = zone['mask']
            zone_data = data[zone_mask]
            zone_depth = depth[zone_mask]
            
            if len(zone_data) < 5:  # Skip very small zones
                continue
            
            # Get auxiliary curves for this zone
            zone_aux_curves = {}
            if auxiliary_curves:
                for aux_name, aux_data in auxiliary_curves.items():
                    zone_aux_curves[aux_name] = aux_data[zone_mask]
            
            # Fill gaps within this zone only
            zone_result = super().fill_gaps(zone_data, curve_type, zone_aux_curves)
            
            # Update results for this zone
            filled_data[zone_mask] = zone_result['filled_data']
            zone_results.append({
                'zone_id': zone['zone_id'],
                'depth_range': (zone['start_depth'], zone['end_depth']),
                'result': zone_result
            })
        
        # Combine zone results
        combined_result = self._combine_zone_results(zone_results, filled_data)
        return combined_result
    
    def _combine_zone_results(self, zone_results, filled_data):
        """Combine results from multiple zones"""
        total_gaps_filled = sum(r['result']['quality_metrics'].get('total_gaps_filled', 0) 
                               for r in zone_results)
        
        avg_confidence = np.mean([r['result']['quality_metrics'].get('average_confidence', 0) 
                                 for r in zone_results]) if zone_results else 0
        
        return {
            'filled_data': filled_data,
            'zone_results': zone_results,
            'quality_metrics': {
                'total_gaps_filled': total_gaps_filled,
                'average_confidence': avg_confidence,
                'zones_processed': len(zone_results),
                'method_used': 'zone_aware_ensemble'
            }
        }

#=============================================================================
# CROSS-WELL PRIOR MANAGER
#=============================================================================

class CrossWellPriorManager:
    """Build and serve cross-well priors for expert two-pass workflows.

    Priors are computed per curve and optionally per depth bins (or zones) across
    a selected cohort of wells. Robust statistics (median/IQR) are used to
    minimize sensitivity to outliers and tool mismatches.
    """

    def __init__(self, log_processing: Optional[Callable[[str], None]] = None):
        self.app = None
        self.priors: Dict[str, Any] = {}
        self.log_processing = log_processing if log_processing is not None else (lambda msg: None)  # No-op if not provided

    def set_application_reference(self, app):
        self.app = app

    def _select_cohort_ids(self) -> List[str]:
        if not self.app or not getattr(self.app, 'well_datasets', None):
            return []
        # Manual selection takes precedence when provided
        if self.app.cohort_selected_well_ids:
            return [wid for wid in self.app.cohort_selected_well_ids if wid in self.app.well_datasets and wid != self.app.active_well_id]
        # Auto-select if enabled: all loaded wells except active
        if getattr(self.app, 'auto_select_cohort_var', None) and self.app.auto_select_cohort_var.get():
            return [wid for wid in self.app.well_datasets.keys() if wid != self.app.active_well_id]
        # Otherwise, no cohort
        return []

    def build_priors(self, depth_binned: bool = True) -> Dict[str, Any]:
        """Build cross-well priors from cohort.

        Returns a dict: { curve -> { 'global': stats, 'bins': [ {depth_min,max, stats} ] } }
        """
        priors: Dict[str, Any] = {'curves': {}, 'families': {}}
        try:
            cohort_ids = self._select_cohort_ids()
            if not cohort_ids:
                return {}
            # Collect per-curve and per-family arrays (from processed data for standardization)
            curve_to_series: Dict[str, List[np.ndarray]] = {}
            depth_to_series: Dict[str, List[np.ndarray]] = {}
            family_to_series: Dict[str, List[np.ndarray]] = {}
            family_depth_series: Dict[str, List[np.ndarray]] = {}
            for wid in cohort_ids:
                ds = self.app.well_datasets.get(wid, {})
                df = ds.get('processed_data') if isinstance(ds.get('processed_data'), pd.DataFrame) else ds.get('current_data')
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                depth_col = 'DEPT' if 'DEPT' in df.columns else (df.columns[0] if len(df.columns) else None)
                if depth_col is None:
                    continue
                depth_vals = pd.to_numeric(df[depth_col], errors='coerce').values
                ds_ci = ds.get('curve_info', {}) or {}
                for col in df.columns:
                    if col == depth_col:
                        continue
                    arr = pd.to_numeric(df[col], errors='coerce').values
                    # Determine curve family from curve info if available
                    cinfo = ds_ci.get(col, {}) if isinstance(ds_ci, dict) else {}
                    ctype = str(cinfo.get('curve_type', 'UNKNOWN'))
                    family = ctype.split('_')[0] if '_' in ctype else ctype
                    curve_to_series.setdefault(col, []).append(arr)
                    depth_to_series.setdefault(col, []).append(depth_vals)
                    family_to_series.setdefault(family, []).append(arr)
                    family_depth_series.setdefault(family, []).append(depth_vals)

            # Compute robust global stats and optional depth bins
            def _compute_stats_from_arrays(arrays_list: List[np.ndarray], depths_list: List[np.ndarray], family_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
                try:
                    stacked = np.concatenate([a[~np.isnan(a)] for a in arrays_list if isinstance(a, np.ndarray)])
                    if stacked.size == 0:
                        return None
                    # Detect if resistivity-like (log-space) using family key if provided
                    use_log = family_key is not None and ('RESISTIVITY' in str(family_key).upper())
                    if use_log:
                        stacked_pos = stacked[stacked > 0]
                        if stacked_pos.size == 0:
                            return None
                        stacked_stat = np.log10(stacked_pos)
                    else:
                        stacked_stat = stacked
                    global_stats = {
                        'median': float(np.median(stacked_stat)),
                        'p10': float(np.percentile(stacked_stat, 10)),
                        'p90': float(np.percentile(stacked_stat, 90)),
                        'mean': float(np.mean(stacked_stat)),
                        'std': float(np.std(stacked_stat)),
                        'count': int(stacked_stat.size),
                        'space': 'log' if use_log else 'linear'
                    }
                    entry = {'global': global_stats, 'bins': []}

                    if depth_binned:
                        # Equal-count bins by depth quantiles (5 bins)
                        all_depths = np.concatenate([d for d in depths_list if isinstance(d, np.ndarray)])
                        if all_depths.size > 20:
                            # Use quantiles for equal-count binning
                            try:
                                depth_edges = np.quantile(all_depths[~np.isnan(all_depths)], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            except (ValueError, IndexError) as quantile_error:
                                # Quantile calculation failed - use linear spacing as fallback
                                depth_edges = np.linspace(np.nanmin(all_depths), np.nanmax(all_depths), 6)
                            except Exception as quantile_error:
                                # Unexpected error in quantile calculation - use centralized handler
                                if hasattr(self, 'app') and hasattr(self.app, 'handle_graceful_degradation'):
                                    self.app.handle_graceful_degradation(
                                        quantile_error,
                                        "Quantile calculation for depth binning",
                                        "Using linear spacing fallback - depth binning may be less accurate"
                                    )
                                elif hasattr(self, 'log_processing'):
                                    try:
                                        self.log_processing(f"Warning: Quantile calculation failed, using linear spacing: {type(quantile_error).__name__}")
                                    except Exception:
                                        pass  # Can't log logging failure
                                depth_edges = np.linspace(np.nanmin(all_depths), np.nanmax(all_depths), 6)
                            for b in range(len(depth_edges)-1):
                                dmin, dmax = depth_edges[b], depth_edges[b+1]
                                bin_vals = []
                                for arr, dvals in zip(arrays_list, depths_list):
                                    mask = (dvals >= dmin) & (dvals < dmax)
                                    bin_vals.append(arr[mask])
                                if bin_vals:
                                    bin_stack = np.concatenate([bv[~np.isnan(bv)] for bv in bin_vals if isinstance(bv, np.ndarray)])
                                    if bin_stack.size > 0:
                                        if use_log:
                                            bin_stack_pos = bin_stack[bin_stack > 0]
                                            if bin_stack_pos.size == 0:
                                                continue
                                            bin_stat = np.log10(bin_stack_pos)
                                        else:
                                            bin_stat = bin_stack
                                        entry['bins'].append({
                                            'depth_min': float(dmin),
                                            'depth_max': float(dmax),
                                            'median': float(np.median(bin_stat)),
                                            'p10': float(np.percentile(bin_stat, 10)),
                                            'p90': float(np.percentile(bin_stat, 90)),
                                            'mean': float(np.mean(bin_stat)),
                                            'std': float(np.std(bin_stat)),
                                            'count': int(bin_stat.size)
                                        })
                    return entry
                except (ValueError, TypeError, IndexError) as stats_error:
                    # Statistical computation failed - use centralized handler
                    if hasattr(self, 'app') and hasattr(self.app, 'handle_graceful_degradation'):
                        self.app.handle_graceful_degradation(
                            stats_error,
                            "Statistical computation for prior entry",
                            "Skipping this entry - prior computation continues with remaining entries"
                        )
                    elif hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Stats computation failed for entry: {type(stats_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
                    return None
                except Exception as stats_error:
                    # Unexpected error in statistics computation - use centralized handler
                    if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                        self.app.handle_processing_error(
                            stats_error,
                            "Statistical computation for prior entry",
                            "Computing statistics for cross-well priors",
                            show_dialog=False
                        )
                    elif hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Unexpected error in stats computation: {type(stats_error).__name__}: {str(stats_error)}")
                        except Exception:
                            pass  # Can't log logging failure
                    return None

            # Curves
            for curve, arrays in curve_to_series.items():
                entry = _compute_stats_from_arrays(arrays, depth_to_series.get(curve, []), None)
                if entry:
                    priors['curves'][curve] = entry

            # Families
            for family, arrays in family_to_series.items():
                entry = _compute_stats_from_arrays(arrays, family_depth_series.get(family, []), family)
                if entry:
                    priors['families'][family] = entry
        except (ValueError, TypeError, AttributeError, KeyError) as prior_error:
            # Prior computation failed - use centralized handler
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    prior_error,
                    "Cross-well prior computation",
                    "Computing cross-well statistical priors",
                    show_dialog=False
                )
            elif hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Prior computation failed: {type(prior_error).__name__}: {str(prior_error)}")
                except Exception:
                    pass  # Can't log logging failure
            return {}
        except Exception as prior_error:
            # Unexpected error in prior computation - use centralized handler
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    prior_error,
                    "Cross-well prior computation",
                    "Computing cross-well statistical priors",
                    show_dialog=False
                )
            elif hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error in prior computation: {type(prior_error).__name__}: {str(prior_error)}")
                except Exception:
                    pass  # Can't log logging failure
            return {}
        self.priors = priors
        return priors

    def _bounds_from_entry(self, entry: Dict[str, Any], depth: Optional[float]) -> Tuple[float, float]:
        # Convert from stat-space to linear if needed
        def _to_linear(v: float, space: str) -> float:
            return float(10 ** v) if space == 'log' else float(v)
        if depth is None or not entry['bins']:
            g = entry['global']
            return (_to_linear(g['p10'], g['space']), _to_linear(g['p90'], g['space']))
        for b in entry['bins']:
            if b['depth_min'] <= depth <= b['depth_max']:
                space = entry['global'].get('space', 'linear')
                return (_to_linear(b['p10'], space), _to_linear(b['p90'], space))
        g = entry['global']
        return (_to_linear(g['p10'], g['space']), _to_linear(g['p90'], g['space']))

    def get_bounds_for_curve(self, curve: str, depth: Optional[float] = None, family: Optional[str] = None) -> Optional[Tuple[float, float]]:
        # Try curve-specific
        if isinstance(self.priors, dict) and 'curves' in self.priors and curve in self.priors['curves']:
            return self._bounds_from_entry(self.priors['curves'][curve], depth)
        # Try family-level
        fam = family
        if fam is None and self.app and hasattr(self.app, 'curve_info') and isinstance(self.app.curve_info, dict):
            ctype = str(self.app.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN'))
            fam = ctype.split('_')[0] if '_' in ctype else ctype
        if fam and 'families' in self.priors and fam in self.priors['families']:
            return self._bounds_from_entry(self.priors['families'][fam], depth)
        return None

    def get_bounds_vector_for_curve(self, curve: str, depths: np.ndarray, family: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Return arrays of low/high bounds per depth sample
        if not isinstance(depths, np.ndarray) or depths.size == 0:
            return None
        # Prefer curve entry
        entry = None
        if 'curves' in self.priors and curve in self.priors['curves']:
            entry = self.priors['curves'][curve]
        else:
            fam = family
            if fam is None and self.app and hasattr(self.app, 'curve_info'):
                ctype = str(self.app.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN'))
                fam = ctype.split('_')[0] if '_' in ctype else ctype
            if fam and 'families' in self.priors and fam in self.priors['families']:
                entry = self.priors['families'][fam]
        if entry is None:
            return None
        # Prepare bin edges and convert bounds to linear space
        def _to_linear(v: float, space: str) -> float:
            return float(10 ** v) if space == 'log' else float(v)
        space = entry['global'].get('space', 'linear')
        if entry['bins']:
            edges = [b['depth_min'] for b in entry['bins']] + [entry['bins'][-1]['depth_max']]
            p10 = np.array([b['p10'] for b in entry['bins']], dtype=float)
            p90 = np.array([b['p90'] for b in entry['bins']], dtype=float)
            # Digitize depths
            idx = np.clip(np.digitize(depths, edges) - 1, 0, len(p10) - 1)
            lows = np.array([_to_linear(p10[i], space) for i in idx], dtype=float)
            highs = np.array([_to_linear(p90[i], space) for i in idx], dtype=float)
            return lows, highs
        # Fallback to global
        g = entry['global']
        low = _to_linear(g['p10'], space)
        high = _to_linear(g['p90'], space)
        return np.full_like(depths, low, dtype=float), np.full_like(depths, high, dtype=float)

    def estimate_coherence(self, curve: str, data: np.ndarray, depths: np.ndarray, family: Optional[str] = None) -> Optional[float]:
        # Compute average exp(-0.5*z^2) where z = (x-mean)/std per depth bin vs priors
        if data is None or depths is None or not isinstance(data, np.ndarray) or data.size == 0:
            return None
        entry = None
        if 'curves' in self.priors and curve in self.priors['curves']:
            entry = self.priors['curves'][curve]
        else:
            fam = family
            if fam is None and self.app and hasattr(self.app, 'curve_info'):
                ctype = str(self.app.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN'))
                fam = ctype.split('_')[0] if '_' in ctype else ctype
            if fam and 'families' in self.priors and fam in self.priors['families']:
                entry = self.priors['families'][fam]
        if entry is None:
            return None
        space = entry['global'].get('space', 'linear')
        def _to_stat(v: np.ndarray) -> np.ndarray:
            if space == 'log':
                vp = v.copy()
                vp[vp <= 0] = np.nan
                return np.log10(vp)
            return v
        x = _to_stat(data.astype(float))
        mask_valid = ~np.isnan(x)
        if not np.any(mask_valid):
            return None
        if entry['bins']:
            edges = [b['depth_min'] for b in entry['bins']] + [entry['bins'][-1]['depth_max']]
            means = np.array([b.get('mean', entry['global']['mean']) for b in entry['bins']], dtype=float)
            stds = np.array([b.get('std', entry['global']['std']) for b in entry['bins']], dtype=float)
            idx = np.clip(np.digitize(depths, edges) - 1, 0, len(means) - 1)
            mu = means[idx]
            sd = stds[idx]
        else:
            mu = np.full_like(x, entry['global']['mean'], dtype=float)
            sd = np.full_like(x, max(entry['global']['std'], 1e-6), dtype=float)
        sd = np.where(sd <= 1e-12, 1e-12, sd)
        z = np.zeros_like(x)
        z[mask_valid] = (x[mask_valid] - mu[mask_valid]) / sd[mask_valid]
        coh = np.nanmean(np.exp(-0.5 * (z ** 2)))
        return float(coh)

#=============================================================================
# PETROPHYSICAL RELATIONSHIP VALIDATION
#=============================================================================
# NOTE: PetrophysicalRelationshipValidator has been extracted to
# core/petrophysical_relationship_validation.py.
# Import maintained here for backward compatibility during modularization.
from core.petrophysical_relationship_validation import PetrophysicalRelationshipValidator

#=============================================================================
# LAS STANDARDS COMPLIANCE
#=============================================================================
# NOTE: LASStandardsCompliance has been extracted to core/las_loading.py.
# Import maintained here for backward compatibility during modularization.
from core.las_loading import LASStandardsCompliance

# NOTE: WellLoadingMixin extracted to core/well_loaders.py.
from core.well_loaders import WellLoadingMixin
from ui.app import AppUIMixin


#=============================================================================
# THREAD SAFE VISUALIZATION MANAGER
#=============================================================================

class ThreadSafeVisualizationManager:
    """Thread-safe matplotlib management for professional applications"""
    
    def __init__(self, log_processing: Optional[Callable[[str], None]] = None, app: Optional[Any] = None):
        self._lock = threading.RLock()
        self._main_thread_id = threading.current_thread().ident
        self._visualization_queue = queue.Queue()
        self._cleanup_scheduled = False
        self.log_processing = log_processing if log_processing is not None else (lambda msg: None)  # No-op if not provided
        self.app = app  # App reference for UI operations
        self.root = None  # Will be set if root window reference is needed
        # Do not set the backend here. Backend is configured once at top of file
        plt.ioff()  # Turn off interactive mode for thread-safety
        
    def create_visualization_threadsafe(self, viz_function, *args, **kwargs):
        """Create visualization in thread-safe manner"""
        
        current_thread = threading.current_thread().ident
        
        if current_thread == self._main_thread_id:
            # We're in main thread, safe to proceed
            return viz_function(*args, **kwargs)
        else:
            # We're in worker thread, queue for main thread execution
            return self._queue_visualization(viz_function, *args, **kwargs)
    
    def _queue_visualization(self, viz_function, *args, **kwargs):
        """Queue visualization for main thread execution"""
        result_container = {'result': None, 'exception': None, 'complete': False}

        def wrapper():
            try:
                result_container['result'] = viz_function(*args, **kwargs)
            except Exception as e:
                result_container['exception'] = e
            finally:
                result_container['complete'] = True

        # Queue for main thread
        self._visualization_queue.put(wrapper)

        # Schedule processing in main thread via injected scheduler
        if hasattr(self, '_schedule_on_main') and callable(self._schedule_on_main):
            self._schedule_on_main(self._process_visualization_queue)

        # Wait for completion (with timeout)
        timeout = 30  # 30 seconds timeout
        start_time = time.time()

        while not result_container['complete'] and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if result_container['exception']:
            raise result_container['exception']

        return result_container['result']
    
    def _process_visualization_queue(self):
        """Process queued visualizations in main thread"""
        
        try:
            while not self._visualization_queue.empty():
                viz_function = self._visualization_queue.get_nowait()
                viz_function()
        except queue.Empty:
            pass  # Queue empty is normal, not an error
        except Exception as e:
            warnings.warn(f"Error processing visualization queue: {str(e)}", UserWarning)
            self.log_processing(f"Visualization queue processing error: {e}")
    
    @contextmanager
    def thread_safe_figure_context(self, figsize=(12, 8)):
        """Context manager for thread-safe figure creation"""
        
        with self._lock:
            # Create figure in thread-safe manner
            fig = None
            try:
                if threading.current_thread().ident == self._main_thread_id:
                    fig = plt.figure(figsize=figsize)
                else:
                    # Create without displaying
                    fig = matplotlib.figure.Figure(figsize=figsize)
                
                yield fig
                
            finally:
                if fig is not None:
                    try:
                        plt.close(fig)
                    except (RuntimeError, OSError) as fig_error:
                        # Figure may already be closed - graceful degradation
                        if hasattr(self, 'app') and hasattr(self.app, 'handle_graceful_degradation'):
                            self.app.handle_graceful_degradation(
                                fig_error,
                                "Figure cleanup in thread-safe context",
                                "Figure may already be closed - continuing cleanup"
                            )
                    except Exception as fig_error:
                        # Unexpected error closing figure
                        if hasattr(self, 'app') and hasattr(self.app, 'handle_ui_error'):
                            self.app.handle_ui_error(
                                fig_error,
                                "Figure cleanup in thread-safe context",
                                "matplotlib figure",
                                graceful_degradation=True
                            )
    
    def safe_cleanup_all_figures(self):
        """Safely cleanup all matplotlib figures"""
        
        with self._lock:
            if not self._cleanup_scheduled:
                self._cleanup_scheduled = True
                
                def cleanup():
                    try:
                        plt.close('all')
                        gc.collect()
                    finally:
                        self._cleanup_scheduled = False
                
                if hasattr(self, 'root'):
                    self.root.after_idle(cleanup)
                else:
                    cleanup()

#=============================================================================
# SCALE AWARE PROCESSOR
#=============================================================================

class ScaleAwareProcessor:
    """Process curves according to their physical scale and distribution"""
    
    def __init__(self):
        self.curve_scales = {
            'log_normal': ['RT', 'RM', 'RS', 'RXO', 'PERM'],  # Resistivity, permeability
            'bounded': ['NPHI', 'PHIE', 'SW', 'SHG'],         # Porosity, saturation (0-1)
            'normal': ['GR', 'SP', 'DT', 'PE'],               # Most other curves
            'discrete': ['FACIES', 'LITH', 'FLAG']            # Discrete classifications
        }
        
        self.processing_methods = {
            'log_normal': {
                'gap_fill': 'log_space_interpolation',
                'denoise': 'log_space_smoothing',
                'outlier_detect': 'log_space_iqr'
            },
            'bounded': {
                'gap_fill': 'constrained_interpolation',
                'denoise': 'edge_preserving',
                'outlier_detect': 'bounded_iqr'
            },
            'normal': {
                'gap_fill': 'standard_interpolation',
                'denoise': 'gaussian_smoothing',
                'outlier_detect': 'standard_iqr'
            },
            'discrete': {
                'gap_fill': 'mode_fill',
                'denoise': 'none',
                'outlier_detect': 'none'
            }
        }
    
    def determine_curve_scale(self, curve_name, data):
        """Determine the appropriate scale/distribution for a curve"""
        
        # Check explicit assignments first
        for scale_type, curve_list in self.curve_scales.items():
            if any(curve_pat in curve_name.upper() for curve_pat in curve_list):
                return scale_type
        
        # Statistical determination for unknown curves
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 10:
            return 'normal'  # Default
        
        # Check if discrete
        unique_values = len(np.unique(clean_data))
        if unique_values < 10 and np.allclose(clean_data, np.round(clean_data)):
            return 'discrete'
        
        # Check if bounded (0-1 range suggests porosity/saturation)
        if np.all(clean_data >= 0) and np.all(clean_data <= 1):
            return 'bounded'
        
        # Check if log-normal (positive skew, multiplicative nature)
        if np.all(clean_data > 0):
            if SCIPY_AVAILABLE:
                skewness = stats.skew(clean_data)
            else:
                d = clean_data - np.mean(clean_data)
                std = np.std(clean_data)
                skewness = np.mean(d ** 3) / (std ** 3) if std > 0 else 0.0
            if skewness > 1.5:  # Highly skewed suggests log-normal
                return 'log_normal'
        
        return 'normal'
    
    def process_curve_scale_aware(self, curve_name, data, processing_type):
        """Process curve according to its scale characteristics"""
        
        scale_type = self.determine_curve_scale(curve_name, data)
        method = self.processing_methods[scale_type][processing_type]
        
        if scale_type == 'discrete':
            # Don't process discrete curves
            return data, {'method': 'none', 'scale_type': scale_type}
        
        # Apply scale-appropriate processing
        if method == 'log_space_interpolation':
            result = self._log_space_gap_fill(data)
        elif method == 'constrained_interpolation':
            result = self._bounded_gap_fill(data)
        elif method == 'log_space_smoothing':
            result = self._log_space_denoise(data)
        elif method == 'edge_preserving':
            result = self._edge_preserving_denoise(data)
        else:
            result = data  # Standard processing
        
        return result, {'method': method, 'scale_type': scale_type}
    
    def _log_space_gap_fill(self, data):
        """Gap filling in log space for log-normal data"""
        
        # Transform to log space
        positive_data = np.maximum(data, 0.01)  # Avoid log(0)
        log_data = np.log10(positive_data)
        
        # Fill gaps in log space
        filled_log = self._interpolate_gaps(log_data)
        
        # Transform back
        result = 10 ** filled_log
        
        return result
    
    def _bounded_gap_fill(self, data):
        """Gap filling for bounded data (0-1 range)"""
        
        # Use logit transform for better interpolation
        epsilon = 1e-6
        bounded_data = np.clip(data, epsilon, 1 - epsilon)
        
        # Logit transform
        logit_data = np.log(bounded_data / (1 - bounded_data))
        
        # Fill gaps in logit space
        filled_logit = self._interpolate_gaps(logit_data)
        
        # Inverse logit transform
        result = 1 / (1 + np.exp(-filled_logit))
        
        return result
    
    def _log_space_denoise(self, data):
        """Denoising in log space for multiplicative noise"""
        
        positive_data = np.maximum(data, 0.01)
        log_data = np.log10(positive_data)
        
        # Apply gentle smoothing in log space
        from scipy.signal import savgol_filter
        smoothed_log = savgol_filter(log_data, window_length=5, polyorder=2)
        
        result = 10 ** smoothed_log
        
        return result
    
    def _edge_preserving_denoise(self, data):
        """Edge-preserving denoising for bounded data"""
        
        # Use bilateral filtering for edge preservation
        from scipy.ndimage import gaussian_filter1d
        
        # Apply gentle Gaussian smoothing
        smoothed = gaussian_filter1d(data, sigma=1.0)
        
        # Ensure bounds are preserved
        result = np.clip(smoothed, 0, 1)
        
        return result
    
    def _interpolate_gaps(self, data):
        """Generic gap interpolation"""
        
        # Simple linear interpolation for gaps
        from scipy.interpolate import interp1d
        
        # Find valid data points
        valid_mask = ~np.isnan(data)
        valid_indices = np.nonzero(valid_mask)[0]
        valid_values = data[valid_mask]
        
        if len(valid_indices) < 2:
            return data  # Not enough data for interpolation
        
        # Create interpolation function
        f = interp1d(valid_indices, valid_values, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        
        # Interpolate all points
        all_indices = np.arange(len(data))
        result = f(all_indices)
        
        return result
#=============================================================================
# PROCESSING HISTORY MANAGER
#=============================================================================

class ProcessingHistoryManager:
    """Manage processing history with undo/redo capabilities"""
    
    def __init__(self, max_history=50, log_processing: Optional[Callable[[str], None]] = None, app: Optional[Any] = None):
        self.history = []
        self.current_position = -1
        self.max_history = max_history
        self.log_processing = log_processing if log_processing is not None else (lambda msg: None)  # No-op if not provided
        self.app = app  # App reference for error handling
        
    def save_state(self, data, curve_info, operation_name, parameters=None):
        """Save current state before operation"""
        
        # Remove any future history if we're in the middle
        if self.current_position < len(self.history) - 1:
            self.history = self.history[:self.current_position + 1]
        
        # Create state snapshot
        state = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation_name,
            'parameters': parameters or {},
            'data': data.copy(),
            'curve_info': copy.deepcopy(curve_info),
            'data_hash': self._calculate_data_hash(data)
        }
        
        self.history.append(state)
        self.current_position = len(self.history) - 1
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.current_position = len(self.history) - 1
    
    def undo(self):
        """Undo last operation"""
        if self.can_undo():
            self.current_position -= 1
            return self.history[self.current_position]
        return None
    
    def redo(self):
        """Redo next operation"""
        if self.can_redo():
            self.current_position += 1
            return self.history[self.current_position]
        return None
    
    def can_undo(self):
        """Check if undo is possible"""
        return self.current_position > 0
    
    def can_redo(self):
        """Check if redo is possible"""
        return self.current_position < len(self.history) - 1
    
    def get_history_summary(self):
        """Get summary of processing history"""
        return [
            {
                'timestamp': state['timestamp'],
                'operation': state['operation'],
                'parameters': state['parameters']
            }
            for state in self.history
        ]
    
    def _calculate_data_hash(self, data):
        """Calculate hash of data for change detection"""
        try:
            # Use pandas hash for efficiency
            return hash(data.to_string())
        except (ValueError, AttributeError, MemoryError) as hash_error:
            # Fallback to simple hash - log for debugging
            if hasattr(self, 'app') and hasattr(self.app, 'handle_graceful_degradation'):
                self.app.handle_graceful_degradation(
                    hash_error,
                    "Data hash calculation",
                    "Using fallback hash method - may affect change detection accuracy"
                )
            return hash(str(data.shape) + str(data.dtypes.tolist()))
        except Exception as hash_error:
            # Unexpected error in hash calculation - use fallback and log
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    hash_error,
                    "Data hash calculation",
                    "Calculating hash for change detection",
                    show_dialog=False
                )
            return hash(str(data.shape) + str(data.dtypes.tolist()))
    
    def get_current_state(self):
        """Get current state without changing position"""
        if self.current_position >= 0 and self.current_position < len(self.history):
            return self.history[self.current_position]
        return None
    
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self.current_position = -1
    
    def export_history(self, filepath):
        """Export processing history to file"""
        try:
            history_data = {
                'max_history': self.max_history,
                'current_position': self.current_position,
                'history_summary': self.get_history_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting history: {e}")
            return False
    
    def import_history(self, filepath):
        """Import processing history from file"""
        try:
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            # Only import summary, not full data (for memory efficiency)
            self.max_history = history_data.get('max_history', 50)
            self.current_position = history_data.get('current_position', -1)
            
            return True
        except Exception as e:
            print(f"Error importing history: {e}")
            return False

#=============================================================================
# PROFESSIONAL USER INTERFACE
#=============================================================================

class PetrophysicalButtons:
    """Professional button system for petrophysical software - designed for international corporate use"""
    
    def __init__(self, root):
        """Initialize professional button styles and resources
        
        Args:
            root: The tkinter root or parent widget
        """
        self.root = root
        
        # Professional color scheme based on industry software standards
        self.colors = {
            'primary': '#2b78c9',      # Blue for primary actions
            'primary_hover': '#1a67b8', # Darker blue for hover
            'primary_active': '#155ba4', # Even darker for pressed
            'success': '#1e9d57',      # Green for completion actions
            'success_hover': '#168a49', # Darker green for hover
            'success_active': '#107c3c', # Even darker for pressed
            'warning': '#eb9c12',      # Amber for caution actions
            'warning_hover': '#d18c10', # Darker amber for hover
            'warning_active': '#bb7e0e', # Even darker for pressed
            'secondary': '#f2f4f7',    # Light gray for secondary actions
            'secondary_hover': '#e6e9ed', # Slightly darker for hover
            'secondary_active': '#d8dce3', # Even darker for pressed
            'text_dark': '#222222',    # Dark text for light backgrounds
            'text_light': '#ffffff',   # Light text for dark backgrounds
            'border': '#c3cad5',       # Border color for buttons
            'disabled': '#cccccc'      # Disabled button color
        }
        
        self._setup_styles()
        
    def _setup_styles(self):
        """Configure ttk styles for professional buttons"""
        self.style = ttk.Style(self.root)
        
        # Use a more compatible approach for cross-platform styling
        # Primary Button Style (Blue) - for main actions
        self.style.configure('Primary.TButton',
                            font=(FONT_DEFAULT, 10, 'bold'),
                            padding=(12, 7))
        
        # Success Button Style (Green) - for completion actions
        self.style.configure('Success.TButton',
                            font=(FONT_DEFAULT, 10, 'bold'),
                            padding=(12, 7))
        
        # Warning Button Style (Amber) - for caution actions
        self.style.configure('Warning.TButton',
                            font=(FONT_DEFAULT, 10, 'bold'),
                            padding=(12, 7))
        
        # Secondary Button Style (Light Gray) - for secondary actions
        self.style.configure('Secondary.TButton',
                            font=(FONT_DEFAULT, 10),
                            padding=(12, 7))
        
        # Modern frame styles for cards
        self.style.configure('Card.TFrame',
                           relief='flat',
                           borderwidth=1)
        
        # Modern label styles
        self.style.configure('Title.TLabel',
                           font=(FONT_DEFAULT, 24, 'bold'))
        
        self.style.configure('Subtitle.TLabel',
                           font=(FONT_DEFAULT, 14))
        
        self.style.configure('Card.TLabel',
                           font=(FONT_DEFAULT, 11))
        
    def create_button(self, parent, text, command=None, button_type='primary', 
                    tooltip=None, width=None, **kwargs):
        """Create a professional button
        
        Args:
            parent: Parent widget
            text: Button text
            command: Button command function
            button_type: 'primary', 'success', 'warning', or 'secondary'
            tooltip: Optional tooltip text
            width: Optional fixed width
            **kwargs: Additional tk.Button parameters
            
        Returns:
            tk.Button widget
        """
        if button_type not in ['primary', 'success', 'warning', 'secondary']:
            button_type = 'primary'
        
        # Get colors for this button type
        bg_color = self.colors[button_type]
        fg_color = self.colors['text_light'] if button_type in ['primary', 'success'] else self.colors['text_dark']
        
        # Create button with proper styling
        button = tk.Button(parent, text=text, command=command,
                          bg=bg_color, fg=fg_color,
                          font=(FONT_DEFAULT, 10, 'bold') if button_type != 'secondary' else (FONT_DEFAULT, 10),
                          relief='flat', borderwidth=0,
                          padx=12, pady=7,
                          cursor='hand2',
                          **kwargs)
        
        # Set fixed width if specified
        if width:
            button.configure(width=width)
        
        # Add hover effects
        def on_enter(e):
            hover_color = self.colors[f'{button_type}_hover']
            button.configure(bg=hover_color)
        
        def on_leave(e):
            button.configure(bg=bg_color)
        
        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)
        
        # Add tooltip if specified
        if tooltip:
            self._create_tooltip(button, tooltip)
            
        return button
    
    def create_toggle_button(self, parent, text, variable, value, **kwargs):
        """Create a toggle button (like a checkbox but styled as a button)
        
        Args:
            parent: Parent widget
            text: Button text
            variable: Variable to track state
            value: Value when selected
            **kwargs: Additional parameters
            
        Returns:
            tk.Checkbutton styled as a button
        """
        toggle = tk.Checkbutton(parent, text=text, variable=variable,
                               bg=self.colors['secondary'], fg=self.colors['text_dark'],
                               font=(FONT_DEFAULT, 10),
                               relief='flat', borderwidth=1,
                               padx=12, pady=7,
                               selectcolor=self.colors['primary'],
                               **kwargs)
        return toggle
    
    def create_button_group(self, parent, buttons, orientation='horizontal'):
        """Create a group of related buttons
        
        Args:
            parent: Parent widget
            buttons: List of button definitions, each a dict with:
                     {'text': button text, 'command': function, 'type': button_type}
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Frame containing the button group
        """
        frame = ttk.Frame(parent)
        
        for i, btn_def in enumerate(buttons):
            btn = self.create_button(
                frame, 
                text=btn_def.get('text', ''),
                command=btn_def.get('command'),
                button_type=btn_def.get('type', 'primary'),
                tooltip=btn_def.get('tooltip')
            )
            
            if orientation == 'horizontal':
                btn.pack(side='left', padx=(0 if i > 0 else 0, 5), pady=5)
            else:
                btn.pack(side='top', padx=0, pady=(0 if i > 0 else 0, 5))
                
        return frame
    
    def create_card(self, parent, title: str, help_text: str = None, **kwargs) -> ttk.Frame:
        """Create a professional card widget with optional info icon.
        
        Args:
            parent: Parent widget
            title: Card title
            help_text: Optional help text shown when clicking the info icon
        """
        card = ttk.Frame(parent, style='Card.TFrame', **kwargs)
        
        # Title row with optional info icon
        header = ttk.Frame(card)
        header.pack(fill='x', padx=20, pady=(15, 5))
        title_label = ttk.Label(header, text=title, style='Subtitle.TLabel')
        title_label.pack(side='left')
        
        if help_text:
            def _show_card_help():
                try:
                    from tkinter import Toplevel
                    dialog = Toplevel(card)
                    dialog.title(f"Help - {title}")
                    dialog.transient(card)
                    dialog.grab_set()
                    dialog.resizable(True, True)
                    body = ttk.Frame(dialog, padding=15)
                    body.pack(fill='both', expand=True)
                    lbl = ttk.Label(body, text=help_text, wraplength=560, justify='left')
                    lbl.pack(fill='both', expand=True)
                    btn = ttk.Button(body, text='Close', command=dialog.destroy)
                    btn.pack(anchor='e', pady=(10, 0))
                    dialog.update_idletasks()
                    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
                    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
                    dialog.geometry(f"+{x}+{y}")
                except (tk.TclError, AttributeError) as dialog_error:
                    # Dialog positioning failed - continue without centering
                    self.handle_ui_error(
                        dialog_error,
                        "Dialog positioning",
                        "dialog window",
                        graceful_degradation=True
                    )
                except Exception as dialog_error:
                    # Unexpected error positioning dialog
                    self.handle_ui_error(
                        dialog_error,
                        "Dialog positioning",
                        "dialog window",
                        graceful_degradation=True
                    )
            info_btn = ttk.Button(header, text='i', width=2, command=_show_card_help)
            self._create_tooltip(info_btn, f"Help: {title}")
            info_btn.pack(side='right')
        
        # Content area
        content = ttk.Frame(card)
        content.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        return card, content
    
    def create_progress_card(self, parent, title: str) -> Tuple[ttk.Frame, ttk.Progressbar, ttk.Label]:
        """Create progress card with bar and status"""
        card, content = self.create_card(parent, title)
        
        progress = ttk.Progressbar(content, mode='determinate', length=400)
        progress.pack(pady=(10, 5))
        
        status_label = ttk.Label(content, text="Ready", style='Card.TLabel')
        status_label.pack()
        
        return card, progress, status_label
    
    def _create_tooltip(self, widget, text):
        """Create a professional tooltip for a widget
        
        Args:
            widget: Widget to add tooltip to
            text: Tooltip text
        """
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, background="#ffffcc",
                            relief="solid", borderwidth=1)
            label.pack()
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)



#=============================================================================
# UNIT STANDARDIZATION SYSTEM
#=============================================================================
# NOTE: IndustryUnitStandardizer has been extracted to core/unit_standardization.py.
# Import maintained here for backward compatibility during modularization.
from core.unit_standardization import IndustryUnitStandardizer

#=============================================================================
# MAIN APPLICATION CLASS
#=============================================================================

class AdvancedPreprocessingApplication(WellLoadingMixin, AppUIMixin):
    """Main application class with advanced preprocessing capabilities"""
    
    def __init__(self):
        # Initialize feature flags first
        if BETA_SYSTEM_AVAILABLE:
            self.feature_flags = BetaFeatureFlags()
            
            # Initialize beta systems
            self.beta_analytics = BetaAnalytics(self.feature_flags)
            self.feedback_collector = BetaFeedbackCollector(self.feature_flags)
        else:
            self.feature_flags = None
            self.beta_analytics = None
            self.feedback_collector = None
        
        self.root = tk.Tk()
        self.root.title("Advanced Wireline Data Preprocessing System")
        self.root.geometry("1400x900")
        
        # Bind window resize event
        self.root.bind(EVENT_CONFIGURE, self.on_window_resize)
        
        # Initialize centralized error handler
        # Note: Will be initialized after setup_ui() when log_processing is available
        self.error_handler = None  # Will be initialized after UI setup
        
        # Initialize components
        self.curve_identifier = CurveIdentificationEngine()
        if hasattr(self, 'standardization_reporter'):
            pass  # Already initialized
        else:
            try:
                from core.reporting import StandardizationReporter
                self.standardization_reporter = StandardizationReporter()
            except ImportError:
                self.standardization_reporter = None
        # Gap filler will be initialized after error handler is available
        self.gap_filler = None  # Will be initialized after error handler
        # Signal processor will be initialized after error handler is available
        self.signal_processor = None  # Will be initialized after error handler
        self.ui = PetrophysicalButtons(self.root)
        self.rrp_model = None  # Will be initialized when needed
        
        # Visualization manager
        self.viz_manager = SecureVisualizationManager()
        
        # Initialize new advanced processing components
        self.depth_validator = DepthValidationManager(log_processing=self.log_processing)
        self.depth_validator.app = self  # Provide app reference for error handling
        self.reservoir_depth_manager = ReservoirDepthManager()
        self.geological_zone_manager = GeologicalZoneManager()
        self.zone_aware_gap_filler = ZoneAwareGapFiller(GapFillingParameters(), self.geological_zone_manager, error_callback=self.show_error_dialog)
        self.petrophysical_validator = PetrophysicalRelationshipValidator()
        self.las_compliance = LASStandardsCompliance()
        # Initialize thread-safe viz manager and inject scheduler callback
        self.thread_safe_viz = ThreadSafeVisualizationManager()
        try:
            # Provide a scheduler hook so the manager doesn't need root reference
            def _schedule_on_main(func: Callable):
                self.root.after_idle(func)
            setattr(self.thread_safe_viz, '_schedule_on_main', _schedule_on_main)
        except (AttributeError, RuntimeError) as callback_error:
            # UI callback setup failed - continue without callback
            self.handle_ui_error(
                callback_error,
                "Thread-safe visualization callback setup",
                "thread_safe_viz._schedule_on_main",
                graceful_degradation=True
            )
        except Exception as callback_error:
            # Unexpected error setting up UI callback
            self.handle_ui_error(
                callback_error,
                "Thread-safe visualization callback setup",
                "thread_safe_viz._schedule_on_main",
                graceful_degradation=True
            )
        self.environmental_corrections = EnvironmentalCorrectionsManager()
        self.scale_aware_processor = ScaleAwareProcessor()
        self.processing_history = ProcessingHistoryManager()
        self.processing_history.app = self  # Provide app reference for error handling
        
        # Geological context for intelligent gap classification
        self.geological_context = GeologicalContext()
        self._gamma_ray_curves = []
        
        # Cross-well priors manager (multiwell intelligence)
        self.crosswell_prior_manager = None  # Will be initialized after class definition
        
        # Initialize unit standardizer
        self.unit_standardizer = IndustryUnitStandardizer()
        self.unit_standardizer.app = self  # Provide app reference for error handling
        
        # Status manager for user feedback (no file logging)
        # Will be initialized after UI creation in setup_ui()
        self.status_manager = None  # Will be set after UI creation
        
        # Application state
        self.current_data = None
        self.processed_data = None
        self.curve_info = {}
        self.processing_results = {}
        self.original_las_header = None  # Initialize LAS header storage
        self.file_path_var = tk.StringVar()  # Initialize file path variable
        self.fig = None  # Initialize matplotlib figure
        self.well_info = {}  # CRITICAL: Well identification for safety
        # Multiwell state
        self.well_datasets: Dict[str, Dict[str, Any]] = {}
        self.active_well_id: Optional[str] = None
        
        # Popup window registry for proper memory management
        self.popup_windows = []  # Track all popup visualization windows
        self.popup_figures = []  # Track figures in popup windows for cleanup
        
        # Initialize UI variables that are referenced in report generation
        self.max_gap_var = tk.IntVar(value=500)
        self.gap_method_var = tk.StringVar(value="auto")
        self.physics_informed_var = tk.BooleanVar(value=True)
        self.multi_curve_var = tk.BooleanVar(value=True)
        self.denoise_method_var = tk.StringVar(value="auto")
        # Default resampling spacing will be adapted to current depth units.
        # Initialize with metric default; will be updated post-load by _sync_depth_spacing_default().
        self.depth_spacing_var = tk.DoubleVar(value=0.1)
        self.rename_curves_var = tk.BooleanVar(value=True)
        self.null_value_var = tk.StringVar(value="-999.25")
        # Phase 0 item 2 (see POLISH_pipeline_contracts.md sections 6 and 8): unit
        # standardization is OFF by default. Converting on load is what breaks an
        # imperial well. On KEOUGH #12-34 the chain is:
        #   DEPT FT -> M (x0.3048), so 0-5359.5 ft becomes 0-1633.678 m; but
        #   depth_spacing_var was already fixed at 0.5 by _sync_depth_spacing_default
        #   while the unit was still FT. The resampler then reads that 0.5 as metres.
        #   1633.678 / 0.5 + 1 = 3268 rows out of 10720. That is the C4 violation:
        #   a parameter derived under "unit is FT" survived the unit changing.
        #   RHOB G/CC -> KG/M3 (x1000) gives ~2000-2700, but BULK_DENSITY carries a
        #   single 'range': [1.0, 3.5] for both G/CC and KG/M3 inputs
        #   (core/curve_identification.py), so range validation empties the curve.
        #   That is the C1 violation.
        # Leaving an already-imperial well alone sidesteps both: DEPT stays FT, 0.5 ft
        # spacing is then correct, 10720 rows survive, and RHOB passes its own range.
        # This defaults the transformation off (C6); it does not repair C1 or C4
        # themselves. Those are Phase 2, and the trap is still live for any user who
        # ticks the box. Whether metric wells are processed at all is an open question
        # (contracts section 9) -- deliberately not assumed either way here.
        self.standardize_units_var = tk.BooleanVar(value=False)
        
        # === NEW PRODUCTION-READY FEATURE VARIABLES ===
        # Environmental Corrections (Priority 1.1)
        self.apply_env_corrections_var = tk.BooleanVar(value=False)
        self.tool_type_var = tk.StringVar(value='generic')
        self.bit_size_var = tk.StringVar(value='8.5')
        self.mud_weight_var = tk.StringVar(value='10.0')
        self.matrix_type_var = tk.StringVar(value='sandstone')
        
        # Saturation Calculation (Priority 1.2)
        self.compute_saturation_var = tk.BooleanVar(value=False)
        self.archie_a_var = tk.StringVar(value='1.0')
        self.archie_m_var = tk.StringVar(value='2.0')
        self.archie_n_var = tk.StringVar(value='2.0')
        self.rw_var = tk.StringVar(value='0.05')
        self.gr_clean_var = tk.StringVar(value='20.0')
        self.gr_shale_var = tk.StringVar(value='120.0')
        self.rsh_var = tk.StringVar(value='2.0')
        
        # Basin Selection (Priority 1.3)
        self.basin_var = tk.StringVar(value='Generic Clean Sandstone')
        self.basin_info_label = None  # Will be created in UI setup
        
        # Batch Processing Manager (Priority 1.5)
        self.batch_manager = None  # Will be initialized when batch tab is created
        self.output_format_var = tk.StringVar(value="Company Standard")
        self.large_gap_var = tk.StringVar(value="formation_based")
        # Threshold (in points) beyond which gaps are considered "large"
        # Used by gap filling logic and bound to the UI entry in the Gap Filling tab
        self.large_gap_threshold_var = tk.IntVar(value=1000)
        # Sync depth spacing default based on detected depth unit (m or ft)
        try:
            self._sync_depth_spacing_default()
        except (AttributeError, ValueError) as sync_error:
            # Depth spacing sync failed - log but don't prevent startup
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Depth spacing sync failed: {type(sync_error).__name__}: {str(sync_error)}")
                except Exception:
                    pass  # Can't log logging failure
        except Exception as sync_error:
            # Unexpected error in depth spacing sync
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error in depth spacing sync: {type(sync_error).__name__}: {str(sync_error)}")
                except Exception:
                    pass  # Can't log logging failure
        # Use instance created during __init__
        self.geological_gap_threshold_var = tk.IntVar(value=GAP_THRESHOLD_GEOLOGICAL)  # Geological gap threshold
        self.qc_enabled_var = tk.BooleanVar(value=True)
        self.outlier_detection_var = tk.BooleanVar(value=True)
        self.range_validation_var = tk.BooleanVar(value=True)
        self.parallel_processing_var = tk.BooleanVar(value=False)
        self.uncertainty_quantification_var = tk.BooleanVar(value=True)
        self.confidence_intervals_var = tk.BooleanVar(value=True)
        self.memory_limit_var = tk.IntVar(value=2048)
        self.auto_cleanup_var = tk.BooleanVar(value=True)
        self.plot_in_new_window_var = tk.BooleanVar(value=True)  # Default to popup windows for professional workflow
        
        # Session preference: standardize units on upload (percent → v/v for fractional families)
        # Allow selection of which curve families to standardize
        self.standardize_porosity_var = tk.BooleanVar(value=True)
        self.standardize_saturation_var = tk.BooleanVar(value=True)
        self.standardize_volume_var = tk.BooleanVar(value=True)
        self.standardize_probability_var = tk.BooleanVar(value=True)
        # Legacy support: if all families are enabled, standardization is "on"
        self.standardize_on_upload_var = tk.BooleanVar(value=True)  # Kept for backward compatibility checks
        self._upload_standardization_note = ""
        
        # Cohort & cross-well priors preferences
        self.use_crosswell_priors_var = tk.BooleanVar(value=False)
        self.two_pass_refinement_var = tk.BooleanVar(value=True)
        self.priors_depth_binning_var = tk.BooleanVar(value=True)
        self.auto_select_cohort_var = tk.BooleanVar(value=True)
        self.cohort_selected_well_ids: List[str] = []
        self.crosswell_priors: Dict[str, Any] = {}
        
        # Configure matplotlib for better memory management
        plt.rcParams['figure.max_open_warning'] = 10
        
        self.setup_ui()
        
        # Initialize centralized error handler after UI setup (log_processing is now available)
        try:
            self.error_handler = CentralizedErrorHandler(
                root=self.root,
                log_callback=self.log_processing
            )
        except Exception as init_error:
            # Fallback if error handler initialization fails - log but continue
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Error handler initialization failed: {type(init_error).__name__}: {str(init_error)}")
                except Exception:
                    pass  # Can't log logging failure
            self.error_handler = None  # Will use fallback error handling
        
        # Initialize gap filler and signal processor with error handler
        if self.gap_filler is None:
            self.gap_filler = AdvancedGapFiller(GapFillingParameters(), error_handler=self.error_handler, log_processing=self.log_processing)
        if self.signal_processor is None:
            self.signal_processor = AdvancedSignalProcessor(error_handler=self.error_handler, log_processing=self.log_processing)
        
        # Setup beta features if in beta mode
        if BETA_SYSTEM_AVAILABLE and self.feature_flags.is_beta_mode():
            self.setup_beta_features()
        
        # ENHANCED: Initialize library availability management
        self.library_status = self.create_robust_gap_filler_with_fallbacks()
        
        # ENHANCED: Initialize comprehensive memory management
        self.implement_comprehensive_memory_management()
        
        # Prompt user once at startup for standardization preference
        try:
            self.root.after(200, self.show_startup_standardization_dialog)
        except (tk.TclError, AttributeError) as startup_error:
            # Startup dialog scheduling failed - optional feature, don't fail initialization
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Startup dialog scheduling failed: {type(startup_error).__name__}")
                except Exception:
                    pass  # Can't log logging failure
        except Exception as startup_error:
            # Unexpected error scheduling startup dialog
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error scheduling startup dialog: {type(startup_error).__name__}: {str(startup_error)}")
                except Exception:
                    pass  # Can't log logging failure

        # Ergonomic enhancement: clarify long-missing-run warnings
        try:
            self._orig_showwarning = messagebox.showwarning
            def _ap_showwarning(title, message, *args, **kwargs):
                try:
                    text = str(message)
                    text_l = text.lower()
                    if ((('100' in text or 'hundred' in text_l) and 'missing' in text_l) or
                       ('consecutive' in text_l or 'in a row' in text_l)):
                        message = text + "\n\nPress OK or Enter to continue. The program did not crash."
                except (AttributeError, TypeError) as msg_error:
                    # Message formatting failed - continue with original message
                    pass
                except Exception as msg_error:
                    # Unexpected error in message formatting - log for debugging
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Message formatting failed: {type(msg_error).__name__}: {str(msg_error)}")
                        except Exception:
                            pass  # Can't log logging failure
                return self._orig_showwarning(title, message, *args, **kwargs)
            messagebox.showwarning = _ap_showwarning
        except Exception as wrap_error:
            # Warning messagebox wrapping failed - optional enhancement, don't fail startup
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Messagebox wrapping failed: {type(wrap_error).__name__}: {str(wrap_error)}")
                except Exception:
                    pass  # Can't log logging failure

    def show_startup_standardization_dialog(self):
        """Show a modal to choose which curve families to standardize on upload."""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Upload Standardization Settings")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.resizable(False, False)
            
            main_frame = ttk.Frame(dialog, padding=20)
            main_frame.pack(fill='both', expand=True)
            
            # Header
            header_msg = ("Select which curve families to standardize on upload:\n"
                         "Convert percent-style fractional curves to decimals (v/v)")
            ttk.Label(main_frame, text=header_msg, wraplength=500, justify='left', 
                     font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 15))
            
            # Options frame
            options_frame = ttk.LabelFrame(main_frame, text="Curve Families", padding=15)
            options_frame.pack(fill='x', pady=(0, 15))
            
            # Individual checkboxes for each curve family
            porosity_chk = ttk.Checkbutton(
                options_frame, 
                text="Porosity curves (NPHI, PHI, PHIT, DPOR, TNPH, MPHI, etc.)",
                variable=self.standardize_porosity_var
            )
            porosity_chk.pack(anchor='w', pady=5)
            
            saturation_chk = ttk.Checkbutton(
                options_frame,
                text="Saturation curves (SW, SO, SG, SAT, etc.)",
                variable=self.standardize_saturation_var
            )
            saturation_chk.pack(anchor='w', pady=5)
            
            volume_chk = ttk.Checkbutton(
                options_frame,
                text="Volume fraction curves (VSH, VCL, VCARB, VMIN, VOL, etc.)",
                variable=self.standardize_volume_var
            )
            volume_chk.pack(anchor='w', pady=5)
            
            probability_chk = ttk.Checkbutton(
                options_frame,
                text="Probability curves (PROB, FACIES_PROB, etc.)",
                variable=self.standardize_probability_var
            )
            probability_chk.pack(anchor='w', pady=5)
            
            # Helper buttons
            helper_frame = ttk.Frame(main_frame)
            helper_frame.pack(fill='x', pady=(0, 15))
            
            def select_all():
                self.standardize_porosity_var.set(True)
                self.standardize_saturation_var.set(True)
                self.standardize_volume_var.set(True)
                self.standardize_probability_var.set(True)
            
            def deselect_all():
                self.standardize_porosity_var.set(False)
                self.standardize_saturation_var.set(False)
                self.standardize_volume_var.set(False)
                self.standardize_probability_var.set(False)
            
            ttk.Button(helper_frame, text=DIALOG_SELECT_ALL, command=select_all, width=12).pack(side='left', padx=(0, 5))
            ttk.Button(helper_frame, text=DIALOG_DESELECT_ALL, command=deselect_all, width=12).pack(side='left')
            
            # Info note
            info_text = ("Note: Only curves in selected families will be considered for standardization.\n"
                        "You can still review and approve each conversion when loading a file.")
            ttk.Label(main_frame, text=info_text, wraplength=500, justify='left', 
                     font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(0, 15))
            
            # OK button
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill='x')
            btn = ttk.Button(btn_frame, text="OK", command=dialog.destroy)
            btn.pack(anchor='e')
            
            # Update legacy compatibility variable based on current selections
            def update_legacy_var(*args):
                all_enabled = (self.standardize_porosity_var.get() and 
                              self.standardize_saturation_var.get() and 
                              self.standardize_volume_var.get() and 
                              self.standardize_probability_var.get())
                self.standardize_on_upload_var.set(all_enabled)
            
            # Track changes to update legacy variable
            self.standardize_porosity_var.trace('w', update_legacy_var)
            self.standardize_saturation_var.trace('w', update_legacy_var)
            self.standardize_volume_var.trace('w', update_legacy_var)
            self.standardize_probability_var.trace('w', update_legacy_var)
            
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
        except (tk.TclError, AttributeError) as pos_error:
            # Dialog positioning failed - cosmetic, continue without centering
            self.handle_ui_error(
                pos_error,
                "Startup standardization dialog positioning",
                "dialog widget",
                graceful_degradation=True
            )
        except Exception as pos_error:
            # Unexpected error positioning dialog - log for debugging
            self.handle_ui_error(
                pos_error,
                "Startup standardization dialog",
                "Tkinter dialog",
                graceful_degradation=True
            )

    def _is_porosity_curve(self, name_upper: str) -> bool:
        """Return True if curve name indicates porosity family."""
        porosity_terms = ['NPHI', 'NPOR', 'PHI', 'PHIT', 'PHIE', 'DPOR', 'TNPH', 'MPHI']
        return any(t in name_upper for t in porosity_terms)
    
    def _is_saturation_curve(self, name_upper: str) -> bool:
        """Return True if curve name indicates saturation family."""
        saturation_terms = ['SW', 'SO', 'SG', 'SAT']
        return any(name_upper.startswith(t) for t in saturation_terms)
    
    def _is_volume_curve(self, name_upper: str) -> bool:
        """Return True if curve name indicates volume fraction family."""
        volume_terms = ['VSH', 'VCL', 'VCARB', 'VMIN', 'VOL']
        return any(name_upper.startswith(t) for t in volume_terms)
    
    def _is_probability_curve(self, name_upper: str) -> bool:
        """Return True if curve name indicates probability family."""
        probability_terms = ['PROB', 'FACIES_PROB']
        return any(t in name_upper for t in probability_terms)
    
    def _is_fractional_family_name(self, name_upper: str) -> bool:
        """Return True if curve name indicates fractional family (porosity/saturation/volume/probability)."""
        return (self._is_porosity_curve(name_upper) or 
                self._is_saturation_curve(name_upper) or 
                self._is_volume_curve(name_upper) or 
                self._is_probability_curve(name_upper))
    
    def _should_standardize_curve_family(self, name_upper: str) -> bool:
        """Check if curve should be considered for standardization based on enabled families."""
        if self._is_porosity_curve(name_upper):
            return self.standardize_porosity_var.get()
        elif self._is_saturation_curve(name_upper):
            return self.standardize_saturation_var.get()
        elif self._is_volume_curve(name_upper):
            return self.standardize_volume_var.get()
        elif self._is_probability_curve(name_upper):
            return self.standardize_probability_var.get()
        else:
            # Not a recognized fractional family - don't standardize
            return False

    def standardize_fractional_curves_on_upload(self):
        """Convert percent-style fractional families to decimals after file load, before validation.
        
        SAFETY FEATURE: Shows preview dialog with conversion details before applying changes.
        This prevents accidental misinterpretation of data (e.g., impedance as porosity).
        Only considers curves in families selected in the startup dialog.
        """
        try:
            self._upload_standardization_note = ""
            
            # Check if any families are enabled for standardization
            any_family_enabled = (self.standardize_porosity_var.get() or 
                                 self.standardize_saturation_var.get() or 
                                 self.standardize_volume_var.get() or 
                                 self.standardize_probability_var.get())
            
            if not any_family_enabled or self.current_data is None or self.current_data.empty:
                return
            
            # First pass: identify potential conversions (only for enabled families)
            conversion_candidates = []
            for col in list(self.current_data.columns):
                name_upper = str(col).upper()
                if name_upper in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']:
                    continue
                
                # Only consider curves in enabled families
                if not self._should_standardize_curve_family(name_upper):
                    continue
                
                series = pd.to_numeric(self.current_data[col], errors='coerce')
                unit = self.curve_info.get(col, {}).get('unit', '')
                unit_upper = str(unit).upper()
                
                should_convert = False
                reason = ""
                
                if '%' in unit_upper or unit_upper in ['PERCENT', 'PCT', 'PERC']:
                    should_convert = True
                    reason = f"Unit indicates percent ({unit})"
                elif self._is_fractional_family_name(name_upper):
                    vals = series.dropna()
                    if len(vals) > 10:
                        med = float(np.median(vals))
                        min_val = float(vals.min())
                        max_val = float(vals.max())
                        # Heuristic: likely percent if median within (1, 100] and few extreme values
                        if 1.0 < med <= 100.0:
                            should_convert = True
                            reason = f"Fractional curve with median {med:.2f} (range: {min_val:.2f}-{max_val:.2f})"
                
                if should_convert:
                    # Determine which family this curve belongs to
                    family_type = "Unknown"
                    if self._is_porosity_curve(name_upper):
                        family_type = "Porosity"
                    elif self._is_saturation_curve(name_upper):
                        family_type = "Saturation"
                    elif self._is_volume_curve(name_upper):
                        family_type = "Volume"
                    elif self._is_probability_curve(name_upper):
                        family_type = "Probability"
                    
                    conversion_candidates.append({
                        'name': col,
                        'unit': unit,
                        'reason': reason,
                        'family': family_type,
                        'median': float(np.median(series.dropna())) if len(series.dropna()) > 0 else 0,
                        'range': f"{series.min():.2f} to {series.max():.2f}"
                    })
            
            # If no conversions needed, return early
            if not conversion_candidates:
                return
            
            # Show confirmation dialog with selective conversion options
            selected_curves = self._show_conversion_confirmation_dialog(conversion_candidates)
            
            if not selected_curves:
                self.log_processing("User declined or cancelled unit conversion")
                return
            
            # User selected specific curves - perform conversions only for selected
            converted = []
            for candidate in conversion_candidates:
                col = candidate['name']
                
                # Only convert if this curve was selected by user
                if col not in selected_curves:
                    continue
                
                series = pd.to_numeric(self.current_data[col], errors='coerce')
                self.current_data[col] = series / 100.0
                if col in self.curve_info:
                    self.curve_info[col]['unit'] = 'v/v'
                    self.curve_info[col]['original_unit'] = candidate['unit'] or self.curve_info[col].get('original_unit', '')
                    # Record in standardization reporter
                    try:
                        if hasattr(self, 'standardization_reporter') and self.standardization_reporter:
                            vals = series.dropna()
                            if vals.size > 0:
                                original_sample = float(np.median(vals))
                                standardized_sample = float(np.median(vals / 100.0))
                                self.standardization_reporter.record_fractional_standardization(
                                    curve_name=col,
                                    original_unit=candidate['unit'] or '%',
                                    original_value_sample=original_sample,
                                    standardized_value_sample=standardized_sample
                                )
                            else:
                                self.standardization_reporter.record_fractional_standardization(
                                    curve_name=col,
                                    original_unit=candidate['unit'] or '%'
                                )
                    except (ValueError, AttributeError, KeyError) as report_error:
                        # Standardization reporting failed - continue without reporting
                        self.handle_graceful_degradation(
                            report_error,
                            "Unit standardization reporting",
                            "Continuing standardization without detailed reporting"
                        )
                    except Exception as report_error:
                        # Unexpected error in standardization reporting
                        self.handle_ui_error(
                            report_error,
                            "Unit standardization reporting",
                            "standardization_reporter",
                            graceful_degradation=True
                        )
                converted.append(col)
            
            if converted:
                # Refresh stats for display
                self.ensure_curve_statistics()
                note = f"Note: Standardized on upload (%→v/v) for {len(converted)} curve(s): "
                sample = ', '.join(converted[:10])
                if len(converted) > 10:
                    sample += ' ...'
                self._upload_standardization_note = note + sample
                self.log_processing(f"Applied automatic conversions to {len(converted)} curves")
                # Update status manager if available
                if hasattr(self, 'status_manager') and self.status_manager:
                    self.status_manager.update_status(self._upload_standardization_note)
        except Exception as e:
            self.log_processing(f"Upload standardization error: {e}")
            # Update status manager if available
            if hasattr(self, 'status_manager') and self.status_manager:
                self.status_manager.update_status(f"Upload standardization skipped due to error: {e}")
        
        # === UNIT AMBIGUITY DETECTION ===
        try:
            if self.current_data is not None:
                ambiguities = self.unit_standardizer.detect_unit_ambiguities(
                    self.current_data, 
                    self.curve_info
                )
                
                if ambiguities:
                    self.log_processing(f"[UNIT AMBIGUITY] Detected {len(ambiguities)} ambiguous units")
                    
                    # Show resolution dialog
                    conversions = self.unit_standardizer.show_ambiguity_resolution_dialog(ambiguities)
                    
                    if conversions:
                        for curve_name, should_convert in conversions.items():
                            if should_convert and curve_name in self.current_data.columns:
                                self.current_data[curve_name] = self.current_data[curve_name] / 100.0
                                if curve_name in self.curve_info:
                                    self.curve_info[curve_name]['unit'] = 'v/v'
                                self.log_processing(f"   Converted {curve_name}: % → v/v")
                        
                        # Refresh statistics
                        self.ensure_curve_statistics()
        except Exception as e:
            self.log_processing(f"[UNIT AMBIGUITY] Error: {str(e)}")
    
    def _show_conversion_confirmation_dialog(self, conversion_candidates):
        """Show dialog with preview of proposed unit conversions with selective conversion options
        
        Args:
            conversion_candidates: List of dicts with conversion details
            
        Returns:
            List[str]: List of curve names that user selected to convert (empty list if cancelled)
        """
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Unit Conversions")
            dialog.geometry("800x600")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.resizable(True, True)  # Allow resizing
            
            # Main frame
            main_frame = ttk.Frame(dialog, padding=15)
            main_frame.pack(fill='both', expand=True)
            
            # Header
            header_text = f"Unit Conversion Selection\n\nDetected {len(conversion_candidates)} curve(s) that appear to be in percent format.\nSelect which conversions to apply:"
            header_label = ttk.Label(main_frame, text=header_text, wraplength=750, justify='left', font=('TkDefaultFont', 10, 'bold'))
            header_label.pack(pady=(0, 10), anchor='w')
            
            # Scrollable frame for conversion list
            canvas_frame = ttk.Frame(main_frame)
            canvas_frame.pack(fill='both', expand=True, pady=(0, 10))
            
            canvas = tk.Canvas(canvas_frame, highlightthickness=0, bg='white')
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            def update_scroll_region(event=None):
                canvas.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
            
            scrollable_frame.bind("<Configure>", update_scroll_region)
            
            canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Make canvas window resize with canvas
            def on_canvas_configure(event):
                canvas_width = event.width
                if canvas_width > 1:  # Only update if canvas has been rendered
                    canvas.itemconfig(canvas_window, width=canvas_width)
            
            canvas.bind(EVENT_CONFIGURE, on_canvas_configure)
            
            # Enable mouse wheel scrolling (works on all platforms)
            def on_mousewheel(event):
                # Windows uses delta, Linux/Mac uses different events
                try:
                    if hasattr(event, 'delta'):
                        # Windows
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    elif hasattr(event, 'num'):
                        # Linux/Mac
                        if event.num == 4:
                            canvas.yview_scroll(-1, "units")  # Scroll up
                        elif event.num == 5:
                            canvas.yview_scroll(1, "units")   # Scroll down
                except (tk.TclError, AttributeError) as scroll_error:
                    # Canvas scrolling failed - continue without scrolling
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas mouse wheel scrolling",
                        "canvas widget",
                        graceful_degradation=True
                    )
                except Exception as scroll_error:
                    # Unexpected error in canvas scrolling
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas mouse wheel scrolling",
                        "canvas widget",
                        graceful_degradation=True
                    )
            
            # Bind mouse wheel for different platforms (bind to canvas and dialog)
            canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
            canvas.bind("<Button-4>", on_mousewheel)   # Linux scroll up
            canvas.bind("<Button-5>", on_mousewheel)   # Linux scroll down
            dialog.bind("<MouseWheel>", on_mousewheel)  # Also bind to dialog
            scrollable_frame.bind("<MouseWheel>", on_mousewheel)  # And scrollable frame
            
            # Set focus to canvas so mouse wheel works immediately
            canvas.focus_set()
            
            # Store checkboxes for each candidate
            checkboxes = {}
            conversion_vars = {}
            
            # Validate conversion candidates
            if not conversion_candidates or len(conversion_candidates) == 0:
                # Show message if no candidates (shouldn't happen, but handle gracefully)
                no_items_label = ttk.Label(scrollable_frame, 
                                          text="No conversion candidates found.",
                                          font=('TkDefaultFont', 10),
                                          foreground='gray')
                no_items_label.pack(pady=20)
            else:
                # Add conversion details with checkboxes
                for i, candidate in enumerate(conversion_candidates, 1):
                    try:
                        curve_name = candidate.get('name', f'Curve_{i}')
                        conversion_vars[curve_name] = tk.BooleanVar(value=True)  # Default to selected
                        
                        # Main frame for each candidate
                        item_frame = ttk.Frame(scrollable_frame)
                        item_frame.pack(fill='x', pady=5, padx=5)
                        
                        # Checkbox frame
                        checkbox_frame = ttk.Frame(item_frame)
                        checkbox_frame.pack(fill='x', pady=(0, 5))
                        
                        chk = ttk.Checkbutton(
                            checkbox_frame,
                            text=f"{i}. {curve_name}",
                            variable=conversion_vars[curve_name]
                        )
                        chk.pack(side='left', anchor='w')
                        checkboxes[curve_name] = chk
                        
                        # Details frame
                        details_frame = ttk.LabelFrame(item_frame, padding=8, text=f"Conversion Details for {curve_name}")
                        details_frame.pack(fill='x', padx=(25, 0), pady=(0, 5))  # Indent details
                        
                        # Safely get candidate details
                        unit = candidate.get('unit', 'Not specified')
                        range_val = candidate.get('range', 'N/A')
                        median = candidate.get('median', 0)
                        reason = candidate.get('reason', 'Detected as percent format')
                        family_type = candidate.get('family', 'Unknown')
                        
                        details = (
                            f"Family: {family_type}\n"
                            f"Current Unit: {unit}\n"
                            f"Current Range: {range_val}\n"
                            f"Median Value: {median:.2f}\n"
                            f"Reason: {reason}\n"
                            f"→ Will convert to: v/v (divide by 100)"
                        )
                        ttk.Label(details_frame, text=details, justify='left', font=('Courier', 9)).pack(anchor='w')
                    except Exception as e:
                        # Log error but continue with other candidates
                        self.log_processing(f"Error adding conversion candidate {i}: {e}")
                        continue
            
            # Pack canvas and scrollbar BEFORE updating scroll region
            canvas.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Update scroll region after all items are added and widgets are packed
            scrollable_frame.update_idletasks()
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
            
            # Update canvas window width to match canvas
            canvas_width = canvas.winfo_width()
            if canvas_width > 1:
                canvas.itemconfig(canvas_window, width=canvas_width)
            
            # Force scroll region update after window is fully rendered
            def final_scroll_update():
                canvas.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
                # Ensure canvas window width is correct
                cw = canvas.winfo_width()
                if cw > 1:
                    canvas.itemconfig(canvas_window, width=cw)
            
            # Update after window is fully rendered (multiple updates to catch all render stages)
            dialog.after(100, final_scroll_update)
            dialog.after(300, final_scroll_update)
            dialog.after(500, final_scroll_update)
            
            # Select All / Deselect All buttons
            select_frame = ttk.Frame(main_frame)
            select_frame.pack(fill='x', pady=(5, 5))
            
            def select_all():
                for var in conversion_vars.values():
                    var.set(True)
            
            def deselect_all():
                for var in conversion_vars.values():
                    var.set(False)
            
            ttk.Button(select_frame, text=DIALOG_SELECT_ALL, command=select_all, width=15).pack(side='left', padx=5)
            ttk.Button(select_frame, text=DIALOG_DESELECT_ALL, command=deselect_all, width=15).pack(side='left', padx=5)
            
            # Warning label
            warning_text = "⚠️ WARNING: Incorrect conversions can corrupt your data. Verify these conversions are appropriate."
            warning_label = ttk.Label(main_frame, text=warning_text, foreground='red', wraplength=750, font=('TkDefaultFont', 9, 'bold'))
            warning_label.pack(pady=10)
            
            # Button frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(10, 0))
            
            result = {'selected': []}  # Changed to return list of selected curve names
            
            def apply_selected():
                # Collect all selected curve names
                result['selected'] = [name for name, var in conversion_vars.items() if var.get()]
                dialog.destroy()
            
            def cancel():
                result['selected'] = []  # Empty list means cancelled/no conversions
                dialog.destroy()
            
            ttk.Button(button_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
            apply_button = ttk.Button(button_frame, text=f"Apply Selected ({len(conversion_candidates)} total)", 
                                     command=apply_selected, style='success.TButton')
            apply_button.pack(side='right', padx=5)
            
            # Update button text when selection changes
            def update_button():
                selected_count = sum(1 for var in conversion_vars.values() if var.get())
                apply_button.config(text=f"Apply Selected ({selected_count} of {len(conversion_candidates)})")
            
            # Bind to checkbox changes
            for var in conversion_vars.values():
                var.trace('w', lambda *args: update_button())
            
            # Center dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Initial button text update
            update_button()
            
            # Wait for user response
            dialog.wait_window()
            
            return result['selected']  # Return list of selected curve names
            
        except Exception as e:
            self.log_processing(f"Error showing conversion dialog: {e}")
            # On error, default to no conversion (safe choice)
            return []
    
    def convert_columns_percent_to_decimal(self):
        """Convert percent-style columns to decimal format (v/v) for manual conversion."""
        try:
            if self.current_data is None or self.current_data.empty:
                messagebox.showwarning("No Data", "Please load data first before converting units.")
                return
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Converting Units")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            progress_dialog.resizable(False, False)
            progress_dialog.geometry("400x200")
            
            # Center dialog
            progress_dialog.update_idletasks()
            x = (progress_dialog.winfo_screenwidth() // 2) - (progress_dialog.winfo_width() // 2)
            y = (progress_dialog.winfo_screenheight() // 2) - (progress_dialog.winfo_height() // 2)
            progress_dialog.geometry(f"+{x}+{y}")
            
            # Progress content
            main_frame = ttk.Frame(progress_dialog, padding=20)
            main_frame.pack(fill='both', expand=True)
            
            ttk.Label(main_frame, text="Analyzing columns for conversion...", 
                     font=('TkDefaultFont', 10, 'bold')).pack(pady=(0, 15))
            
            progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
            progress_bar.pack(fill='x', pady=(0, 15))
            progress_bar.start()
            
            status_label = ttk.Label(main_frame, text="Starting analysis...", wraplength=350)
            status_label.pack(pady=(0, 15))
            
            # Update status function
            def update_status(message):
                status_label.config(text=message)
                progress_dialog.update_idletasks()
            
            converted = []
            total_columns = len([col for col in self.current_data.columns 
                               if str(col).upper() not in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']])
            processed = 0
            
            for col in list(self.current_data.columns):
                name_upper = str(col).upper()
                if name_upper in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']:
                    continue
                
                processed += 1
                update_status(f"Analyzing column {processed}/{total_columns}: {col}")
                
                series = pd.to_numeric(self.current_data[col], errors='coerce')
                unit = self.curve_info.get(col, {}).get('unit', '')
                unit_upper = str(unit).upper()
                
                should_convert = False
                conversion_reason = ""
                
                # Check if unit indicates percent
                if '%' in unit_upper or unit_upper in ['PERCENT', 'PCT', 'PERC']:
                    should_convert = True
                    conversion_reason = f"Unit indicates percent ({unit})"
                # Check if curve name suggests fractional family
                elif self._is_fractional_family_name(name_upper):
                    vals = series.dropna()
                    if len(vals) > 10:
                        med = float(np.median(vals))
                        # Heuristic: likely percent if median within (1, 100] and few extreme values
                        if 1.0 < med <= 100.0:
                            should_convert = True
                            conversion_reason = f"Fractional family curve with median value {med:.2f} (likely percent)"
                
                if should_convert:
                    update_status(f"Converting {col}: {conversion_reason}")
                    
                    # Convert from percent to decimal
                    self.current_data[col] = series / 100.0
                    
                    # Update curve info
                    if col in self.curve_info:
                        self.curve_info[col]['unit'] = 'v/v'
                        self.curve_info[col]['original_unit'] = unit or self.curve_info[col].get('original_unit', '')
                    
                    converted.append(col)
                else:
                    update_status(f"Skipping {col}: No conversion needed")
            
            # Close progress dialog
            progress_dialog.destroy()
            
            if converted:
                # Refresh statistics and display
                self.ensure_curve_statistics()
                self.update_data_display()
                
                # Show comprehensive success message
                note = f"Successfully converted {len(converted)} column(s) from percent to decimal (v/v):\n\n"
                sample = '\n'.join([f"• {col}" for col in converted[:15]])
                if len(converted) > 15:
                    sample += f"\n• ... and {len(converted) - 15} more columns"
                
                details = f"{note}{sample}\n\n"
                details += "Conversion Details:\n"
                details += f"• Total columns analyzed: {total_columns}\n"
                details += f"• Columns converted: {len(converted)}\n"
                details += f"• Columns unchanged: {total_columns - len(converted)}\n"
                details += "• All converted columns now use v/v (volume/volume) units\n"
                details += "• Original units have been preserved in curve metadata"
                
                messagebox.showinfo("Conversion Complete", details)
                
                # Update status manager if available
                if hasattr(self, 'status_manager') and self.status_manager:
                    self.status_manager.update_status(f"Unit conversion completed: {len(converted)} columns converted from % to v/v")
            else:
                messagebox.showinfo("No Conversion Needed", 
                                  "No columns were identified for conversion.\n\n"
                                  "Analysis Results:\n"
                                  f"• Total columns analyzed: {total_columns}\n"
                                  "• Columns are either already in decimal format\n"
                                  "• Or don't appear to be percent-based\n\n"
                                  "No action was taken.")
                
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Unit Conversion",
                    component="UnitStandardizer",
                    user_action="Converting units",
                    remediation_hint="Please check the data and conversion parameters."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Conversion Error", error_msg)
            
            # Update status manager if available
            if hasattr(self, 'status_manager') and self.status_manager:
                self.status_manager.update_status(f"Conversion error: {str(e)}")
    
    def open_percent_conversion_dialog(self):
        """Open dialog for manual column selection and conversion."""
        try:
            if self.current_data is None or self.current_data.empty:
                messagebox.showwarning("No Data", "Please load data first before opening conversion dialog.")
                return
            
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Columns for Percent to Decimal Conversion")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.resizable(False, False)
            dialog.geometry("500x400")
            
            # Main frame
            main_frame = ttk.Frame(dialog, padding=15)
            main_frame.pack(fill='both', expand=True)
            
            # Instructions label (will be updated if no columns found)
            instructions_label = ttk.Label(main_frame, text="Select columns to convert from percent (%) to decimal (v/v):", 
                     font=('TkDefaultFont', 10, 'bold'))
            instructions_label.pack(anchor='w', pady=(0, 10))
            
            # Create scrollable frame for checkboxes
            canvas = tk.Canvas(main_frame, highlightthickness=0)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            def update_scroll_region(event=None):
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            scrollable_frame.bind("<Configure>", update_scroll_region)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Variables to store checkbox states
            checkbox_vars = {}
            columns_added = 0
            
            # Create checkboxes for each column
            for col in self.current_data.columns:
                name_upper = str(col).upper()
                if name_upper in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']:
                    continue
                
                # Determine if column should be pre-checked
                unit = self.curve_info.get(col, {}).get('unit', '')
                unit_upper = str(unit).upper()
                is_percent = ('%' in unit_upper or unit_upper in ['PERCENT', 'PCT', 'PERC'])
                is_fractional = self._is_fractional_family_name(name_upper)
                
                var = tk.BooleanVar(value=is_percent or is_fractional)
                checkbox_vars[col] = var
                
                # Create frame for each checkbox with additional info
                row_frame = ttk.Frame(scrollable_frame)
                row_frame.pack(fill='x', padx=5, pady=2)
                
                # Checkbox
                cb = ttk.Checkbutton(row_frame, text=col, variable=var)
                cb.pack(side='left')
                
                # Additional info label
                info_text = []
                if is_percent:
                    info_text.append("Unit: %")
                if is_fractional:
                    info_text.append("Fractional family")
                
                if info_text:
                    info_label = ttk.Label(row_frame, text=f" ({', '.join(info_text)})", 
                                         foreground='blue', font=('TkDefaultFont', 8))
                    info_label.pack(side='left', padx=(5, 0))
                
                columns_added += 1
            
            # If no columns found, show a message
            if columns_added == 0:
                no_columns_label = ttk.Label(scrollable_frame, 
                                           text="No convertible columns found.\n\nAll available columns are either depth columns\nor do not require percent conversion.", 
                                           font=('TkDefaultFont', 10),
                                           foreground='gray',
                                           justify='center')
                no_columns_label.pack(fill='both', expand=True, pady=20)
                
                # Update instructions
                instructions_label.config(
                    text="No columns available for conversion.\nAll columns are either depth columns or already in decimal format.",
                    foreground='gray',
                    font=('TkDefaultFont', 9)
                )
            
            # Pack canvas and scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Update canvas scroll region after widgets are added
            scrollable_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            
            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(15, 0))
            
            # Convert selected button
            def convert_selected():
                selected_cols = [col for col, var in checkbox_vars.items() if var.get()]
                if not selected_cols:
                    messagebox.showwarning("No Selection", "Please select at least one column to convert.")
                    return
                
                try:
                    converted = []
                    for col in selected_cols:
                        series = pd.to_numeric(self.current_data[col], errors='coerce')
                        unit = self.curve_info.get(col, {}).get('unit', '')
                        
                        # Convert from percent to decimal
                        self.current_data[col] = series / 100.0
                        
                        # Update curve info
                        if col in self.curve_info:
                            self.curve_info[col]['unit'] = 'v/v'
                            self.curve_info[col]['original_unit'] = unit or self.curve_info[col].get('original_unit', '')
                        
                        converted.append(col)
                    
                    # Refresh display
                    self.ensure_curve_statistics()
                    self.update_data_display()
                    
                    # Close dialog and show success message
                    dialog.destroy()
                    messagebox.showinfo("Conversion Complete", 
                                      f"Successfully converted {len(converted)} column(s) from percent to decimal (v/v).")
                    
                    # Update status manager if available
                    if hasattr(self, 'status_manager') and self.status_manager:
                        self.status_manager.update_status(f"Manual conversion completed: {len(converted)} columns converted from % to v/v")
                    
                except Exception as e:
                    if self.error_handler:
                        context = self.error_handler.create_context(
                            operation="Unit Conversion",
                            component="UnitStandardizer",
                            user_action="Converting units",
                            remediation_hint="Please check the data and conversion parameters."
                        )
                        self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
                    else:
                        messagebox.showerror("Conversion Error", f"Error during conversion: {str(e)}")
            
            convert_btn = ttk.Button(button_frame, text="Convert Selected", command=convert_selected)
            convert_btn.pack(side='left', padx=(0, 10))
            
            # Disable convert button if no columns available
            if columns_added == 0:
                convert_btn.config(state='disabled')
            
            # Cancel button
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
            cancel_btn.pack(side='left')
            
            # Center dialog on screen
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
        except Exception as e:
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Unit Conversion Dialog",
                    component="UI",
                    user_action="Opening conversion dialog",
                    remediation_hint="Please try again or restart the application."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Dialog Error", f"Error opening conversion dialog: {str(e)}")
    
    def categorize_error(self, error: Exception, operation: str) -> str:
        """Categorize errors for better user feedback"""
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        if "memory" in error_str or "MemoryError" in error_type:
            return "MEMORY_ERROR"
        elif "import" in error_str or "ModuleNotFoundError" in error_type or "ImportError" in error_type:
            return "DEPENDENCY_ERROR"
        elif "ValueError" in error_type or "IndexError" in error_type or "KeyError" in error_type:
            return "DATA_ERROR"
        elif "timeout" in error_str or "TimeoutError" in error_type:
            return "TIMEOUT_ERROR"
        elif "permission" in error_str or "PermissionError" in error_type:
            return "PERMISSION_ERROR"
        elif "file" in error_str or "FileNotFoundError" in error_type or "OSError" in error_type:
            return "FILE_ERROR"
        elif "network" in error_str or "ConnectionError" in error_type:
            return "NETWORK_ERROR"
        else:
            return "GENERAL_ERROR"
    
    def on_window_resize(self, event):
        """Handle window resize events"""
        if hasattr(self, 'main_canvas') and hasattr(self, 'main_canvas_window'):
            try:
                # Update scroll region
                bbox = self.main_canvas.bbox("all")
                if bbox:
                    self.main_canvas.configure(scrollregion=bbox)
                
                # Update canvas window width to match canvas width
                canvas_width = self.main_canvas.winfo_width()
                if canvas_width > 1:
                    self.main_canvas.itemconfig(self.main_canvas_window, width=canvas_width)
                
                # Force scrollbar update
                self.root.update_idletasks()
            except (tk.TclError, RuntimeError) as update_error:
                # Root window update failed - graceful degradation
                self.handle_ui_error(
                    update_error,
                    "Root window update (scrollbar)",
                    DIALOG_ROOT_WINDOW,
                    graceful_degradation=True
                )
            except Exception as update_error:
                # Unexpected error updating root window
                self.handle_ui_error(
                    update_error,
                    "Root window update (scrollbar)",
                    DIALOG_ROOT_WINDOW,
                    graceful_degradation=True
                )
    
    def cleanup_visualization(self):
        """Clean up visualization resources to prevent memory leaks and duplicate toolbars"""
        try:
            # CRITICAL: Remove all widgets from viz_content FIRST to prevent toolbar accumulation
            # This must happen before destroying canvas/figures to avoid orphaned widgets
            if hasattr(self, 'viz_content') and self.viz_content:
                try:
                    # Get all child widgets before destroying (avoids modification during iteration)
                    children = list(self.viz_content.winfo_children())
                    for widget in children:
                        try:
                            widget.destroy()
                        except Exception as e:
                            # Log but continue cleaning up other widgets
                            self.log_processing(f"Warning: Error destroying widget: {e}")
                except Exception as e:
                    warnings.warn(f"Error cleaning up viz_content widgets: {e}", UserWarning)
            
            # Clean up canvas if it exists
            if hasattr(self, 'canvas') and self.canvas is not None:
                try:
                    canvas_widget = self.canvas.get_tk_widget()
                    if canvas_widget and canvas_widget.winfo_exists():
                        canvas_widget.destroy()
                except Exception as e:
                    warnings.warn(f"Error cleaning up canvas: {e}", UserWarning)
                finally:
                    self.canvas = None

            # Clean up figure if it exists
            if hasattr(self, 'fig') and self.fig is not None:
                try:
                    plt.close(self.fig)
                except Exception as e:
                    warnings.warn(f"Error closing figure: {e}", UserWarning)
                finally:
                    self.fig = None

            # Clean up any remaining matplotlib figures
            try:
                plt.close('all')
            except Exception as e:
                warnings.warn(f"Error in global matplotlib cleanup: {e}", UserWarning)

            # Force garbage collection
            try:
                gc.collect()
            except Exception as e:
                warnings.warn(f"Error in garbage collection: {e}", UserWarning)

            return True
        except Exception as e:
            warnings.warn(f"Critical error in visualization cleanup: {e}", UserWarning)
            warnings.warn(
                f"CRITICAL: Visualization cleanup failed completely: {str(e)}. "
                f"Memory leaks likely. Consider restarting application.",
                UserWarning
            )
            self.log_processing(f"ERROR: Critical visualization cleanup failure: {e}")
            try:
                plt.close('all')
                gc.collect()
            except (RuntimeError, OSError) as cleanup_error:
                # Matplotlib cleanup failed - graceful degradation
                self.handle_graceful_degradation(
                    cleanup_error,
                    "Matplotlib cleanup in reset",
                    "Some figures may remain open - continuing reset"
                )
            except Exception as cleanup_error:
                # Unexpected error in matplotlib cleanup
                self.handle_ui_error(
                    cleanup_error,
                    "Matplotlib cleanup in reset",
                    "matplotlib",
                    graceful_degradation=True
                )
            return False
    
    # =============================================================================
    # MEDIUM PRIORITY FIX #1: ADVANCED LIBRARY AVAILABILITY MANAGEMENT
    # Enterprise-grade graceful degradation for optional dependencies
    # =============================================================================
    
    def validate_advanced_libraries_availability(self):
        """Comprehensive validation of optional advanced libraries with graceful degradation"""
        
        library_status = {
            'scipy': {'available': SCIPY_AVAILABLE, 'features': [], 'fallbacks': []},
            'sklearn': {'available': SKLEARN_AVAILABLE, 'features': [], 'fallbacks': []},
            'pywavelets': {'available': PYWT_AVAILABLE, 'features': [], 'fallbacks': []},
            'lasio': {'available': LASIO_AVAILABLE, 'features': [], 'fallbacks': []},
            'psutil': {'available': PSUTIL_AVAILABLE, 'features': [], 'fallbacks': []}
        }
        
        # Detailed feature mapping and fallback strategies
        if SCIPY_AVAILABLE:
            library_status['scipy']['features'] = [
                'Advanced interpolation (cubic spline, kriging)',
                'Signal processing (Savitzky-Golay, bilateral filtering)', 
                'Statistical analysis (correlation, regression)',
                'Optimization algorithms'
            ]
        else:
            library_status['scipy']['fallbacks'] = [
                'Linear interpolation using numpy.interp()',
                'Basic smoothing using numpy.convolve()',
                'Simple correlation using numpy.corrcoef()',
                'Gradient descent optimization'
            ]
        
        if SKLEARN_AVAILABLE:
            library_status['sklearn']['features'] = [
                'Gaussian Process gap filling',
                'Machine learning outlier detection',
                'Advanced preprocessing (scaling, PCA)',
                'Performance metrics (R², RMSE)'
            ]
        else:
            library_status['sklearn']['fallbacks'] = [
                'Statistical interpolation methods',
                'Z-score outlier detection', 
                'Manual data scaling',
                'Custom metric calculations'
            ]
        
        if PYWT_AVAILABLE:
            library_status['pywavelets']['features'] = [
                'Wavelet denoising (Daubechies, Coiflets)',
                'Multi-resolution signal analysis',
                'Adaptive thresholding',
                'Curve-specific wavelet optimization'
            ]
        else:
            library_status['pywavelets']['fallbacks'] = [
                'Moving average smoothing',
                'Median filtering',
                'Simple threshold denoising',
                'Generic smoothing parameters'
            ]
        
        # Log comprehensive library status for enterprise deployment

        for lib_name, status in library_status.items():
            if status['available']:
                self.log_processing(f"[AVAILABLE] {lib_name}: Available - Features: {len(status['features'])}")
            else:
                self.log_processing(f"WARNING: {lib_name}: Not available - Using fallbacks")
                if status['fallbacks']:
                    self.log_processing(f"  Fallbacks: {', '.join(status['fallbacks'][:3])}")
        
        return library_status

    def create_robust_gap_filler_with_fallbacks(self):
        """Create gap filler with intelligent fallback for missing libraries"""
        
        # Validate library availability
        lib_status = self.validate_advanced_libraries_availability()
        
        # Configure gap filling parameters based on available libraries
        gap_params = GapFillingParameters()
        
        # Adjust parameters based on library availability
        if not SKLEARN_AVAILABLE:
            # Disable ML-based methods if sklearn not available
            # Machine learning capabilities disabled due to missing dependencies
            gap_params.multi_curve_correlation = False
            
        if not SCIPY_AVAILABLE:
            # Disable advanced interpolation if scipy not available  
            # Advanced interpolation methods disabled due to missing dependencies
            pass
            gap_params.physics_informed = False
        
        # Create gap filler with appropriate configuration
        self.gap_filler = AdvancedGapFiller(gap_params)
        try:
            def ui_notify(title: str, message: str):
                self.root.after(0, lambda: messagebox.showerror(title, message))
            setattr(self.gap_filler, 'ui_notify', ui_notify)
        except Exception:
            pass
        
        # Configure fallback processing methods
        self._configure_processing_fallbacks(lib_status)
        
        return lib_status

    def _configure_processing_fallbacks(self, lib_status):
        """Configure processing fallbacks based on library availability"""
        
        self.processing_capabilities = {
            'advanced_gap_filling': lib_status['sklearn']['available'] and lib_status['scipy']['available'],
            'wavelet_denoising': lib_status['pywavelets']['available'],
            'advanced_interpolation': lib_status['scipy']['available'],
            'performance_monitoring': lib_status['psutil']['available'],
            'las_file_support': lib_status['lasio']['available']
        }
        
        # Configure method priorities based on available capabilities
        if not self.processing_capabilities['wavelet_denoising']:
            # Update denoising method options for UI
            if hasattr(self, 'denoise_method_var'):
                # Remove wavelet option if PyWavelets not available
                current_methods = ['auto', 'bilateral', 'savgol', 'median']  # No wavelet
                # Information logging removed
                        # Wavelet denoising unavailable - using alternative methods
                pass
        
        if not self.processing_capabilities['advanced_gap_filling']:
            # Limit gap filling options
            pass  # info removed("Advanced gap filling limited - using basic interpolation methods")

    # =============================================================================
    # MEDIUM PRIORITY FIX #2: COMPREHENSIVE MEMORY MANAGEMENT ENHANCEMENT
    # Enterprise-grade memory optimization and leak prevention
    # =============================================================================

    def implement_comprehensive_memory_management(self):
        """Enterprise-grade memory management with monitoring and optimization"""
        
        # Initialize memory monitoring
        self.memory_monitor = {
            'initial_usage': self._get_current_memory_usage(),
            'peak_usage': 0,
            'cleanup_threshold': self.memory_limit_var.get() if hasattr(self, 'memory_limit_var') else 2048,
            'cleanup_frequency': 10,  # Cleanup every 10 operations
            'operation_count': 0
        }
        
        # Information logging removed
        # System status handled - operation continues
        pass  # f"Memory management initialized - Limit: {self.memory_monitor['cleanup_threshold']}MB")

    def _get_current_memory_usage(self):
        """Get current memory usage with fallback for systems without psutil"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                return {
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': process.memory_percent()
                }
            else:
                # Fallback memory estimation using resource module
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                # Convert to MB (ru_maxrss is in KB on Linux, bytes on macOS)
                import sys
                if sys.platform == 'darwin':  # macOS
                    rss_mb = usage.ru_maxrss / (1024 * 1024)
                else:  # Linux
                    rss_mb = usage.ru_maxrss / 1024
                
                return {
                    'rss_mb': rss_mb,
                    'vms_mb': rss_mb * 1.2,  # Estimate
                    'percent': min(100, rss_mb / 4096 * 100)  # Assume 4GB total
                }
        except Exception as e:
            # Debug information removed for security
            # Operation result handled - continuing safely
            pass  # f"Memory usage detection failed: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

    def monitor_and_cleanup_memory(self, operation_name: str = "unknown"):
        """Monitor memory usage and trigger cleanup when needed"""
        
        try:
            # Increment operation counter
            self.memory_monitor['operation_count'] += 1
            
            # Get current memory usage
            current_memory = self._get_current_memory_usage()
            current_mb = current_memory['rss_mb']
            
            # Update peak usage tracking
            if current_mb > self.memory_monitor['peak_usage']:
                self.memory_monitor['peak_usage'] = current_mb
            
            # Check if cleanup is needed
            cleanup_needed = False
            cleanup_reason = ""
            
            # Threshold-based cleanup (80% of limit)
            memory_limit_mb = getattr(self, 'memory_limit_var', None)
            if memory_limit_mb:
                memory_limit = memory_limit_mb.get() if hasattr(memory_limit_mb, 'get') else MEMORY_LIMIT_DEFAULT_MB
            else:
                memory_limit = MEMORY_LIMIT_DEFAULT_MB
            
            threshold_80_percent = memory_limit * 0.8
            if current_mb > threshold_80_percent:
                cleanup_needed = True
                cleanup_reason = f"Memory threshold exceeded (80%): {current_mb:.1f}MB > {threshold_80_percent:.1f}MB"
            
            # Frequency-based cleanup (every 10 operations instead of default)
            elif self.memory_monitor['operation_count'] % 10 == 0:
                cleanup_needed = True
                cleanup_reason = f"Scheduled cleanup after {self.memory_monitor['operation_count']} operations"
            
            # Pressure-based cleanup: if memory increased significantly since last cleanup
            elif hasattr(self.memory_monitor, 'last_cleanup_memory'):
                memory_increase = current_mb - self.memory_monitor.get('last_cleanup_memory', current_mb)
                if memory_increase > 500:  # More than 500MB increase
                    cleanup_needed = True
                    cleanup_reason = f"Memory pressure detected: {memory_increase:.1f}MB increase since last cleanup"
            
            # Perform cleanup if needed
            if cleanup_needed:
                memory_before = current_mb
                self._perform_comprehensive_memory_cleanup(operation_name)
                
                # Measure cleanup effectiveness
                memory_after = self._get_current_memory_usage()['rss_mb']
                memory_freed = memory_before - memory_after
                
                # Store last cleanup memory for pressure detection
                if not hasattr(self.memory_monitor, 'last_cleanup_memory'):
                    self.memory_monitor['last_cleanup_memory'] = memory_after
                else:
                    self.memory_monitor['last_cleanup_memory'] = memory_after
                
                # Log cleanup if significant memory was freed
                if memory_freed > 50:  # Only log if more than 50MB freed
                    self.log_processing(f"Memory cleanup triggered: {cleanup_reason}")
                    self.log_processing(f"Memory freed: {memory_freed:.1f}MB ({memory_before:.1f}MB → {memory_after:.1f}MB)")
                
                # Reset operation counter
                self.memory_monitor['operation_count'] = 0
            
            # Enterprise monitoring: log significant memory increases
            if current_mb > self.memory_monitor['initial_usage']['rss_mb'] * 2:
                import warnings
                warnings.warn(
                    f"Memory usage doubled: {current_mb:.1f}MB vs initial {self.memory_monitor['initial_usage']['rss_mb']:.1f}MB. "
                    f"Consider restarting application or reducing data size.",
                    UserWarning
                )
                self.log_processing(f"WARNING: Memory doubled during {operation_name}")
            
        except Exception as e:
            self.log_processing(f"Memory monitoring failed for '{operation_name}': {str(e)}")

    def _perform_comprehensive_memory_cleanup(self, context: str):
        """Comprehensive memory cleanup with detailed tracking and aggressive strategies"""
        
        cleanup_actions = []
        memory_before = self._get_current_memory_usage()['rss_mb'] if hasattr(self, '_get_current_memory_usage') else 0
        
        try:
            # Phase 1: Matplotlib cleanup - more aggressive
            if hasattr(self, 'fig') and self.fig is not None:
                try:
                    plt.close(self.fig)
                    self.fig = None
                    cleanup_actions.append("Main figure closed")
                except Exception:
                    pass
            if hasattr(self, 'canvas') and self.canvas is not None:
                try:
                    self.canvas.destroy()
                    self.canvas = None
                    cleanup_actions.append("Canvas destroyed")
                except Exception:
                    pass
            # Cleanup all popup figures
            if hasattr(self, 'popup_figures'):
                for fig in self.popup_figures[:]:
                    try:
                        plt.close(fig)
                        self.popup_figures.remove(fig)
                    except Exception:
                        pass
                if self.popup_figures:
                    cleanup_actions.append(f"Closed {len(self.popup_figures)} popup figures")
            # Call visualization cleanup
            if hasattr(self, 'cleanup_visualization'):
                try:
                    self.cleanup_visualization()
                    cleanup_actions.append("Visualization cleanup")
                except Exception:
                    pass
            
            # Phase 2: Large data structure cleanup - more aggressive
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                # Clean up any cached computations in DataFrame
                if hasattr(self.processed_data, '_mgr'):
                    try:
                        # Trigger pandas memory consolidation
                        self.processed_data._consolidate_inplace()
                        cleanup_actions.append("DataFrame consolidation")
                    except Exception:
                        pass
                # Clear DataFrame caches
                if hasattr(self.processed_data, '_cache'):
                    try:
                        self.processed_data._cache.clear()
                        cleanup_actions.append("DataFrame cache cleared")
                    except Exception:
                        pass
            
            # Phase 3: Processing results cleanup - more aggressive (reduce from 50 to 20)
            if hasattr(self, 'processing_results') and len(self.processing_results) > 20:
                # Keep only recent processing results to prevent memory bloat
                recent_results = dict(list(self.processing_results.items())[-20:])
                removed_count = len(self.processing_results) - len(recent_results)
                self.processing_results = recent_results
                cleanup_actions.append(f"Processing results trimmed ({removed_count} removed)")
            
            # Phase 4: Clear any cached curve computations - more aggressive
            if hasattr(self, 'curve_identifier') and hasattr(self.curve_identifier, '_curve_info'):
                # Clear any cached curve analysis that might be holding references
                cache_cleared = 0
                for curve_info in self.curve_identifier._curve_info.values():
                    if hasattr(curve_info, '_cached_data') and curve_info._cached_data is not None:
                        curve_info._cached_data = None
                        cache_cleared += 1
                    # Clear other potential cache attributes
                    for attr in ['_cached_stats', '_cached_analysis', '_computed_values']:
                        if hasattr(curve_info, attr):
                            setattr(curve_info, attr, None)
                if cache_cleared > 0:
                    cleanup_actions.append(f"Curve cache cleared ({cache_cleared} curves)")
            
            # Phase 5: Clear intermediate processing variables
            if hasattr(self, 'auxiliary_curves_dict'):
                try:
                    del self.auxiliary_curves_dict
                    cleanup_actions.append("Auxiliary curves dict cleared")
                except Exception:
                    pass
            
            # Phase 6: Clear numpy array caches
            if hasattr(self, '_numpy_cache'):
                try:
                    self._numpy_cache.clear()
                    cleanup_actions.append("NumPy cache cleared")
                except Exception:
                    pass
            
            # Phase 7: Python garbage collection with multiple passes (increased from 3 to 5)
            total_collected = 0
            for i in range(5):
                collected = gc.collect()
                total_collected += collected
                if collected > 0:
                    cleanup_actions.append(f"GC pass {i+1}: {collected} objects")
            if total_collected > 0:
                cleanup_actions.append(f"Total GC: {total_collected} objects")
            
            # Phase 8: Clear import caches if available
            try:
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                    cleanup_actions.append("Type cache clearing")
            except (AttributeError, RuntimeError) as cache_error:
                # Type cache clearing failed - graceful degradation
                if hasattr(self, 'handle_graceful_degradation'):
                    self.handle_graceful_degradation(
                        cache_error,
                        "Type cache clearing in cleanup",
                        "Type cache may not be cleared - continuing cleanup"
                    )
            except Exception as cache_error:
                # Unexpected error clearing type cache
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        cache_error,
                        "Type cache clearing in cleanup",
                        "sys._clear_type_cache",
                        graceful_degradation=True
                    )
            
            # Phase 9: Clear matplotlib backend caches
            try:
                import matplotlib
                matplotlib.pyplot.close('all')  # Close all figures
                cleanup_actions.append("All matplotlib figures closed")
            except Exception:
                pass
            
            # Log cleanup summary
            memory_after = self._get_current_memory_usage()['rss_mb'] if hasattr(self, '_get_current_memory_usage') else 0
            memory_freed = memory_before - memory_after
            if memory_freed > 0:
                self.log_processing(f"Memory cleanup for '{context}': Freed {memory_freed:.1f}MB ({len(cleanup_actions)} actions)")
            
        except Exception as e:
            self.log_processing(f"ERROR: Comprehensive memory cleanup failed: {str(e)}")
            warnings.warn(
                f"Memory cleanup failed: {str(e)}. Memory leaks likely. Consider restarting application.",
                UserWarning
            )

    # =============================================================================
    # LOW PRIORITY FIX #1: BETA ANALYTICS INTEGRATION ROBUSTNESS
    # Professional-grade analytics with comprehensive error handling
    # =============================================================================

    def track_event_with_enhanced_error_handling(self, event_type: str, details: Dict = None):
        """Enhanced event tracking with comprehensive error handling and validation"""
        
        if not self.feature_flags.should_collect_analytics():
            return
        
        try:
            # Validate event data before processing
            validated_details = self._validate_and_sanitize_event_data(details or {})
            
            # Create comprehensive event record
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': validated_details,
                'session_id': getattr(self, 'session_id', 'unknown'),
                'app_version': self.feature_flags.get_beta_version(),
                'system_info': self._get_system_context()
            }
            
            # Add to session data with size management
            self.session_data['events'].append(event)
            
            # Prevent unbounded growth of event list
            max_events = 1000
            if len(self.session_data['events']) > max_events:
                # Keep most recent events
                self.session_data['events'] = self.session_data['events'][-max_events:]
                # Beta analytics removed for production security
                # Operation result handled - continuing safely
            pass  # f"Event list trimmed to {max_events} most recent events")
            
            # Periodic analytics persistence for enterprise reliability
            if len(self.session_data['events']) % 50 == 0:
                self._persist_analytics_safely()
            
        except Exception as e:
            # Event tracking failed - continuing operation silently
            pass

    def _validate_and_sanitize_event_data(self, details: Dict) -> Dict:
        """Validate and sanitize event data for enterprise compliance"""
        
        sanitized = {}
        
        try:
            for key, value in details.items():
                # Validate key format
                if not isinstance(key, str) or len(key) > 100:
                    # Beta analytics removed for production security
                    # Operation result handled - continuing safely
                    continue
                
                # Sanitize and validate values
                if isinstance(value, (str, int, float, bool)):
                    # Basic types - validate size
                    if isinstance(value, str) and len(value) > 1000:
                        sanitized[key] = value[:1000] + "...[truncated]"
                        # Beta analytics removed for production security
                        # Operation result handled - continuing safely
                    else:
                        sanitized[key] = value
                elif isinstance(value, (list, tuple)):
                    # Arrays - limit size and validate elements
                    if len(value) > 100:
                        sanitized[key] = list(value[:100]) + ["...[truncated]"]
                    else:
                        sanitized[key] = [v for v in value if isinstance(v, (str, int, float, bool))]
                elif isinstance(value, dict):
                    # Nested objects - recursive validation
                    sanitized[key] = self._validate_and_sanitize_event_data(value)
                else:
                    # Other types - convert to string representation
                    sanitized[key] = str(value)[:200]  # Limit string representation
            
            return sanitized
            
        except Exception as e:
            # beta logger removed(f"Event data sanitization failed: {e}")
            return {"sanitization_error": str(e)}

    def _get_system_context(self) -> Dict:
        """Get system context for analytics with comprehensive error handling"""
        
        context = {}
        
        try:
            # Platform information
            context['platform'] = {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'python_version': platform.python_version()
            }
            
            # Memory information if available
            if PSUTIL_AVAILABLE:
                try:
                    virtual_memory = psutil.virtual_memory()
                    context['memory'] = {
                        'total_gb': round(virtual_memory.total / (1024**3), 1),
                        'available_gb': round(virtual_memory.available / (1024**3), 1),
                        'percent_used': virtual_memory.percent
                    }
                except Exception:
                    context['memory'] = {'status': 'unavailable'}
            
            # Library availability context
            context['libraries'] = {
                'scipy': SCIPY_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE, 
                'pywavelets': PYWT_AVAILABLE,
                'lasio': LASIO_AVAILABLE,
                'psutil': PSUTIL_AVAILABLE
            }
            
        except Exception as e:
            # beta logger removed(f"System context collection failed: {e}")
            context['error'] = str(e)
        
        return context

    def _persist_analytics_safely(self):
        """Safely persist analytics data with error recovery"""
        
        try:
            # Use existing SafeFileHandler for robust file operations
            if hasattr(self, 'analytics_file'):
                success = SafeFileHandler.safe_write_json(self.analytics_file, self.session_data)
                if success:
                    # Analytics persistence successful - no action required
                    pass
                else:
                    # Analytics persistence failed - continuing operation
                    pass
            
        except Exception as e:
            # Analytics persistence error - operation continues safely
            pass

    def get_comprehensive_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics with detailed analytics"""
        
        try:
            events = self.session_data.get('events', [])
            
            # Basic event counting
            basic_stats = {
                'total_events': len(events),
                'files_loaded': len([e for e in events if e['event_type'] == 'file_loaded']),
                'processing_runs': len([e for e in events if e['event_type'] == 'processing_completed']),
                'exports_completed': len([e for e in events if e['event_type'] == 'export_attempt']),
                'visualizations_created': len([e for e in events if e['event_type'] == 'visualization_created']),
                'errors_encountered': len([e for e in events if e['event_type'] == 'error'])
            }
            
            # Advanced analytics for enterprise insights
            advanced_stats = {}
            
            # Performance analytics
            processing_events = [e for e in events if e['event_type'] == 'processing_completed']
            if processing_events:
                processing_times = [e['details'].get('processing_time_seconds', 0) for e in processing_events]
                advanced_stats['performance'] = {
                    'avg_processing_time': sum(processing_times) / len(processing_times),
                    'max_processing_time': max(processing_times),
                    'total_curves_processed': sum(e['details'].get('curve_count', 0) for e in processing_events)
                }
            
            # Error analytics
            error_events = [e for e in events if e['event_type'] == 'error']
            if error_events:
                error_types = {}
                for event in error_events:
                    error_type = event['details'].get('error_type', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                advanced_stats['errors'] = {
                    'error_types': error_types,
                    'error_rate': len(error_events) / max(1, len(events))
                }
            
            # Combine basic and advanced statistics
            comprehensive_stats = {**basic_stats, **advanced_stats}
            
            return comprehensive_stats
            
        except Exception as e:
            # beta logger removed(f"Error generating comprehensive usage stats: {e}")
            return {
                'total_events': 0,
                'files_loaded': 0,
                'processing_runs': 0,
                'exports_completed': 0,
                'visualizations_created': 0,
                'errors_encountered': 0,
                'stats_error': str(e)
            }
    def ensure_figure_exists(self):
        """Create a clean matplotlib figure with proper memory management"""
        try:
            # Always clean up existing figure first
            if hasattr(self, 'fig') and self.fig is not None:
                try:
                    plt.close(self.fig)
                except Exception as e:
                    warnings.warn(f"Error closing previous figure: {e}", UserWarning)
            
            # Create new figure with standard settings
            self.fig = Figure(figsize=(12, 8), dpi=100, tight_layout=True)
            
            # Configure for professional output
            self.fig.patch.set_facecolor('white')
            self.fig.patch.set_alpha(1.0)
            
            # Debug information removed for security
            # Operation result handled - continuing safely
            return self.fig
            
        except Exception as e:
            # Log figure creation failure and attempt fallback
            warnings.warn(
                f"Primary figure creation failed: {str(e)}. "
                f"Attempting fallback with minimal settings.",
                UserWarning
            )
            
            # Fallback: minimal figure creation
            try:
                self.fig = Figure(figsize=(8, 6), dpi=72)
                self.log_processing("Using fallback figure settings due to creation error")
                return self.fig
            except Exception as fe:
                # Complete failure - log and return None
                self.log_processing(f"ERROR: Both primary and fallback figure creation failed: {fe}")
                import warnings
                warnings.warn(
                    f"CRITICAL: Cannot create matplotlib figure. Visualization will not work. "
                    f"Error: {str(fe)}. Check matplotlib installation.",
                    UserWarning
                )
                self.fig = None
                return None
    
    def run_visualization_in_main_thread(self, viz_function, *args):
        """Run visualization functions in the main thread"""
        # Schedule visualization on main thread
        self.root.after(0, lambda: viz_function(*args))
    
    def schedule_ui_update(self, update_type: str, **kwargs):
        """Professional thread-safe UI update scheduling with performance tracking"""
        try:
            # Validate threading context
            if threading.current_thread() == threading.main_thread():
                # Direct execution if already on main thread
                self._execute_ui_update(update_type, **kwargs)
            else:
                # Schedule on main thread with performance monitoring
                start_time = time.time()
                
                def monitored_update():
                    try:
                        self._execute_ui_update(update_type, **kwargs)
                        
                        # Performance monitoring for enterprise deployment
                        elapsed = time.time() - start_time
                        if elapsed > 0.1:  # Performance monitoring threshold
                            # Slow UI update detected - performance optimization may be needed
                            pass
                            
                    except Exception as e:
                        # Log scheduled UI update failure
                        import warnings
                        warnings.warn(
                            f"Scheduled UI update failed for '{update_type}': {str(e)}",
                            UserWarning
                        )
                
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after_idle(monitored_update)
                
        except Exception as e:
            # Log UI update scheduling failure
            warnings.warn(
                f"UI update scheduling failed for '{update_type}': {str(e)}. "
                f"UI may not reflect current state.",
                UserWarning
            )

    def _execute_ui_update(self, update_type: str, **kwargs):
        """Execute UI updates with comprehensive validation"""
        try:
            if update_type == 'progress':
                progress_value = kwargs.get('progress', 0)
                status_text = kwargs.get('status', '')
                
                # Update progress bar with validation
                if hasattr(self, 'progress_bar') and self.progress_bar:
                    try:
                        self.progress_bar.configure(value=progress_value)
                    except tk.TclError as e:
                        # Debug information removed for security
                        # Operation result handled - continuing safely
                        pass
                
                # Update status label with validation
                if hasattr(self, 'status_label') and self.status_label:
                    try:
                        self.status_label.config(text=status_text)
                    except tk.TclError as e:
                        # Debug information removed for security
                        # Operation result handled - continuing safely
                        pass
            
            elif update_type == 'results':
                message = kwargs.get('message', '')
                if hasattr(self, 'results_text') and self.results_text:
                    try:
                        self.results_text.insert(tk.END, f"{message}\n")
                        self.results_text.see(tk.END)
                        self.results_text.update_idletasks()
                    except tk.TclError as e:
                        # Tcl error in results text update - widget may be destroyed
                        import warnings
                        warnings.warn(
                            f"Results text update failed (widget may be destroyed): {str(e)}",
                            UserWarning
                        )
            
        except Exception as e:
            # Log UI update execution failure
            warnings.warn(
                f"UI update execution failed for '{update_type}': {str(e)}. "
                f"Check widget availability and thread context.",
                UserWarning
            )
    
    def _get_memory_usage(self):
        """Get current memory usage in MB for performance monitoring"""
        if not PSUTIL_AVAILABLE:
            return 0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ProcessLookupError as e:
            # Process may have terminated - not critical for monitoring
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Memory usage check failed: Process not found: {str(e)}")
        except AttributeError as e:
            # psutil may not have expected attributes - not critical
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Memory usage check failed: Attribute error: {str(e)}")
        except (RuntimeError, OSError) as e:
            # Process access issues or OS-level errors - not critical for monitoring
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Memory usage check failed: {type(e).__name__}: {str(e)}")
        except Exception as e:
            # Log unexpected errors for debugging
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Unexpected error in memory usage check: {type(e).__name__}: {str(e)}")
        return 0
    
    def setup_beta_features(self):
        """Setup beta-specific features"""
        # Update title
        current_title = self.root.title()
        self.root.title(f"{current_title} - Beta Version {self.feature_flags.get_beta_version()}")
        
        # Add beta menu
        self.create_beta_menu()
        
        # Track app start
        self.beta_analytics.track_event('app_started', {
            'version': self.feature_flags.get_beta_version()
        })
    
    def create_beta_menu(self):
        """Create beta testing menu"""
        # Create menubar if it doesn't exist
        try:
            menubar = self.root.cget('menu')
            if not menubar:
                menubar = tk.Menu(self.root)
                self.root.config(menu=menubar)
        except (tk.TclError, AttributeError) as e:
            # Tkinter widget may not be fully initialized or accessed incorrectly
            try:
                menubar = tk.Menu(self.root)
                self.root.config(menu=menubar)
            except Exception as menu_error:
                # Log if menu creation completely fails
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"Failed to create beta menu: {type(menu_error).__name__}: {str(menu_error)}")
                return  # Cannot create menu, skip beta menu
        except Exception as e:
            # Unexpected error - log for debugging
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Unexpected error creating beta menu: {type(e).__name__}: {str(e)}")
            return  # Cannot create menu, skip beta menu
        
        # Beta menu
        beta_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Beta Testing", menu=beta_menu)
        
        beta_menu.add_command(label="Submit Feedback...", 
                             command=lambda: self.feedback_collector.show_feedback_dialog(self.root))
        beta_menu.add_separator()
        beta_menu.add_command(label="Usage Statistics", command=self.show_usage_stats)
        beta_menu.add_command(label="System Information", command=self.show_system_info)
        beta_menu.add_separator()
        beta_menu.add_command(label="About Beta Program", command=self.show_beta_info)
    
    def show_usage_stats(self):
        """Show usage statistics"""
        if not BETA_SYSTEM_AVAILABLE or not self.beta_analytics:
            return
            
        stats = self.beta_analytics.get_usage_stats()
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Usage Statistics")
        stats_window.geometry("400x300")
        stats_window.transient(self.root)
        stats_window.grab_set()
        
        stats_text = f"""
BETA TESTING USAGE STATISTICS

Total Events Recorded: {stats.get('total_events', 0)}
Files Loaded: {stats.get('files_loaded', 0)}
Processing Operations: {stats.get('processing_runs', 0)}
Exports Completed: {stats.get('exports_completed', 0)}
Visualizations Created: {stats.get('visualizations_created', 0)}
Errors Encountered: {stats.get('errors_encountered', 0)}

User ID: {self.feature_flags.user_id}
Version: {self.feature_flags.get_beta_version()}

These statistics help improve software performance and reliability.
        """
        
        text_widget = scrolledtext.ScrolledText(stats_window, wrap='word', font=('Arial', 10))
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, stats_text.strip())
        text_widget.config(state='disabled')
        
        close_btn = tk.Button(stats_window, text="Close", command=stats_window.destroy,
                             font=('Arial', 10), padx=20)
        close_btn.pack(pady=(0, 20))
    
    def show_system_info(self):
        """Show system information"""
        import platform
        
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("400x300")
        info_window.transient(self.root)
        info_window.grab_set()
        
        system_info = f"""
SYSTEM INFORMATION

Operating System: {platform.system()}
OS Version: {platform.version()}
Architecture: {platform.architecture()[0]}
Machine: {platform.machine()}
Processor: {platform.processor()}
Python Version: {platform.python_version()}

Beta Version: {self.feature_flags.get_beta_version() if BETA_SYSTEM_AVAILABLE else 'N/A'}
User ID: {self.feature_flags.user_id if BETA_SYSTEM_AVAILABLE else 'N/A'}
        """
        
        text_widget = scrolledtext.ScrolledText(info_window, wrap='word', font=('Arial', 10))
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, system_info.strip())
        text_widget.config(state='disabled')
        
        close_btn = tk.Button(info_window, text="Close", command=info_window.destroy,
                             font=('Arial', 10), padx=20)
        close_btn.pack(pady=(0, 20))
    
    def show_beta_info(self):
        """Show beta program information"""
        if not BETA_SYSTEM_AVAILABLE:
            return
            
        info_text = f"""
BETA TESTING PROGRAM
Version {self.feature_flags.get_beta_version()}

TESTING SCOPE:
• Advanced data processing algorithms
• Export functionality and file formats
• User interface and workflow efficiency
• Performance with large datasets
• Cross-platform compatibility

COMPLIMENTARY FEATURES (Beta Period):
• Unlimited data processing operations
• All visualization capabilities
• Data export functionality (normally premium)
• Advanced gap filling algorithms
• Professional quality reporting

FEEDBACK PRIORITIES:
• Processing accuracy and reliability
• Software performance and speed
• User interface usability
• File format compatibility
• Feature requests and improvements

Technical Support: {self.feature_flags.flags['feedback_email']}

Thank you for participating in our beta testing program.
Your feedback contributes to software quality and reliability.
        """
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Beta Testing Information")
        info_window.geometry("500x400")
        info_window.transient(self.root)
        info_window.grab_set()
        
        text_widget = scrolledtext.ScrolledText(info_window, wrap='word', font=('Arial', 10))
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state='disabled')
        
        close_btn = tk.Button(info_window, text="Close", command=info_window.destroy,
                             font=('Arial', 10), padx=20)
        close_btn.pack(pady=(0, 20))
    
    def log_processing(self, message: str) -> None:
        """Route processing messages to the on-screen status UI only (no file logging)."""
        try:
            # Error handler should be initialized in __init__ after setup_ui()
            # This is a fallback only if initialization failed
            if self.error_handler is None and hasattr(self, 'root'):
                try:
                    self.error_handler = CentralizedErrorHandler(
                        root=self.root,
                        log_callback=self.log_processing
                    )
                except Exception:
                    # Fallback if error handler initialization fails
                    self.error_handler = None
            
            if hasattr(self, 'status_manager') and self.status_manager:
                self.status_manager.update_status(message)
                return
            # Fallbacks if status_manager is not available yet
            if hasattr(self, 'results_text') and self.results_text:
                try:
                    self.results_text.insert(tk.END, f"{message}\n")
                    self.results_text.see(tk.END)
                    self.results_text.update_idletasks()
                except (tk.TclError, AttributeError) as ui_error:
                    # UI widget may be destroyed or invalid - graceful degradation
                    pass
                except Exception as ui_error:
                    # Unexpected UI error - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Results text update failed: {type(ui_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
            if hasattr(self, 'status_label') and self.status_label:
                try:
                    self.status_label.config(text=message)
                except (tk.TclError, AttributeError) as status_error:
                    # Status label may be destroyed or invalid - graceful degradation
                    pass
                except Exception as status_error:
                    # Unexpected status update error - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Status label update failed: {type(status_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
        except Exception:
            # Absolutely no file logging, and avoid raising during UI init
            pass
    
    def _log_processing_internal(self, message: str) -> None:
        """Internal logging method for error handler (prevents recursion)."""
        try:
            if hasattr(self, 'status_manager') and self.status_manager:
                self.status_manager.update_status(message)
                return
            # Fallbacks if status_manager is not available yet
            if hasattr(self, 'results_text') and self.results_text:
                try:
                    self.results_text.insert(tk.END, f"{message}\n")
                    self.results_text.see(tk.END)
                    self.results_text.update_idletasks()
                except (tk.TclError, AttributeError):
                    pass
            if hasattr(self, 'status_label') and self.status_label:
                try:
                    self.status_label.config(text=message)
                except (tk.TclError, AttributeError):
                    pass
        except Exception:
            # Last resort: print to console
            print(f"[LOG] {message}")

    def _begin_operation(self, message: str) -> None:
        """Unified user feedback when a long-running operation starts."""
        try:
            self.log_processing(message)
            if hasattr(self, 'status_label'):
                self.status_label.config(text=message)
            if hasattr(self, 'progress_bar'):
                try:
                    self.progress_bar.config(mode='indeterminate')
                    self.progress_bar.start(10)
                except (tk.TclError, AttributeError) as progress_error:
                    # Progress bar may not be initialized or destroyed - graceful degradation
                    pass
                except Exception as progress_error:
                    # Unexpected error with progress bar - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Progress bar start failed: {type(progress_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
            try:
                self.root.update_idletasks()
            except (tk.TclError, RuntimeError) as update_error:
                # Root window may be destroyed - graceful degradation
                pass
            except Exception as update_error:
                # Unexpected error updating UI - log if possible
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: UI update failed: {type(update_error).__name__}")
                    except Exception:
                        pass  # Can't log logging failure
        except Exception as begin_error:
            # Unexpected error in begin operation - log if possible
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Begin operation failed: {type(begin_error).__name__}: {str(begin_error)}")
                except Exception:
                    pass  # Can't log logging failure

    def _end_operation(self, message: str) -> None:
        """Unified user feedback when an operation completes successfully."""
        try:
            if hasattr(self, 'progress_bar'):
                try:
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate', value=100)
                except (tk.TclError, AttributeError) as progress_error:
                    # Progress bar may not be initialized or destroyed - graceful degradation
                    pass
                except Exception as progress_error:
                    # Unexpected error with progress bar - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Progress bar stop failed: {type(progress_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
            if hasattr(self, 'status_label'):
                try:
                    self.status_label.config(text=message)
                except (tk.TclError, AttributeError):
                    # Status label may be destroyed - graceful degradation
                    pass
            self.log_processing(message)
            try:
                self.root.update_idletasks()
            except (tk.TclError, RuntimeError):
                # Root window may be destroyed - graceful degradation
                pass
            except Exception as update_error:
                # Unexpected error updating UI - log if possible
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: UI update failed: {type(update_error).__name__}")
                    except Exception:
                        pass  # Can't log logging failure
        except Exception as end_error:
            # Unexpected error in end operation - log if possible
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: End operation failed: {type(end_error).__name__}: {str(end_error)}")
                except Exception:
                    pass  # Can't log logging failure

    def _fail_operation(self, title: str, message: str) -> None:
        """Unified error feedback with dialog and status label."""
        try:
            if hasattr(self, 'progress_bar'):
                try:
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate', value=0)
                except (tk.TclError, AttributeError) as progress_error:
                    # Progress bar may not be initialized or destroyed - graceful degradation
                    pass
                except Exception as progress_error:
                    # Unexpected error with progress bar - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Progress bar reset failed: {type(progress_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
            if hasattr(self, 'status_label'):
                try:
                    self.status_label.config(text=message)
                except (tk.TclError, AttributeError):
                    # Status label may be destroyed - graceful degradation
                    pass
                except Exception as status_error:
                    # Unexpected error updating status - log if possible
                    if hasattr(self, 'log_processing'):
                        try:
                            self.log_processing(f"Warning: Status label update failed: {type(status_error).__name__}")
                        except Exception:
                            pass  # Can't log logging failure
        except Exception as fail_error:
            # Unexpected error in fail operation - log if possible
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Fail operation setup failed: {type(fail_error).__name__}: {str(fail_error)}")
                except Exception:
                    pass  # Can't log logging failure
        try:
            messagebox.showerror(title, message)
        except (tk.TclError, RuntimeError) as dialog_error:
            # Dialog display failed (may be off main thread or window destroyed) - log but continue
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Error dialog display failed: {type(dialog_error).__name__}: {str(dialog_error)}")
                except Exception:
                    pass  # Can't log logging failure
            # Last resort: print to console
            print(f"ERROR: {title}: {message}")
        except Exception as dialog_error:
            # Unexpected error displaying dialog - log and print to console
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error displaying error dialog: {type(dialog_error).__name__}: {str(dialog_error)}")
                except Exception:
                    pass  # Can't log logging failure
            print(f"ERROR: {title}: {message}")
    
    # ========================================================================
    # CENTRALIZED ERROR HANDLING HELPERS
    # ========================================================================
    # Professional error handling wrappers for common patterns
    # These methods provide consistent error handling across the application
    
    def handle_ui_error(self, 
                       error: Exception,
                       operation: str,
                       widget_name: str = None,
                       graceful_degradation: bool = True) -> None:
        """
        Handle UI-related errors with graceful degradation.
        
        Args:
            error: The exception that occurred
            operation: Description of the UI operation
            widget_name: Name of the widget (optional)
            graceful_degradation: If True, continue without crashing (default: True)
        """
        if self.error_handler is None:
            # Fallback if error handler not initialized
            print(f"[UI ERROR] {operation}: {error}")
            return
        
        context = self.error_handler.create_context(
            operation=operation,
            component=widget_name or "UI",
            user_action=f"UI operation: {operation}",
            remediation_hint="UI widget may have been destroyed or invalid. This is usually non-critical."
        )
        
        # UI errors are typically non-critical - use warning severity for graceful degradation
        severity = ErrorSeverity.WARNING if graceful_degradation else ErrorSeverity.ERROR
        
        # Don't show dialog for UI errors - just log (too disruptive)
        self.error_handler.handle_error(
            error=error,
            context=context,
            severity=severity,
            show_dialog=False,  # UI errors shouldn't interrupt user workflow
            log_error=True
        )
    
    def handle_processing_error(self,
                               error: Exception,
                               operation: str,
                               data_context: str = None,
                               show_dialog: bool = True) -> None:
        """
        Handle data processing errors with user notification.
        
        Args:
            error: The exception that occurred
            operation: Description of the processing operation
            data_context: Context about what data was being processed
            show_dialog: Whether to show error dialog to user (default: True)
        """
        if self.error_handler is None:
            # Fallback if error handler not initialized
            print(f"[PROCESSING ERROR] {operation}: {error}")
            if show_dialog:
                try:
                    messagebox.showerror(ERROR_TITLE_PROCESSING, f"{operation}: {error}")
                except (tk.TclError, RuntimeError) as msg_error:
                    # Messagebox display failed - graceful degradation
                    # Already in error handler, so just log to console
                    print(f"[ERROR] Failed to display error dialog: {msg_error}")
                except Exception as msg_error:
                    # Unexpected error displaying messagebox
                    print(f"[ERROR] Unexpected error displaying error dialog: {msg_error}")
            return
        
        context = self.error_handler.create_context(
            operation=operation,
            component="Data Processing",
            user_action=data_context or "Processing data",
            remediation_hint="Please check your input data and try again. Verify data format and quality.",
            data_info=data_context
        )
        
        self.error_handler.handle_error(
            error=error,
            context=context,
            severity=ErrorSeverity.ERROR,
            show_dialog=show_dialog,
            log_error=True
        )
    
    def handle_file_error(self,
                         error: Exception,
                         operation: str,
                         file_path: str = None,
                         file_operation: str = None) -> None:
        """
        Handle file operation errors with user-friendly messages.
        
        Args:
            error: The exception that occurred
            operation: Description of the file operation
            file_path: Path to the file (will be sanitized for display)
            file_operation: Type of file operation (read/write/validate)
        """
        if self.error_handler is None:
            # Fallback if error handler not initialized
            print(f"[FILE ERROR] {operation}: {error}")
            try:
                sanitized = SafeFileHandler.sanitize_path_for_display(file_path) if file_path else "unknown"
                if self.error_handler:
                    context = self.error_handler.create_context(
                        operation=operation,
                        component="FileOperations",
                        user_action="File operation",
                        remediation_hint=f"Please check file: {sanitized}",
                        additional_info={"file_path": sanitized}
                    )
                    self.error_handler.handle_error(Exception(str(error)), context, severity=ErrorSeverity.ERROR)
                else:
                    messagebox.showerror(ERROR_TITLE_FILE, f"{operation}: {error}\nFile: {sanitized}")
            except Exception:
                pass
            return
        
        # Sanitize file path for privacy
        sanitized_path = SafeFileHandler.sanitize_path_for_display(file_path) if file_path else "unknown"
        
        remediation = "Please check file path, permissions, and ensure file is not open in another program."
        if file_operation == "write":
            remediation += " Verify you have write permissions to the target directory."
        elif file_operation == "read":
            remediation += " Verify the file exists and is accessible."
        
        context = self.error_handler.create_context(
            operation=operation,
            component="File Operations",
            user_action=f"File {file_operation or 'operation'}",
            remediation_hint=remediation,
            file_path=sanitized_path
        )
        
        self.error_handler.handle_error(
            error=error,
            context=context,
            severity=ErrorSeverity.ERROR,
            show_dialog=True,
            log_error=True
        )
    
    def handle_graceful_degradation(self,
                                   error: Exception,
                                   operation: str,
                                   fallback_message: str = None) -> None:
        """
        Handle non-critical errors that allow graceful degradation.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation
            fallback_message: Message to display/log if fallback behavior is used
        """
        if self.error_handler is None:
            # Fallback if error handler not initialized
            print(f"[WARNING] {operation}: {error}")
            return
        
        context = self.error_handler.create_context(
            operation=operation,
            component="Application",
            user_action="Non-critical operation",
            remediation_hint=fallback_message or "Operation failed but application continues normally."
        )
        
        # Use warning severity - not critical, don't interrupt user
        self.error_handler.handle_warning(
            message=f"{operation}: {error}",
            context=context,
            show_dialog=False  # Warnings shouldn't interrupt workflow
        )
    
    def check_system_resources(self):
        """Monitor system resources periodically"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)
                
                if memory_usage > 1000:  # Over 1GB
                    import warnings
                    warnings.warn(
                        f"High memory usage detected: {memory_usage:.1f}MB. "
                        f"Triggering garbage collection. Consider closing unused visualizations.",
                        UserWarning
                    )
                    self.log_processing(f"High memory usage: {memory_usage:.1f}MB - running garbage collection")
                    gc.collect()
        except Exception:
            pass  # psutil not available or other issues
        
        # Schedule next check
        self.root.after(10000, self.check_system_resources)  # Every 10 seconds
    
    def clear_data(self):
        """Clear loaded and processed data and reset UI elements safely."""
        # Use the comprehensive reset method
        self.reset_application_state(prompt_if_unsaved=False)

    # ============================
    # Multiwell management methods
    # ============================
    def _snapshot_single_state(self) -> Dict[str, Any]:
        return {
            'current_data': self.current_data.copy() if isinstance(self.current_data, pd.DataFrame) else None,
            'processed_data': self.processed_data.copy() if isinstance(self.processed_data, pd.DataFrame) else None,
            'curve_info': copy.deepcopy(self.curve_info) if isinstance(self.curve_info, dict) else {},
            'processing_results': copy.deepcopy(self.processing_results) if isinstance(self.processing_results, dict) else {},
            'original_las_header': self.original_las_header,
            'well_info': copy.deepcopy(self.well_info) if isinstance(self.well_info, dict) else {},
            'file_path': self.file_path_var.get() if hasattr(self, 'file_path_var') else ''
        }

    def _apply_dataset_to_single_state(self, dataset: Dict[str, Any]) -> None:
        # Replace current in-memory single-well state with dataset contents
        self.current_data = dataset.get('current_data')
        self.processed_data = dataset.get('processed_data')
        self.curve_info = copy.deepcopy(dataset.get('curve_info', {}))
        self.processing_results = copy.deepcopy(dataset.get('processing_results', {}))
        self.original_las_header = dataset.get('original_las_header')
        self.well_info = copy.deepcopy(dataset.get('well_info', {}))
        try:
            self.file_path_var.set(dataset.get('file_path', ''))
        except (tk.TclError, AttributeError) as path_error:
            # UI variable may not be initialized - log but continue
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: File path variable update failed: {type(path_error).__name__}")
                except Exception:
                    pass  # Can't log logging failure
        except Exception as path_error:
            # Unexpected error setting file path - log for debugging
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error setting file path: {type(path_error).__name__}: {str(path_error)}")
                except Exception:
                    pass  # Can't log logging failure
        # Refresh UI elements tied to single state
        try:
            self._update_well_info_display()
            self.update_curve_options()
            self.update_data_display()
        except (tk.TclError, AttributeError) as ui_refresh_error:
            # UI refresh failed - log but don't fail dataset loading
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: UI refresh failed during dataset load: {type(ui_refresh_error).__name__}: {str(ui_refresh_error)}")
                except Exception:
                    pass  # Can't log logging failure
        except Exception as ui_refresh_error:
            # Unexpected error refreshing UI - log for debugging
            if hasattr(self, 'log_processing'):
                try:
                    self.log_processing(f"Warning: Unexpected error refreshing UI: {type(ui_refresh_error).__name__}: {str(ui_refresh_error)}")
                except Exception:
                    pass  # Can't log logging failure

    def _dataset_from_current_state(self, file_path: str) -> Dict[str, Any]:
        return {
            'file_path': file_path,
            'current_data': self.current_data.copy() if isinstance(self.current_data, pd.DataFrame) else None,
            'processed_data': self.processed_data.copy() if isinstance(self.processed_data, pd.DataFrame) else None,
            'curve_info': copy.deepcopy(self.curve_info) if isinstance(self.curve_info, dict) else {},
            'processing_results': copy.deepcopy(self.processing_results) if isinstance(self.processing_results, dict) else {},
            'original_las_header': self.original_las_header,
            'well_info': copy.deepcopy(self.well_info) if isinstance(self.well_info, dict) else {},
        }

    def _gen_well_id_from_info(self, filepath: str) -> str:
        try:
            uwi = str(self.well_info.get('uwi', '')).strip()
            name = str(self.well_info.get('well_name', '')).strip()
            base = os.path.splitext(os.path.basename(filepath or ''))[0]
            candidate = uwi or name or base or f"well_{len(self.well_datasets)+1}"
            candidate = candidate.replace(' ', '_')
        except Exception:
            candidate = f"well_{len(self.well_datasets)+1}"
        # Ensure uniqueness
        unique = candidate
        idx = 2
        while unique in self.well_datasets:
            unique = f"{candidate}_{idx}"
            idx += 1
        return unique

    def _save_active_well_to_dataset(self) -> None:
        try:
            if not self.active_well_id:
                return
            self.well_datasets[self.active_well_id] = self._dataset_from_current_state(
                self.file_path_var.get() if hasattr(self, 'file_path_var') else ''
            )
        except (tk.TclError, AttributeError) as file_var_error:
            # UI variable access failed - use empty string and log
            self.handle_ui_error(
                file_var_error,
                "File path variable access in well info update",
                "file_path_var",
                graceful_degradation=True
            )
        except Exception as file_var_error:
            # Unexpected error accessing file path variable
            self.handle_ui_error(
                file_var_error,
                "File path variable access in well info update",
                "file_path_var",
                graceful_degradation=True
            )

    def set_active_well(self, well_id: str) -> None:
        if well_id not in self.well_datasets:
            messagebox.showwarning("Well Selection", f"Well '{well_id}' not found")
            return
        self.active_well_id = well_id
        self._apply_dataset_to_single_state(self.well_datasets[well_id])
        try:
            self.status_label.config(text=f"Active well: {well_id}")
        except (tk.TclError, AttributeError) as status_error:
            # Status label update failed - graceful degradation
            self.handle_ui_error(
                status_error,
                "Status label update (set active well)",
                "status_label",
                graceful_degradation=True
            )
        except Exception as status_error:
            # Unexpected error updating status label
            self.handle_ui_error(
                status_error,
                "Status label update (set active well)",
                "status_label",
                graceful_degradation=True
            )
        # Refresh lists
        self.update_well_list_display()
        try:
            if hasattr(self, 'cohort_listbox') and self.cohort_listbox:
                self.cohort_listbox.delete(0, tk.END)
                for wid in self.well_datasets.keys():
                    if wid != self.active_well_id:
                        self.cohort_listbox.insert(tk.END, wid)
        except (tk.TclError, AttributeError) as listbox_error:
            # Listbox update failed - graceful degradation
            self.handle_ui_error(
                listbox_error,
                "Cohort listbox update",
                "cohort_listbox",
                graceful_degradation=True
            )
        except Exception as listbox_error:
            # Unexpected error updating listbox
            self.handle_ui_error(
                listbox_error,
                "Cohort listbox update",
                "cohort_listbox",
                graceful_degradation=True
            )

    def update_well_list_display(self) -> None:
        try:
            if not hasattr(self, 'well_listbox') or self.well_listbox is None:
                return
            self.well_listbox.delete(0, tk.END)
            for wid, ds in self.well_datasets.items():
                wi = ds.get('well_info', {}) or {}
                label = wi.get('well_name') or wi.get('uwi') or wid
                rows = len(ds.get('current_data')) if isinstance(ds.get('current_data'), pd.DataFrame) else 0
                cols = len(ds.get('current_data').columns) if isinstance(ds.get('current_data'), pd.DataFrame) else 0
                self.well_listbox.insert(tk.END, f"{wid}  |  {label}  |  {rows}x{cols}")
        except (tk.TclError, AttributeError) as listbox_error:
            # Listbox update failed - graceful degradation
            self.handle_ui_error(
                listbox_error,
                "Well listbox update",
                "well_listbox",
                graceful_degradation=True
            )
        except Exception as listbox_error:
            # Unexpected error updating listbox
            self.handle_ui_error(
                listbox_error,
                "Well listbox update",
                "well_listbox",
                graceful_degradation=True
            )

    def on_set_active_well(self):
        try:
            sel = self.well_listbox.curselection()
            if not sel:
                messagebox.showwarning("Well Selection", "Select a well from the list")
                return
            display = self.well_listbox.get(sel[0])
            wid = display.split("  |  ")[0]
            self.set_active_well(wid)
        except Exception as e:
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Well Selection",
                    component="MultiWellManager",
                    user_action="Selecting well",
                    remediation_hint="Please check well data and try again."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Selection Error", f"Failed to set active well: {e}")

    def on_remove_selected_wells(self):
        try:
            sel = self.well_listbox.curselection()
            if not sel:
                return
            display = self.well_listbox.get(sel[0])
            wid = display.split("  |  ")[0]
            if wid in self.well_datasets:
                del self.well_datasets[wid]
            if self.active_well_id == wid:
                self.active_well_id = None
                self.reset_application_state(prompt_if_unsaved=False)
            self.update_well_list_display()
        except Exception as remove_error:
            # Error removing well - log and continue
            self.handle_processing_error(
                remove_error,
                "Remove selected wells",
                "Removing selected wells from dataset",
                show_dialog=False
            )

    def load_multiple_files(self):
        try:
            filetypes = [("LAS files", "*.las"), ("DLIS/LIS files", "*.dlis *.lis"), ("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
            filenames = filedialog.askopenfilenames(title="Select Multiple Data Files", filetypes=filetypes)
            if not filenames:
                return
            # Load each file into datasets without disturbing final active selection until the end
            first_well_id = None
            for fp in filenames:
                try:
                    # Security: Validate and normalize file path
                    validated_path = SafeFileHandler.validate_file_path(fp)
                    if not validated_path:
                        sanitized = SafeFileHandler.sanitize_path_for_display(fp)
                        self.log_processing(f"Security: Invalid file path skipped: {sanitized}")
                        continue
                    
                    # Security: Validate file size
                    if not SafeFileHandler.validate_file_size(str(validated_path)):
                        size_mb = os.path.getsize(str(validated_path)) / (1024 * 1024)
                        max_mb = SafeFileHandler.MAX_FILE_SIZE_MB
                        self.log_processing(f"File too large skipped: {SafeFileHandler.sanitize_path_for_display(fp)} ({size_mb:.1f}MB > {max_mb}MB)")
                        continue
                    
                    # Security: Validate file extension
                    if not SafeFileHandler.validate_file_extension(str(validated_path), mode='read'):
                        ext = os.path.splitext(str(validated_path))[1].lower()
                        self.log_processing(f"Invalid file type skipped: {SafeFileHandler.sanitize_path_for_display(fp)} (extension: {ext})")
                        continue
                    
                    # Use validated path
                    fp = str(validated_path)
                    
                    # Clear transient state
                    self.reset_application_state(prompt_if_unsaved=False)
                    ext = os.path.splitext(fp)[1].lower()
                    if ext == '.las':
                        df = self.load_las_file(fp)
                    elif ext == '.csv':
                        df = self.load_csv_file(fp)
                    elif ext in ['.xlsx', '.xls']:
                        df = self.load_excel_file(fp)
                    else:
                        self.log_processing(f"Unsupported file format skipped: {fp}")
                        continue
                    self.current_data = df
                    # Identify curves (lightweight)
                    self.analyze_curves()
                    # Build dataset
                    wid = self._gen_well_id_from_info(fp)
                    self.well_datasets[wid] = self._dataset_from_current_state(fp)
                    if first_well_id is None:
                        first_well_id = wid
                    self.log_processing(f"Loaded well '{wid}' from {fp}")
                except Exception as e:
                    if self.error_handler:
                        context = self.error_handler.create_context(
                            operation="File Loading",
                            component="FileLoader",
                            user_action="Loading file",
                            remediation_hint=f"Please check file: {fp}",
                            additional_info={"file_path": fp}
                        )
                        self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
                    else:
                        messagebox.showerror("Load Error", f"Failed to load file {fp}: {e}")
            # Set active to the first loaded well and refresh list
            if first_well_id:
                self.set_active_well(first_well_id)
            self.update_well_list_display()
            # After all wells are in well_datasets, sync NULL (prompt only on conflict)
            self._reconcile_null_value_convention(allow_prompt=True)
        except Exception as e:
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Batch File Loading",
                    component="FileLoader",
                    user_action="Loading multiple files",
                    remediation_hint="Please check file paths and formats."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Load Error", f"Failed to load multiple files: {e}")

    def process_current_well_blocking(self):
        try:
            # Spawn processing thread but keep UI responsive
            t = threading.Thread(target=self.process_data_thread, daemon=True)
            t.start()
            while t.is_alive():
                try:
                    self.root.update()
                except (tk.TclError, RuntimeError) as update_error:
                    # Root window update failed - continue without update
                    self.handle_ui_error(
                        update_error,
                        "Root window update during file load",
                        DIALOG_ROOT_WINDOW,
                        graceful_degradation=True
                    )
                except Exception as update_error:
                    # Unexpected error updating root window
                    self.handle_ui_error(
                        update_error,
                        "Root window update during file load",
                        DIALOG_ROOT_WINDOW,
                        graceful_degradation=True
                    )
                time.sleep(0.05)
            # Persist results into dataset
            self._save_active_well_to_dataset()
        except Exception as e:
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Well Processing",
                    component="ProcessingPipeline",
                    user_action="Processing well data",
                    remediation_hint="Please check data quality and processing parameters."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror(ERROR_TITLE_PROCESSING, f"Failed to process well: {e}")

    def process_all_wells(self):
        try:
            if not self.well_datasets:
                messagebox.showwarning("Process All Wells", "No wells loaded. Use 'Load Multiple Files' first.")
                return
            # Show processing tab for visual feedback
            try:
                self.notebook.select(1)
            except Exception:
                pass
            self._begin_operation("Processing all wells...")
            ordered_ids = list(self.well_datasets.keys())
            total = len(ordered_ids)
            for i, wid in enumerate(ordered_ids, start=1):
                self.set_active_well(wid)
                try:
                    self.status_label.config(text=f"Processing well {i}/{total}: {wid}")
                except (tk.TclError, AttributeError) as status_error:
                    # Status label update failed - graceful degradation
                    self.handle_ui_error(
                        status_error,
                        "Status label update (process all wells)",
                        "status_label",
                        graceful_degradation=True
                    )
                except Exception as status_error:
                    # Unexpected error updating status label
                    self.handle_ui_error(
                        status_error,
                        "Status label update (process all wells)",
                        "status_label",
                        graceful_degradation=True
                    )
                self.process_current_well_blocking()
            self._end_operation(f"Processed all wells ({total})")
            try:
                messagebox.showinfo("Process All Wells", f"Completed processing {total} well(s).")
            except (tk.TclError, RuntimeError) as msg_error:
                # Messagebox display failed - graceful degradation
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (process all wells)",
                    "messagebox",
                    graceful_degradation=True
                )
            except Exception as msg_error:
                # Unexpected error displaying messagebox
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (process all wells)",
                    "messagebox",
                    graceful_degradation=True
                )
        except Exception as e:
            self._fail_operation("Process All Wells", f"Failed to process all wells: {e}")

    def process_selected_wells(self):
        try:
            if not hasattr(self, 'well_listbox') or self.well_listbox is None:
                self._fail_operation("Process Selected", "Well list is not available.")
                return
            sel = self.well_listbox.curselection()
            if not sel:
                messagebox.showwarning("Process Selected", "Select one or more wells in the list.")
                return
            selected_ids = []
            for idx in sel:
                display = self.well_listbox.get(idx)
                wid = display.split("  |  ")[0]
                if wid in self.well_datasets:
                    selected_ids.append(wid)
            if not selected_ids:
                messagebox.showwarning("Process Selected", "No valid wells selected.")
                return
            # Switch to Processing tab for progress visibility
            try:
                self.notebook.select(1)
            except Exception:
                pass
            self._begin_operation(f"Processing {len(selected_ids)} selected well(s)...")
            for i, wid in enumerate(selected_ids, start=1):
                self.set_active_well(wid)
                try:
                    self.status_label.config(text=f"Processing well {i}/{len(selected_ids)}: {wid}")
                except (tk.TclError, AttributeError) as status_error:
                    # Status label update failed - graceful degradation
                    self.handle_ui_error(
                        status_error,
                        "Status label update (process selected wells)",
                        "status_label",
                        graceful_degradation=True
                    )
                except Exception as status_error:
                    # Unexpected error updating status label
                    self.handle_ui_error(
                        status_error,
                        "Status label update (process selected wells)",
                        "status_label",
                        graceful_degradation=True
                    )
                self.process_current_well_blocking()
            self._end_operation(f"Processed {len(selected_ids)} selected well(s)")
            try:
                messagebox.showinfo("Process Selected", f"Completed processing {len(selected_ids)} well(s).")
            except (tk.TclError, RuntimeError) as msg_error:
                # Messagebox display failed - graceful degradation
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (process selected wells)",
                    "messagebox",
                    graceful_degradation=True
                )
            except Exception as msg_error:
                # Unexpected error displaying messagebox
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (process selected wells)",
                    "messagebox",
                    graceful_degradation=True
                )
        except Exception as e:
            self._fail_operation("Process Selected", f"Failed to process selected wells: {e}")

    def _format_cross_well_summary(self) -> str:
        lines: List[str] = []
        lines.append("CROSS-WELL ANALYSIS SUMMARY")
        lines.append("=" * 80)
        
        if not self.well_datasets:
            lines.append("No wells loaded.")
            return "\n".join(lines)
        
        num_wells = len(self.well_datasets)
        lines.append(f"Dataset Overview: {num_wells} well(s) loaded")
        lines.append("")
        
        # 1. WELL INFORMATION SUMMARY
        lines.append("1. WELL INFORMATION")
        lines.append("-" * 40)
        for wid, ds in self.well_datasets.items():
            wi = ds.get('well_info', {}) or {}
            well_name = wi.get('well_name', 'Unknown')
            uwi = wi.get('uwi', 'N/A')
            field = wi.get('field', 'Unknown')
            company = wi.get('company', 'Unknown')
            depth_range = wi.get('depth_range', 'N/A')
            
            lines.append(f"  • {well_name}")
            lines.append(f"    UWI: {uwi}")
            lines.append(f"    Field: {field}")
            lines.append(f"    Company: {company}")
            lines.append(f"    Depth Range: {depth_range}")
            lines.append("")
        
        # 2. DATA QUALITY ASSESSMENT
        lines.append("2. DATA QUALITY ASSESSMENT")
        lines.append("-" * 40)
        
        total_curves = 0
        total_data_points = 0
        total_missing = 0
        curve_quality_stats = {}
        
        for wid, ds in self.well_datasets.items():
            df = ds.get('current_data')
            if isinstance(df, pd.DataFrame):
                total_curves += len(df.columns)
                total_data_points += df.size
                missing_count = df.isna().sum().sum()
                total_missing += missing_count
                
                # Per-well quality
                well_missing_pct = (missing_count / df.size) * 100 if df.size > 0 else 0
                lines.append(f"  {wid}: {len(df.columns)} curves, {len(df)} points, {well_missing_pct:.1f}% missing")
                
                # Track curve quality
                for col in df.columns:
                    if col not in curve_quality_stats:
                        curve_quality_stats[col] = {'total_points': 0, 'missing_points': 0, 'wells': 0}
                    
                    series = pd.to_numeric(df[col], errors='coerce')
                    curve_quality_stats[col]['total_points'] += len(series)
                    curve_quality_stats[col]['missing_points'] += series.isna().sum()
                    curve_quality_stats[col]['wells'] += 1
        
        overall_missing_pct = (total_missing / total_data_points) * 100 if total_data_points > 0 else 0
        lines.append(f"")
        lines.append(f"  Overall: {total_curves} total curves, {total_data_points:,} data points")
        lines.append(f"  Missing Data: {total_missing:,} points ({overall_missing_pct:.1f}%)")
        lines.append("")
        
        # 3. COMMON CURVES ANALYSIS
        lines.append("3. COMMON CURVES ANALYSIS")
        lines.append("-" * 40)
        
        # Find common curves
        common_curves = None
        for ds in self.well_datasets.values():
            df = ds.get('current_data')
            if isinstance(df, pd.DataFrame):
                cols = set(df.columns)
                common_curves = cols if common_curves is None else (common_curves & cols)
        
        if not common_curves:
            lines.append("No common curves across all wells.")
        else:
            lines.append(f"Found {len(common_curves)} common curves across all wells:")
            lines.append("")
            
            # Analyze each common curve
            for curve in sorted(list(common_curves)):
                if curve in curve_quality_stats:
                    stats = curve_quality_stats[curve]
                    completeness = ((stats['total_points'] - stats['missing_points']) / stats['total_points']) * 100
                    lines.append(f"  • {curve}:")
                    lines.append(f"    - Present in {stats['wells']} wells")
                    lines.append(f"    - Data completeness: {completeness:.1f}%")
                    
                    # Get statistical summary
                    all_values = []
                    for ds in self.well_datasets.values():
                        df = ds.get('current_data')
                        if isinstance(df, pd.DataFrame) and curve in df.columns:
                            series = pd.to_numeric(df[curve], errors='coerce')
                            vals = series.dropna()
                            if len(vals) > 0:
                                all_values.extend(vals.tolist())
                    
                    if all_values:
                        all_values = np.array(all_values)
                        lines.append(f"    - Range: {all_values.min():.3g} to {all_values.max():.3g}")
                        lines.append(f"    - Mean: {all_values.mean():.3g}")
                        lines.append(f"    - Std Dev: {all_values.std():.3g}")
                        lines.append(f"    - Median: {np.median(all_values):.3g}")
                    lines.append("")
        
        # 4. PROCESSING STATUS
        lines.append("4. PROCESSING STATUS")
        lines.append("-" * 40)
        
        processed_wells = 0
        for wid, ds in self.well_datasets.items():
            if ds.get('processed_data') is not None:
                processed_wells += 1
        
        lines.append(f"Processed Wells: {processed_wells}/{num_wells} ({processed_wells/num_wells*100:.1f}%)")
        
        if processed_wells > 0:
            lines.append("")
            lines.append("Processing Results Summary:")
            for wid, ds in self.well_datasets.items():
                if ds.get('processed_data') is not None:
                    processed_df = ds['processed_data']
                    original_df = ds.get('current_data')
                    
                    if isinstance(processed_df, pd.DataFrame) and isinstance(original_df, pd.DataFrame):
                        original_missing = original_df.isna().sum().sum()
                        processed_missing = processed_df.isna().sum().sum()
                        improvement = original_missing - processed_missing
                        
                        lines.append(f"  • {wid}: {improvement:,} data points filled")
        
        # 5. RECOMMENDATIONS
        lines.append("")
        lines.append("5. RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if overall_missing_pct > 50:
            lines.append("• High missing data percentage - consider data quality improvement")
        elif overall_missing_pct > 20:
            lines.append("• Moderate missing data - gap filling recommended")
        else:
            lines.append("• Good data quality - ready for analysis")
        
        if len(common_curves) < 3:
            lines.append("• Limited common curves - consider standardizing curve names")
        else:
            lines.append("• Good curve coverage across wells")
        
        if processed_wells < num_wells:
            lines.append("• Some wells not processed - run processing pipeline")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("End of Cross-Well Analysis Summary")
        
        return "\n".join(lines)

    def show_cross_well_summary(self):
        try:
            summary = self._format_cross_well_summary()
            if hasattr(self, 'report_text') and self.report_text:
                self.report_text.config(state='normal')
                self.report_text.delete('1.0', 'end')
                self.report_text.insert('1.0', summary)
                self.report_text.config(state='disabled')
                try:
                    self.status_label.config(text="Cross-well summary generated")
                except (tk.TclError, RuntimeError) as msg_error:
                    # Messagebox display failed - graceful degradation
                    self.handle_ui_error(
                        msg_error,
                        "Messagebox display (cross-well summary)",
                        "messagebox",
                        graceful_degradation=True
                    )
                except Exception as msg_error:
                    # Unexpected error displaying messagebox
                    self.handle_ui_error(
                        msg_error,
                        "Messagebox display (cross-well summary)",
                        "messagebox",
                        graceful_degradation=True
                    )
            else:
                messagebox.showinfo("Cross-Well Summary", summary)
        except Exception as e:
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Cross-Well Summary",
                    component="Reporting",
                    user_action="Generating summary",
                    remediation_hint="Please check well data availability."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Cross-Well Summary", f"Failed to generate cross-well summary: {e}")

    def export_all_processed(self):
        """Export all processed wells with security validation for path traversal protection."""
        try:
            if not self.well_datasets:
                messagebox.showwarning("Export All", "No wells loaded.")
                return
            target_dir = filedialog.askdirectory(title="Select export directory")
            if not target_dir:
                return
            
            # Security: Validate target directory
            validated_dir = SafeFileHandler.validate_file_path(target_dir)
            if not validated_dir or not validated_dir.is_dir():
                sanitized = SafeFileHandler.sanitize_path_for_display(target_dir)
                if self.error_handler:
                    context = self.error_handler.create_context(
                        operation="Export Directory Validation",
                        component="Security",
                        user_action="Exporting data",
                        remediation_hint=f"Please use a valid directory path.",
                        additional_info={"directory": sanitized}
                    )
                    self.error_handler.handle_error(Exception(f"Invalid export directory: {sanitized}"), context, severity=ErrorSeverity.ERROR)
                else:
                    messagebox.showerror(ERROR_TITLE_SECURITY, f"Invalid export directory: {sanitized}")
                return
            
            target_dir = str(validated_dir)
            exported = 0
            
            # Optionally rebuild priors before export for audit trail
            if self.use_crosswell_priors_var.get() and self.crosswell_prior_manager:
                try:
                    self.crosswell_priors = self.crosswell_prior_manager.build_priors(
                        depth_binned=self.priors_depth_binning_var.get()
                    )
                except (ValueError, AttributeError, KeyError) as prior_error:
                    # Prior building failed for this well - continue with others
                    self.handle_graceful_degradation(
                        prior_error,
                        "Cross-well prior building",
                        f"Skipping well {wid} - continuing with remaining wells"
                    )
                except Exception as prior_error:
                    # Unexpected error building priors
                    self.handle_processing_error(
                        prior_error,
                        "Cross-well prior building",
                        f"Building priors for well {wid}",
                        show_dialog=False
                    )
            
            for wid, ds in self.well_datasets.items():
                pdf = ds.get('processed_data')
                if not isinstance(pdf, pd.DataFrame) or pdf.empty:
                    continue
                ci = ds.get('curve_info', {}) or {}
                null_value = str(self.null_value_var.get()) if hasattr(self, 'null_value_var') else '-999.25'
                las_text = self._generate_las_text_from_dataframe(pdf, ci, null_value, max_rows=None)
                
                # Security: Validate output filename and ensure it stays within target directory
                # Sanitize well ID to prevent path traversal in filename
                safe_wid = "".join(c for c in str(wid) if c.isalnum() or c in ('-', '_', '.'))
                if not safe_wid:
                    safe_wid = "well"
                
                out_path = os.path.join(target_dir, f"{safe_wid}_processed.las")
                
                # Security: Final validation - ensure output path is within target directory
                final_validated = SafeFileHandler.validate_file_path(out_path, allowed_dir=target_dir)
                if not final_validated:
                    sanitized = SafeFileHandler.sanitize_path_for_display(out_path)
                    if self.error_handler:
                        context = self.error_handler.create_context(
                            operation="Export Path Generation",
                            component="Security",
                            user_action="Exporting data",
                            remediation_hint="Please check export settings and try again.",
                            additional_info={"path": sanitized}
                        )
                        self.error_handler.handle_error(Exception(f"Invalid export path generated: {sanitized}"), context, severity=ErrorSeverity.ERROR)
                    else:
                        messagebox.showerror(ERROR_TITLE_SECURITY, f"Invalid export path generated: {sanitized}")
                    continue
                
                with open(str(final_validated), 'w', encoding='utf-8') as f:
                    f.write(las_text)
                exported += 1
            
            messagebox.showinfo("Export All", f"Exported {exported} processed well(s) to {target_dir}")
        except Exception as e:
            sanitized = SafeFileHandler.sanitize_path_for_display(target_dir if 'target_dir' in locals() else "unknown")
            if self.error_handler:
                context = self.error_handler.create_context(
                    operation="Batch Export",
                    component="ExportManager",
                    user_action="Exporting data",
                    remediation_hint="Please check export settings and file permissions."
                )
                self.error_handler.handle_error(e, context, severity=ErrorSeverity.ERROR)
            else:
                messagebox.showerror("Export All", f"Failed to export: {e}")

    def build_crosswell_priors(self):
        try:
            if not self.use_crosswell_priors_var.get():
                messagebox.showinfo("Cross-Well Priors", "Enable Cross-Well Priors first.")
                return
            if not self.crosswell_prior_manager:
                if self.error_handler:
                    context = self.error_handler.create_context(
                        operation="Cross-Well Priors",
                        component="CrossWellPriorManager",
                        user_action="Accessing priors",
                        remediation_hint="Please initialize cross-well prior manager first."
                    )
                    self.error_handler.handle_error(Exception("Prior manager unavailable."), context, severity=ErrorSeverity.WARNING)
                else:
                    messagebox.showerror("Cross-Well Priors", "Prior manager unavailable.")
                return
            self._begin_operation("Building cross-well priors...")
            priors = self.crosswell_prior_manager.build_priors(depth_binned=self.priors_depth_binning_var.get())
            self.crosswell_priors = priors or {}
            count = len(self.crosswell_priors)
            self._end_operation(f"Cross-well priors built for {count} curves")
            try:
                messagebox.showinfo("Cross-Well Priors", f"Built priors for {count} curves.")
            except (tk.TclError, RuntimeError) as msg_error:
                # Messagebox display failed - graceful degradation
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (cross-well priors)",
                    "messagebox",
                    graceful_degradation=True
                )
            except Exception as msg_error:
                # Unexpected error displaying messagebox
                self.handle_ui_error(
                    msg_error,
                    "Messagebox display (cross-well priors)",
                    "messagebox",
                    graceful_degradation=True
                )
        except Exception as e:
            self._fail_operation("Cross-Well Priors", f"Failed to build priors: {e}")
    
    def reset_application_state(self, prompt_if_unsaved=True):
        """Comprehensive application state reset to prevent cross-contamination between wells
        
        SAFETY CRITICAL: This method ensures complete cleanup between well loads to prevent:
        - Processing results from previous well affecting new well
        - Curve information bleed-over
        - Visualization artifacts from previous data
        - Memory leaks from accumulated state
        
        Args:
            prompt_if_unsaved: If True, warn user if processed data exists and hasn't been saved
        """
        # Check for unsaved processed data
        if prompt_if_unsaved and self.processed_data is not None:
            response = messagebox.askyesno(
                "Unsaved Processed Data",
                "You have processed data that hasn't been saved.\n\n"
                "Loading a new file will discard this data.\n\n"
                "Continue anyway?",
                icon='warning'
            )
            if not response:
                return False  # User cancelled
        
        try:
            # Log reset for audit trail
            self.log_processing("="*60)
            self.log_processing("APPLICATION STATE RESET - Clearing all data")
            self.log_processing("="*60)
            
            # Reset core data structures
            self.current_data = None
            self.processed_data = None
            self.curve_info = {}
            self.processing_results = {}
            self.original_las_header = None
            self._upload_standardization_note = ""
            
            # Reset well information (CRITICAL for safety)
            self.well_info = {
                'well_name': 'UNKNOWN',
                'uwi': 'UNKNOWN',
                'field': 'UNKNOWN',
                'company': 'UNKNOWN'
            }
            
            # Reset geological context
            if hasattr(self, 'geological_context'):
                self.geological_context = GeologicalContext()
            
            # Reset processing history
            if hasattr(self, 'processing_history'):
                self.processing_history.clear_history()
            
            # Reset RRP model
            self.rrp_model = None
            
            # Close all popup visualization windows (prevents memory leaks)
            if hasattr(self, 'popup_windows') and self.popup_windows:
                popup_count = len(self.popup_windows)
                for popup_window in self.popup_windows[:]:  # Copy list to avoid modification during iteration
                    try:
                        popup_window.destroy()
                    except Exception:
                        pass  # Window might already be closed
                self.popup_windows = []
                self.log_processing(f"Closed {popup_count} popup visualization windows")
            
            # Clean up popup figures
            if hasattr(self, 'popup_figures') and self.popup_figures:
                for fig in self.popup_figures[:]:
                    try:
                        plt.close(fig)
                    except (RuntimeError, OSError) as fig_error:
                        # Figure may already be closed - graceful degradation
                        self.handle_graceful_degradation(
                            fig_error,
                            "Popup figure cleanup in reset",
                            "Figure may already be closed - continuing cleanup"
                        )
                    except Exception as fig_error:
                        # Unexpected error closing popup figure
                        self.handle_ui_error(
                            fig_error,
                            "Popup figure cleanup in reset",
                            "matplotlib figure",
                            graceful_degradation=True
                        )
                self.popup_figures = []
            
            # Clear visualization state
            self.cleanup_visualization()
            
            # Reset window title
            self.root.title("Advanced Wireline Data Preprocessing System")
            
            # Update well info display to show "not loaded" state
            self._update_well_info_display()
            
            self.log_processing("Core data structures cleared")
            
        except Exception as e:
            self.log_processing(f"Error during core data reset: {e}")
            # Don't fail silently - this is critical
        
        # Clear data tree
        try:
            if hasattr(self, 'data_tree') and self.data_tree:
                for item in self.data_tree.get_children():
                    self.data_tree.delete(item)
        except (tk.TclError, AttributeError) as tree_error:
            # Tree widget clearing failed - graceful degradation
            self.handle_ui_error(
                tree_error,
                "Data tree widget clearing",
                "data_tree",
                graceful_degradation=True
            )
        except Exception as tree_error:
            # Unexpected error clearing tree widget
            self.handle_ui_error(
                tree_error,
                "Data tree widget clearing",
                "data_tree",
                graceful_degradation=True
            )

    def show_error_dialog(self, title: str, message: str) -> None:
        """Thread-safe error reporting helper shared across background workers.
        
        This method now uses the centralized error handler if available,
        falling back to direct messagebox for backward compatibility.
        """
        # Use centralized error handler if available
        if self.error_handler:
            try:
                context = self.error_handler.create_context(
                    operation="Error Display",
                    component="UI",
                    user_action="Error occurred",
                    remediation_hint="Please check the error message and try again."
                )
                # Determine severity from title
                severity = ErrorSeverity.ERROR
                if "Critical" in title or "Fatal" in title:
                    severity = ErrorSeverity.CRITICAL
                elif "Warning" in title:
                    severity = ErrorSeverity.WARNING
                
                # Create a simple exception for the error handler
                error = Exception(message)
                self.error_handler.handle_error(error, context, severity=severity, show_dialog=True, log_error=True)
                return
            except Exception as handler_error:
                # Centralized handler failed - fall back to direct display
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: Centralized error handler failed: {type(handler_error).__name__}: {str(handler_error)}")
                    except Exception:
                        pass

        def _display():
            try:
                self.log_processing(f"[ERROR] {title}: {message}")
            except Exception as log_error:
                # Logging failed - continue without logging
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        log_error,
                        "Error logging in show_error_dialog",
                        "log_processing",
                        graceful_degradation=True
                    )
            try:
                messagebox.showerror(title, message)
            except (tk.TclError, RuntimeError) as msg_error:
                # Messagebox display failed - graceful degradation
                self.handle_ui_error(
                    msg_error,
                    "Error dialog display",
                    "messagebox",
                    graceful_degradation=True
                )
            except Exception as msg_error:
                # Unexpected error displaying error dialog
                self.handle_ui_error(
                    msg_error,
                    "Error dialog display",
                    "messagebox",
                    graceful_degradation=True
                )

        if threading.current_thread() is threading.main_thread():
            _display()
        else:
            try:
                self.root.after(0, _display)
            except Exception:
                _display()
        
        # Clear results text
        try:
            if hasattr(self, 'results_text') and self.results_text:
                self.results_text.delete('1.0', 'end')
        except (tk.TclError, AttributeError) as text_error:
            # Text widget clearing failed - graceful degradation
            self.handle_ui_error(
                text_error,
                "Results text widget clearing",
                "results_text",
                graceful_degradation=True
            )
        except Exception as text_error:
            # Unexpected error clearing text widget
            self.handle_ui_error(
                text_error,
                "Results text widget clearing",
                "results_text",
                graceful_degradation=True
            )
        
        # Clear report text
        try:
            if hasattr(self, 'report_text') and self.report_text:
                self.report_text.delete('1.0', 'end')
        except (tk.TclError, AttributeError) as text_error:
            # Text widget clearing failed - graceful degradation
            self.handle_ui_error(
                text_error,
                "Report text widget clearing",
                "report_text",
                graceful_degradation=True
            )
        except Exception as text_error:
            # Unexpected error clearing text widget
            self.handle_ui_error(
                text_error,
                "Report text widget clearing",
                "report_text",
                graceful_degradation=True
            )
        
        # Clear LAS preview panes (original and processed)
        try:
            if hasattr(self, 'original_las_preview_text') and self.original_las_preview_text:
                self.original_las_preview_text.config(state='normal')
                self.original_las_preview_text.delete('1.0', 'end')
                self.original_las_preview_text.config(state='disabled')
        except (tk.TclError, AttributeError) as text_error:
            # Text widget clearing failed - graceful degradation
            self.handle_ui_error(
                text_error,
                "Original LAS preview text widget clearing",
                "original_las_preview_text",
                graceful_degradation=True
            )
        except Exception as text_error:
            # Unexpected error clearing text widget
            self.handle_ui_error(
                text_error,
                "Original LAS preview text widget clearing",
                "original_las_preview_text",
                graceful_degradation=True
            )
        try:
            if hasattr(self, 'processed_las_preview_text') and self.processed_las_preview_text:
                self.processed_las_preview_text.config(state='normal')
                self.processed_las_preview_text.delete('1.0', 'end')
                self.processed_las_preview_text.config(state='disabled')
        except (tk.TclError, AttributeError) as text_error:
            # Text widget clearing failed - graceful degradation
            self.handle_ui_error(
                text_error,
                "Processed LAS preview text widget clearing",
                "processed_las_preview_text",
                graceful_degradation=True
            )
        except Exception as text_error:
            # Unexpected error clearing text widget
            self.handle_ui_error(
                text_error,
                "Processed LAS preview text widget clearing",
                "processed_las_preview_text",
                graceful_degradation=True
            )
        
        # Clear visualization resources
        try:
            self.cleanup_visualization()
        except Exception as cleanup_error:
            # Visualization cleanup failed - log and continue
            self.handle_ui_error(
                cleanup_error,
                "Visualization cleanup in reset",
                "cleanup_visualization",
                graceful_degradation=True
            )
        
        # Reset file path field
        try:
            if hasattr(self, 'file_path_var') and self.file_path_var:
                self.file_path_var.set("")
        except (tk.TclError, AttributeError) as var_error:
            # File path variable reset failed - graceful degradation
            self.handle_ui_error(
                var_error,
                "File path variable reset",
                "file_path_var",
                graceful_degradation=True
            )
        except Exception as var_error:
            # Unexpected error resetting file path variable
            self.handle_ui_error(
                var_error,
                "File path variable reset",
                "file_path_var",
                graceful_degradation=True
            )
        
        # Refresh any dependent UI choices
        try:
            self.update_curve_options()
        except Exception:
            pass
    
    def on_viz_type_change(self, event=None):
        """Handle visualization type changes"""
        viz_type = self.viz_type_var.get()
        
        # Show/hide appropriate controls based on viz type
        if viz_type == "multi_curve":
            self.multi_curve_frame.pack(fill='x', pady=5)
        else:
            self.multi_curve_frame.pack_forget()
        
        # Enable/disable secondary curve combobox based on viz type
        if viz_type in ["3d_visualization"]:
            self.viz_curve2_combo['state'] = 'readonly'
        else:
            self.viz_curve2_combo['state'] = 'disabled'

    def update_curve_options(self):
        """Update curve selection options"""
        if self.current_data is None:
            return
        
        curves = list(self.current_data.columns)
        
        # Update comboboxes
        self.viz_curve_combo['values'] = curves
        self.viz_curve2_combo['values'] = curves
        self.viz_curve3_combo['values'] = curves
        
        if curves:
            self.viz_curve_combo.current(0)
            # Set secondary curve to second option or first if only one available
            self.viz_curve2_combo.current(min(1, len(curves)-1))
        
        # Update multi-select listbox
        self.curve_listbox.delete(0, tk.END)
        for curve in curves:
            self.curve_listbox.insert(tk.END, curve)
        
        # Select first few curves by default
        num_to_select = min(3, len(curves))
        for i in range(num_to_select):
            self.curve_listbox.selection_set(i)

    def update_visualization(self):
        """Professional visualization pipeline with comprehensive error recovery"""
        
        # Pre-flight validation
        validation_result = self._validate_visualization_prerequisites()
        if not validation_result['valid']:
            messagebox.showwarning("Visualization Warning", validation_result['message'])
            return
        
        curve = self.viz_curve_var.get()
        viz_type = self.viz_type_var.get()
        
        # Determine which data source to use for visualization
        data_source = validation_result.get('data_source', 'processed_data')
        if data_source == 'current_data':
            # Use current_data for unprocessed visualization
            self._visualize_unprocessed_data(curve, viz_type)
            return
        
        # Performance and memory optimization
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Phase 1: Resource cleanup with enhanced method
            cleanup_success = self.cleanup_visualization()
            if not cleanup_success:
                import warnings
                warnings.warn(
                    "Visualization cleanup reported errors. "
                    "Memory leaks possible. Check cleanup logs for details.",
                    UserWarning
                )

            # --- Remove all widgets from viz_content to prevent duplicate toolbars/canvases ---
            if hasattr(self, 'viz_content') and self.viz_content:
                for widget in self.viz_content.winfo_children():
                    widget.destroy()
                self.canvas = None

            # Phase 2: Figure creation with optimization
            fig = self.ensure_figure_exists()
            if not fig:
                raise RuntimeError("Failed to create visualization figure")

            # Phase 3: Plotting with method-specific error handling
            plot_method_map = {
                "single_curve": self.plot_single_curve,
                "single_curve_comparison": self.plot_single_curve_comparison,
                "comparison": self.plot_comparison,
                "uncertainty": self.plot_uncertainty,
                "quality_metrics": self.plot_quality_metrics,
                "correlation_matrix": self.plot_correlation_matrix,
                "scatter_plot": self.plot_scatter,
                "3d_visualization": self.plot_3d_visualization,
                "multi_curve": self.plot_multi_curve,
                "log_display": self.plot_log_display,
                "unprocessed_curves": self.plot_unprocessed_curves,
                "quality_overview": self.plot_curve_quality_overview,
                "histogram": self.plot_histogram
            }

            plot_method = plot_method_map.get(viz_type)
            if not plot_method:
                raise ValueError(f"Unknown visualization type: {viz_type}")

            # Execute plotting with parameters based on method requirements
            # Types that require a curve parameter
            if viz_type in ["single_curve", "comparison", "uncertainty", "quality_metrics", "scatter_plot", "3d_visualization", "histogram"]:
                plot_method(curve)
            else:
                # Types that don't require a curve parameter or handle curve selection internally
                plot_method()

            # Phase 4: Canvas creation with enhanced error handling
            if self.fig and hasattr(self, 'viz_content') and self.viz_content:
                # Create canvas using existing professional pattern
                self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
                self.canvas.draw()
                
                # Create navigation toolbar for professional interaction
                toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                toolbar.update()
                toolbar.pack(side='top', fill='x')
                
                # Pack canvas below toolbar
                self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
                
                # Add mouse scroll functionality
                def on_scroll(event):
                    try:
                        if event.inaxes:
                            # Get current axis limits
                            xlim = event.inaxes.get_xlim()
                            ylim = event.inaxes.get_ylim()
                            
                            # Calculate zoom factor
                            zoom_factor = 1.1 if event.button == 'up' else 0.9
                            
                            # Get mouse position in data coordinates
                            x_center = event.xdata
                            y_center = event.ydata
                            
                            if x_center is not None and y_center is not None:
                                # Calculate new limits centered on mouse position
                                x_range = (xlim[1] - xlim[0]) * zoom_factor
                                y_range = (ylim[1] - ylim[0]) * zoom_factor
                                
                                event.inaxes.set_xlim([x_center - x_range/2, x_center + x_range/2])
                                event.inaxes.set_ylim([y_center - y_range/2, y_center + y_range/2])
                                
                                self.canvas.draw()
                    except Exception:
                        pass  # Silent fail for zoom operations
                
                # Connect scroll event
                self.canvas.mpl_connect('scroll_event', on_scroll)
                
                # Performance monitoring
                elapsed_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                
                # Information logging removed
                # System status handled - operation continues
                if memory_after > memory_before:
                    memory_delta = memory_after - memory_before
                    if memory_delta > 50:  # MB
                        import warnings
                        warnings.warn(
                            f"Visualization created significant memory delta: {memory_delta:.1f}MB. "
                            f"Consider simplifying visualization or reducing data size.",
                            UserWarning
                        )
                        self.log_processing(f"High memory delta from visualization: {memory_delta:.1f}MB")
                
                # Analytics tracking for enterprise monitoring
                if BETA_SYSTEM_AVAILABLE and hasattr(self, 'beta_analytics'):
                    curve_count = 1  # Single curve visualization
                    if viz_type == "multi_curve":
                        selected_indices = self.curve_listbox.curselection()
                        curve_count = len(selected_indices) if selected_indices else 1
                    elif viz_type == "correlation_matrix":
                        curve_count = len(self.processed_data.columns)
                    
                    self.beta_analytics.track_visualization_created(viz_type, curve_count)
                    self.beta_analytics.track_event('visualization_created', {
                        'type': viz_type,
                        'processing_time': elapsed_time,
                        'memory_delta_mb': memory_after - memory_before
                    })
            
        except Exception as e:
            # Log visualization update failure with diagnostic information
            self.log_processing(f"ERROR: Visualization update failed for {viz_type}: {str(e)}")
            warnings.warn(
                f"Visualization update failed for '{viz_type}': {str(e)}. "
                f"Check data availability and visualization parameters.",
                UserWarning
            )
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, 
                               f"Failed to create {viz_type} visualization:\n{str(e)}\n\n"
                               f"Check the processing log for details.")
            
            # Track error with analytics
            if BETA_SYSTEM_AVAILABLE and hasattr(self, 'beta_analytics'):
                self.beta_analytics.track_error("visualization_failed", str(e), f"viz_type_{viz_type}")
            
            # Cleanup on failure
            try:
                self.cleanup_visualization()
            except Exception as cleanup_error:
                # Log cleanup errors but don't fail the main operation
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"Warning: Visualization cleanup failed: {type(cleanup_error).__name__}: {str(cleanup_error)}")

    def _validate_visualization_prerequisites(self):
        """Comprehensive validation of visualization prerequisites"""
        # Check if we have any data available (either current or processed)
        if self.processed_data is None and self.current_data is None:
            return {'valid': False, 'message': "No data available for visualization"}
        
        if not hasattr(self, 'viz_content') or not self.viz_content:
            return {'valid': False, 'message': "Visualization interface not ready"}
        
        curve = self.viz_curve_var.get() if hasattr(self, 'viz_curve_var') else None
        if not curve:
            return {'valid': False, 'message': "No curve selected for visualization"}
        
        # Check if curve exists in either processed_data or current_data
        data_source = None
        if self.processed_data is not None and curve in self.processed_data.columns:
            data_source = 'processed_data'
        elif self.current_data is not None and curve in self.current_data.columns:
            data_source = 'current_data'
        else:
            return {'valid': False, 'message': f"Selected curve '{curve}' not found in available data"}
        
        return {'valid': True, 'message': "Prerequisites validated", 'data_source': data_source}
    
    def _schedule_visualization_update_safely(self):
        """Thread-safe visualization update with comprehensive error handling"""
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                # Use existing thread-safe pattern with enhanced error recovery
                self.root.after_idle(lambda: self._execute_visualization_update_safely())
            else:
                # Root window not available for visualization update
                import warnings
                warnings.warn(
                    "Root window unavailable for visualization update. Skipping update.",
                    UserWarning
                )
        except Exception as e:
            # Log thread marshalling failure
            warnings.warn(
                f"Thread marshalling failed for visualization update: {str(e)}. "
                f"Visualization may not update automatically.",
                UserWarning
            )

    def _execute_visualization_update_safely(self):
        """Execute visualization update with enterprise-grade error handling"""
        try:
            # Validate application state before proceeding
            if not hasattr(self, 'processed_data') or self.processed_data is None:
                # Warning removed - operation continues
                # Status notification handled - continuing operation
                return
            
            # Use existing preview method with enhanced safety
            self.preview_processed_las()
            
            # Track successful update for analytics if available
            if BETA_SYSTEM_AVAILABLE and hasattr(self, 'beta_analytics'):
                self.beta_analytics.track_event('visualization_updated', {
                    'update_source': 'processing_completion',
                    'data_size': len(self.processed_data)
                })
                
        except Exception as e:
            # Log visualization update execution failure
            self.log_processing(f"ERROR: Visualization update execution failed: {str(e)}")
            warnings.warn(
                f"Visualization update execution failed: {str(e)}. "
                f"Preview may not reflect processed data.",
                UserWarning
            )
            # Graceful degradation - inform user without crashing
            if hasattr(self, 'status_label'):
                try:
                    self.status_label.config(text="Processing completed (preview update failed)")
                except (tk.TclError, AttributeError) as ui_error:
                    # UI widget may have been destroyed or accessed incorrectly
                    # Log but don't fail - this is graceful degradation
                    if hasattr(self, 'log_processing'):
                        self.log_processing(f"Warning: Status label update failed: {type(ui_error).__name__}: {str(ui_error)}")
                except Exception as ui_error:
                    # Unexpected error - log for debugging
                    if hasattr(self, 'log_processing'):
                        self.log_processing(f"Warning: Unexpected error updating status label: {type(ui_error).__name__}: {str(ui_error)}")

    def plot_multi_curve(self):
        """Plot multiple curves in petroleum industry standard format with depth on Y-axis"""
        # Clean up previous visualization resources
        self.cleanup_visualization()
        
        # Validate data availability
        if not hasattr(self, 'current_data') or self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded. Please load a file first.")
            return
        
        # Get selected curves from listbox
        selected_indices = self.curve_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one curve to plot")
            return
        
        selected_curves = [self.curve_listbox.get(i) for i in selected_indices]
        
        # Validate that selected curves exist in data
        available_curves = self.current_data.columns.tolist()
        valid_curves = [curve for curve in selected_curves if curve in available_curves]
        
        if not valid_curves:
            messagebox.showwarning("Warning", "None of the selected curves are available in the loaded data.")
            return
        
        # Use industry-standard colors for log curves (API & SPWLA standards)
        industry_colors = PHYSICAL_CONSTANTS.LOG_COLORS
        
        # Create a depth track layout based on number of curves
        num_curves = len(selected_curves)
        
        # Ensure we have a valid figure
        self.ensure_figure_exists()
        
        # Set figure title
        self.fig.suptitle(f'Multi-Curve Log Display - {len(valid_curves)} Curves', fontsize=14, fontweight='bold')
        
        if len(valid_curves) <= 3:
            # For 1-3 curves, use a single track with shared Y-axis
            ax = self.fig.add_subplot(111)
            self._plot_depth_based_curves(ax, valid_curves, industry_colors)
            
            # Add professional styling
            ax.set_ylabel(LABEL_DEPTH_M, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
            
        else:
            # For 4+ curves, use multiple tracks (industry standard)
            # Determine number of tracks needed (maximum 3 curves per track)
            num_tracks = (len(valid_curves) + 2) // 3  # Ceiling division
            
            # Create a grid of tracks sharing the same y-axis
            axes = []
            for i in range(num_tracks):
                if i == 0:
                    ax = self.fig.add_subplot(1, num_tracks, i+1)
                    axes.append(ax)
                else:
                    ax = self.fig.add_subplot(1, num_tracks, i+1, sharey=axes[0])
                    axes.append(ax)
            
            # Distribute curves among tracks
            for i, track_ax in enumerate(axes):
                # Get curves for this track
                start_idx = i * 3
                end_idx = min((i + 1) * 3, len(valid_curves))
                track_curves = valid_curves[start_idx:end_idx]
                
                # Plot curves on this track
                self._plot_depth_based_curves(track_ax, track_curves, industry_colors)
                
                # Add professional styling
                track_ax.grid(True, alpha=0.3)
                track_ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
                
                # Only show depth labels on the first track
                if i == 0:
                    track_ax.set_ylabel(LABEL_DEPTH_M, fontsize=10, fontweight='bold')
                else:
                    track_ax.set_ylabel('')
        
        # Apply tight layout with proper spacing
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _plot_depth_based_curves(self, ax, curves, industry_colors):
        """Plot curves in petroleum industry standard with depth on Y-axis.

        Depth is resolved per curve from the same frame that supplied that
        curve's values. processing_results arrays share processed_data's grid;
        current_data may still sit on the as-loaded grid after resampling, so a
        single depth array taken from current_data cannot be shared across the
        three-way value fallback below.
        """
        # Identify a depth mnemonic only to exclude it from the value list and
        # to choose the axis-limit branch (set_ylim vs invert_yaxis). The depth
        # ordinate itself is resolved per curve after the value source is known.
        depth_curve = None
        for curve in curves:
            curve_type = self.curve_info.get(curve, {}).get('curve_type', '')
            if 'DEPTH' in curve_type:
                depth_curve = curve
                break

        plot_curves = [c for c in curves if c != depth_curve] if depth_curve else list(curves)

        # Create twin axes for different scales if needed
        twin_axes = []
        depth_min = None
        depth_max = None

        # Plot each curve with appropriate styling
        for i, curve in enumerate(plot_curves):
            # Get curve data with proper validation; depth follows the winning frame
            curve_data = None
            curve_status = 'unknown'
            depth = None

            try:
                if hasattr(self, 'processing_results') and self.processing_results and curve in self.processing_results:
                    curve_data = self.processing_results[curve]['final_data']
                    curve_status = 'processed'
                    depth = self._get_depth_for_frame(self.processed_data)
                elif hasattr(self, 'processed_data') and self.processed_data is not None and curve in self.processed_data.columns:
                    curve_data = self.processed_data[curve].values
                    curve_status = 'unprocessed'
                    depth = self._get_depth_for_frame(self.processed_data)
                elif hasattr(self, 'current_data') and self.current_data is not None and curve in self.current_data.columns:
                    curve_data = self.current_data[curve].values
                    curve_status = 'original'
                    depth = self._get_depth_for_frame(self.current_data)
                else:
                    warnings.warn(f"Curve '{curve}' not found in any data source", UserWarning)
                    continue

                # Validate curve data
                if curve_data is None or len(curve_data) == 0:
                    warnings.warn(f"Curve '{curve}' has no valid data", UserWarning)
                    continue

                # Index-fallback path: when the caller did not pass a depth
                # mnemonic in `curves`, preserve the historical behaviour of
                # plotting against row index rather than looking up DEPT from
                # the frame. _get_depth_for_frame would otherwise return the
                # real depth column and change the ordinate silently.
                if depth_curve is None:
                    depth = np.arange(len(curve_data))

                if depth is None or len(depth) != len(curve_data):
                    warnings.warn(
                        f"Depth length {0 if depth is None else len(depth)} does not "
                        f"match curve '{curve}' length {len(curve_data)}",
                        UserWarning)
                    continue

                # Convert null values to NaN for proper line breaking (for visualization only)
                # Uses helper method to ensure consistent null detection
                curve_data = self._convert_nulls_to_nan(curve_data)

            except Exception as e:
                warnings.warn(f"Error accessing curve '{curve}': {e}", UserWarning)
                continue

            # Expand shared Y limits across every frame that contributed a curve
            c_min, c_max = self._get_depth_limits(depth)
            if depth_min is None:
                depth_min, depth_max = c_min, c_max
            else:
                depth_min = min(depth_min, c_min)
                depth_max = max(depth_max, c_max)

            curve_type = self.curve_info.get(curve, {}).get('curve_type', '')
            curve_family = curve_type.split('_')[0] if '_' in curve_type else curve_type

            # Determine if this curve should use log scale
            use_log_scale = False
            log_scale_families = ['RESISTIVITY', 'PERMEABILITY']
            if curve_family in log_scale_families:
                use_log_scale = True

            # Determine color based on industry standards
            if curve_family in industry_colors:
                color = industry_colors[curve_family]
            else:
                # Use a color cycle for non-standard curves
                color = plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)]

            # Determine line style and width based on processing status
            if curve_status == 'processed':
                line_style = '-'
                line_width = 2.0
            elif curve_status == 'unprocessed':
                line_style = '--'
                line_width = 1.5
            else:  # original
                line_style = ':'
                line_width = 1.0

            # For multiple curves with different scales, create twin axes
            if i > 0 and use_log_scale != (ax.get_xscale() == 'log'):
                twin_ax = ax.twiny()
                twin_axes.append(twin_ax)
                current_ax = twin_ax
                # Position the axis at the top for the second curve
                current_ax.xaxis.set_ticks_position('top')
                current_ax.xaxis.set_label_position('top')
            else:
                current_ax = ax

            # Set appropriate scale for logarithmic curves
            if use_log_scale:
                # Handle zeros and negatives for log scale
                valid_data = curve_data[curve_data > 0]
                if len(valid_data) > 0:
                    min_val = np.min(valid_data)
                    current_ax.set_xscale('log')
                    # Set standard track scales for this curve type if available
                    if curve_family in PHYSICAL_CONSTANTS.LOG_TRACK_SCALES:
                        current_ax.set_xlim(PHYSICAL_CONSTANTS.LOG_TRACK_SCALES[curve_family])
                    else:
                        # Fallback to reasonable log bounds
                        current_ax.set_xlim([min_val * 0.5, np.max(valid_data) * 2])

            # Handle missing data (NaN values break lines properly)
            valid_mask = ~np.isnan(curve_data) & np.isfinite(curve_data)
            if np.any(valid_mask):
                valid_data = curve_data[valid_mask]
                valid_depth = depth[valid_mask]

                # Plot with depth on Y-axis (inverted)
                legend_label = f"{curve} ({curve_status})"
                current_ax.plot(valid_data, valid_depth, color=color, linestyle=line_style,
                              linewidth=line_width, label=legend_label)

            # Add gridlines
            current_ax.grid(True, alpha=0.3, which='both')

            # Set labels
            unit = self.curve_info.get(curve, {}).get('unit', '')
            current_ax.set_xlabel(f'{curve} ({unit})')

        # CRITICAL: Set axis limits to ACTUAL data range (once for shared Y-axis).
        # apply_depth_axis encodes downward depth via set_ylim alone; never pair
        # that with invert_yaxis.
        if depth_min is not None:
            label = (
                f'Depth ({self.curve_info.get(depth_curve, {}).get("unit", "m")})'
                if depth_curve else 'Depth (index)'
            )
            self.apply_depth_axis(ax, np.array([depth_min, depth_max]), label=label)

        # Optional: draw formation tops and zone shading
        try:
            if hasattr(self, 'geological_context') and self.geological_context:
                # Tops as horizontal lines
                for top_name, top_depth in getattr(self.geological_context, 'formation_tops', {}).items():
                    ax.axhline(y=top_depth, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7)
                # Open-hole interval shading
                ohs = getattr(self.geological_context, 'open_hole_start', None)
                ohe = getattr(self.geological_context, 'open_hole_end', None)
                if ohs is not None and ohe is not None and ohs < ohe:
                    ax.axhspan(ohs, ohe, color='#f0f8ff', alpha=0.25)
        except Exception:
            pass

        # Add legends (outside, consistent)
        handles, labels = ax.get_legend_handles_labels()
        for twin_ax in twin_axes:
            twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
            handles.extend(twin_handles)
            labels.extend(twin_labels)

        if handles:
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0.0, frameon=False, ncol=1)
            self.fig.subplots_adjust(right=0.82)

        # Add processing status note if any curves are unprocessed
        unprocessed_curves = [curve for curve in plot_curves if curve not in self.processing_results]
        if unprocessed_curves:
            status_text = f"Note: {len(unprocessed_curves)} curve(s) not yet processed (dashed/dotted lines)"
            ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _get_industry_color(self, curve_type: str, curve_name: str = '') -> str:
        """
        Get industry-standard color for a curve type.
        
        Checks curve_info first, then falls back to mnemonic library database,
        then to default industry colors.
        
        Args:
            curve_type: The identified curve type (e.g., 'GAMMA_RAY_TOTAL')
            curve_name: Optional curve name for fallback lookup
            
        Returns:
            Hex color code (e.g., '#008000')
        """
        # First check if color is stored in curve_info
        if curve_name and curve_name in self.curve_info:
            color = self.curve_info[curve_name].get('industry_color', '')
            if color and color != '#000000':  # Valid color
                return color
        
        # Try to get from unified curve identifier database
        try:
            if hasattr(self, 'curve_identifier') and self.curve_identifier:
                db = self.curve_identifier.mnemonic_database
                if curve_type in db and 'industry_color' in db[curve_type]:
                    color = db[curve_type]['industry_color']
                    if color and color != '#000000':
                        return color
        except Exception:
            pass
        
        # Fallback to default industry colors based on curve type
        default_colors = {
            'GAMMA_RAY_TOTAL': '#008000',      # Green
            'GAMMA_RAY_SPECTRAL': '#008000',   # Green
            'SPONTANEOUS_POTENTIAL': '#FFA500', # Orange
            'RESISTIVITY_DEEP': '#FF0000',     # Red
            'RESISTIVITY_MEDIUM': '#FF4444',   # Light red
            'RESISTIVITY_SHALLOW': '#FF8888',  # Lighter red
            'RESISTIVITY_MICRO': '#FFAAAA',    # Lightest red
            'NEUTRON_POROSITY': '#0000FF',     # Blue
            'BULK_DENSITY': '#FF0000',         # Red
            'SONIC_COMPRESSIONAL': '#800080',  # Purple
            'CALIPER_SINGLE': '#000000',       # Black
            'CALIPER_MULTI': '#000000',        # Black
            'PHOTOELECTRIC_FACTOR': '#FF00FF', # Magenta
        }
        
        return default_colors.get(curve_type, '#333333')  # Default dark gray
    
    def _add_qc_indicators(self, ax, curve_name: str, curve_data: np.ndarray, depth: np.ndarray):
        """
        Add QC indicators to a curve plot:
        - Confidence markers (from standardization reporter)
        - Gap indicators (from processing results)
        - Quality flags
        
        Args:
            ax: Matplotlib axis to add indicators to
            curve_name: Name of the curve
            curve_data: Data values for the curve
            depth: Depth values
        """
        # Confidence indicators from standardization reporter
        if hasattr(self, 'standardization_reporter') and self.standardization_reporter:
            # Find confidence for this curve
            for ident in self.standardization_reporter.curve_identifications:
                if ident['original_name'] == curve_name:
                    confidence = ident['confidence']
                    
                    # Add confidence marker at regular intervals
                    if len(curve_data) > 0 and len(depth) > 0:
                        # Sample at regular intervals (every 50th point or so)
                        sample_step = max(1, len(curve_data) // 50)
                        sample_indices = np.arange(0, len(curve_data), sample_step)
                        
                        valid_mask = ~np.isnan(curve_data[sample_indices]) & ~np.isnan(depth[sample_indices])
                        if np.any(valid_mask):
                            valid_indices = sample_indices[valid_mask]
                            x_positions = curve_data[valid_indices]
                            y_positions = depth[valid_indices]
                            
                            # Color based on confidence: green (≥0.8), yellow (0.5-0.8), red (<0.5)
                            if confidence >= 0.8:
                                marker_color = 'green'
                                marker_alpha = 0.6
                            elif confidence >= 0.5:
                                marker_color = 'orange'
                                marker_alpha = 0.5
                            else:
                                marker_color = 'red'
                                marker_alpha = 0.4
                            
                            # Small circular markers
                            ax.scatter(x_positions, y_positions, 
                                     c=marker_color, s=15, alpha=marker_alpha, 
                                     marker='o', edgecolors='none', zorder=4, label='_nolegend_')
                    break
        
        # Gap indicators from processing results
        if hasattr(self, 'processing_results') and curve_name in self.processing_results:
            proc_result = self.processing_results[curve_name]
            
            if 'gap_filling' in proc_result:
                gap_info = proc_result['gap_filling'].get('gaps_filled', [])
                
                # Mark gap locations with small triangles
                for gap_data in gap_info[:20]:  # Limit to first 20 gaps to avoid clutter
                    gap = gap_data.get('gap', {})
                    gap_start = gap.get('start', 0)
                    gap_end = gap.get('end', 0)
                    
                    if gap_start < len(depth) and gap_end < len(depth):
                        gap_center_idx = (gap_start + gap_end) // 2
                        if gap_center_idx < len(curve_data) and not np.isnan(curve_data[gap_center_idx]):
                            # Small triangle marker at gap location
                            ax.scatter(curve_data[gap_center_idx], depth[gap_center_idx],
                                     c='blue', s=30, alpha=0.5, marker='^',
                                     edgecolors='none', zorder=4, label='_nolegend_')
        
        # Processing history badges (add to legend via label)
        processing_operations = []
        if hasattr(self, 'processing_results') and curve_name in self.processing_results:
            proc_result = self.processing_results[curve_name]
            if 'gap_filling' in proc_result:
                processing_operations.append('Gap Filled')
            if 'denoising' in proc_result:
                processing_operations.append('Denoised')
            if 'normalization' in proc_result:
                processing_operations.append('Normalized')
        
        # Return operations list for badge display
        return processing_operations
    
    def _setup_log_display_figure(self, data_source: pd.DataFrame) -> Tuple[List[Any], np.ndarray, str, Dict[str, List[str]], float]:
        """Setup figure and axes for industry log display.
        
        Returns:
            Tuple of (axes, depth, depth_unit, curve_by_type, null_value)
        """
        # Clean up previous visualization resources
        self.cleanup_visualization()
        
        # Get null value for data conversion
        if hasattr(self, 'null_value_var') and self.null_value_var.get():
            try:
                null_value = float(self.null_value_var.get())
            except (ValueError, AttributeError):
                null_value = -999.25  # Default LAS null value
        else:
            null_value = -999.25  # Default LAS null value
        
        # Identify curves by type
        curve_by_type = {}
        for curve in data_source.columns:
            curve_type = self.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN')
            if curve_type not in curve_by_type:
                curve_by_type[curve_type] = []
            curve_by_type[curve_type].append(curve)
        
        # Use proper figure management with larger size for 4-track display
        self.ensure_figure_exists()
        self.fig.set_size_inches(20, 10)  # Wider for 4 tracks
        
        # Set figure title
        self.fig.suptitle('Industry Standard Log Display', fontsize=16, fontweight='bold')
        
        # Create a 4-track display using proper method (industry standard)
        axes = self.fig.subplots(1, 4, sharey=True)
        
        # Add professional styling to all axes
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
        
        # Use depth for Y-axis
        depth_curves = curve_by_type.get('DEPTH_MEASURED', []) + curve_by_type.get('DEPTH_TRUE_VERTICAL', [])
        if depth_curves:
            depth = data_source[depth_curves[0]].values
            depth_unit = self.curve_info.get(depth_curves[0], {}).get('unit', 'm')
        else:
            depth = np.arange(len(data_source))
            depth_unit = 'index'
        
        # Shared Y-axis: apply limits once and label only the first track.
        self.apply_depth_axis_shared(
            axes, depth, label=f'Depth ({depth_unit})')
        
        return axes, depth, depth_unit, curve_by_type, null_value
    
    def _plot_lithology_track(self, ax: Any, data_source: pd.DataFrame, depth: np.ndarray, 
                              curve_by_type: Dict[str, List[str]],
                              null_value: float) -> LithologyTrackCurves:
        """Plot Track 1: GR, SP, Caliper (Lithology Track).
        
        Returns the GR and SP curve names and their null-converted arrays so the
        caller can attach QC indicators to the same data that was plotted. See
        LithologyTrackCurves for why this is returned rather than recomputed.
        """
        ax.set_title('Track 1: GR/SP/CAL', fontsize=12, fontweight='bold')
        track_curves = LithologyTrackCurves()
        
        # GR with industry-standard zone shading
        gr_curves = curve_by_type.get('GAMMA_RAY_TOTAL', [])
        if gr_curves:
            gr_curve_name = gr_curves[0]
            gr_data = data_source[gr_curve_name].values
            gr_data = self._convert_nulls_to_nan(gr_data)
            track_curves.gr_curves = gr_curves
            track_curves.gr_data = gr_data
            gr_color = self._get_industry_color('GAMMA_RAY_TOTAL', gr_curve_name)
            
            ax.plot(gr_data, depth, color=gr_color, linewidth=1.5, label=gr_curve_name, zorder=3)
            
            # Industry-standard GR zone shading
            valid_mask = ~np.isnan(gr_data) & ~np.isnan(depth)
            if np.any(valid_mask):
                valid_gr = gr_data[valid_mask]
                valid_depth = depth[valid_mask]
                ax.fill_betweenx(valid_depth, 0, valid_gr, where=(valid_gr < 60), 
                                 color='green', alpha=0.15, label='Clean Zone')
                ax.fill_betweenx(valid_depth, 60, valid_gr, 
                                 where=(valid_gr >= 60) & (valid_gr < 90), 
                                 color='gold', alpha=0.15, label='Transition Zone')
                ax.fill_betweenx(valid_depth, 90, valid_gr, where=(valid_gr >= 90), 
                                 color='red', alpha=0.15, label='Shale Zone')
            
            ax.set_xlim([0, 150])
            ax.set_xlabel('GR (API)', fontsize=10)
        
        # SP with industry color
        sp_curves = curve_by_type.get('SPONTANEOUS_POTENTIAL', [])
        if sp_curves:
            sp_curve_name = sp_curves[0]
            sp_color = self._get_industry_color('SPONTANEOUS_POTENTIAL', sp_curve_name)
            twin1 = ax.twiny()
            sp_data = self._convert_nulls_to_nan(data_source[sp_curve_name].values, null_value)
            track_curves.sp_curves = sp_curves
            track_curves.sp_data = sp_data
            twin1.plot(sp_data, depth, color=sp_color, linewidth=1.5, label=sp_curve_name, zorder=2)
            twin1.set_xlim([-100, 100])
            twin1.xaxis.set_ticks_position('top')
            twin1.xaxis.set_label_position('top')
        
        # Caliper with industry color
        cal_curves = curve_by_type.get('CALIPER_SINGLE', []) + curve_by_type.get('CALIPER_MULTI', [])
        if cal_curves:
            cal_curve_name = cal_curves[0]
            cal_color = self._get_industry_color('CALIPER_SINGLE', cal_curve_name)
            twin1_2 = ax.twiny()
            cal_data = data_source[cal_curve_name].values
            twin1_2.plot(cal_data, depth, color=cal_color, linewidth=1.5, label=cal_curve_name, zorder=2)
            twin1_2.xaxis.set_ticks_position('top')
            twin1_2.spines['top'].set_position(('outward', 40))
        
        return track_curves
    
    def plot_log_display(self):
        """Create a standard industry log display with multiple tracks"""
        # Validate data availability
        if not hasattr(self, 'current_data') or self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded. Please load a file first.")
            return
        
        data_source = self.current_data
        axes, depth, depth_unit, curve_by_type, null_value = self._setup_log_display_figure(data_source)
        
        # Track 1 is drawn by a helper; the QC-indicator and badge blocks below
        # need the curves it selected, so they are carried back explicitly.
        track1 = self._plot_lithology_track(axes[0], data_source, depth, curve_by_type, null_value)
        
        # Track 2: Resistivity curves (log scale, industry standard)
        ax2 = axes[1]
        ax2.set_title('Track 2: Resistivity', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Resistivity (ohm-m)', fontsize=10)
        
        res_types = ['RESISTIVITY_DEEP', 'RESISTIVITY_MEDIUM', 'RESISTIVITY_SHALLOW', 'RESISTIVITY_MICRO']
        resistivity_curves_data = {}  # Store for QC indicators
        
        has_res = False
        for res_type in res_types:
            res_curves = curve_by_type.get(res_type, [])
            if res_curves:
                has_res = True
                res_curve_name = res_curves[0]
                res_color = self._get_industry_color(res_type, res_curve_name)
                res_data = data_source[res_curve_name].values
                # Convert null values to NaN for proper line breaking
                res_data = self._convert_nulls_to_nan(res_data, null_value)
                # Handle zeros and negatives for log scale
                res_data = np.maximum(res_data, 0.1)  # Clamp to minimum 0.1 ohm-m (after null conversion)
                ax2.plot(res_data, depth, color=res_color, linewidth=1.5, label=res_curve_name)
                resistivity_curves_data[res_curve_name] = res_data
        
        if has_res:
            ax2.set_xscale('log')
            ax2.set_xlim([0.1, 1000])
        
        # Track 3: Porosity curves with RHOB-NPHI crossover highlighting
        ax3 = axes[2]
        ax3.set_title('Track 3: Porosity', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Porosity (v/v)', fontsize=10)
        
        # Neutron with industry color
        neutron_data = None
        neutron_curves = curve_by_type.get('NEUTRON_POROSITY', [])
        if neutron_curves:
            neutron_curve_name = neutron_curves[0]
            neutron_color = self._get_industry_color('NEUTRON_POROSITY', neutron_curve_name)
            neutron_data = data_source[neutron_curve_name].values
            # Convert null values to NaN for proper line breaking
            neutron_data = self._convert_nulls_to_nan(neutron_data, null_value)
            ax3.plot(neutron_data, depth, color=neutron_color, linewidth=1.5, label=neutron_curve_name)
        
        # Density with industry color and crossover detection
        density_data = None
        density_curves = curve_by_type.get('BULK_DENSITY', [])
        if density_curves:
            density_curve_name = density_curves[0]
            density_color = self._get_industry_color('BULK_DENSITY', density_curve_name)
            density_data = data_source[density_curve_name].values
            # Convert null values to NaN for proper line breaking
            density_data = self._convert_nulls_to_nan(density_data, null_value)
            
            if not neutron_curves:
                ax3.plot(density_data, depth, color=density_color, linewidth=1.5, label=density_curve_name)
            else:
                # If both neutron and density are present, plot density on the same scale but reversed
                twin3 = ax3.twiny()
                twin3.plot(density_data, depth, color=density_color, linewidth=1.5, label=density_curve_name)
                # Set same range but reversed
                if ax3.get_xlim()[1] > ax3.get_xlim()[0]:
                    neutron_min, neutron_max = ax3.get_xlim()
                    twin3.set_xlim([neutron_max, neutron_min])
                twin3.xaxis.set_ticks_position('top')
                twin3.xaxis.set_label_position('top')
        
        # RHOB-NPHI crossover highlighting (gas detection)
        # Industry standard: Highlight where density is low AND neutron is high (gas crossover)
        if neutron_data is not None and density_data is not None:
            valid_mask = ~np.isnan(neutron_data) & ~np.isnan(density_data) & ~np.isnan(depth)
            if np.any(valid_mask):
                valid_neutron = neutron_data[valid_mask]
                valid_density = density_data[valid_mask]
                valid_depth_cross = depth[valid_mask]
                
                # Typical values for gas detection:
                # Low density (<2.35 g/cm³) AND high neutron (>0.35 v/v) suggests gas
                # Convert neutron to density equivalent if needed, or use normalized crossover
                gas_threshold_density = 2.35  # g/cm³ - below this suggests gas
                gas_threshold_neutron = 0.35  # v/v - above this suggests gas/high porosity
                
                # Find gas crossover zones (low density AND high neutron)
                gas_mask = (valid_density < gas_threshold_density) & (valid_neutron > gas_threshold_neutron)
                
                if np.any(gas_mask):
                    # Highlight gas crossover zones with subtle shading
                    ax3.fill_betweenx(valid_depth_cross[gas_mask], 
                                     ax3.get_xlim()[0], ax3.get_xlim()[1],
                                     alpha=0.2, color='yellow', label='Gas Crossover', zorder=0)
        
        # Sonic with industry color
        sonic_curves = curve_by_type.get('SONIC_COMPRESSIONAL', [])
        if sonic_curves:
            sonic_curve_name = sonic_curves[0]
            sonic_color = self._get_industry_color('SONIC_COMPRESSIONAL', sonic_curve_name)
            twin3_2 = ax3.twiny()
            sonic_data = data_source[sonic_curve_name].values
            twin3_2.plot(sonic_data, depth, color=sonic_color, linewidth=1.5, label=sonic_curve_name)
            # Position x-axis
            twin3_2.xaxis.set_ticks_position('top')
            twin3_2.spines['top'].set_position(('outward', 40))
        
        # Track 4: Computed/Derived parameters
        ax4 = axes[3]
        ax4.set_title('Track 4: Computed', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Computed Parameters', fontsize=10)
        
        # Identify computed curves (saturation, porosity, shale volume, etc.)
        computed_curve_types = [
            'WATER_SATURATION', 'OIL_SATURATION', 'GAS_SATURATION',
            'POROSITY_COMPUTED', 'SHALE_VOLUME', 'PERMEABILITY',
            'EFFECTIVE_POROSITY', 'TOTAL_POROSITY'
        ]
        
        has_computed = False
        computed_curves_plotted = []
        for comp_type in computed_curve_types:
            comp_curves = curve_by_type.get(comp_type, [])
            if comp_curves:
                has_computed = True
                comp_curve_name = comp_curves[0]
                comp_color = self._get_industry_color(comp_type, comp_curve_name)
                comp_data = data_source[comp_curve_name].values
                ax4.plot(comp_data, depth, color=comp_color, linewidth=1.5, label=comp_curve_name)
                computed_curves_plotted.append(comp_curve_name)
        
        if not has_computed:
            # If no computed curves, show placeholder message
            ax4.text(0.5, 0.5, 'No computed parameters\navailable', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=11, style='italic', color='gray')
        
        # Common settings for all tracks. Depth orientation and the shared Y
        # label were applied once in _setup_log_display_figure; do not flip
        # each sharey track here (even track counts would cancel the flip).
        for ax in axes:
            ax.grid(True, alpha=0.3)
            
            # Enhanced formation tops with labels
            try:
                if hasattr(self, 'geological_context') and self.geological_context:
                    formation_tops = getattr(self.geological_context, 'formation_tops', {})
                    
                    # Draw formation top lines with labels
                    for top_name, top_depth in formation_tops.items():
                        # Check if top is within depth range
                        if np.min(depth) <= top_depth <= np.max(depth):
                            # Draw dashed line
                            ax.axhline(y=top_depth, color='#666666', linestyle='--', 
                                      linewidth=1.0, alpha=0.7, zorder=1)
                            
                            # Add formation name label at right edge
                            ax.text(ax.get_xlim()[1], top_depth, f'  {top_name}', 
                                   fontsize=8, verticalalignment='center',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            edgecolor='#666666', alpha=0.8),
                                   zorder=5)
                    
                    # Open-hole shading
                    ohs = getattr(self.geological_context, 'open_hole_start', None)
                    ohe = getattr(self.geological_context, 'open_hole_end', None)
                    if ohs is not None and ohe is not None and ohs < ohe:
                        ax.axhspan(ohs, ohe, color='#f0f8ff', alpha=0.25, zorder=0)
            except Exception:
                pass
        
        # Hide y-axis labels for all but the first track
        for ax in axes[1:]:
            ax.set_ylabel('')
        
        # Add QC indicators and collect processing badges
        all_processing_badges = {}  # Track processing operations per curve
        
        # Add QC indicators to curves in each track
        # Track 1: GR, SP, Caliper
        if track1.gr_curves:
            badges = self._add_qc_indicators(axes[0], track1.gr_curves[0], track1.gr_data, depth)
            if badges:
                all_processing_badges[track1.gr_curves[0]] = badges
        if track1.sp_curves:
            badges = self._add_qc_indicators(axes[0], track1.sp_curves[0], track1.sp_data, depth)
            if badges:
                all_processing_badges[track1.sp_curves[0]] = badges
        
        # Track 2: Resistivity
        for res_curve_name, res_data in resistivity_curves_data.items():
            badges = self._add_qc_indicators(ax2, res_curve_name, res_data, depth)
            if badges:
                all_processing_badges[res_curve_name] = badges
        
        # Track 3: Porosity
        if neutron_curves:
            badges = self._add_qc_indicators(ax3, neutron_curves[0], neutron_data, depth)
            if badges:
                all_processing_badges[neutron_curves[0]] = badges
        if density_curves:
            badges = self._add_qc_indicators(ax3, density_curves[0], density_data, depth)
            if badges:
                all_processing_badges[density_curves[0]] = badges
        
        # Track 4: Computed
        for comp_curve_name in computed_curves_plotted:
            comp_data = data_source[comp_curve_name].values
            badges = self._add_qc_indicators(ax4, comp_curve_name, comp_data, depth)
            if badges:
                all_processing_badges[comp_curve_name] = badges
        
        # Add legends to each track with processing badges
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            # Get handles and labels from twin axes too
            for child in ax.get_children():
                if isinstance(child, plt.Axes):
                    twin_handles, twin_labels = child.get_legend_handles_labels()
                    handles.extend(twin_handles)
                    labels.extend(twin_labels)
            
            # Add processing badges to legend if any curves on this axis have been processed
            # Note: We'll show badges in track headers instead to keep legends clean
            if handles:
                ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         ncol=len(handles), fontsize=9)
        
        # Add processing history badges to track titles
        for i, ax in enumerate(axes):
            curve_names_on_axis = []
            # Determine which curves are on this axis
            if i == 0:  # Track 1
                if track1.gr_curves:
                    curve_names_on_axis.append(track1.gr_curves[0])
                if track1.sp_curves:
                    curve_names_on_axis.append(track1.sp_curves[0])
            elif i == 1:  # Track 2
                for res_type in res_types:
                    res_curves_list = curve_by_type.get(res_type, [])
                    if res_curves_list:
                        curve_names_on_axis.append(res_curves_list[0])
            elif i == 2:  # Track 3
                if neutron_curves:
                    curve_names_on_axis.append(neutron_curves[0])
                if density_curves:
                    curve_names_on_axis.append(density_curves[0])
            elif i == 3:  # Track 4
                curve_names_on_axis.extend(computed_curves_plotted)
            
            # Add badge text to title if any processing occurred
            badge_texts = []
            for curve_name in curve_names_on_axis:
                if curve_name in all_processing_badges:
                    badge_texts.extend(all_processing_badges[curve_name])
            
            if badge_texts:
                unique_badges = list(set(badge_texts))  # Remove duplicates
                badge_str = f" [{', '.join(unique_badges)}]"
                current_title = ax.get_title()
                ax.set_title(current_title + badge_str, fontsize=11, fontweight='bold')
        
        # Apply enhanced spacing for 4-track display
        self.fig.tight_layout()
        # Reserve space for legends and formation top labels
        self.fig.subplots_adjust(left=0.06, right=0.92, bottom=0.15, top=0.90, wspace=0.20)
    
    def plot_standardization_summary(self):
        """
        Create comprehensive standardization visualization showing:
        - Before/after standardization comparison
        - Confidence mapping across all curves
        - Transformation documentation
        """
        if not hasattr(self, 'standardization_reporter') or not self.standardization_reporter:
            messagebox.showinfo("Info", "No standardization data available to visualize.")
            return
        
        if self.standardization_reporter.total_operations == 0:
            messagebox.showinfo("Info", "No standardization operations recorded.")
            return
        
        # Clean up previous visualization
        self.cleanup_visualization()
        self.ensure_figure_exists()
        self.fig.set_size_inches(16, 12)
        self.fig.suptitle('Standardization Summary Report', fontsize=16, fontweight='bold')
        
        # Create subplot layout: 2x2 grid
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)
        
        # Panel 1: Confidence Distribution (Top Left)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.set_title('Identification Confidence Distribution', fontsize=12, fontweight='bold')
        
        if self.standardization_reporter.curve_identifications:
            confidences = [ident['confidence'] for ident in self.standardization_reporter.curve_identifications]
            
            # Create histogram with color coding
            n, bins, patches = ax1.hist(confidences, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
            
            # Color bars based on bin midpoint confidence
            for i, patch in enumerate(patches):
                bin_mid = (bins[i] + bins[i+1]) / 2
                if bin_mid < 0.5:
                    patch.set_facecolor('red')
                elif bin_mid < 0.8:
                    patch.set_facecolor('orange')
                else:
                    patch.set_facecolor('green')
            
            ax1.set_xlabel('Confidence Score', fontsize=10)
            ax1.set_ylabel(LABEL_NUMBER_OF_CURVES, fontsize=10)
            ax1.set_xlim([0, 1])
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            avg_conf = np.mean(confidences)
            high_conf = sum(1 for c in confidences if c >= 0.8)
            stats_text = f"Avg: {avg_conf:.2f}\nHigh (≥0.8): {high_conf}/{len(confidences)}"
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax1.text(0.5, 0.5, 'No identification data', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=11, style='italic', color='gray')
        
        # Panel 2: Standardization Operations Summary (Top Right)
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.set_title('Standardization Operations Count', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Create summary text box
        summary_data = {
            'Curve Identifications': len(self.standardization_reporter.curve_identifications),
            'Curve Renames': len(self.standardization_reporter.curve_renames),
            'Unit Conversions': len(self.standardization_reporter.unit_conversions),
            'Fractional Standardizations': len(self.standardization_reporter.fractional_standardizations),
            'Conflicts Resolved': len(self.standardization_reporter.conflicts)
        }
        
        summary_text = "STANDARDIZATION SUMMARY\n" + "=" * 30 + "\n\n"
        for operation, count in summary_data.items():
            summary_text += f"{operation:.<25} {count:>5}\n"
        
        summary_text += f"\n{'Total Operations':.<25} {self.standardization_reporter.total_operations:>5}"
        
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Panel 3: Curve Renames Comparison (Middle Left)
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax3.set_title('Curve Name Standardizations', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        if self.standardization_reporter.curve_renames:
            rename_text = "ORIGINAL → STANDARDIZED\n" + "=" * 35 + "\n\n"
            rename_items = list(self.standardization_reporter.curve_renames.items())[:15]  # Limit to 15
            
            for original, rename_info in rename_items:
                standardized = rename_info['standardized_name']
                confidence = rename_info['confidence']
                conf_symbol = '✓' if confidence >= 0.8 else '~' if confidence >= 0.5 else '?'
                rename_text += f"{conf_symbol} {original:<20} → {standardized}\n"
            
            if len(self.standardization_reporter.curve_renames) > 15:
                rename_text += f"\n... and {len(self.standardization_reporter.curve_renames) - 15} more"
            
            ax3.text(0.05, 0.98, rename_text, transform=ax3.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No curve renames recorded', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=11, style='italic', color='gray')
        
        # Panel 4: Unit Conversions List (Middle Right)
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.set_title('Unit Conversions Applied', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        if self.standardization_reporter.unit_conversions:
            conv_text = "CURVE | FROM → TO | METHOD\n" + "=" * 40 + "\n\n"
            conv_items = self.standardization_reporter.unit_conversions[:15]  # Limit to 15
            
            for conv in conv_items:
                curve = conv['curve_name'][:15]  # Truncate long names
                from_unit = conv['original_unit'][:8]
                to_unit = conv['standardized_unit'][:8]
                method = conv['method'][:8]
                factor = conv.get('conversion_factor', None)
                
                if factor:
                    factor_str = f"×{factor:.3f}"
                else:
                    factor_str = "function"
                
                conv_text += f"{curve:<15} {from_unit:>6} → {to_unit:<6} [{method}] {factor_str}\n"
            
            if len(self.standardization_reporter.unit_conversions) > 15:
                conv_text += f"\n... and {len(self.standardization_reporter.unit_conversions) - 15} more"
            
            ax4.text(0.05, 0.98, conv_text, transform=ax4.transAxes,
                    fontsize=8, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No unit conversions recorded', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=11, style='italic', color='gray')
        
        # Panel 5: Method Distribution (Bottom, spans both columns)
        ax5 = self.fig.add_subplot(gs[2, :])
        ax5.set_title('Identification Methods Used', fontsize=12, fontweight='bold')
        
        if self.standardization_reporter.curve_identifications:
            methods = [ident['method'] for ident in self.standardization_reporter.curve_identifications]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            if method_counts:
                methods_list = list(method_counts.keys())
                counts_list = list(method_counts.values())
                
                bars = ax5.bar(methods_list, counts_list, color='steelblue', alpha=0.7, edgecolor='black')
                ax5.set_xlabel('Identification Method', fontsize=10)
                ax5.set_ylabel('Count', fontsize=10)
                ax5.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        self.fig.tight_layout()
    
    def plot_standardization_comparison(self, curve_name: str):
        """
        Create before/after standardization comparison for a specific curve.
        Shows original vs standardized name/unit with confidence.
        
        Args:
            curve_name: Name of the curve to compare
        """
        if not hasattr(self, 'standardization_reporter') or not self.standardization_reporter:
            messagebox.showinfo("Info", "No standardization data available.")
            return
        
        # Find standardization info for this curve
        rename_info = None
        unit_conv = None
        fractional_std = None
        identification = None
        
        # Check for rename
        if curve_name in self.standardization_reporter.curve_renames:
            rename_info = self.standardization_reporter.curve_renames[curve_name]
        
        # Check for unit conversion
        for conv in self.standardization_reporter.unit_conversions:
            if conv['curve_name'] == curve_name:
                unit_conv = conv
                break
        
        # Check for fractional standardization
        for frac in self.standardization_reporter.fractional_standardizations:
            if frac['curve_name'] == curve_name:
                fractional_std = frac
                break
        
        # Check for identification
        for ident in self.standardization_reporter.curve_identifications:
            if ident['original_name'] == curve_name:
                identification = ident
                break
        
        # If no standardization data, show info message
        if not any([rename_info, unit_conv, fractional_std, identification]):
            messagebox.showinfo("Info", f"No standardization operations recorded for curve '{curve_name}'.")
            return
        
        # Create visualization
        self.cleanup_visualization()
        self.ensure_figure_exists()
        self.fig.set_size_inches(14, 10)
        self.fig.suptitle(f'Standardization Comparison: {curve_name}', fontsize=14, fontweight='bold')
        
        # Check if curve data exists
        if curve_name not in self.current_data.columns:
            messagebox.showwarning("Warning", f"Curve '{curve_name}' not found in current data.")
            return
        
        # Get depth and data
        depth_curves = [col for col in self.current_data.columns 
                       if 'DEPTH' in self.curve_info.get(col, {}).get('curve_type', '')]
        if depth_curves:
            depth = self.current_data[depth_curves[0]].values
            depth_unit = self.curve_info.get(depth_curves[0], {}).get('unit', 'm')
        else:
            depth = np.arange(len(self.current_data))
            depth_unit = 'index'
        
        curve_data = self.current_data[curve_name].values
        
        # Create comparison plot (overlay mode like processing comparison)
        ax = self.fig.add_subplot(111)
        
        # Plot the curve
        curve_color = self._get_industry_color(
            self.curve_info.get(curve_name, {}).get('curve_type', 'UNKNOWN'),
            curve_name
        )
        
        ax.plot(curve_data, depth, color=curve_color, linewidth=2.0, alpha=0.9, label=f'{curve_name} (Standardized)')
        
        # Add information panel
        info_text = f"STANDARDIZATION DETAILS\n{'=' * 40}\n\n"
        
        if identification:
            info_text += f"Identified Type: {identification['identified_type']}\n"
            info_text += f"Confidence: {identification['confidence']*100:.1f}%\n"
            info_text += f"Method: {identification['method']}\n\n"
        
        if rename_info:
            info_text += f"Name Change:\n"
            info_text += f"  {curve_name} → {rename_info['standardized_name']}\n"
            info_text += f"  Confidence: {rename_info['confidence']*100:.1f}%\n\n"
        
        if unit_conv:
            factor_str = f"×{unit_conv['conversion_factor']:.6f}" if unit_conv.get('conversion_factor') else "function"
            info_text += f"Unit Conversion:\n"
            info_text += f"  {unit_conv['original_unit']} → {unit_conv['standardized_unit']}\n"
            info_text += f"  Method: {unit_conv['method']} | Factor: {factor_str}\n\n"
        
        if fractional_std:
            info_text += f"Fractional Standardization:\n"
            info_text += f"  {fractional_std['original_unit']} → {fractional_std['standardized_unit']}\n"
            if fractional_std.get('original_sample') is not None:
                info_text += f"  Sample: {fractional_std['original_sample']:.3f}% → {fractional_std['standardized_sample']:.3f} v/v\n"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel(f'{curve_name} ({self.curve_info.get(curve_name, {}).get("unit", "UNIT")})', fontsize=11)
        ax.grid(True, alpha=0.3)
        self.apply_depth_axis(ax, depth, label=f'Depth ({depth_unit})')
        ax.legend(loc=LABEL_UPPER_RIGHT, fontsize=10)
        
        self.fig.tight_layout()
    
    def _visualize_unprocessed_data(self, curve: str, viz_type: str):
        """Visualize unprocessed data from current_data"""
        try:
            # Clean up previous visualization resources
            self.cleanup_visualization()
            
            # Ensure we have a valid figure
            self.ensure_figure_exists()
            
            # Handle different visualization types
            if viz_type == "multi_curve":
                # For multi-curve, get selected curves from listbox
                selected_indices = self.curve_listbox.curselection()
                if not selected_indices:
                    messagebox.showwarning("Warning", "Please select at least one curve to plot")
                    return
                
                selected_curves = [self.curve_listbox.get(i) for i in selected_indices]
                self._plot_unprocessed_multi_curve(selected_curves)
            else:
                # Single curve visualization
                self._plot_unprocessed_single_curve(curve)
            
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, f"Failed to visualize unprocessed data: {e}")
    
    def _plot_unprocessed_single_curve(self, curve: str):
        """Plot a single unprocessed curve"""
        # Get the curve data from current_data
        if curve not in self.current_data.columns:
            messagebox.showerror("Error", f"Curve '{curve}' not found in unprocessed data")
            return
        
        curve_data = self.current_data[curve].values
        
        # Find depth curve if available
        depth_curve = None
        for col in self.current_data.columns:
            if 'DEPT' in col.upper() or 'DEPTH' in col.upper():
                depth_curve = col
                break
        
        # Use depth for Y-axis if available, otherwise use index
        if depth_curve:
            depth = self.current_data[depth_curve].values
            depth_unit = 'm'  # Default unit
            y_label = f'Depth ({depth_unit})'
        else:
            depth = np.arange(len(curve_data))
            y_label = 'Depth (index)'
        
        # Create the plot
        ax = self.fig.add_subplot(111)
        
        # Determine curve type and styling
        curve_type = self.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN')
        curve_family = curve_type.split('_')[0] if '_' in curve_type else curve_type
        
        # Use industry-standard colors
        industry_colors = PHYSICAL_CONSTANTS.LOG_COLORS
        if curve_family in industry_colors:
            color = industry_colors[curve_family]
        else:
            color = '#000000'  # Default black
        
        # Determine if this curve should use log scale
        use_log_scale = False
        log_scale_families = ['RESISTIVITY', 'PERMEABILITY']
        if curve_family in log_scale_families:
            use_log_scale = True
        
        # Plot the curve
        ax.plot(curve_data, depth, color=color, linewidth=1.5, label=f'{curve} (Unprocessed)')
        
        # Set appropriate scale
        if use_log_scale:
            # Handle zeros and negatives for log scale
            valid_data = curve_data[curve_data > 0]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                ax.set_xscale('log')
                # Set standard track scales if available
                if curve_family in PHYSICAL_CONSTANTS.LOG_TRACK_SCALES:
                    ax.set_xlim(PHYSICAL_CONSTANTS.LOG_TRACK_SCALES[curve_family])
                else:
                    # Fallback to reasonable log bounds
                    ax.set_xlim([min_val * 0.5, np.max(valid_data) * 2])
        
        # Set labels and title
        unit = self.curve_info.get(curve, {}).get('unit', '')
        ax.set_xlabel(f'{curve} ({unit})')
        ax.set_title(f'Unprocessed Data: {curve}', fontsize=14, fontweight='bold')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.apply_depth_axis(ax, depth, label=y_label)
        
        # Create canvas and display
        self._create_visualization_canvas("Note: Displaying unprocessed data. Run processing to see enhanced results.")
    
    def _plot_unprocessed_multi_curve(self, selected_curves: list):
        """Plot multiple unprocessed curves"""
        # Use industry-standard colors for log curves (API & SPWLA standards)
        industry_colors = PHYSICAL_CONSTANTS.LOG_COLORS
        
        # Create a depth track layout based on number of curves
        num_curves = len(selected_curves)
        
        if num_curves <= 3:
            # For 1-3 curves, use a single track with shared Y-axis
            ax = self.fig.add_subplot(111)
            self._plot_unprocessed_depth_based_curves(ax, selected_curves, industry_colors)
        else:
            # For 4+ curves, use multiple tracks (industry standard)
            # Determine number of tracks needed (maximum 3 curves per track)
            num_tracks = (num_curves + 2) // 3  # Ceiling division
            
            # Create a grid of tracks sharing the same y-axis
            axes = []
            for i in range(num_tracks):
                if i == 0:
                    ax = self.fig.add_subplot(1, num_tracks, i+1)
                    axes.append(ax)
                else:
                    ax = self.fig.add_subplot(1, num_tracks, i+1, sharey=axes[0])
                    axes.append(ax)
            
            # Distribute curves among tracks
            for i, track_ax in enumerate(axes):
                # Get curves for this track
                start_idx = i * 3
                end_idx = min((i + 1) * 3, num_curves)
                track_curves = selected_curves[start_idx:end_idx]
                
                # Plot curves on this track
                self._plot_unprocessed_depth_based_curves(track_ax, track_curves, industry_colors)
                
                # Only show depth labels on the first track
                if i > 0:
                    track_ax.set_ylabel('')
            
            self.fig.tight_layout()
        
        # Create canvas and display
        self._create_visualization_canvas("Note: Displaying unprocessed data. Run processing to see enhanced results.")
    
    def _plot_unprocessed_depth_based_curves(self, ax, curves, industry_colors):
        """Plot unprocessed curves in petroleum industry standard with depth on Y-axis"""
        # Find depth curve if available
        depth_curve = None
        for curve in curves:
            if 'DEPT' in curve.upper() or 'DEPTH' in curve.upper():
                depth_curve = curve
                break
        
        # If no explicit depth curve, use index
        if depth_curve:
            depth = self.current_data[depth_curve].values
            # Remove depth from plotting curves
            plot_curves = [c for c in curves if c != depth_curve]
        else:
            # Use row index as depth
            depth = np.arange(len(self.current_data))
            plot_curves = curves
        
        # Get actual depth range for proper axis limits
        depth_min, depth_max = self._get_depth_limits(depth)
        
        # Create twin axes for different scales if needed
        twin_axes = []
        
        # Plot each curve with appropriate styling
        for i, curve in enumerate(plot_curves):
            if curve not in self.current_data.columns:
                continue
            
            curve_data = self.current_data[curve].values
            
            # CRITICAL: Convert null values to NaN for proper line breaking
            curve_data = self._convert_nulls_to_nan(curve_data)
            
            # Skip if entire curve is NaN
            if np.all(np.isnan(curve_data)):
                continue
            curve_type = self.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN')
            curve_family = curve_type.split('_')[0] if '_' in curve_type else curve_type
            
            # Determine if this curve should use log scale
            use_log_scale = False
            log_scale_families = ['RESISTIVITY', 'PERMEABILITY']
            if curve_family in log_scale_families:
                use_log_scale = True
                
            # Determine color based on industry standards
            if curve_family in industry_colors:
                color = industry_colors[curve_family]
            else:
                # Use a color cycle for non-standard curves
                color = plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)]
            
            # Determine line style and width
            line_style = '-'
            line_width = 1.5
            
            # For multiple curves with different scales, create twin axes
            if i > 0 and use_log_scale != (ax.get_xscale() == 'log'):
                twin_ax = ax.twiny()
                twin_axes.append(twin_ax)
                current_ax = twin_ax
                # Position the axis at the top for the second curve
                current_ax.xaxis.set_ticks_position('top')
                current_ax.xaxis.set_label_position('top')
            else:
                current_ax = ax
            
            # Handle missing data (NaN values break lines properly)
            valid_mask = ~np.isnan(curve_data) & np.isfinite(curve_data)
            if np.any(valid_mask):
                valid_data = curve_data[valid_mask]
                valid_depth = depth[valid_mask]
                
                # Set appropriate scale for logarithmic curves
                if use_log_scale:
                    # Handle zeros and negatives for log scale
                    positive_mask = valid_data > 0
                    if np.any(positive_mask):
                        log_data = valid_data[positive_mask]
                        log_depth = valid_depth[positive_mask]
                        current_ax.set_xscale('log')
                        # Set standard track scales for this curve type if available
                        if curve_family in PHYSICAL_CONSTANTS.LOG_TRACK_SCALES:
                            current_ax.set_xlim(PHYSICAL_CONSTANTS.LOG_TRACK_SCALES[curve_family])
                        else:
                            # Fallback to reasonable log bounds
                            min_val = np.min(log_data)
                            current_ax.set_xlim([min_val * 0.5, np.max(log_data) * 2])
                        
                        # Plot with depth on Y-axis
                        current_ax.plot(log_data, log_depth, color=color, linestyle=line_style, 
                                      linewidth=line_width, label=f'{curve} (Unprocessed)')
                    else:
                        # No positive values for log scale, use linear
                        current_ax.plot(valid_data, valid_depth, color=color, linestyle=line_style, 
                                      linewidth=line_width, label=f'{curve} (Unprocessed)')
                else:
                    # Linear scale - plot with depth on Y-axis
                    current_ax.plot(valid_data, valid_depth, color=color, linestyle=line_style, 
                                  linewidth=line_width, label=f'{curve} (Unprocessed)')
                
                # Add gridlines
                current_ax.grid(True, alpha=0.3, which='both')
                
                # Set labels
                unit = self.curve_info.get(curve, {}).get('unit', '')
                current_ax.set_xlabel(f'{curve} ({unit})')
        
        # CRITICAL: Set axis limits to ACTUAL data range (not default range).
        if depth_curve:
            depth_unit = 'm'  # Default unit
            label = f'Depth ({depth_unit})'
        else:
            label = 'Depth (index)'
        self.apply_depth_axis(ax, depth, label=label)
        
        # Add legends
        handles, labels = ax.get_legend_handles_labels()
        for twin_ax in twin_axes:
            twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
            handles.extend(twin_handles)
            labels.extend(twin_labels)
        
        if handles:
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=min(3, len(handles)))
    
    def _create_visualization_canvas(self, status_message: str):
        """Create and display the visualization canvas with status message"""
        if hasattr(self, 'viz_content') and self.viz_content:
            self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
            try:
                self.canvas.draw_idle()
            except Exception:
                self.canvas.draw()
            
            # Create navigation toolbar
            if NavigationToolbar2Tk is not None:
                toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                toolbar.update()
                toolbar.pack(side='top', fill='x')
            
            # Pack canvas below toolbar
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
            
            # Add status note
            status_note = ttk.Label(self.viz_content, text=status_message, style='Info.TLabel')
            status_note.pack(side='bottom', pady=5)

            # Add export buttons
            export_frame = ttk.Frame(self.viz_content)
            export_frame.pack(side='bottom', fill='x', pady=(5, 10))
            def _export_fig(dpi=150):
                try:
                    from tkinter import filedialog as _fd
                    path = _fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
                    if not path:
                        return
                    fmt = 'pdf' if path.lower().endswith('.pdf') else 'png'
                    # Set publication or screen margins via bbox_inches
                    self.fig.savefig(path, dpi=dpi, format=fmt, bbox_inches='tight', facecolor='white')
                    self.status_label.config(text=f"Figure exported: {path}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to export figure: {e}")
            ttk.Button(export_frame, text="Export (Screen)", command=lambda: _export_fig(150)).pack(side='left', padx=(0, 8))
            ttk.Button(export_frame, text="Export (Publication)", command=lambda: _export_fig(300)).pack(side='left')
    
    def plot_comparison(self, curve: str):
        """Plot original vs processed comparison - overlay mode with same X-Y axes and toggle controls"""
        # Check if curve has been processed
        if curve in self.processing_results:
            original = self.processing_results[curve]['original_data']
            processed = self.processing_results[curve]['final_data']
            has_processed = True
        elif self.current_data is not None and curve in self.current_data.columns:
            # Show only original data if not processed
            original = self.current_data[curve].values
            processed = None
            has_processed = False
        else:
            messagebox.showwarning("Warning", f"Curve '{curve}' not found in data")
            return
        
        # Ensure we have a valid figure with good size for overlay viewing
        self.ensure_figure_exists()
        self.fig.set_size_inches(12, 10)
        
        # Get curve color from industry standards, or use neutral color
        curve_info = self.curve_info.get(curve, {})
        curve_type = curve_info.get('curve_type', 'UNKNOWN')
        industry_color = curve_info.get('industry_color', '#333333')  # Default to dark gray
        
        # Use industry color if available, otherwise neutral dark color
        base_color = industry_color if industry_color != '#000000' else '#333333'
        
        # Find depth curve if available
        depth_curve = None
        data_source = self.processed_data if has_processed else self.current_data
        for col in data_source.columns:
            curve_type_col = self.curve_info.get(col, {}).get('curve_type', '')
            if 'DEPTH' in curve_type_col:
                depth_curve = col
                break
        
        # Use depth for Y-axis if available, otherwise use index
        if depth_curve:
            depth = data_source[depth_curve].values
            depth_unit = self.curve_info.get(depth_curve, {}).get('unit', 'm')
            y_label = f'Depth ({depth_unit})'
            # Get actual depth range for proper axis limits
            depth_min, depth_max = self._get_depth_limits(depth)
        else:
            depth = np.arange(len(original))
            y_label = 'Depth (index)'
            depth_min, depth_max = self._get_depth_limits(depth)
        
        # Single plot area - overlay mode with toggle capability
        if has_processed and processed is not None:
            ax = self.fig.add_subplot(111)
            
            # Store plot objects for toggle functionality (stored in figure for persistence)
            plot_objects = {}
            
            # Convert null values to NaN for proper line breaking (for visualization only)
            original_plot = self._convert_nulls_to_nan(original)
            processed_plot = self._convert_nulls_to_nan(processed)
            
            # Plot Original - lower opacity (always present for toggle)
            line_orig = ax.plot(original_plot, depth, color=base_color, alpha=0.4, label='Original', 
                    linewidth=1.5, linestyle='-', visible=True)[0]
            plot_objects['original'] = line_orig
            
            # Plot Processed - higher opacity (always present for toggle)
            line_proc = ax.plot(processed_plot, depth, color=base_color, alpha=0.9, label='Processed', 
                    linewidth=2.0, linestyle='-', visible=True)[0]
            plot_objects['processed'] = line_proc
            
            # Store in figure for toggle access
            self.fig._comparison_plot_objects = plot_objects
            self.fig._comparison_curve = curve
            
            # Mark significant changes
            valid_mask = ~np.isnan(original) & ~np.isnan(processed)
            if np.any(valid_mask):
                changes = np.abs(original[valid_mask] - processed[valid_mask])
                if len(changes) > 0:
                    # Find points with significant changes (top 5%)
                    threshold = np.percentile(changes, 95) if len(changes) > 20 else np.max(changes) * 0.5
                    significant_idx = np.nonzero((np.abs(original - processed) > threshold) & valid_mask)[0]
                    
                    # Mark points with significant changes (subtle marker)
                    if len(significant_idx) > 0:
                        x_proc = processed[significant_idx]
                        y_proc = depth[significant_idx]
                        scatter = ax.scatter(x_proc, y_proc, color=base_color, s=30, alpha=0.6, 
                                  marker='o', edgecolors='none', label=LABEL_SIGNIFICANT_CHANGES, zorder=3)
                        plot_objects['changes'] = scatter
            
            # Add gap annotations if available
            if 'gap_filling' in self.processing_results[curve]:
                gap_info = self.processing_results[curve]['gap_filling'].get('gaps_filled', [])
                if gap_info:
                    for i, gap in enumerate(gap_info[:3]):  # Limit to first 3 gaps
                        gap_start = gap['gap']['start']
                        gap_end = gap['gap']['end']
                        gap_center = (gap_start + gap_end) // 2
                        if gap_center < len(processed) and not np.isnan(processed[gap_center]):
                            ax.annotate(f'Gap {i+1}',
                                       xy=(processed[gap_center], depth[gap_center]),
                                       xytext=(10, 20),
                                       textcoords=LABEL_OFFSET_POINTS,
                                       fontsize=8,
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                                                      color=base_color, alpha=0.6))
            
            # CRITICAL: Set axis limits to ACTUAL data range
            self.apply_depth_axis(ax, depth, label=y_label)
            
            # Configure axes
            ax.set_title(f'Processing Comparison: {curve} (Click legend to toggle)', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel(f'{curve} ({curve_info.get("unit", "UNIT")})', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Legend with toggle capability
            legend = ax.legend(loc=LABEL_UPPER_RIGHT, fontsize=10, framealpha=0.9)
            self.fig._comparison_legend = legend
            self.fig._comparison_plot_objects = plot_objects
            self.fig._comparison_ax = ax  # Store for event handler
            
            # Make legend items clickable for toggle
            def on_legend_click(event):
                """Toggle plot visibility when legend item is clicked"""
                if not hasattr(self.fig, '_comparison_plot_objects'):
                    return
                    
                stored_ax = self.fig._comparison_ax
                stored_legend = self.fig._comparison_legend
                stored_objects = self.fig._comparison_plot_objects
                
                if event.inaxes != stored_ax:
                    return
                
                # Check if legend was clicked
                if stored_legend.contains(event)[0]:
                    handles = stored_legend.legendHandles
                    texts = [t.get_text() for t in stored_legend.get_texts()]
                    
                    # Simple click detection on legend items
                    clicked = False
                    for handle, label in zip(handles, texts):
                        # Toggle visibility based on label
                        if label == 'Original':
                            new_visibility = not stored_objects['original'].get_visible()
                            stored_objects['original'].set_visible(new_visibility)
                            if hasattr(handle, 'set_alpha'):
                                handle.set_alpha(1.0 if new_visibility else 0.3)
                            clicked = True
                        elif label == 'Processed':
                            new_visibility = not stored_objects['processed'].get_visible()
                            stored_objects['processed'].set_visible(new_visibility)
                            if hasattr(handle, 'set_alpha'):
                                handle.set_alpha(1.0 if new_visibility else 0.3)
                            clicked = True
                        elif 'changes' in stored_objects and label == LABEL_SIGNIFICANT_CHANGES:
                            new_visibility = not stored_objects['changes'].get_visible()
                            stored_objects['changes'].set_visible(new_visibility)
                            if hasattr(handle, 'set_alpha'):
                                handle.set_alpha(1.0 if new_visibility else 0.3)
                            clicked = True
                    
                    if clicked:
                        # Redraw
                        self.fig.canvas.draw_idle()
            
            # Connect click event to legend
            self.fig.canvas.mpl_connect('button_press_event', on_legend_click)
            
            # Add statistics comparison panels below plot
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(3, 2, figure=self.fig, height_ratios=[10, 1, 1], hspace=0.4)
            
            # Move main plot to use GridSpec
            ax.remove()
            ax = self.fig.add_subplot(gs[0, :])
            
            # Re-plot everything on new axes (overlay mode - same X-Y)
            # Use already converted data (original_plot and processed_plot)
            line_orig = ax.plot(original_plot, depth, color=base_color, alpha=0.4, label='Original', 
                    linewidth=1.5, linestyle='-', visible=True)[0]
            plot_objects['original'] = line_orig
            
            line_proc = ax.plot(processed_plot, depth, color=base_color, alpha=0.9, label='Processed', 
                    linewidth=2.0, linestyle='-', visible=True)[0]
            plot_objects['processed'] = line_proc
            
            # Re-add significant changes if available (use converted data)
            valid_mask_plot = ~np.isnan(original_plot) & ~np.isnan(processed_plot)
            if np.any(valid_mask_plot):
                changes = np.abs(original_plot[valid_mask_plot] - processed_plot[valid_mask_plot])
                if len(changes) > 0:
                    threshold = np.percentile(changes, 95) if len(changes) > 20 else np.max(changes) * 0.5
                    significant_idx = np.nonzero((np.abs(original_plot - processed_plot) > threshold) & valid_mask_plot)[0]
                    if len(significant_idx) > 0:
                        x_proc = processed_plot[significant_idx]
                        y_proc = depth[significant_idx]
                        scatter = ax.scatter(x_proc, y_proc, color=base_color, s=30, alpha=0.6, 
                                  marker='o', edgecolors='none', label=LABEL_SIGNIFICANT_CHANGES, zorder=3)
                        plot_objects['changes'] = scatter
            
            # Re-add gap annotations
            if 'gap_filling' in self.processing_results[curve]:
                gap_info = self.processing_results[curve]['gap_filling'].get('gaps_filled', [])
                if gap_info:
                    for i, gap in enumerate(gap_info[:3]):
                        gap_start = gap['gap']['start']
                        gap_end = gap['gap']['end']
                        gap_center = (gap_start + gap_end) // 2
                        if gap_center < len(processed) and not np.isnan(processed[gap_center]):
                            ax.annotate(f'Gap {i+1}',
                                       xy=(processed[gap_center], depth[gap_center]),
                                       xytext=(10, 20),
                                       textcoords=LABEL_OFFSET_POINTS,
                                       fontsize=8,
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                                                      color=base_color, alpha=0.6))
            
            ax.set_title(f'Processing Comparison: {curve} (Click legend to toggle)', fontsize=14, fontweight='bold', pad=10)
            # CRITICAL: Set axis limits to ACTUAL data range
            self.apply_depth_axis(ax, depth, label=y_label)
            
            ax.set_xlabel(f'{curve} ({curve_info.get("unit", "UNIT")})', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            legend = ax.legend(loc=LABEL_UPPER_RIGHT, fontsize=10, framealpha=0.9)
            self.fig._comparison_legend = legend
            self.fig._comparison_plot_objects = plot_objects
            self.fig._comparison_ax = ax
            
            # Reconnect toggle handler
            self.fig.canvas.mpl_connect('button_press_event', on_legend_click)
            
            # Statistics panels (side-by-side below plot)
            valid_orig = original[~np.isnan(original)]
            valid_proc = processed[~np.isnan(processed)]
            
            # Original stats panel
            ax_stats_orig = self.fig.add_subplot(gs[1:, 0])
            ax_stats_orig.axis('off')
            if len(valid_orig) > 0:
                stats_orig = (
                    "Original Statistics:\n"
                    f"Min: {np.min(valid_orig):.3f}\n"
                    f"Max: {np.max(valid_orig):.3f}\n"
                    f"Mean: {np.mean(valid_orig):.3f}\n"
                    f"Std: {np.std(valid_orig):.3f}\n"
                    f"Missing: {np.sum(np.isnan(original))/len(original)*100:.1f}%"
                )
                ax_stats_orig.text(0.1, 0.5, stats_orig, transform=ax_stats_orig.transAxes,
                        fontsize=9, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
            # Processed stats panel
            ax_stats_proc = self.fig.add_subplot(gs[1:, 1])
            ax_stats_proc.axis('off')
            if len(valid_proc) > 0:
                stats_proc = (
                    "Processed Statistics:\n"
                    f"Min: {np.min(valid_proc):.3f}\n"
                    f"Max: {np.max(valid_proc):.3f}\n"
                    f"Mean: {np.mean(valid_proc):.3f}\n"
                    f"Std: {np.std(valid_proc):.3f}\n"
                    f"Missing: {np.sum(np.isnan(processed))/len(processed)*100:.1f}%"
                )
                ax_stats_proc.text(0.1, 0.5, stats_proc, transform=ax_stats_proc.transAxes,
                        fontsize=9, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
            self.fig.tight_layout()
            
        else:
            # Single panel if not processed
            ax = self.fig.add_subplot(111)
            ax.plot(original, depth, color=base_color, alpha=0.7, label=LABEL_ORIGINAL_DATA, linewidth=2)
            ax.set_title(f'Original Data: {curve} (Not Yet Processed)', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{curve} ({curve_info.get("unit", "UNIT")})', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            self.apply_depth_axis(ax, depth, label=y_label)
            
            # Statistics box
            valid_orig = original[~np.isnan(original)]
            if len(valid_orig) > 0:
                stats_text = (
                    f"Min: {np.min(valid_orig):.3f}\n"
                    f"Max: {np.max(valid_orig):.3f}\n"
                    f"Mean: {np.mean(valid_orig):.3f}\n"
                    f"Std: {np.std(valid_orig):.3f}\n"
                    f"Missing: {np.sum(np.isnan(original))/len(original)*100:.1f}%"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
            
            self.fig.tight_layout()
    

    
    def generate_report(self):
        """Generate the comprehensive processing report and display it in the report tab.
        Uses the in-class create_comprehensive_report() which assembles header, LAS preview, and
        full analysis details. Robustly updates the UI text widget and surfaces errors to user.
        """
        try:
            if not hasattr(self, 'report_text'):
                messagebox.showerror("Report", "Report view is not initialized yet.")
                return
            # Build report from current application state
            report_text = self.create_comprehensive_report()
            # Update UI text widget
            self.report_text.config(state='normal')
            self.report_text.delete('1.0', 'end')
            self.report_text.insert('1.0', report_text)
            self.report_text.config(state='disabled')
            try:
                self.status_label.config(text="Report generated")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {e}")
            try:
                self.status_label.config(text="Report generation failed")
            except Exception:
                pass

    def _generate_las_text_from_dataframe(self, df: pd.DataFrame, curve_info: dict, null_value: str, max_rows: int = None) -> str:
        """Create a minimally compliant LAS v2.0 text from a DataFrame and curve metadata.
        - Ensures depth curve appears first when available
        - Uses units/descriptions from curve_info where possible
        - Replaces NaN with provided null_value
        - If max_rows provided, limits ASCII section to that many rows (for preview)
        """
        try:
            from datetime import datetime as _dt
            # Determine depth curve preference
            preferred_depth_names = ["DEPT", "DEPTH", "MD", "MDMSL"]
            columns = list(df.columns)
            depth_col = next((c for c in preferred_depth_names if c in columns), None)
            if depth_col:
                ordered_cols = [depth_col] + [c for c in columns if c != depth_col]
            else:
                ordered_cols = columns

            # Header blocks
            lines = []
            lines.append("~Version")
            lines.append("VERS.                  2.0:   CWLS LOG ASCII STANDARD - VERSION 2.0")
            lines.append("WRAP.                   NO:   One line per depth step")
            lines.append("DLM .                 SPACE:   Column Data Section Delimiter")
            lines.append("")
            lines.append("~Well")
            try:
                file_label = self.file_path_var.get() if hasattr(self, 'file_path_var') else ''
            except (tk.TclError, AttributeError) as file_var_error:
                # UI variable access failed - use empty string and log
                self.handle_ui_error(
                    file_var_error,
                    "File path variable access in LAS export",
                    "file_path_var",
                    graceful_degradation=True
                )
                file_label = ''
            except Exception as file_var_error:
                # Unexpected error accessing file path variable
                self.handle_ui_error(
                    file_var_error,
                    "File path variable access in LAS export",
                    "file_path_var",
                    graceful_degradation=True
                )
                file_label = ''
            lines.append(f"FILE. {file_label} :   Source file path")
            lines.append(f"DATE. {_dt.now().strftime('%Y-%m-%d %H:%M:%S')} :   Export timestamp")
            lines.append(f"NULL. {null_value} :   Null value")
            lines.append("")

            # Curve Information block
            lines.append("~Curve")
            for col in ordered_cols:
                info = curve_info.get(col, {}) if isinstance(curve_info, dict) else {}
                unit = info.get('unit', '') if isinstance(info, dict) else ''
                desc = info.get('description', '') if isinstance(info, dict) else ''
                safe_unit = unit if unit is not None else ''
                safe_desc = desc if desc is not None else ''
                lines.append(f"{col}. {safe_unit} : {safe_desc}")
            lines.append("")

            # ASCII data block
            lines.append("~Ascii")
            # Prepare values with null substitution
            nv = null_value
            # Limit rows for preview if requested
            data_iter = df[ordered_cols].itertuples(index=False, name=None)
            count = 0
            for row in data_iter:
                if max_rows is not None and count >= max_rows:
                    break
                formatted = []
                for val in row:
                    if pd.isna(val):
                        formatted.append(str(nv))
                    else:
                        # Use a sensible formatting to avoid scientific notation explosions
                        try:
                            if isinstance(val, (int,)):
                                formatted.append(f"{val}")
                            else:
                                formatted.append(f"{float(val):.6g}")
                        except Exception:
                            formatted.append(str(val))
                lines.append(" ".join(formatted))
                count += 1

            return "\n".join(lines)
        except Exception as e:
            # In case of unexpected failure, surface a simple, direct representation
            fallback = ["~Ascii", str(e)]
            try:
                fallback.extend(df.head(50).to_string(index=False).splitlines())
            except Exception:
                pass
            return "\n".join(fallback)

    def preview_original_las(self):
        """Display the original LAS header and raw content preview in the Original LAS tab."""
        try:
            if not hasattr(self, 'original_las_preview_text'):
                messagebox.showerror("Preview", "Original LAS preview is not initialized yet.")
                return

            # Start fresh
            self.original_las_preview_text.config(state='normal')
            self.original_las_preview_text.delete('1.0', 'end')

            # Include header if available
            if hasattr(self, 'original_las_header') and self.original_las_header:
                self.original_las_preview_text.insert('end', "ORIGINAL LAS HEADER\n")
                self.original_las_preview_text.insert('end', "-" * 40 + "\n")
                self.original_las_preview_text.insert('end', self.original_las_header + "\n\n")

            # Try to include first 100 lines of the original LAS file
            raw_added = False
            try:
                from pathlib import Path
                las_filepath = None
                try:
                    las_filepath = Path(self.file_path_var.get()) if self.file_path_var.get() else None
                except Exception:
                    las_filepath = None
                if las_filepath and las_filepath.exists():
                    with las_filepath.open('r', encoding='utf-8', errors='replace') as f:
                        raw_lines = f.read().splitlines()
                    self.original_las_preview_text.insert('end', "FIRST 100 LINES OF LAS FILE (raw)\n")
                    self.original_las_preview_text.insert('end', "-" * 40 + "\n")
                    for i, ln in enumerate(raw_lines[:100], start=1):
                        self.original_las_preview_text.insert('end', f"{i:03}: {ln}\n")
                    raw_added = True
            except Exception:
                pass

            if not raw_added and not (hasattr(self, 'original_las_header') and self.original_las_header):
                self.original_las_preview_text.insert('end', "No original LAS content available. Load a LAS file first.\n")

            self.original_las_preview_text.config(state='disabled')
            try:
                self.status_label.config(text="Original LAS preview updated")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to generate Original LAS preview: {e}")
            try:
                self.status_label.config(text="Original LAS preview failed")
            except Exception:
                pass

    def preview_processed_las(self):
        """Display a LAS-format preview synthesized from the processed DataFrame and curve metadata."""
        try:
            if not hasattr(self, 'processed_las_preview_text'):
                messagebox.showerror("Preview", "Processed LAS preview is not initialized yet.")
                return
            if self.processed_data is None or not isinstance(self.processed_data, pd.DataFrame) or self.processed_data.empty:
                messagebox.showerror("Preview", "No processed data available. Run processing first.")
                return

            # Build LAS text limited to a reasonable number of rows for UI responsiveness
            null_value = self.null_value_var.get() if hasattr(self, 'null_value_var') else "-999.25"
            las_text = self._generate_las_text_from_dataframe(self.processed_data, self.curve_info or {}, str(null_value), max_rows=300)

            # Update UI text widget
            self.processed_las_preview_text.config(state='normal')
            self.processed_las_preview_text.delete('1.0', 'end')
            self.processed_las_preview_text.insert('1.0', las_text)
            self.processed_las_preview_text.config(state='disabled')
            try:
                self.status_label.config(text="Processed LAS preview updated")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to generate Processed LAS preview: {e}")
            try:
                self.status_label.config(text="Processed LAS preview failed")
            except Exception:
                pass

    def export_data(self):
        """Export processed data to CSV, Excel, or LAS using a simple, robust writer.
        - CSV: comma-separated values without index
        - Excel: .xlsx via pandas (if engine available)
        - LAS: text assembled from DataFrame and curve metadata
        Includes security validation for path traversal protection.
        """
        try:
            if self.processed_data is None or not isinstance(self.processed_data, pd.DataFrame) or self.processed_data.empty:
                messagebox.showerror("Export", "No processed data available to export.")
                return
            filetypes = [
                ("LAS file", "*.las"),
                ("CSV file", "*.csv"),
                ("Excel workbook", "*.xlsx"),
            ]
            save_path = filedialog.asksaveasfilename(defaultextension=".las", filetypes=filetypes)
            if not save_path:
                return
            
            # Security: Validate and normalize export path
            validated_path = SafeFileHandler.validate_file_path(save_path)
            if not validated_path:
                sanitized = SafeFileHandler.sanitize_path_for_display(save_path)
                messagebox.showerror(ERROR_TITLE_SECURITY, f"Invalid export path: {sanitized}")
                return
            
            # Security: Validate file extension
            if not SafeFileHandler.validate_file_extension(str(validated_path), mode='write'):
                ext = os.path.splitext(str(validated_path))[1].lower()
                messagebox.showerror("File Type Error", 
                                   f"Invalid export file type: {ext}\nAllowed types: .las, .csv, .xlsx")
                return
            
            sp = str(validated_path)
            sp_lower = sp.lower()
            export_df = self._dataframe_with_uncertainty_bands(self.processed_data)
            # Missing data is held as NaN internally, so the sentinel is
            # written here at the export boundary. Without na_rep the tabular
            # writers would emit empty cells instead of the declared null.
            null_value = self.null_value_var.get() if hasattr(self, 'null_value_var') else "-999.25"
            if sp_lower.endswith('.csv'):
                export_df.to_csv(sp, index=False, na_rep=str(null_value))
            elif sp_lower.endswith('.xlsx'):
                try:
                    export_df.to_excel(sp, index=False, na_rep=str(null_value))
                except Exception as ex:
                    messagebox.showerror("Export", f"Excel export failed: {ex}")
                    return
            else:
                # Default to LAS
                # Merge uncertainty into curve_info for LAS headers when present
                export_info = dict(self.curve_info or {})
                for col in export_df.columns:
                    if col.endswith('_UNC') or col.endswith('_CONF'):
                        export_info.setdefault(col, {
                            'unit': '' if col.endswith('_CONF') else export_info.get(col.replace('_UNC', '').replace('_CONF', ''), {}).get('unit', ''),
                            'description': 'Gap-fill uncertainty' if col.endswith('_UNC') else 'Gap-fill confidence (0-1)',
                            'curve_type': 'QC',
                        })
                las_text = self._generate_las_text_from_dataframe(export_df, export_info, str(null_value), max_rows=None)
                with open(sp, 'w', encoding='utf-8') as f:
                    f.write(las_text)
            messagebox.showinfo("Export", f"Exported data to: {sp}")
            try:
                self.status_label.config(text="Data exported")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
            try:
                self.status_label.config(text="Export failed")
            except Exception:
                pass

    def _dataframe_with_uncertainty_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach per-curve uncertainty and confidence columns from gap-fill results when available."""
        out = df.copy()
        results = getattr(self, 'processing_results', None) or {}
        for curve, result in results.items():
            if curve not in out.columns:
                continue
            gap = result.get('gap_filling') or {}
            unc = gap.get('uncertainty')
            conf = gap.get('confidence')
            n = len(out)
            if unc is not None and len(unc) == n and np.nanmax(unc) > 0:
                out[f'{curve}_UNC'] = np.asarray(unc, dtype=float)
            if conf is not None and len(conf) == n:
                # Only export confidence where uncertainty was meaningful (filled gaps)
                if unc is not None and len(unc) == n and np.nanmax(unc) > 0:
                    out[f'{curve}_CONF'] = np.asarray(conf, dtype=float)
        return out
    
    def browse_batch_directory(self):
        """Browse for batch processing input directory"""
        directory = filedialog.askdirectory(title="Select Directory with LAS Files")
        if directory:
            self.batch_directory_var.set(directory)
    
    def browse_batch_output_directory(self):
        """Browse for batch processing output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.batch_output_dir_var.set(directory)
    
    def scan_batch_directory(self):
        """Scan selected directory for LAS files"""
        try:
            directory = self.batch_directory_var.get()
            if not directory:
                messagebox.showwarning("Warning", "Please select a directory first.")
                return
            
            recursive = self.batch_recursive_var.get()
            files = self.batch_manager.load_directory(directory, recursive=recursive)
            
            # Update listbox
            self.batch_file_listbox.delete(0, tk.END)
            for file in files:
                self.batch_file_listbox.insert(tk.END, os.path.basename(file))
            
            # Update status
            count = len(files)
            self.batch_status_label.config(text=f"Found {count} file(s)")
            
            if count == 0:
                messagebox.showinfo("Info", "No LAS files found in the selected directory.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan directory:\n{str(e)}")
    
    def start_batch_processing(self):
        """Start batch processing in a separate thread"""
        try:
            if not self.batch_directory_var.get():
                messagebox.showwarning("Warning", "Please select an input directory first.")
                return
            
            output_dir = self.batch_output_dir_var.get()
            if not output_dir:
                messagebox.showwarning("Warning", "Please select an output directory first.")
                return
            
            # Disable start button, enable stop button
            self.batch_process_btn.config(state='disabled')
            self.batch_stop_btn.config(state='normal')
            
            # Set processing flag
            if self.batch_manager:
                self.batch_manager.is_processing = True
            
            # Start processing in thread
            thread = threading.Thread(target=self._batch_processing_thread, daemon=True)
            thread.start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start batch processing:\n{str(e)}")
            self.batch_process_btn.config(state='normal')
            self.batch_stop_btn.config(state='disabled')
    
    def _batch_processing_thread(self):
        """Background thread for batch processing"""
        try:
            directory = self.batch_directory_var.get()
            output_dir = self.batch_output_dir_var.get()
            recursive = self.batch_recursive_var.get()
            
            # Load files
            files = self.batch_manager.load_directory(directory, recursive=recursive)
            
            if not files:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No files to process."))
                return
            
            total = len(files)
            
            # Process each file
            for i, file_path in enumerate(files):
                if self.batch_manager.is_processing is False:
                    break
                
                try:
                    # Update progress
                    progress = int((i / total) * 100)
                    self.root.after(0, lambda p=progress, f=os.path.basename(file_path): 
                                   self._update_batch_progress(p, f))
                    
                    # Process file using parent app's methods
                    # This would integrate with the main processing pipeline
                    # For now, just log the file
                    self.root.after(0, lambda f=file_path: self.log_processing(f"Processing: {f}"))
                    
                except Exception as e:
                    self.root.after(0, lambda e=e, f=file_path: 
                                   self.log_processing(f"Error processing {f}: {e}"))
            
            # Complete
            self.root.after(0, lambda: self._batch_processing_complete())
        
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Batch processing failed:\n{str(e)}"))
            self.root.after(0, lambda: self._batch_processing_complete())
    
    def _update_batch_progress(self, progress: int, filename: str):
        """Update batch processing progress (called from main thread)"""
        self.batch_progress_bar['value'] = progress
        self.batch_progress_label.config(text=f"Processing: {filename} ({progress}%)")
        self.root.update_idletasks()
    
    def _batch_processing_complete(self):
        """Called when batch processing completes"""
        self.batch_progress_bar['value'] = 100
        self.batch_progress_label.config(text="Processing complete")
        self.batch_process_btn.config(state='normal')
        self.batch_stop_btn.config(state='disabled')
        messagebox.showinfo("Complete", "Batch processing completed.")
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        try:
            if self.batch_manager:
                self.batch_manager.is_processing = False
            self.batch_progress_label.config(text="Stopping...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop processing:\n{str(e)}")
    
    def browse_file(self):
        """Browse for data file"""
        filetypes = [
            ("LAS files", "*.las"),
            ("DLIS/LIS files", "*.dlis *.lis"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    def load_file(self):
        """Load and analyze data file with analytics and security validation"""
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a valid file")
            return
        
        # Security: Validate and normalize file path
        validated_path = SafeFileHandler.validate_file_path(filepath)
        if not validated_path:
            sanitized = SafeFileHandler.sanitize_path_for_display(filepath)
            messagebox.showerror(ERROR_TITLE_SECURITY, f"Invalid or inaccessible file path: {sanitized}")
            return
        
        # Security: Validate file size before loading
        if not SafeFileHandler.validate_file_size(str(validated_path)):
            size_mb = os.path.getsize(str(validated_path)) / (1024 * 1024)
            max_mb = SafeFileHandler.MAX_FILE_SIZE_MB
            messagebox.showerror("File Size Error", 
                               f"File is too large: {size_mb:.1f}MB\nMaximum allowed: {max_mb}MB")
            return
        
        # Security: Validate file extension
        if not SafeFileHandler.validate_file_extension(str(validated_path), mode='read'):
            ext = os.path.splitext(str(validated_path))[1].lower()
            messagebox.showerror("File Type Error", 
                               f"Invalid file type: {ext}\nAllowed types: .las, .csv, .xlsx, .xls")
            return
        
        # Use validated path
        filepath = str(validated_path)
        
        try:
            # Clear existing data before loading new file with unsaved data check
            if self.reset_application_state(prompt_if_unsaved=True) == False:
                # User cancelled due to unsaved data
                return
            
            self.status_label.config(text="Loading file...")
            self.progress_bar['value'] = 10
            
            # Load file based on extension
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.las':
                self.current_data = self.load_las_file(filepath)
                # Well info is extracted in load_las_file, so it's already set
            elif ext in ('.dlis', '.lis'):
                self.current_data = self.load_dlis_file(filepath)
            elif ext == '.csv':
                self.current_data = self.load_csv_file(filepath)
                # For CSV/Excel, create basic well info from filename if not set
                if not hasattr(self, 'well_info') or not self.well_info or self.well_info.get('well_name') == 'UNKNOWN':
                    self.well_info = {
                        'well_name': os.path.splitext(os.path.basename(filepath))[0],
                        'uwi': 'N/A',
                        'field': 'N/A',
                        'company': 'N/A',
                        'start_depth': 'N/A',
                        'stop_depth': 'N/A',
                        'depth_unit': 'm'
                    }
            elif ext in ['.xlsx', '.xls']:
                self.current_data = self.load_excel_file(filepath)
                # For Excel, create basic well info from filename if not set
                if not hasattr(self, 'well_info') or not self.well_info or self.well_info.get('well_name') == 'UNKNOWN':
                    self.well_info = {
                        'well_name': os.path.splitext(os.path.basename(filepath))[0],
                        'uwi': 'N/A',
                        'field': 'N/A',
                        'company': 'N/A',
                        'start_depth': 'N/A',
                        'stop_depth': 'N/A',
                        'depth_unit': 'm'
                    }
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # CRITICAL: Update well info display immediately after loading
            # This ensures users can see the well information right away
            self._update_well_info_display()
            self._update_window_title_with_well_info()
            
            # CRITICAL: Add well to well_datasets so it appears in "Loaded Wells" section
            if filepath:
                well_id = self._gen_well_id_from_info(filepath)
                self.well_datasets[well_id] = self._dataset_from_current_state(filepath)
                self.active_well_id = well_id
                # Update the well listbox to show the loaded well
                self.update_well_list_display()

            # Apply LAS-declared NULL (prompt only if multiple loaded wells disagree)
            self._reconcile_null_value_convention(allow_prompt=True)
            
            self.progress_bar['value'] = 50
            self.status_label.config(text="Analyzing curves...")
            
            # Analyze curves
            self.analyze_curves()
            
            # Skip auto-fix for performance - curves are already identified
            pass
            
            # Optional standardization on upload (fractional families: % → v/v) before any validation/preview
            try:
                if self.standardize_on_upload_var.get():
                    self.standardize_fractional_curves_on_upload()
            except Exception as e:
                self.log_processing(f"Standardize-on-upload failed: {e}")
            
            self.progress_bar['value'] = 100
            self.status_label.config(text="File loaded successfully")
            
            # Update UI
            self.update_data_display()
            self.update_curve_options()
            
            # Ensure all curves have statistics calculated
            self.ensure_curve_statistics()
            
            # Automatically update original LAS preview
            self.preview_original_las()
            # Prepend note about upload standardization if any
            try:
                if getattr(self, '_upload_standardization_note', '') and hasattr(self, 'original_las_preview_text'):
                    self.original_las_preview_text.config(state='normal')
                    self.original_las_preview_text.insert('1.0', self._upload_standardization_note + "\n\n")
                    self.original_las_preview_text.config(state='disabled')
            except Exception:
                pass
            
            # Track file loading with analytics
            if BETA_SYSTEM_AVAILABLE and self.beta_analytics:
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                curve_count = len(self.current_data.columns) if self.current_data is not None else 0
                self.beta_analytics.track_file_loaded(ext, file_size_mb, curve_count)
            
            # Information logging removed
            # System status handled - operation continues
            
        except Exception as e:
            # Track error with analytics
            if BETA_SYSTEM_AVAILABLE and self.beta_analytics:
                self.beta_analytics.track_error("file_load_failed", str(e), filepath)
            
            messagebox.showerror("File Load Error", f"Failed to load file: {e}")
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_label.config(text="Failed to load file")
            self.progress_bar['value'] = 0
    
    def load_data(self, filepath: str) -> None:
        """Headless/batch entry point: load a file into current_data without UI dialogs."""
        validated_path = SafeFileHandler.validate_file_path(filepath)
        if not validated_path:
            raise ValueError(f"Invalid or inaccessible file path: {filepath}")
        filepath = str(validated_path)
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.las':
            self.current_data = self.load_las_file(filepath)
        elif ext in ('.dlis', '.lis'):
            self.current_data = self.load_dlis_file(filepath)
        elif ext == '.csv':
            self.current_data = self.load_csv_file(filepath)
        elif ext in ('.xlsx', '.xls'):
            self.current_data = self.load_excel_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        self.processed_data = self.current_data.copy() if self.current_data is not None else None
        # Headless: adopt file NULL without UI prompts
        self._reconcile_null_value_convention(allow_prompt=False)
        if hasattr(self, 'analyze_curves'):
            try:
                self.analyze_curves()
            except Exception as e:
                self.log_processing(f"analyze_curves after load_data: {e}")


    
    
            # This is not a critical error - just means we don't have automatic formation detection

    
    
    def _format_null_value_for_ui(self, raw: Any) -> Optional[str]:
        """Normalize a LAS/well NULL declaration to a Combobox-friendly string.

        Returns None when the well did not declare a usable NULL (CSV/Excel stubs,
        missing header, UNKNOWN placeholders).
        """
        if raw is None:
            return None
        text = str(raw).strip()
        if not text or text.upper() in ('UNKNOWN', 'N/A', 'NONE', 'NULL'):
            # Missing / placeholder declarations are not usable sentinels.
            return None
        if text.upper() in ('NAN', 'NA'):
            return 'NaN'
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return 'NaN'
        # Prefer the Combobox's canonical labels for common LAS sentinels.
        known = (
            (-999.25, '-999.25'),
            (-999.0, '-999'),
            (-9999.0, '-9999'),
            (99999.0, '99999'),
            (-99999.0, '-99999'),
        )
        for target, label in known:
            if abs(value - target) < 1e-9:
                return label
        # Preserve other declared numerics without scientific noise.
        if float(value).is_integer():
            return str(int(value))
        return format(value, 'g')

    def _null_value_strings_agree(self, a: str, b: str) -> bool:
        """True when two UI null strings represent the same convention."""
        if a == b:
            return True
        if a == 'NaN' or b == 'NaN':
            return a == b
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False

    def _set_null_value_var(self, ui_value: str, *, source: str = '') -> None:
        """Apply a null convention to the session variable and Combobox options."""
        if not hasattr(self, 'null_value_var'):
            return
        if hasattr(self, 'null_value_combo') and self.null_value_combo is not None:
            try:
                options = list(self.null_value_combo.cget('values') or ())
                if ui_value not in options:
                    options.append(ui_value)
                    self.null_value_combo.configure(values=options)
            except (tk.TclError, AttributeError):
                pass
        try:
            self.null_value_var.set(ui_value)
        except (tk.TclError, AttributeError):
            return
        if source:
            self.log_processing(f"Using declared NULL value {ui_value} ({source})")
        else:
            self.log_processing(f"Using declared NULL value {ui_value}")

    def _collect_declared_nulls_by_well(self) -> Dict[str, str]:
        """Map well_id -> formatted NULL for wells that declare one."""
        declared: Dict[str, str] = {}
        datasets = getattr(self, 'well_datasets', None) or {}
        for well_id, dataset in datasets.items():
            well_info = (dataset or {}).get('well_info') or {}
            formatted = self._format_null_value_for_ui(well_info.get('null_value'))
            if formatted is not None:
                declared[well_id] = formatted
        # Fall back to current well_info when datasets are empty (headless load_data).
        if not declared and getattr(self, 'well_info', None):
            formatted = self._format_null_value_for_ui(self.well_info.get('null_value'))
            if formatted is not None:
                label = str(self.well_info.get('well_name') or self.well_info.get('uwi') or 'active')
                declared[label] = formatted
        return declared

    def _prompt_null_value_conflict(self, declared_by_well: Dict[str, str]) -> Optional[str]:
        """Ask the user to pick a NULL when loaded wells disagree. Returns UI string or None."""
        # Group wells by equivalent null convention.
        groups: List[Tuple[str, List[str]]] = []
        for well_id, ui_null in declared_by_well.items():
            placed = False
            for canonical, wells in groups:
                if self._null_value_strings_agree(canonical, ui_null):
                    wells.append(well_id)
                    placed = True
                    break
            if not placed:
                groups.append((ui_null, [well_id]))

        if len(groups) <= 1:
            return groups[0][0] if groups else None

        dialog = tk.Toplevel(self.root)
        dialog.title("Null Value Conflict")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("520x360")

        ttk.Label(
            dialog,
            text=(
                "Loaded wells declare different NULL values.\n"
                "Choose which convention to use for processing and export."
            ),
            wraplength=480,
            justify='left',
        ).pack(anchor='w', padx=12, pady=(12, 8))

        choice = tk.StringVar(value=groups[0][0])
        for ui_null, wells in groups:
            well_list = ', '.join(wells[:6])
            if len(wells) > 6:
                well_list += f', … (+{len(wells) - 6} more)'
            ttk.Radiobutton(
                dialog,
                text=f"{ui_null}  —  {well_list}",
                variable=choice,
                value=ui_null,
            ).pack(anchor='w', padx=20, pady=3)

        result: Dict[str, Optional[str]] = {'value': None}

        def on_ok() -> None:
            result['value'] = choice.get()
            dialog.destroy()

        def on_cancel() -> None:
            result['value'] = None
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=12, padx=12)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left')
        ttk.Button(button_frame, text="Use Selected NULL", command=on_ok).pack(side='right')

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        self.root.wait_window(dialog)
        return result['value']

    def _reconcile_null_value_convention(self, *, allow_prompt: bool = True) -> None:
        """Set null_value_var from declared LAS NULL values; prompt only on conflict.

        - Single file / all wells agree → apply silently.
        - Multiple wells with disagreeing NULL → prompt (unless allow_prompt=False).
        - No declared NULL → leave the current session value unchanged.
        """
        declared = self._collect_declared_nulls_by_well()
        if not declared:
            self.log_processing(
                "No LAS-declared NULL value found; keeping session null "
                f"{self.null_value_var.get() if hasattr(self, 'null_value_var') else '-999.25'}"
            )
            return

        unique: List[str] = []
        for ui_null in declared.values():
            if not any(self._null_value_strings_agree(ui_null, existing) for existing in unique):
                unique.append(ui_null)

        if len(unique) == 1:
            wells = ', '.join(list(declared.keys())[:4])
            self._set_null_value_var(unique[0], source=f"from {wells}")
            return

        self.log_processing(
            "NULL conflict across loaded wells: "
            + '; '.join(f"{wid}={val}" for wid, val in declared.items())
        )
        if allow_prompt:
            chosen = self._prompt_null_value_conflict(declared)
            if chosen is not None:
                self._set_null_value_var(chosen, source='user selection after conflict')
                return
            self.log_processing(
                "NULL conflict dialog cancelled; keeping session null "
                f"{self.null_value_var.get()}"
            )
            return

        # Headless / non-interactive: prefer the active well, else first seen.
        active = getattr(self, 'active_well_id', None)
        if active and active in declared:
            self._set_null_value_var(declared[active], source=f'active well {active} (no prompt)')
        else:
            first_well, first_val = next(iter(declared.items()))
            self._set_null_value_var(first_val, source=f'{first_well} (no prompt)')

    def _get_null_value(self) -> float:
        """Get the configured null value with proper error handling.
        
        Returns:
            float: The null value to use for data processing and visualization.
                  Defaults to -999.25 if not configured.
        """
        try:
            if hasattr(self, 'null_value_var') and self.null_value_var.get():
                return float(self.null_value_var.get())
        except (ValueError, AttributeError) as e:
            self.log_processing(f"Warning: Could not parse null value, using default: {e}")
        return -999.25
    
    def _convert_nulls_to_nan(self, data: np.ndarray, null_value: Optional[float] = None, tolerance: float = 0.01) -> np.ndarray:
        """Convert null values to NaN in a copy of the data for visualization.
        
        This method creates a copy of the input data and replaces null values
        (within tolerance) with NaN for proper matplotlib visualization (line breaking).
        The original data remains unchanged.
        
        Args:
            data: Input data array (will be copied before modification)
            null_value: Null value to detect. If None, uses configured null_value_var.
            tolerance: Tolerance for null value detection (default: 0.01)
        
        Returns:
            np.ndarray: Copy of data with null values converted to NaN
        """
        if null_value is None:
            null_value = self._get_null_value()
        
        # Create a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Convert null values to NaN using tolerance-based detection
        null_mask = np.abs(data_copy - null_value) < tolerance
        data_copy[null_mask] = np.nan
        
        return data_copy
    
    def _update_window_title_with_well_info(self):
        """Update main window title to include well name for safety"""
        try:
            if hasattr(self, 'well_info') and self.well_info:
                well_name = self.well_info.get('well_name', 'UNKNOWN')
                if well_name and well_name != 'UNKNOWN':
                    self.root.title(f"Advanced Wireline Data Preprocessing System - Well: {well_name}")
                else:
                    self.root.title("Advanced Wireline Data Preprocessing System")
            else:
                self.root.title("Advanced Wireline Data Preprocessing System")
        except Exception as e:
            self.log_processing(f"Warning: Could not update window title: {e}")
    
    def _update_well_info_display(self):
        """Update well information display in Data Tab
        
        SAFETY CRITICAL: Updates the prominent well identification card
        to ensure users always know which well they're working with.
        """
        try:
            if not hasattr(self, 'well_info') or not self.well_info:
                # No well info available - show "not loaded" state
                if hasattr(self, 'well_name_label'):
                    self.well_name_label.config(text="Well: Not loaded", foreground='#CC0000')
                if hasattr(self, 'field_label'):
                    self.field_label.config(text="Field: Not loaded")
                if hasattr(self, 'uwi_label'):
                    self.uwi_label.config(text="UWI: Not loaded")
                if hasattr(self, 'company_label'):
                    self.company_label.config(text="Company: Not loaded")
                if hasattr(self, 'depth_range_label'):
                    self.depth_range_label.config(text="Depth Range: Not loaded")
                return
            
            # Extract well information
            well_name = self.well_info.get('well_name', 'UNKNOWN')
            field = self.well_info.get('field', 'UNKNOWN')
            uwi = self.well_info.get('uwi', 'UNKNOWN')
            company = self.well_info.get('company', 'UNKNOWN')
            start_depth = self.well_info.get('start_depth', 'UNKNOWN')
            stop_depth = self.well_info.get('stop_depth', 'UNKNOWN')
            
            # Color code well name based on whether it's known
            if well_name and well_name != 'UNKNOWN':
                well_color = '#006400'  # Dark green for loaded
                well_text = f"Well: {well_name}"
            else:
                well_color = '#CC6600'  # Orange for unknown
                well_text = "Well: UNKNOWN - Please verify well identification"
            
            # Update UI labels if they exist
            if hasattr(self, 'well_name_label'):
                self.well_name_label.config(text=well_text, foreground=well_color)
            
            if hasattr(self, 'field_label'):
                field_text = f"Field: {field}" if field and field != 'UNKNOWN' else "Field: Not specified"
                self.field_label.config(text=field_text)
            
            if hasattr(self, 'uwi_label'):
                uwi_text = f"UWI: {uwi}" if uwi and uwi != 'UNKNOWN' else "UWI: Not specified"
                self.uwi_label.config(text=uwi_text)
            
            if hasattr(self, 'company_label'):
                company_text = f"Company: {company}" if company and company != 'UNKNOWN' else "Company: Not specified"
                self.company_label.config(text=company_text)
            
            if hasattr(self, 'depth_range_label'):
                if start_depth != 'UNKNOWN' and stop_depth != 'UNKNOWN':
                    depth_text = f"Depth Range: {start_depth} to {stop_depth}"
                else:
                    depth_text = "Depth Range: Not available"
                self.depth_range_label.config(text=depth_text)
            
            self.log_processing("Well information display updated in UI")
            
        except Exception as e:
            self.log_processing(f"Error updating well info display: {e}")
            # Don't fail silently - log the error but continue
            import traceback
            self.log_processing(f"Traceback: {traceback.format_exc()}")
    

    
    
    



    
    
    
    def analyze_curves(self):
        """Analyze loaded curves with mnemonic identification"""
        if self.current_data is None:
            return
        
        for column in self.current_data.columns:
            # Initialize curve info if not exists
            if column not in self.curve_info:
                self.curve_info[column] = {
                    'unit': '',
                    'description': '',
                    'curve_type': 'UNKNOWN',
                    'type_confidence': 0.0,
                    'curve_data': {},
                    'statistics': {}
                }
            
            # Get curve info
            unit = self.curve_info[column].get('unit', '')
            description = self.curve_info[column].get('description', '')
            
            # Identify curve type
            curve_type, confidence, curve_data = self.curve_identifier.identify_curve(
                column, unit, description
            )
            
            # CRITICAL: Record standardization operation for audit trail
            if hasattr(self, 'standardization_reporter'):
                self.standardization_reporter.record_curve_identification(
                    original_name=column,
                    curve_type=curve_type,
                    confidence=confidence,
                    method='analyze_curves',
                    unit=unit,
                    description=description
                )
            
            # Calculate statistics
            data = self.current_data[column].dropna()
            stats = {
                'count': len(data),
                'missing': self.current_data[column].isna().sum(),
                'missing_percent': (self.current_data[column].isna().sum() / len(self.current_data)) * 100,
                'min': data.min() if len(data) > 0 else np.nan,
                'max': data.max() if len(data) > 0 else np.nan,
                'mean': data.mean() if len(data) > 0 else np.nan,
                'std': data.std() if len(data) > 0 else np.nan
            }
            
            # Update curve info
            self.curve_info[column].update({
                'curve_type': curve_type,
                'type_confidence': confidence,
                'curve_data': curve_data,
                'statistics': stats
            })
            
            # Store basic curve info only
            pass
        
        # === DUPLICATE DETECTION AND RESOLUTION ===
        try:
            # Detect duplicates
            duplicate_info = self.curve_identifier.detect_and_resolve_duplicates(self.curve_info)
            
            if duplicate_info['duplicates_found']:
                # Log what was found
                self.log_processing(f"[DUPLICATE DETECTION] Found {len(duplicate_info['duplicates_found'])} duplicate curve types")
                
                # Apply auto-resolved duplicates
                curves_to_remove = []
                for curve_type, selected_name in duplicate_info['auto_resolved'].items():
                    candidates = duplicate_info['duplicates_found'][curve_type]
                    for candidate in candidates:
                        if candidate['name'] != selected_name:
                            curves_to_remove.append(candidate['name'])
                            self.log_processing(f"   Auto-removed: {candidate['name']} (kept {selected_name})")
                
                # Get user input for remaining duplicates
                if duplicate_info['resolution_needed']:
                    user_selections = self.show_duplicate_resolution_dialog(duplicate_info)
                    
                    if user_selections:
                        for curve_type, selected_name in user_selections.items():
                            candidates = duplicate_info['duplicates_found'][curve_type]
                            for candidate in candidates:
                                if candidate['name'] != selected_name:
                                    curves_to_remove.append(candidate['name'])
                                    self.log_processing(f"   User removed: {candidate['name']} (kept {selected_name})")
                
                # Remove duplicate curves from dataset
                if curves_to_remove:
                    for curve_name in curves_to_remove:
                        if curve_name in self.current_data.columns:
                            self.current_data.drop(columns=[curve_name], inplace=True)
                        if curve_name in self.curve_info:
                            del self.curve_info[curve_name]
                    
                    self.log_processing(f"[DUPLICATE RESOLUTION] Removed {len(curves_to_remove)} duplicate curves")
                    
                    # Record in standardization reporter
                    if hasattr(self, 'standardization_reporter') and self.standardization_reporter:
                        for curve_name in curves_to_remove:
                            self.standardization_reporter.record_operation(
                                operation_type='duplicate_removal',
                                curve_name=curve_name,
                                details=f"Removed as duplicate"
                            )
        
        except Exception as e:
            self.log_processing(f"[DUPLICATE DETECTION] Error: {str(e)}")
    
    def show_duplicate_resolution_dialog(self, duplicate_info: Dict) -> Dict[str, str]:
        """
        Show dialog for user to select which curves to keep from duplicates
        
        Args:
            duplicate_info: Result from detect_and_resolve_duplicates()
        
        Returns:
            Dict[curve_type, selected_curve_name] - user selections
        """
        if not duplicate_info['resolution_needed']:
            return {}
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Resolve Duplicate Curves")
        dialog.geometry("900x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header = ttk.Label(
            dialog,
            text="Duplicate Curves Detected\n\nMultiple curves identified as same type. Please select which to keep:",
            font=('TkDefaultFont', 10, 'bold'),
            justify='left'
        )
        header.pack(pady=10, padx=10, anchor='w')
        
        # Scrollable frame for duplicates
        canvas_frame = ttk.Frame(dialog)
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store user selections
        selections = {}
        
        # Create selection UI for each duplicate group
        for idx, curve_type in enumerate(duplicate_info['resolution_needed']):
            candidates = duplicate_info['duplicates_found'][curve_type]
            
            # Group frame
            group_frame = ttk.LabelFrame(
                scrollable_frame,
                text=f"Curve Type: {curve_type}",
                padding=10
            )
            group_frame.pack(fill='x', pady=5, padx=5)
            
            # Radio button variable
            var = tk.StringVar(value=candidates[0]['name'])
            selections[curve_type] = var
            
            # Create radio button for each candidate
            for candidate in candidates:
                quality_text = f"{candidate['name']}"
                quality_text += f" | Confidence: {candidate['confidence']:.2f}"
                quality_text += f" | Missing: {candidate['missing_pct']:.1f}%"
                quality_text += f" | Unit: {candidate['unit']}"
                
                radio = ttk.Radiobutton(
                    group_frame,
                    text=quality_text,
                    value=candidate['name'],
                    variable=var
                )
                radio.pack(anchor='w', pady=2)
            
            # Add visual separator
            if idx < len(duplicate_info['resolution_needed']) - 1:
                ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        
        result = {}
        
        def on_confirm():
            for curve_type, var in selections.items():
                result[curve_type] = var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Confirm Selections", command=on_confirm).pack(side='left', padx=10)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left')
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result
    
    def ensure_curve_statistics(self):
        """Ensure all curves have statistics calculated - prevents KeyError issues"""
        if self.current_data is None:
            return
        
        for column in self.current_data.columns:
            # Check if curve_info exists and has statistics
            if column not in self.curve_info:
                # Create complete curve_info entry
                self.curve_info[column] = {
                    'unit': '',
                    'description': '',
                    'curve_type': 'UNKNOWN',
                    'type_confidence': 0.0,
                    'curve_data': {},
                    'statistics': {}
                }
            
            # Check if statistics are missing or incomplete
            if 'statistics' not in self.curve_info[column] or not self.curve_info[column]['statistics']:
                # Calculate statistics
                data = self.current_data[column].dropna()
                stats = {
                    'count': len(data),
                    'missing': self.current_data[column].isna().sum(),
                    'missing_percent': (self.current_data[column].isna().sum() / len(self.current_data)) * 100,
                    'min': data.min() if len(data) > 0 else np.nan,
                    'max': data.max() if len(data) > 0 else np.nan,
                    'mean': data.mean() if len(data) > 0 else np.nan,
                    'std': data.std() if len(data) > 0 else np.nan
                }
                
                # Update statistics
                self.curve_info[column]['statistics'] = stats
                
                # Also ensure other required fields exist, but preserve existing curve identification
                if 'curve_type' not in self.curve_info[column]:
                    self.curve_info[column]['curve_type'] = 'UNKNOWN'
                if 'type_confidence' not in self.curve_info[column]:
                    self.curve_info[column]['type_confidence'] = 0.0
                if 'unit' not in self.curve_info[column]:
                    self.curve_info[column]['unit'] = ''
                if 'description' not in self.curve_info[column]:
                    self.curve_info[column]['description'] = ''
                if 'curve_data' not in self.curve_info[column]:
                    self.curve_info[column]['curve_data'] = {}
                
                # Lightweight handling - no heavy re-identification calls
                pass
    
    # Removed heavy methods - back to simple working system
    pass
    
    def update_data_display(self):
        """Update the data display tree"""
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.current_data is None:
            return
        
        # Add curve information
        for column in self.current_data.columns:
            info = self.curve_info[column]
            stats = info.get('statistics', {})
            
            # Determine quality based on missing percentage using scientific thresholds
            # NEW: Distinguish between data errors and geological gaps
            missing_pct = stats.get('missing_percent', 0.0)
            
            # Analyze gap patterns to exclude geological gaps from quality assessment
            # Use depth-aware threshold to maintain consistent physical distance
            depth_params = self.get_depth_aware_parameters()
            geological_threshold = depth_params['geological_gap_threshold']
            gap_sizes = self._count_consecutive_missing(self.current_data[column])
            
            # Separate geological gaps from data error gaps
            geological_gaps = [g for g in gap_sizes if g >= geological_threshold]
            data_error_gaps = [g for g in gap_sizes if g < geological_threshold]
            
            # Calculate adjusted missing percentage (excluding geological gaps)
            total_points = len(self.current_data[column])
            data_error_missing = sum(data_error_gaps)
            adjusted_missing_pct = (data_error_missing / total_points * 100) if total_points > 0 else 0
            
            # Determine quality based on adjusted percentage (excludes geological gaps)
            if adjusted_missing_pct < PetrophysicalConstants.DATA_QUALITY["EXCELLENT"]:
                quality = "Excellent"
            elif adjusted_missing_pct < PetrophysicalConstants.DATA_QUALITY["GOOD"]:
                quality = "Good"
            elif adjusted_missing_pct < PetrophysicalConstants.DATA_QUALITY["FAIR"]:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Add note if geological gaps were excluded
            if geological_gaps:
                quality += f" ({len(geological_gaps)} geo)"
            
            # Format range
            if not np.isnan(stats['min']) and not np.isnan(stats['max']):
                range_str = f"{stats['min']:.2f} - {stats['max']:.2f}"
            else:
                range_str = "N/A"
            
            self.data_tree.insert('', 'end', values=(
                column,
                info['curve_type'],
                info['unit'],
                range_str,
                quality,
                f"{missing_pct:.1f}%"
            ))
        
        # Add helpful message about unprocessed curve visualization
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Data loaded successfully. Use 'Multi-Curve' plot to visualize all curves (processed and unprocessed).")
    
    def start_processing(self):
        """Start the processing pipeline with enterprise memory management and analytics"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        # Prevent concurrent processing runs
        if getattr(self, '_processing_active', False):
            messagebox.showinfo("Processing", "Processing is already running.")
            return
        self._processing_active = True
        
        # Track processing start with analytics
        if BETA_SYSTEM_AVAILABLE and self.beta_analytics:
            curve_count = len(self.current_data.columns)
            self.beta_analytics.track_event('processing_started', {
                'curve_count': curve_count,
                'data_points': len(self.current_data)
            })
        
        # Force garbage collection before processing large datasets
        gc.collect()
        
        # Set lower memory consumption options for matplotlib
        plt.rcParams['figure.dpi'] = 100  # Lower DPI for visualization
        plt.rcParams['figure.max_open_warning'] = 10
        
        # Run processing in separate thread with proper exception handling
        processing_thread = threading.Thread(target=self.process_data_thread)
        processing_thread.daemon = True
        processing_thread.start()
    def _initialize_processing_pipeline(self) -> None:
        """Initialize processing pipeline with data setup and validation."""
        # Schedule UI updates on main thread
        self.root.after(0, lambda: self.progress_bar.configure(value=0))
        self.root.after(0, lambda: self.status_label.config(text="Initializing processing..."))
        
        # Initialize processed data
        self.processed_data = self.current_data.copy()
        self.processing_results = {}
        
        # Save initial state for undo/redo
        self.processing_history.save_state(
            self.processed_data, 
            self.curve_info, 
            "Initial Data Load"
        )
        
        # Debug: Log available columns and curve_info
        self.log_processing(f"Available columns in data: {list(self.processed_data.columns)}")
        self.log_processing(f"Available curve_info keys: {list(self.curve_info.keys())}")
        
        # Check for missing curve_info entries
        missing_curve_info = [col for col in self.processed_data.columns if col not in self.curve_info]
        if missing_curve_info:
            self.log_processing(f"WARNING: Missing curve_info for columns: {missing_curve_info}")
            # Create default curve_info for missing columns
            for col in missing_curve_info:
                self.curve_info[col] = {
                    'curve_type': 'UNKNOWN',
                    'unit': '',
                    'description': f'Unknown curve: {col}',
                    'quality': 0.5
                }
                self.log_processing(f"Created default curve_info for: {col}")
        
        # Memory monitoring: Before processing starts
        self.monitor_and_cleanup_memory("Processing Start")
    
    def normalize_processed_data(self) -> None:
        """Normalize numeric log curves (excludes depth and discrete flag columns)."""
        if self.processed_data is None or self.processed_data.empty:
            return

        depth_cols = {'DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS', 'DEPTH_PRIMARY'}
        method = 'zscore'
        if hasattr(self, 'normalize_method_var'):
            method = self.normalize_method_var.get() or 'zscore'

        normalized_count = 0
        for col in self.processed_data.columns:
            if str(col).upper() in depth_cols:
                continue
            series = pd.to_numeric(self.processed_data[col], errors='coerce')
            arr = series.to_numpy(dtype=float)
            if np.sum(~np.isnan(arr)) < 2:
                continue
            curve_type = str(self.curve_info.get(col, {}).get('curve_type', '')).upper()
            if curve_type in ('FACIES', 'LITH', 'FLAG', 'DISCRETE'):
                continue

            if method == 'zscore':
                mean_val = np.nanmean(arr)
                std_val = np.nanstd(arr)
                if std_val <= 0:
                    continue
                out = (arr - mean_val) / std_val
            else:
                vmin = np.nanmin(arr)
                vmax = np.nanmax(arr)
                if vmax <= vmin:
                    continue
                out = (arr - vmin) / (vmax - vmin)

            self.processed_data[col] = pd.Series(out, index=self.processed_data.index)
            normalized_count += 1

        self.log_processing(
            f"Normalization ({method}) applied to {normalized_count} curve(s); depth columns excluded"
        )
    
    def _validate_and_standardize_depth(self) -> None:
        """Validate and standardize depth reference for processing."""
        self.root.after(0, lambda: self.status_label.config(text="Validating depth reference..."))
        self.log_processing("Starting enhanced depth validation...")
        
        try:
            # Validate and identify depth curve
            depth_curve = self.depth_validator.validate_and_identify_depth(
                self.processed_data.columns, 
                self.curve_info, 
                self.processed_data
            )
            self.log_processing(f"Validated depth curve: {depth_curve}")
            
            # Standardize depth reference for reservoir work
            selected_depth, depth_metadata = self.reservoir_depth_manager.standardize_depth_reference(
                self.processed_data, 
                self.curve_info
            )
            self.log_processing(f"Standardized depth reference: {selected_depth}")
            self.log_processing(f"Depth metadata: {depth_metadata}")

            # After depth reference is known, sync default resampling spacing
            try:
                self._sync_depth_spacing_default()
            except Exception:
                pass
            
        except Exception as e:
            error_category = self.categorize_error(e, "depth_validation")
            error_msg = f"[{error_category}] Depth validation failed: {e}"
            self.log_processing(f"ERROR: {error_msg}")
            
            # Provide category-specific user feedback
            if error_category == "MEMORY_ERROR":
                self.show_error_dialog(ERROR_TITLE_MEMORY, 
                    "Insufficient memory for depth validation. Try processing smaller datasets.")
            elif error_category == "DATA_ERROR":
                self.show_error_dialog(ERROR_TITLE_DATA, 
                    "Invalid depth data format detected. Check your input files.")
            elif error_category == "FILE_ERROR":
                self.show_error_dialog(ERROR_TITLE_FILE, 
                    "Unable to access depth data file. Check file permissions and path.")
            else:
                self.show_error_dialog(ERROR_TITLE_PROCESSING, error_msg)
            
            self.root.after(0, lambda: self.status_label.config(text="Depth validation failed - continuing with defaults"))
            self.log_processing("Continuing with existing depth reference...")
    
    def _detect_geological_zones(self) -> List[Any]:
        """Detect geological zones from gamma ray data.
        
        Returns:
            List of zone masks for zone-aware processing
        """
        zones = []  # Initialize zones as empty list
        self._gamma_ray_curves = [
            col for col in self.processed_data.columns
            if 'GR' in col.upper() or 'GAMMA' in col.upper()
        ]
        gamma_ray_curves = self._gamma_ray_curves
        if gamma_ray_curves and 'DEPT' in self.processed_data.columns:
            self.root.after(0, lambda: self.status_label.config(text="Detecting geological boundaries..."))
            self.log_processing("Starting geological boundary detection...")
            
            try:
                depth_data = self.processed_data['DEPT'].values
                gamma_ray_data = self.processed_data[gamma_ray_curves[0]].values
                
                boundary_depths = self.geological_zone_manager.detect_geological_boundaries(
                    depth_data, 
                    gamma_ray_data
                )
                self.log_processing(f"Detected {len(boundary_depths)} geological boundaries")
                
                # Create zone masks for zone-aware processing
                zones = self.geological_zone_manager.create_zone_masks(depth_data, boundary_depths)
                self.log_processing(f"Created {len(zones)} processing zones")
                
            except Exception as e:
                error_category = self.categorize_error(e, "geological_detection")
                error_msg = f"[{error_category}] Geological boundary detection failed: {e}"
                self.log_processing(f"ERROR: {error_msg}")
                
                # Provide category-specific user feedback
                if error_category == "DATA_ERROR":
                    self.show_error_dialog(ERROR_TITLE_DATA, 
                        "Unable to detect geological boundaries. Check gamma ray data quality.")
                elif error_category == "MEMORY_ERROR":
                    self.show_error_dialog(ERROR_TITLE_MEMORY, 
                        "Insufficient memory for geological analysis. Try processing smaller datasets.")
                else:
                    self.show_error_dialog(ERROR_TITLE_PROCESSING, error_msg)
                
                self.root.after(0, lambda: self.status_label.config(text="Geological detection failed - continuing without zones"))
                self.log_processing("Continuing without geological zones...")
                zones = []
        else:
            zones = []
            self._gamma_ray_curves = []
            self.log_processing("No gamma ray data available for geological boundary detection")
        
        return zones
    
    def _apply_environmental_corrections(self) -> None:
        """Apply environmental corrections to log data."""
        self.root.after(0, lambda: self.status_label.config(text="Applying environmental corrections..."))
        self.log_processing("Starting environmental corrections...")
        
        try:
            # Default well parameters (can be enhanced with actual well data)
            well_parameters = {
                'HOLE_SIZE': 8.5,  # inches
                'MUD_RESISTIVITY': 1.0,  # ohm-m
                'BHT': 150  # °F
            }
            
            corrected_data, corrections_applied = self.environmental_corrections.apply_environmental_corrections(
                self.processed_data, 
                self.curve_info, 
                well_parameters
            )
            self.processed_data = corrected_data
            self.log_processing(f"Applied environmental corrections: {corrections_applied}")
            
        except Exception as e:
            error_category = self.categorize_error(e, "environmental_corrections")
            error_msg = f"[{error_category}] Environmental corrections failed: {e}"
            self.log_processing(f"ERROR: {error_msg}")
            
            # Provide category-specific user feedback
            if error_category == "DATA_ERROR":
                self.show_error_dialog(ERROR_TITLE_DATA, 
                    "Unable to apply environmental corrections. Check well parameters and data quality.")
            elif error_category == "MEMORY_ERROR":
                self.show_error_dialog(ERROR_TITLE_MEMORY, 
                    "Insufficient memory for environmental corrections. Try processing smaller datasets.")
            else:
                self.show_error_dialog(ERROR_TITLE_PROCESSING, error_msg)
            
            self.root.after(0, lambda: self.status_label.config(text="Environmental corrections failed - continuing without corrections"))
            self.log_processing("Continuing without environmental corrections...")
    
    def _uniformize_data(self) -> None:
        """Uniformize curve names and units, resample to standard spacing."""
        if self.rename_curves_var.get() or self.standardize_units_var.get():
            self.root.after(0, lambda: self.status_label.config(text="Uniformizing data..."))
            self.log_processing("Starting data uniformization...")
            
            # Standardize curve names and units
            self.uniformize_curves()
            
            # Resample to standard depth spacing if needed
            if 'DEPT' in self.processed_data.columns:
                depth_spacing = self.depth_spacing_var.get()
                self.log_processing(f"Resampling to standard depth spacing: {depth_spacing} m")
                self.resample_to_standard_spacing('DEPT', depth_spacing)
    
    def process_data_thread(self):
        """Process data in separate thread with standardization"""
        try:
            # Initialize processing pipeline
            self._initialize_processing_pipeline()
            
            # Step 1: Depth Validation and Standardization
            self._validate_and_standardize_depth()
            
            # Optional normalization step
            try:
                if hasattr(self, 'normalize_var') and self.normalize_var.get():
                    self.root.after(0, lambda: self.status_label.config(text="Applying normalization..."))
                    self.log_processing("Applying normalization to processed data...")
                    self.normalize_processed_data()
                    self.log_processing("Normalization complete.")
            except Exception as e:
                self.log_processing(f"WARNING: Normalization failed: {e}")
            
            # Step 2: Geological Zone Detection
            zones = self._detect_geological_zones()
            
            # Memory monitoring: After zone detection
            self.monitor_and_cleanup_memory("After Zone Detection")
            
            # Step 2b: Build cross-well priors early if enabled
            if self.use_crosswell_priors_var.get() and self.crosswell_prior_manager:
                try:
                    self.root.after(0, lambda: self.status_label.config(text="Building cross-well priors..."))
                    self.crosswell_priors = self.crosswell_prior_manager.build_priors(
                        depth_binned=self.priors_depth_binning_var.get()
                    )
                    self.log_processing(f"Cross-well priors ready for {len(self.crosswell_priors)} curves")
                except Exception as e:
                    self.log_processing(f"Cross-well priors build skipped: {e}")
            
            # Step 3: Environmental Corrections
            self._apply_environmental_corrections()
            
            # Step 4: Uniformization
            self._uniformize_data()
            
            total_curves = len(self.processed_data.columns)
            
            # Get depth-aware parameters (adjusts for depth spacing)
            depth_params = self.get_depth_aware_parameters()
            
            self.log_processing("=" * 50)
            self.log_processing("DEPTH-AWARE PARAMETER ADJUSTMENT")
            self.log_processing(f"Depth Spacing: {depth_params['depth_spacing']} m")
            self.log_processing(f"Scaling Ratio: {depth_params['spacing_ratio']:.2f}x")
            self.log_processing(f"Geological Gap Threshold: {depth_params['geological_gap_threshold']} pts ({depth_params['geological_gap_meters']:.1f} m)")
            self.log_processing(f"Large Gap Threshold: {depth_params['large_gap_threshold']} pts ({depth_params['large_gap_meters']:.1f} m)")
            self.log_processing(f"Max Gap Size: {depth_params['max_gap_size']} pts ({depth_params['max_gap_meters']:.1f} m)")
            self.log_processing("=" * 50)
            
            # Get UI parameters for gap filling
            if hasattr(self, 'large_gap_threshold_var'):
                self.large_gap_threshold = depth_params['large_gap_threshold']  # Use depth-aware value
            else:
                self.large_gap_threshold = 500
                
            if hasattr(self, 'large_gap_var'):
                self.large_gap_treatment = self.large_gap_var.get()
            else:
                self.large_gap_treatment = "formation_based"
            
            # Initialize gap_params once outside the loop with depth-aware values
            gap_params = GapFillingParameters(
                max_gap_size=depth_params['max_gap_size'],  # Use depth-aware max gap size
                physics_informed=self.physics_informed_var.get(),
                multi_curve_correlation=self.multi_curve_var.get(),
                geological_gap_threshold=depth_params['geological_gap_threshold']  # Use depth-aware geological threshold
            )
            
            # Prepare auxiliary curves once for all columns
            auxiliary_curves_dict = {}
            if self.multi_curve_var.get():
                for column in self.processed_data.columns:
                    auxiliary_curves_dict[column] = {}
                    for other_col in self.processed_data.columns:
                        if other_col != column:
                            auxiliary_curves_dict[column][other_col] = self.processed_data[other_col].values
            
            for i, column in enumerate(self.processed_data.columns):
                # Skip depth column for processing
                if column in ['DEPT', 'DEPTH', 'MD', 'TVD']:
                    continue
                
                # Memory monitoring: Every 5 curves
                if i % 5 == 0:
                    self.monitor_and_cleanup_memory(f"Processing Curve {i}/{total_curves}")
                    
                curve_progress = (i / total_curves) * 100
                # Schedule UI updates on main thread
                self.root.after(0, lambda progress=curve_progress: self.progress_bar.configure(value=progress))
                self.root.after(0, lambda col=column: self.status_label.config(text=f"Processing {col}..."))
                
                # Get curve data and info
                data = self.processed_data[column].values.copy()  # Create a copy to avoid modifying original
                
                # Industry-flexible viability gate
                try:
                    decision, validity_ratio = self.evaluate_curve_viability(column, data)
                    if decision == 'SKIP_INSUFFICIENT_DATA':
                        _skip_msg = f"Skipping {column}: insufficient valid data ({validity_ratio:.1%})"
                        self.log_processing(_skip_msg)
                        continue
                except Exception:
                    # On error, proceed with processing rather than skipping
                    pass

                # Check if curve_info exists for this column
                if column not in self.curve_info:
                    self.log_processing(f"WARNING: No curve info found for column '{column}', creating default info")
                    self.curve_info[column] = {
                        'curve_type': 'UNKNOWN',
                        'unit': '',
                        'description': f'Unknown curve: {column}',
                        'quality': 0.5
                    }
                
                curve_info = self.curve_info[column]
                curve_type = curve_info.get('curve_type', 'UNKNOWN')
                
                # ENHANCED DATA QUALITY: Apply range validation and outlier detection
                self.log_processing(f"Applying data quality validation for {column}...")
                
                # Step 1: Range validation based on mnemonic database
                if self.range_validation_var.get():
                    original_data_count = np.sum(~np.isnan(data))
                    data = self.apply_range_validation(column, data)
                    validated_data_count = np.sum(~np.isnan(data))
                    removed_count = original_data_count - validated_data_count
                    if removed_count > 0:
                        self.log_processing(f"Range validation: Removed {removed_count} out-of-range values from {column}")
                
                # Step 2: Outlier detection using IQR method
                if self.outlier_detection_var.get():
                    outlier_mask = self.detect_outliers_iqr(data)
                    outlier_count = np.sum(outlier_mask)
                    if outlier_count > 0:
                        self.log_processing(f"Outlier detection: {outlier_count} outliers identified in {column}")
                        
                        # Convert outliers to NaN for gap filling
                        data[outlier_mask] = np.nan
                        self.log_processing(f"Outlier removal: Converted {outlier_count} outliers to gaps in {column}")
                
                # Log data quality summary
                final_valid_count = np.sum(~np.isnan(data))
                total_count = len(data)
                quality_percentage = (final_valid_count / total_count) * 100 if total_count > 0 else 0
                _quality_msg = (
                    f"Data quality summary for {column}: "
                    f"{final_valid_count}/{total_count} valid points ({quality_percentage:.1f}%)"
                )
                self.log_processing(_quality_msg)
                
                # Update the processed data with validated data (convert back to pandas Series)
                self.processed_data[column] = pd.Series(data, index=self.processed_data.index)
                
                # ENHANCED PROCESSING: Scale-aware and Zone-aware Processing
                self.log_processing(f"Enhanced processing for {column}...")
                
                # Step 1: Determine curve scale and apply scale-aware processing
                scale_type = self.scale_aware_processor.determine_curve_scale(column, data)
                self.log_processing(f"Detected scale type for {column}: {scale_type}")
                
            # Step 2: Zone-aware gap filling (if zones detected)
                if zones and 'DEPT' in self.processed_data.columns:
                    self.log_processing(f"Zone-aware gap filling for {column}...")
                    
                    depth_data = self.processed_data['DEPT'].values
                    gamma_ray_data = (
                        self.processed_data[self._gamma_ray_curves[0]].values
                        if getattr(self, '_gamma_ray_curves', None) else None
                    )
                    
                    # Get auxiliary curves for zone-aware processing
                    auxiliary_curves = auxiliary_curves_dict.get(column, {}) if self.multi_curve_var.get() else {}
                    
                    try:
                        if hasattr(self, 'zone_aware_gap_filler') and self.zone_aware_gap_filler is not None:
                            zone_gap_result = self.zone_aware_gap_filler.fill_gaps_with_zone_awareness(
                                data, curve_type, depth_data, gamma_ray_data, auxiliary_curves
                            )
                            filled_data = zone_gap_result['filled_data']
                            self.log_processing(f"Zone-aware gap filling completed for {column}")
                        else:
                            warnings.warn("Zone-aware gap filler not available, skipping zone-aware processing", UserWarning)
                            filled_data = data  # Use original data if zone-aware processing not available
                        # Ensure gap_result is defined for downstream use
                        try:
                            valid_before = int(np.sum(~np.isnan(data)))
                            valid_after = int(np.sum(~np.isnan(filled_data)))
                            total_points_filled = max(0, valid_after - valid_before)
                            data_completeness = (valid_after / len(filled_data)) * 100 if len(filled_data) > 0 else 0.0
                            conf = zone_gap_result.get('confidence', None)
                            if isinstance(conf, np.ndarray):
                                avg_conf = float(np.nanmean(conf)) if conf.size > 0 else 0.8
                            elif isinstance(conf, (list, tuple)):
                                avg_conf = float(np.nanmean(np.array(conf, dtype=float))) if len(conf) > 0 else 0.8
                            elif isinstance(conf, (int, float)):
                                avg_conf = float(conf)
                            else:
                                avg_conf = 0.8
                            quality_metrics = zone_gap_result.get('quality_metrics', {
                                'total_gaps_filled': 0,
                                'total_points_filled': total_points_filled,
                                'data_completeness': data_completeness,
                                'methods_used': ['zone_aware'],
                                'average_confidence': avg_conf
                            })
                            gap_result = {
                                'filled_data': filled_data,
                                'quality_metrics': quality_metrics,
                                'gaps_filled': zone_gap_result.get('gaps_filled', [])
                            }
                        except Exception:
                            # Fallback minimal structure to avoid runtime errors
                            gap_result = {
                                'filled_data': filled_data,
                                'quality_metrics': {
                                    'total_gaps_filled': 0,
                                    'total_points_filled': 0,
                                    'data_completeness': 0.0,
                                    'methods_used': ['zone_aware'],
                                    'average_confidence': 0.8
                                },
                                'gaps_filled': []
                            }
                    except Exception as e:
                        self.log_processing(f"Zone-aware gap filling failed for {column}: {e}")
                        self.log_processing("Falling back to standard gap filling...")
                        
                        # Fallback to standard gap filling
                        auxiliary_curves = None
                        if self.multi_curve_var.get():
                            auxiliary_curves = {}
                            for other_col in self.processed_data.columns:
                                if other_col != column:
                                    auxiliary_curves[other_col] = self.processed_data[other_col].values

                        try:
                            gap_result = self.gap_filler.fill_gaps(
                                data, 
                                curve_type, 
                                auxiliary_curves,
                                curve_name=column
                            )
                            filled_data = gap_result['filled_data']
                        except Exception as e:
                            self.log_processing(f"Fallback gap filling failed for {column}: {e}")
                            filled_data = data.copy()  # Use original data as fallback
                            gap_result = {
                                'filled_data': filled_data,
                                'quality_metrics': {
                                    'total_gaps_filled': 0,
                                    'total_points_filled': 0, 
                                    'data_completeness': 100.0,
                                    'methods_used': [],
                                    'average_confidence': 1.0
                                }
                            }
                else:
                    # Standard gap filling when no zones detected
                    self.log_processing(f"Standard gap filling for {column}...")

                    auxiliary_curves = None
                    if self.multi_curve_var.get():
                        auxiliary_curves = {}
                        for other_col in self.processed_data.columns:
                            if other_col != column:
                                auxiliary_curves[other_col] = self.processed_data[other_col].values

                    try:
                        gap_result = self.gap_filler.fill_gaps(
                            data,
                            curve_type,
                            auxiliary_curves,
                            curve_name=column
                        )
                        filled_data = gap_result['filled_data']
                    except Exception as e:
                        self.log_processing(f"Gap filling failed for {column}: {e}")
                        filled_data = data.copy()  # Use original data as fallback
                        gap_result = {
                            'filled_data': filled_data,
                            'quality_metrics': {
                                'total_gaps_filled': 0,
                                'total_points_filled': 0,
                                'data_completeness': 100.0,
                                'methods_used': [],
                                'average_confidence': 1.0
                            }
                        }
                    # Duplicate standard gap-filling block removed to prevent double execution

                # Optional PASS 2: Cross-well prior-constrained refinement
                try:
                    if self.use_crosswell_priors_var.get() and self.crosswell_prior_manager and self.two_pass_refinement_var.get():
                        if not self.crosswell_priors:
                            # Build priors lazily if not present
                            self.crosswell_priors = self.crosswell_prior_manager.build_priors(
                                depth_binned=self.priors_depth_binning_var.get()
                            )
                        # Vector bounds per depth if available
                        depth_vals = self.processed_data['DEPT'].values if 'DEPT' in self.processed_data.columns else None
                        if depth_vals is not None and self.priors_depth_binning_var.get():
                            vec = self.crosswell_prior_manager.get_bounds_vector_for_curve(column, depth_vals)
                            if vec is not None:
                                lows, highs = vec
                                clipped = np.minimum(np.maximum(filled_data, lows), highs)
                                if np.any(~np.isclose(clipped, filled_data, equal_nan=True)):
                                    filled_data = clipped
                                    gap_result['quality_metrics']['methods_used'] = list(set(gap_result['quality_metrics'].get('methods_used', []) + ['crosswell_prior']))
                                    self.log_processing(f"Applied cross-well prior vector bounds to {column}")
                        else:
                            bounds = self.crosswell_prior_manager.get_bounds_for_curve(column, None)
                            if bounds is not None:
                                low, high = bounds
                                clipped = np.clip(filled_data, low, high)
                                if np.any(~np.isclose(clipped, filled_data, equal_nan=True)):
                                    filled_data = clipped
                                    gap_result['quality_metrics']['methods_used'] = list(set(gap_result['quality_metrics'].get('methods_used', []) + ['crosswell_prior']))
                                    self.log_processing(f"Applied cross-well prior bounds to {column}: [{low:.3g}, {high:.3g}]")
                except Exception as e:
                    self.log_processing(f"Cross-well prior refinement skipped for {column}: {e}")
                
                # Step 3: Scale-aware denoising
                self.log_processing(f"Scale-aware denoising for {column}...")
                
                try:
                    # Apply scale-aware denoising
                    denoised_data, denoise_info = self.scale_aware_processor.process_curve_scale_aware(
                        column, filled_data, 'denoise'
                    )
                    final_data = denoised_data
                    denoise_result = {
                        'denoised': final_data,
                        'method': 'scale_aware',
                        'quality': 0.8,
                        'info': denoise_info
                    }
                    self.log_processing(f"Scale-aware denoising completed for {column}: {denoise_info}")
                    
                except Exception as e:
                    self.log_processing(f"Scale-aware denoising failed for {column}: {e}")
                    self.log_processing("Falling back to standard denoising...")
                    
                    # Fallback to standard denoising
                    try:
                        denoise_result = self.signal_processor.denoise_signal(
                            filled_data, 
                            curve_type, 
                            self.denoise_method_var.get()
                        )
                        final_data = denoise_result['denoised']
                        
                    except Exception as e:
                        self.log_processing(f"Denoising failed for {column}: {e}")
                        final_data = filled_data.copy()  # Use gap-filled data as fallback
                        denoise_result = {
                            'denoised': final_data,
                            'method': 'none',
                            'quality': 0.5
                        }
                
                # Update processed data
                self.processed_data[column] = final_data
                
                # Store enhanced processing results
                self.processing_results[column] = {
                    'original_data': data,
                    'final_data': final_data,
                    'gap_filling': gap_result,
                    'denoising': denoise_result
                }
                
                self.log_processing(f"Completed processing for {column}")
                self.log_processing(f"  - Scale type: {scale_type}")
                self.log_processing(f"  - Processing method: {'Zone-aware' if zones else 'Standard'}")
                
                # Memory cleanup: After each curve (aggressive for large datasets)
                if total_curves > 20 and i % 3 == 0:  # Every 3 curves for large datasets
                    self.monitor_and_cleanup_memory(f"After Curve {i}/{total_curves}")
            
            # ENHANCED PROCESSING: Final Validation and Quality Assurance
            self.root.after(0, lambda: self.status_label.config(text="Validating petrophysical relationships..."))
            self.log_processing("Starting petrophysical relationship validation...")
            
            try:
                # Validate that processing preserved known petrophysical relationships
                validation_results, warnings = self.petrophysical_validator.validate_relationships(
                    self.processed_data, 
                    self.curve_info
                )
                
                if warnings:
                    for warning in warnings:
                        self.log_processing(f"VALIDATION WARNING: {warning}")
                else:
                    self.log_processing("All petrophysical relationships validated successfully")
                
                # Log validation summary
                valid_relationships = sum(1 for result in validation_results.values() if result['valid'])
                total_relationships = len(validation_results)
                self.log_processing(f"Petrophysical validation: {valid_relationships}/{total_relationships} relationships valid")
                
            except Exception as e:
                error_category = self.categorize_error(e, "petrophysical_validation")
                error_msg = f"[{error_category}] Petrophysical validation failed: {e}"
                self.log_processing(f"ERROR: {error_msg}")
                
                # Provide category-specific user feedback
                if error_category == "DATA_ERROR":
                    self.show_error_dialog(ERROR_TITLE_DATA, 
                        "Unable to validate petrophysical relationships. Check data quality and curve correlations.")
                elif error_category == "MEMORY_ERROR":
                    self.show_error_dialog(ERROR_TITLE_MEMORY, 
                        "Insufficient memory for petrophysical validation. Try processing smaller datasets.")
                else:
                    self.show_error_dialog(ERROR_TITLE_PROCESSING, error_msg)
                
                self.root.after(0, lambda: self.status_label.config(text="Petrophysical validation failed - continuing without validation"))
                self.log_processing("Continuing without relationship validation...")
            
            # Memory monitoring: After all curves processed
            self.monitor_and_cleanup_memory("After All Curves Processed")
            
            # Cleanup intermediate DataFrames and variables
            if hasattr(self, 'auxiliary_curves_dict'):
                try:
                    del auxiliary_curves_dict
                except Exception:
                    pass
            
            # Force DataFrame memory consolidation
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                try:
                    self.processed_data._consolidate_inplace()
                except Exception:
                    pass
            
            # Save final processing state for undo/redo
            self.processing_history.save_state(
                self.processed_data, 
                self.curve_info, 
                "Enhanced Processing Complete",
                {
                    'curves_processed': len(self.processed_data.columns),
                    'zones_detected': len(zones) if zones else 0,
                    'environmental_corrections_applied': True,
                    'scale_aware_processing': True,
                    'zone_aware_processing': bool(zones)
                }
            )
            
            # Step 3: Apply final uniformization
            self.log_processing("Applying final uniformization...")
            self.finalize_uniformization()
            
            # Clear any temporary objects to free memory
            auxiliary_curves_dict.clear()
            
            # Final UI updates
            self.root.after(0, lambda: self.progress_bar.configure(value=100))
            self.root.after(0, lambda: self.status_label.config(text="Processing completed successfully"))
            self.log_processing("=" * 50)
            self.log_processing("PROCESSING COMPLETED SUCCESSFULLY")
            self.log_processing("=" * 50)
            
            # Track processing completion with analytics
            if BETA_SYSTEM_AVAILABLE and self.beta_analytics:
                curve_count = len(self.processed_data.columns) if self.processed_data is not None else 0
                total_gaps_filled = self._calculate_total_gaps_filled()
                self.beta_analytics.track_processing_completed(
                    0, curve_count, total_gaps_filled  # Processing time not tracked in this version
                )
            
            # Automatically update processed LAS preview with enhanced threading
            self.root.after(0, self._schedule_visualization_update_safely)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_category = self.categorize_error(e, "main_processing")
            error_msg = f"[{error_category}] Processing failed: {e}"
            
            # Show category-specific error to user
            if error_category == "MEMORY_ERROR":
                self.show_error_dialog(ERROR_TITLE_MEMORY, 
                    "Insufficient memory for processing. Try processing smaller datasets or close other applications.")
            elif error_category == "DATA_ERROR":
                self.show_error_dialog(ERROR_TITLE_DATA, 
                    "Invalid data format detected. Check your input files and data quality.")
            elif error_category == "FILE_ERROR":
                self.show_error_dialog(ERROR_TITLE_FILE, 
                    "Unable to access data files. Check file permissions and paths.")
            elif error_category == "DEPENDENCY_ERROR":
                self.show_error_dialog("Dependency Error", 
                    "Required libraries not available. Check your Python environment setup.")
            else:
                self.show_error_dialog(ERROR_TITLE_PROCESSING, error_msg)
            
            # Track error with analytics
            if BETA_SYSTEM_AVAILABLE and self.beta_analytics:
                self.beta_analytics.track_error("processing_failed", str(e), "process_data_thread")
            
            # Update UI status
            self.root.after(0, lambda: self.status_label.config(text="Processing failed"))
            self.log_processing(f"ERROR: Processing failed - {str(e)}")
            self.log_processing(f"ERROR DETAILS: {error_details}")
        finally:
            # Allow new processing runs after completion or failure
            try:
                self._processing_active = False
            except Exception:
                pass
    def create_comprehensive_report(self) -> str:
        """Create detailed processing report with robust error handling"""
        report = []
        
        # Header
        report.append("╔" + "═" * 78 + "╗")
        report.append("║" + " " * 15 + "ADVANCED WIRELINE DATA PREPROCESSING REPORT" + " " * 20 + "║")
        report.append("╚" + "═" * 78 + "╝")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"File: {self.file_path_var.get()}")
        report.append("")
        
        # CRITICAL: Well Identification Box
        if hasattr(self, 'well_info') and self.well_info:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 26 + "WELL IDENTIFICATION" + " " * 33 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            well_name = self.well_info.get('well_name', 'UNKNOWN')
            uwi = self.well_info.get('uwi', 'UNKNOWN')
            field = self.well_info.get('field', 'UNKNOWN')
            company = self.well_info.get('company', 'UNKNOWN')
            date = self.well_info.get('date', 'UNKNOWN')
            start = self.well_info.get('start_depth', 'UNKNOWN')
            stop = self.well_info.get('stop_depth', 'UNKNOWN')
            unit = self.well_info.get('depth_unit', 'm')
            
            report.append(f"║  Well Name:    {well_name:<62} ║")
            report.append(f"║  UWI:          {uwi:<62} ║")
            report.append(f"║  Field:        {field:<62} ║")
            report.append(f"║  Company:      {company:<62} ║")
            report.append(f"║  Date:         {date:<62} ║")
            depth_range_text = f"{start} - {stop} {unit}"
            report.append(f"║  Depth Range:  {depth_range_text:<62} ║")
            report.append("╚" + "═" * 78 + "╝")
            report.append("")

        # === IMPROVEMENT 1: Trim LAS header to first 30 lines only ===
        try:
            from pathlib import Path

            # Original LAS header captured during file load (trimmed)
            if hasattr(self, 'original_las_header') and self.original_las_header:
                all_header_lines = self.original_las_header.split('\n')
                header_lines = all_header_lines[:30]  # Reduced from 100 to 30
                report.append("ORIGINAL LAS HEADER (First 30 lines):")
                report.append("-" * 80)
                report.extend(header_lines)
                if len(all_header_lines) > 30:
                    omitted_count = len(all_header_lines) - 30
                    report.append(f"... ({omitted_count} additional lines omitted)")
            else:
                report.append("ORIGINAL LAS HEADER: Not available")
            
            report.append("")
        except Exception:
            report.append("ORIGINAL LAS HEADER: Failed to include")
            report.append("")
        
        # Check if we have data to report on
        if self.current_data is None:
            report.append("NO DATA AVAILABLE")
            report.append("Please load and process data before generating a report.")
            return "\n".join(report)
        
        # === IMPROVEMENT 2: Add Processing Dashboard ===
        total_curves = len(self.current_data.columns)
        total_points = len(self.current_data)
        
        # Calculate overall statistics (handle case where no processing has been done)
        if self.processing_results:
            total_gaps_filled = sum(
                result.get('gap_filling', {}).get('quality_metrics', {}).get('total_points_filled', 0)
                for result in self.processing_results.values()
            )
            
            # Calculate average completeness with fallback to original statistics
            completeness_values = []
            for curve, result in self.processing_results.items():
                completeness = result.get('gap_filling', {}).get('quality_metrics', {}).get('data_completeness', 0)
                if completeness == 0:
                    # Fallback to original statistics if gap filling completeness not available
                    curve_stats = self.curve_info.get(curve, {}).get('statistics', {})
                    if curve_stats:
                        completeness = 100.0 - curve_stats.get('missing_percent', 0.0)
                completeness_values.append(completeness)
            
            avg_completeness = np.mean(completeness_values) if completeness_values else 0
            
            avg_denoise_quality = np.mean([
                result.get('denoising', {}).get('quality', 0) * 100
                for result in self.processing_results.values()
            ]) if self.processing_results else 0
        else:
            # No processing done - calculate original completeness from curve statistics
            total_gaps_filled = 0
            completeness_values = []
            for curve in self.current_data.columns:
                curve_stats = self.curve_info.get(curve, {}).get('statistics', {})
                if curve_stats:
                    completeness = 100.0 - curve_stats.get('missing_percent', 0.0)
                    completeness_values.append(completeness)
            
            avg_completeness = np.mean(completeness_values) if completeness_values else 0
            avg_denoise_quality = 0
        
        # Grade the overall processing quality
        if avg_completeness >= 95 and avg_denoise_quality >= 80:
            overall_grade = "EXCELLENT"
        elif avg_completeness >= 90 and avg_denoise_quality >= 70:
            overall_grade = "GOOD"
        elif avg_completeness >= 80 and avg_denoise_quality >= 60:
            overall_grade = "SATISFACTORY"
        else:
            overall_grade = "NEEDS IMPROVEMENT"
        
        # Processing Dashboard
        report.append("╔" + "═" * 78 + "╗")
        report.append("║" + " " * 25 + "PROCESSING DASHBOARD" + " " * 33 + "║")
        report.append("╠" + "═" * 78 + "╣")
        report.append(f"║  Curves Processed: {total_curves:>6}        Overall Grade: {overall_grade:<20} ║")
        report.append(f"║  Total Data Points: {total_points:>6,}      Avg Completeness: {avg_completeness:>5.1f}%{' ' * 15} ║")
        report.append(f"║  Gaps Filled: {total_gaps_filled:>6,}           Avg Denoise Quality: {avg_denoise_quality:>5.1f}%{' ' * 10} ║")
        report.append("╚" + "═" * 78 + "╝")
        report.append("")
        
        # === IMPROVEMENT 3: Add Depth-Aware Parameters Section ===
        depth_params = self.get_depth_aware_parameters()
        report.append("DEPTH-AWARE PARAMETER CONFIGURATION")
        report.append("=" * 80)
        report.append(f"Depth Spacing: {depth_params['depth_spacing']} m")
        report.append(f"Scaling Ratio: {depth_params['spacing_ratio']:.2f}x (relative to 0.5m reference)")
        report.append("")
        report.append("Adjusted Thresholds (Points | Physical Distance):")
        report.append(f"  Geological Gap Threshold:  {depth_params['geological_gap_threshold']:>4} pts | {depth_params['geological_gap_meters']:>6.1f} m")
        report.append(f"  Large Gap Threshold:       {depth_params['large_gap_threshold']:>4} pts | {depth_params['large_gap_meters']:>6.1f} m")
        report.append(f"  Max Gap Size:              {depth_params['max_gap_size']:>4} pts | {depth_params['max_gap_meters']:>6.1f} m")
        report.append("")
        report.append("Filter Windows (Adjusted for depth spacing):")
        report.append(f"  Savitzky-Golay: {depth_params['savgol_window']} pts")
        report.append(f"  Median Filter:  {depth_params['median_window']} pts")
        report.append(f"  Bilateral:      {depth_params['bilateral_window']} pts")
        report.append("")
        
        # === IMPROVEMENT 4 & 5: Add Gap Analysis Summary ===
        report.append("GAP ANALYSIS SUMMARY")
        report.append("=" * 80)
        
        # Collect gap statistics from all curves
        total_gaps = 0
        geological_gaps_count = 0
        data_error_gaps_count = 0
        gaps_filled_count = 0
        gaps_skipped_count = 0
        
        gap_details = []  # For per-curve table
        
        for curve in self.current_data.columns:
            curve_type = self.curve_info.get(curve, {}).get('curve_type', '')
            if 'DEPTH' in curve_type:
                continue  # Skip depth curves
            
            # Count gaps
            gap_sizes = self._count_consecutive_missing(self.current_data[curve])
            curve_total_gaps = len(gap_sizes)
            total_gaps += curve_total_gaps
            
            # Classify gaps
            geo_threshold = depth_params['geological_gap_threshold']
            curve_geo_gaps = sum(1 for g in gap_sizes if g >= geo_threshold)
            curve_data_gaps = sum(1 for g in gap_sizes if g < geo_threshold)
            
            geological_gaps_count += curve_geo_gaps
            data_error_gaps_count += curve_data_gaps
            
            # Determine if curve was processed
            if curve in self.processing_results:
                result = self.processing_results[curve]
                gaps_filled = result.get('gap_filling', {}).get('quality_metrics', {}).get('total_gaps_filled', 0)
            else:
                gaps_filled = 0
            
            gaps_filled_count += gaps_filled
            gaps_skipped = curve_total_gaps - gaps_filled
            gaps_skipped_count += gaps_skipped
            
            # Get quality
            stats = self.curve_info.get(curve, {}).get('statistics', {})
            missing_pct = stats.get('missing_percent', 0.0)
            if missing_pct < 5.0:
                quality = "Excellent"
            elif missing_pct < 15.0:
                quality = "Good"
            elif missing_pct < 30.0:
                quality = "Fair"
            else:
                quality = "Poor"
            
            if curve_geo_gaps > 0:
                quality += f" ({curve_geo_gaps} geo)"
            
            gap_details.append({
                'curve': curve,
                'total': curve_total_gaps,
                'data_err': curve_data_gaps,
                'geo': curve_geo_gaps,
                'filled': gaps_filled,
                'quality': quality
            })
        
        # Overall gap statistics
        report.append(f"Total Gaps Found: {total_gaps}")
        report.append("")
        report.append("Gap Classification:")
        if total_gaps > 0:
            data_pct = (data_error_gaps_count / total_gaps * 100) if total_gaps > 0 else 0
            geo_pct = (geological_gaps_count / total_gaps * 100) if total_gaps > 0 else 0
            report.append(f"  Data Errors (<{depth_params['geological_gap_threshold']} pts):        {data_error_gaps_count:>4} gaps ({data_pct:>5.1f}%)")
            report.append(f"  Geological Features (≥{depth_params['geological_gap_threshold']} pts):  {geological_gaps_count:>4} gaps ({geo_pct:>5.1f}%)")
        else:
            report.append("  No gaps found")
        report.append("")
        report.append("Gap Filling Results:")
        report.append(f"  Gaps Filled:                 {gaps_filled_count:>4} gaps")
        report.append(f"  Gaps Preserved (geological): {gaps_skipped_count:>4} gaps")
        report.append("")
        
        # Per-Curve Gap Table
        report.append("Per-Curve Gap Analysis:")
        report.append("-" * 80)
        report.append(f"{'Curve':<12} | {'Gaps':>5} | {'Data':>5} | {'Geo':>4} | {'Filled':>6} | {'Quality':<15}")
        report.append("-" * 80)
        
        for detail in gap_details[:20]:  # Show first 20 curves
            report.append(f"{detail['curve']:<12} | {detail['total']:>5} | {detail['data_err']:>5} | {detail['geo']:>4} | {detail['filled']:>6} | {detail['quality']:<15}")
        
        if len(gap_details) > 20:
            report.append(f"... and {len(gap_details) - 20} more curves")
        
        report.append("")
        
        # === IMPROVEMENT 3: Gap Classification Methodology Explanation ===
        report.append("GAP CLASSIFICATION METHODOLOGY")
        report.append("=" * 80)
        report.append("How the System Classifies Missing Data:")
        report.append("")
        report.append("The system distinguishes between two fundamentally different types of gaps:")
        report.append("")
        report.append("1. DATA ERRORS (Small Gaps)")
        report.append(f"   Definition: Consecutive missing points < Geological Gap Threshold")
        report.append(f"   Current Threshold: {self.geological_gap_threshold_var.get()} points ({depth_params['geological_gap_meters']:.1f} m)")
        report.append("   Characteristics:")
        report.append("     • Short duration gaps (typically <100m)")
        report.append("     • Caused by: Tool failures, data transmission errors, sensor issues")
        report.append("     • Processing: Should be filled using interpolation methods")
        report.append("   Examples:")
        report.append("     • Tool malfunction: 5-50 point gaps")
        report.append("     • Data transmission glitch: 10-100 point gaps")
        report.append("     • Sensor noise spike: 3-20 point gaps")
        report.append("")
        report.append("2. GEOLOGICAL/LOGGING FEATURES (Large Gaps)")
        report.append(f"   Definition: Consecutive missing points ≥ Geological Gap Threshold")
        report.append(f"   Current Threshold: {self.geological_gap_threshold_var.get()} points ({depth_params['geological_gap_meters']:.1f} m)")
        report.append("   Characteristics:")
        report.append("     • Extended duration gaps (typically >100m)")
        report.append("     • Caused by: Intentional non-logging, cased holes, interval logging")
        report.append("     • Processing: Should be preserved, NOT filled")
        report.append("   Examples:")
        report.append("     • Cased hole sections: 200-1000+ point gaps")
        report.append("     • Interval logging (open hole only): 300-500 point gaps")
        report.append("     • Zones where specific tools not run: 150-400 point gaps")
        report.append("")
        report.append("Classification Logic (Applied to Each Gap):")
        report.append("  ┌─────────────────────────────────────────────────────────────────┐")
        report.append("  │ For each gap found in curve data:                              │")
        report.append("  │   Step 1: Count consecutive missing points = gap_size          │")
        report.append("  │                                                                  │")
        threshold_pts = self.geological_gap_threshold_var.get()
        threshold_m = depth_params['geological_gap_meters']
        report.append(f"  │   Step 2: IF gap_size >= {threshold_pts} points ({threshold_m:.1f}m):               │")
        report.append("  │            → Classify as: GEOLOGICAL FEATURE                    │")
        report.append("  │            → Action: Preserve gap (do NOT fill)                 │")
        report.append("  │            → Quality: Exclude from error metrics                │")
        report.append("  │                                                                  │")
        report.append(f"  │   Step 3: ELSE (gap_size < {threshold_pts} points):                       │")
        report.append("  │            → Classify as: DATA ERROR                            │")
        report.append("  │            → Action: Attempt to fill using interpolation        │")
        report.append("  │            → Quality: Include in error metrics                  │")
        report.append("  └─────────────────────────────────────────────────────────────────┘")
        report.append("")
        report.append("Why This Matters for Quality Assessment:")
        report.append("  • Data quality grades (Excellent/Good/Fair/Poor) are calculated using")
        report.append("    ONLY the data error gaps")
        report.append("  • Geological gaps are excluded because they represent intentional")
        report.append("    non-logging, not data quality problems")
        report.append("  • This provides accurate quality metrics that reflect actual acquisition")
        report.append("    issues, not logging program decisions")
        report.append("")
        report.append("Threshold Adjustment Guidance:")
        report.append("  • Increase threshold (300-500 pts): Wells with longer cased sections")
        report.append("  • Decrease threshold (100-150 pts): High-quality continuous logging")
        report.append("  • Consider depth spacing: Threshold scales automatically with sampling rate")
        spacing_ratio = depth_params['spacing_ratio']
        report.append(f"  • Current scaling: {spacing_ratio:.2f}x (for {depth_params['depth_spacing']}m spacing)")
        report.append("")
        report.append("Physical Distance Interpretation:")
        report.append(f"  At current settings:")
        spacing_val = self.depth_spacing_var.get()
        threshold_val = self.geological_gap_threshold_var.get()
        physical_dist = depth_params['geological_gap_meters']
        report.append(f"    {threshold_val} points × {spacing_val} m/point = {physical_dist:.1f} meters")
        report.append("")
        report.append(f"  Gaps >= {physical_dist:.1f}m are considered geological features.")
        report.append("=" * 80)
        report.append("")
        
        # Detailed Curve Analysis
        report.append("DETAILED CURVE ANALYSIS")
        report.append("=" * 80)
        
        for curve in self.current_data.columns:
            curve_info = self.curve_info.get(curve, {})
            processing_result = self.processing_results.get(curve, {})
            
            report.append(f"\nCURVE: {curve}")
            report.append(f"  Type: {curve_info.get('curve_type', 'UNKNOWN')} (Confidence: {curve_info.get('type_confidence', 0.0):.2f})")
            report.append(f"  Unit: {curve_info.get('unit', 'UNKNOWN')}")
            report.append(f"  Description: {curve_info.get('description', 'No description available')}")
            
            # Safely access statistics with fallback values
            stats = curve_info.get('statistics', {})
            if stats:
                report.append(f"  Original Data:")
                report.append(f"    Valid Points: {stats.get('count', 0):,}")
                report.append(f"    Missing Points: {stats.get('missing', 0):,} ({stats.get('missing_percent', 0.0):.1f}%)")
                
                if not np.isnan(stats.get('mean', float('nan'))):
                    report.append(f"    Range: {stats.get('min', 0.0):.3f} to {stats.get('max', 0.0):.3f}")
                    report.append(f"    Mean: {stats.get('mean', 0.0):.3f}, Std: {stats.get('std', 0.0):.3f}")
            else:
                report.append(f"  Original Data: Statistics not available")
            
            # Gap filling results
            if 'gap_filling' in processing_result:
                gap_result = processing_result['gap_filling']
                gap_metrics = gap_result.get('quality_metrics', {})
                
                report.append(f"  Gap Filling:")
                report.append(f"    Gaps Filled: {gap_metrics.get('total_gaps_filled', 0)}")
                report.append(f"    Points Filled: {gap_metrics.get('total_points_filled', 0)}")
                report.append(f"    Methods Used: {', '.join(gap_metrics.get('methods_used', []))}")
                report.append(f"    Average Confidence: {gap_metrics.get('average_confidence', 0):.3f}")
                
                # Calculate final completeness: use gap filling result if available, otherwise calculate from original stats
                final_completeness = gap_metrics.get('data_completeness', 0)
                if final_completeness == 0 and stats:
                    # If no gap filling data completeness recorded, calculate from original statistics
                    final_completeness = 100.0 - stats.get('missing_percent', 0.0)
                
                report.append(f"    Final Completeness: {final_completeness:.1f}%")
            else:
                # No gap filling processed - show original completeness
                if stats:
                    original_completeness = 100.0 - stats.get('missing_percent', 0.0)
                    report.append(f"  Gap Filling: Not processed (Original Completeness: {original_completeness:.1f}%)")
                else:
                    report.append(f"  Gap Filling: Not processed")
            
            # Denoising results
            if 'denoising' in processing_result:
                denoise_result = processing_result['denoising']
                
                report.append(f"  Denoising:")
                report.append(f"    Method: {denoise_result.get('method', 'unknown')}")
                report.append(f"    Quality Score: {denoise_result.get('quality', 0):.3f}")
                
                if 'noise_reduction_db' in denoise_result:
                    report.append(f"    Noise Reduction: {denoise_result['noise_reduction_db']:.1f} dB")
                
                if denoise_result.get('method') == 'wavelet':
                    report.append(f"    Wavelet Used: {denoise_result.get('wavelet_used', 'unknown')}")
                    report.append(f"    Decomposition Levels: {denoise_result.get('levels', 0)}")
            else:
                report.append(f"  Denoising: Not processed")
        
        # === IMPROVEMENT 6: Processing Configuration with Actual UI Values ===
        report.append("PROCESSING CONFIGURATION")
        report.append("=" * 80)
        report.append("Gap Filling Parameters:")
        report.append(f"  Max Gap Size: {depth_params['max_gap_size']} points ({depth_params['max_gap_meters']:.1f} m)")
        report.append(f"  Large Gap Threshold: {depth_params['large_gap_threshold']} points ({depth_params['large_gap_meters']:.1f} m)")
        report.append(f"  Large Gap Treatment: {self.large_gap_var.get()}")
        report.append(f"  Geological Gap Threshold: {depth_params['geological_gap_threshold']} points ({depth_params['geological_gap_meters']:.1f} m)")
        report.append(f"  Method Priority: {self.gap_method_var.get()}")
        report.append(f"  Physics-Informed: {self.physics_informed_var.get()}")
        report.append(f"  Multi-Curve Correlation: {self.multi_curve_var.get()}")
        report.append("")
        report.append("Denoising Parameters:")
        report.append(f"  Method: {self.denoise_method_var.get()}")
        report.append("")
        report.append("Uniformization Parameters:")
        report.append(f"  Depth Spacing: {self.depth_spacing_var.get()} m")
        report.append(f"  Rename Curves: {self.rename_curves_var.get()}")
        report.append(f"  Standardize Units: {self.standardize_units_var.get()}")
        report.append(f"  Null Value: {self.null_value_var.get()}")
        report.append(f"  Output Format: {self.output_format_var.get()}")
        report.append("")
        report.append("Quality Control Parameters:")
        report.append(f"  QC Enabled: {self.qc_enabled_var.get()}")
        report.append(f"  Outlier Detection: {self.outlier_detection_var.get()}")
        report.append(f"  Range Validation: {self.range_validation_var.get()}")
        report.append(f"  Uncertainty Quantification: {self.uncertainty_quantification_var.get()}")
        
        # Unit Standardization Analysis
        if hasattr(self, 'unit_standardizer'):
            unit_analysis = self.unit_standardizer.get_unit_analysis_for_report()
            
            if any(unit_analysis.values()):  # Only show if there's unit analysis data
                report.append(f"\nUNIT STANDARDIZATION ANALYSIS")
                report.append("-" * 40)
                
                # Show conversions planned
                if unit_analysis['conversions_planned']:
                    report.append("Conversions Planned:")
                    for conv in unit_analysis['conversions_planned']:
                        report.append(f"  {conv['curve']}: {conv['from_unit']} → {conv['to_unit']} ({conv['description']})")
                
                # Show conversions applied
                if unit_analysis['conversions_applied']:
                    report.append("\nConversions Applied:")
                    for conv in unit_analysis['conversions_applied']:
                        report.append(f"  {conv['curve']}: {conv['from_unit']} → {conv['to_unit']} (×{conv['factor']:.4f})")
                
                # Show already standard units
                if unit_analysis['no_conversion_needed']:
                    report.append(f"\nAlready Standard Units: {len(unit_analysis['no_conversion_needed'])} curves")
                    for item in unit_analysis['no_conversion_needed'][:10]:  # Show first 10
                        report.append(f"  {item}")
                    if len(unit_analysis['no_conversion_needed']) > 10:
                        report.append(f"   ... and {len(unit_analysis['no_conversion_needed']) - 10} others")
                
                # Show unknown/unsupported units
                if unit_analysis['unknown_units']:
                    report.append(f"\nUnknown/Unsupported Units: {len(unit_analysis['unknown_units'])} curves")
                    for item in unit_analysis['unknown_units'][:10]:  # Show first 10
                        report.append(f"   ? {item}")
                    if len(unit_analysis['unknown_units']) > 10:
                        report.append(f"   ... and {len(unit_analysis['unknown_units']) - 10} others")
                
                # Show conversion errors if any
                if unit_analysis['conversion_errors']:
                    report.append(f"\nConversion Errors: {len(unit_analysis['conversion_errors'])}")
                    for error in unit_analysis['conversion_errors']:
                        report.append(f"  {error['curve']}: {error['from_unit']} → {error['to_unit']} - {error['reason']}")
                
                # Show depth validation update status
                if unit_analysis['depth_validation_updated']:
                    report.append(f"\nDepth Validation: Updated to match converted units")
                
                # Summary
                total_conversions = len(unit_analysis['conversions_planned'])
                total_applied = len(unit_analysis['conversions_applied'])
                total_standard = len(unit_analysis['no_conversion_needed'])
                total_unknown = len(unit_analysis['unknown_units'])
                total_errors = len(unit_analysis['conversion_errors'])
                
                report.append(f"\nUnit Analysis Summary:")
                report.append(f"  Conversions Planned: {total_conversions}")
                report.append(f"  Conversions Applied: {total_applied}")
                report.append(f"  Already Standard: {total_standard}")
                report.append(f"  Unknown Units: {total_unknown}")
                report.append(f"  Conversion Errors: {total_errors}")
        
        # === IMPROVEMENT 8: Add Export Metadata Section ===
        report.append("EXPORT INFORMATION")
        report.append("=" * 80)
        report.append(f"Null Value Used: {self.null_value_var.get()}")
        report.append(f"Output Format: {self.output_format_var.get()}")
        report.append(f"Depth Spacing: {self.depth_spacing_var.get()} m (standardized)")
        report.append(f"Unit Standard: {'SI Modified' if self.standardize_units_var.get() else 'Original'}")
        report.append(f"Curves Renamed: {'Yes' if self.rename_curves_var.get() else 'No'}")
        report.append("")
        
        # Quality Assessment
        report.append("QUALITY ASSESSMENT")
        report.append("=" * 80)
        report.append(f"Overall Processing Grade: {overall_grade}")
        report.append(f"Average Data Completeness: {avg_completeness:.1f}%")
        report.append(f"Average Denoising Quality: {avg_denoise_quality:.1f}%")
        report.append("")
        
        # === IMPROVEMENT 9: Enhanced Recommendations ===
        report.append("RECOMMENDATIONS AND INSIGHTS")
        report.append("=" * 80)
        
        recommendations = []
        
        # Geological gap recommendations
        if geological_gaps_count > data_error_gaps_count:
            recommendations.append(f"INFO: {geological_gaps_count} geological gaps detected (cased holes or interval logging).")
            recommendations.append(f"  → This is normal for interval curves. Current threshold: {self.geological_gap_threshold_var.get()} pts ({depth_params['geological_gap_meters']:.1f}m)")
        
        # Gap filling recommendations
        if data_error_gaps_count > total_curves * 2:
            recommendations.append(f"ATTENTION: {data_error_gaps_count} data error gaps detected across {total_curves} curves.")
            recommendations.append(f"  → Consider reviewing data acquisition quality or increasing gap fill threshold")
        
        # Depth spacing recommendations
        if self.depth_spacing_var.get() != 0.5:
            recommendations.append(f"NOTE: Non-standard depth spacing ({self.depth_spacing_var.get()}m) detected.")
            recommendations.append(f"  → All parameters automatically adjusted by {depth_params['spacing_ratio']:.2f}x to maintain physical distances")
        
        # Denoising recommendations
        if avg_denoise_quality < 70 and avg_denoise_quality > 0:
            recommendations.append(f"SUGGESTION: Low denoising quality ({avg_denoise_quality:.1f}%).")
            recommendations.append(f"  → Try different denoising method or adjust parameters")
        
        # Curve-specific recommendations
        high_missing_curves = [
            (curve, self.curve_info.get(curve, {}).get('statistics', {}).get('missing_percent', 0.0))
            for curve in self.current_data.columns
            if self.curve_info.get(curve, {}).get('statistics', {}).get('missing_percent', 0.0) > 30
        ]
        
        if high_missing_curves:
            recommendations.append(f"WARNING: {len(high_missing_curves)} curve(s) with >30% missing data:")
            for curve, pct in sorted(high_missing_curves, key=lambda x: x[1], reverse=True)[:5]:
                recommendations.append(f"  → {curve}: {pct:.1f}% missing")
            if len(high_missing_curves) > 5:
                recommendations.append(f"  → ... and {len(high_missing_curves) - 5} more")
        
        # Processing success recommendations
        excellent_curves = [
            curve for curve in self.current_data.columns
            if self.curve_info.get(curve, {}).get('statistics', {}).get('missing_percent', 0.0) < 5.0
        ]
        
        if len(excellent_curves) > total_curves * 0.7:
            recommendations.append(f"EXCELLENT: {len(excellent_curves)} curves ({len(excellent_curves)/total_curves*100:.1f}%) have <5% missing data.")
            recommendations.append(f"  → Data quality is very good, processing should be reliable")
        
        if not recommendations:
            recommendations.append("✓ Data quality is good. No specific recommendations.")
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        # Footer
        report.append(f"\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_scatter(self, curve: str):
        """Create a 2D scatter plot with industry-standard marginal histograms.

        Creates a scatter plot between two processed curves with histograms on top
        and right side showing the distribution of each variable. This follows
        industry standards for petrophysical data visualization.

        Uses the primary curve provided by the visualization controller and the
        secondary curve selected in `viz_curve2_var`. All user communication is
        performed via UI dialogs and `log_processing` only (no file logging).
        """
        try:
            curve2 = self.viz_curve2_var.get() if hasattr(self, 'viz_curve2_var') else None
            if not curve2 or curve2 not in getattr(self, 'processed_data', pd.DataFrame()).columns:
                messagebox.showwarning(ERROR_TITLE_VISUALIZATION, "Please select a valid secondary curve for the scatter plot")
                return

            # Check if curves have been processed, otherwise use original data
            if curve in self.processing_results:
                x = self.processing_results[curve]['final_data']
                x_status = 'processed'
            elif self.current_data is not None and curve in self.current_data.columns:
                x = self.current_data[curve].values
                x_status = 'original'
            else:
                messagebox.showwarning(ERROR_TITLE_VISUALIZATION, f"Curve '{curve}' not found in data")
                return
                
            if curve2 in self.processing_results:
                y = self.processing_results[curve2]['final_data']
                y_status = 'processed'
            elif self.current_data is not None and curve2 in self.current_data.columns:
                y = self.current_data[curve2].values
                y_status = 'original'
            else:
                messagebox.showwarning(ERROR_TITLE_VISUALIZATION, f"Curve '{curve2}' not found in data")
                return

            self.ensure_figure_exists()
            self.fig.set_size_inches(12, 10)
            
            # Clear the figure and create a new layout with marginal histograms
            self.fig.clear()
            
            # Create GridSpec for the layout: main scatter plot + top histogram + right histogram
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(3, 3, figure=self.fig)
            
            # Main scatter plot (bottom-left, spanning 2x2)
            ax_scatter = self.fig.add_subplot(gs[1:, :-1])
            
            # Top histogram (top, spanning 2 columns)
            ax_hist_x = self.fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
            
            # Right histogram (right, spanning 2 rows)
            ax_hist_y = self.fig.add_subplot(gs[1:, -1], sharey=ax_scatter)

            # Data already retrieved above
            valid_mask = (~np.isnan(x)) & (~np.isnan(y))
            if not np.any(valid_mask):
                messagebox.showwarning(ERROR_TITLE_VISUALIZATION, "No valid data points available for the scatter plot")
                return

            # Filter valid data
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # Create main scatter plot
            scatter = ax_scatter.scatter(x_valid, y_valid, s=20, alpha=0.7, 
                                       color='tab:blue', edgecolors='none', zorder=2)
            
            # Add trend line if sufficient data points
            if len(x_valid) > 10:
                try:
                    # Calculate trend line using numpy polyfit
                    z = np.polyfit(x_valid, y_valid, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(np.min(x_valid), np.max(x_valid), 100)
                    y_trend = p(x_trend)
                    ax_scatter.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2, 
                                  label=f'Trend (R² = {np.corrcoef(x_valid, y_valid)[0,1]:.3f})')
                    ax_scatter.legend(loc=LABEL_UPPER_LEFT)
                except Exception:
                    pass  # Continue without trend line if calculation fails
            
            # Create top histogram (X-axis distribution) - PETROPHYSICAL STANDARD
            bins_x = min(50, max(10, len(x_valid)//10))  # Ensure minimum bins for visibility
            n_x, bins_x_vals, patches_x = ax_hist_x.hist(x_valid, bins=bins_x, alpha=0.75, 
                          color='#1565C0', edgecolor='darkblue', linewidth=1.0)
            
            # Color-code top histogram bins based on frequency
            max_count_x = max(n_x) if len(n_x) > 0 else 1
            for patch, count in zip(patches_x, n_x):
                intensity = count / max_count_x
                patch.set_facecolor(plt.cm.Blues(0.3 + 0.5 * intensity))
            
            ax_hist_x.grid(True, alpha=0.3, axis='y')  # Grid on frequency axis
            # Set title with processing status
            title = f"Scatter Plot: {curve} vs {curve2}"
            if x_status != y_status:
                title += f' ({x_status} vs {y_status})'
            elif x_status == 'original':
                title += ' (Original Data)'
            else:
                title += ' (Processed Data)'
            ax_hist_x.set_title(title, fontsize=14, fontweight='bold')
            ax_hist_x.set_ylabel("Frequency")
            
            # Create right histogram (Y-axis distribution) - PETROPHYSICAL STANDARD
            # Use industry-standard color and styling for right-side histogram
            bins_y = min(50, max(10, len(y_valid)//10))  # Ensure minimum bins for visibility
            n, bins, patches = ax_hist_y.hist(y_valid, bins=bins_y, alpha=0.75, 
                          color='#2E7D32', edgecolor='darkgreen', linewidth=1.0, 
                          orientation='horizontal')
            
            # Enhance right histogram with petrophysical styling
            # Color-code bins based on frequency for better visualization
            max_count = max(n) if len(n) > 0 else 1
            for patch, count in zip(patches, n):
                # Gradient color based on frequency (darker = higher frequency)
                intensity = count / max_count
                patch.set_facecolor(plt.cm.Greens(0.3 + 0.5 * intensity))
            
            ax_hist_y.set_xlabel("Frequency", fontsize=10, fontweight='bold')
            ax_hist_y.grid(True, alpha=0.3, axis='x')  # Grid on frequency axis
            
            # Set labels for main scatter plot
            ax_scatter.set_xlabel(f"{curve} ({self.curve_info.get(curve, {}).get('unit', 'UNIT')})", 
                                fontsize=11, fontweight='bold')
            ax_scatter.set_ylabel(f"{curve2} ({self.curve_info.get(curve2, {}).get('unit', 'UNIT')})", 
                                fontsize=11, fontweight='bold')
            
            # Add grid to main scatter plot
            ax_scatter.grid(True, alpha=0.3, zorder=1, linestyle='--')
            
            # Remove axis labels from histograms to avoid duplication (keep for clarity)
            ax_hist_x.set_xlabel("")  # Shared with scatter plot below
            ax_hist_y.set_ylabel("")  # Shared with scatter plot on left
            
            # Keep some tick labels for histograms but make them subtle
            ax_hist_x.tick_params(labelbottom=False, labelsize=8)  # Hide bottom labels, keep side
            ax_hist_y.tick_params(labelleft=False, labelsize=8)  # Hide left labels, keep bottom
            
            # Add statistics text box on the scatter plot
            if len(x_valid) > 0:
                correlation = np.corrcoef(x_valid, y_valid)[0, 1]
                stats_text = f"Data Points: {len(x_valid)}\nCorrelation: {correlation:.3f}"
                ax_scatter.text(0.02, 0.98, stats_text, transform=ax_scatter.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', 
                               facecolor='white', alpha=0.8), fontsize=10)
            
            # Adjust layout to prevent overlap with proper spacing for histograms
            # Use padding to ensure right histogram is fully visible
            self.fig.tight_layout(rect=[0, 0.03, 0.97, 0.97])  # Leave space on right for labels
            
            # Ensure right histogram is clearly visible and properly sized
            # Adjust subplot parameters for better histogram visibility
            self.fig.subplots_adjust(right=0.92, top=0.90, hspace=0.35, wspace=0.35)
            
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, f"Failed to create scatter plot: {str(e)}")
    
    def plot_3d_visualization(self, curve: str):
        """Create a 3D visualization with 3 curves plus depth (industry standard)"""
        try:
            # Get secondary and tertiary curves
            curve2 = self.viz_curve2_var.get()
            curve3 = self.viz_curve3_var.get()
            
            if not curve2 or curve2 not in self.processed_data.columns:
                messagebox.showwarning("Warning", "Please select a valid secondary curve for 3D visualization")
                return
                
            if not curve3 or curve3 not in self.processed_data.columns:
                messagebox.showwarning("Warning", "Please select a valid third curve for 3D visualization")
                return
            
            # Check if curves have been processed, otherwise use original data
            curves_data = {}
            curves_status = {}
            
            for curve_name in [curve, curve2, curve3]:
                if curve_name in self.processing_results:
                    curves_data[curve_name] = self.processing_results[curve_name]['final_data']
                    curves_status[curve_name] = 'processed'
                elif self.current_data is not None and curve_name in self.current_data.columns:
                    curves_data[curve_name] = self.current_data[curve_name].values
                    curves_status[curve_name] = 'original'
                else:
                    messagebox.showwarning("Warning", f"Curve '{curve_name}' not found in data")
                    return
            
            curve1_data = curves_data[curve]
            curve2_data = curves_data[curve2]
            curve3_data = curves_data[curve3]
            
            # Ensure we have a valid figure with good size for 3D visualization
            self.ensure_figure_exists()
            self.fig.set_size_inches(12, 10)
            
            # Create 3D subplot
            ax = self.fig.add_subplot(111, projection='3d')
            
            # Find depth curve if available
            depth_curve = None
            for col in self.processed_data.columns:
                curve_type = self.curve_info.get(col, {}).get('curve_type', '')
                if 'DEPTH' in curve_type:
                    depth_curve = col
                    break
            
            # Use depth for Z-axis if available, otherwise use index
            if depth_curve:
                depth = self.processed_data[depth_curve].values
                depth_unit = self.curve_info.get(depth_curve, {}).get('unit', 'm')
                z_label = f'Depth ({depth_unit})'
                # Get actual depth range for proper axis limits
                depth_min, depth_max = self._get_depth_limits(depth)
            else:
                depth = np.arange(len(curve1_data))
                z_label = 'Depth (index)'
                depth_min, depth_max = self._get_depth_limits(depth)
            
            # Convert null values to NaN for proper visualization
            curve1_plot = self._convert_nulls_to_nan(curve1_data)
            curve2_plot = self._convert_nulls_to_nan(curve2_data)
            curve3_plot = self._convert_nulls_to_nan(curve3_data)
            
            # Filter out NaN values for all three curves
            valid_mask = (~np.isnan(curve1_plot) & 
                         ~np.isnan(curve2_plot) & 
                         ~np.isnan(curve3_plot))
            if not np.any(valid_mask):
                messagebox.showwarning("Warning", "No valid data points for 3D visualization")
                return
            
            valid_x = curve1_plot[valid_mask]
            valid_y = curve2_plot[valid_mask]
            valid_z = curve3_plot[valid_mask]
            valid_depth = depth[valid_mask]
            
            # Get industry-standard colors
            colors = PHYSICAL_CONSTANTS.VISUALIZATION_COLORS["3D_SCATTER"]
            
            # CRITICAL: Set axis limits to ACTUAL data range
            ax.set_zlim(depth_max, depth_min)  # Inverted for depth
            
            # Create 3D scatter plot with industry-standard coloring
            # Use depth for color mapping (industry standard)
            scatter = ax.scatter(valid_x, valid_y, valid_z, c=valid_depth, 
                                cmap=PHYSICAL_CONSTANTS.COLORMAP_STANDARDS["depth"], 
                                s=30, alpha=0.8, marker='o', edgecolors='black', linewidth=0.5)
            
            # Add professional color bar
            cbar = self.fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label(f'Depth ({depth_unit})', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # Set professional labels with units
            curve1_unit = self.curve_info.get(curve, {}).get("unit", "UNIT")
            curve2_unit = self.curve_info.get(curve2, {}).get("unit", "UNIT")
            curve3_unit = self.curve_info.get(curve3, {}).get("unit", "UNIT")
            
            ax.set_xlabel(f'{curve} ({curve1_unit})', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{curve2} ({curve2_unit})', fontsize=12, fontweight='bold')
            ax.set_zlabel(f'{curve3} ({curve3_unit})', fontsize=12, fontweight='bold')
            
            # Set professional title
            status_text = "Processed" if all(s == 'processed' for s in curves_status.values()) else "Mixed Data"
            title = f'3D Log Visualization: {curve} vs {curve2} vs {curve3} ({status_text})'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set initial view angle
            ax.view_init(elev=30, azim=45)
            
            # Invert Z-axis to show increasing depth downward (industry standard)
            ax.invert_zaxis()
            
            # Add a line connecting points in sequence
            ax.plot(valid_x, valid_y, valid_z, color='gray', alpha=0.5, linewidth=1)
            
            self.fig.tight_layout()
            
        except Exception as e:
            messagebox.showerror("3D Visualization Error", f"Failed to create 3D visualization: {str(e)}")
    def evaluate_curve_viability(self, curve_name: str, curve_data: np.ndarray) -> Tuple[str, float]:
        """Universal curve viability assessment with category-aware thresholds.

        Returns a tuple of (decision, validity_ratio) where decision is one of:
        - 'PROCESS_CRITICAL', 'PROCESS_STANDARD', 'PROCESS_MINIMAL', 'SKIP_INSUFFICIENT_DATA'
        Validity ratio is computed using non-NaN values as valid.
        """
        try:
            total_count = len(curve_data)
            if total_count == 0:
                return 'SKIP_INSUFFICIENT_DATA', 0.0
            valid_mask = ~np.isnan(curve_data)
            valid_count = int(np.sum(valid_mask))
            validity_ratio = valid_count / total_count if total_count > 0 else 0.0

            curve_category = self.detect_curve_category(curve_name, curve_data)

            if curve_category == 'DEPTH':
                decision = 'PROCESS_CRITICAL'
            elif curve_category == 'ESSENTIAL' and validity_ratio >= 0.30:
                decision = 'PROCESS_STANDARD'
            elif curve_category in ['RESISTIVITY', 'POROSITY', 'DENSITY'] and validity_ratio >= 0.20:
                decision = 'PROCESS_STANDARD'
            elif validity_ratio >= 0.15:
                decision = 'PROCESS_MINIMAL'
            else:
                decision = 'SKIP_INSUFFICIENT_DATA'

            timestamp = datetime.now().strftime('%H:%M:%S')
            self.log_processing(f"[{timestamp}] Quality gate for {curve_name} ({curve_category}): {validity_ratio:.1%} valid → {decision}")
            return decision, validity_ratio
        except Exception:
            # On any error, default to processing minimally to avoid skipping useful data
            return 'PROCESS_MINIMAL', 1.0

    def detect_outliers_iqr(self, data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method with professional logging"""
        if not self.outlier_detection_var.get():
            return np.zeros(len(data), dtype=bool)
        
        try:
            # Remove NaN values for calculation
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) < 4:
                self.log_processing("IQR outlier detection: Insufficient data points for reliable outlier detection")
                return np.zeros(len(data), dtype=bool)
            
            # Calculate quartiles
            Q1 = np.nanpercentile(valid_data, 25)
            Q3 = np.nanpercentile(valid_data, 75)
            IQR = Q3 - Q1
            
            # Set bounds with configurable multiplier
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Create outlier mask
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
            # Log outlier detection results
            outlier_count = np.sum(outlier_mask)
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(data)) * 100
                self.log_processing(f"IQR outlier detection: {outlier_count} outliers detected ({outlier_percentage:.1f}% of data)")
                
                # Log outlier statistics
                outlier_data = data[outlier_mask]
                if len(outlier_data) > 0:
                    min_outlier = np.min(outlier_data)
                    max_outlier = np.max(outlier_data)
                    self.log_processing(f"Outlier range: [{min_outlier:.3f}, {max_outlier:.3f}]")
                    self.log_processing(f"Data bounds: [{lower_bound:.3f}, {upper_bound:.3f}] (IQR multiplier: {multiplier})")
            else:
                self.log_processing(f"IQR outlier detection: No outliers detected (IQR multiplier: {multiplier})")
            
            return outlier_mask
            
        except Exception as e:
            error_category = self.categorize_error(e, "outlier_detection")
            error_msg = f"[{error_category}] IQR outlier detection failed: {e}"
            self.log_processing(f"ERROR: {error_msg}")
            
            # Return no outliers if detection fails
            if hasattr(self, 'root'):
                self.root.after(0, lambda: self.status_label.config(text="Outlier detection failed - continuing without outlier removal"))
            
            return np.zeros(len(data), dtype=bool)

    def apply_comprehensive_data_quality_validation(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply comprehensive data quality validation to all curves"""
        if not (self.range_validation_var.get() or self.outlier_detection_var.get()):
            self.log_processing("Data quality validation disabled - skipping validation")
            return data_dict
        
        self.log_processing("Starting comprehensive data quality validation...")
        validated_data = {}
        validation_summary = {}
        
        total_curves = len(data_dict)
        for i, (curve_name, data) in enumerate(data_dict.items()):
            # Update progress
            progress = (i / total_curves) * 100
            if hasattr(self, 'root'):
                self.root.after(0, lambda p=progress: self.progress_bar.configure(value=p))
                self.root.after(0, lambda c=curve_name: self.status_label.config(text=f"Validating {c}..."))
            
            original_count = np.sum(~np.isnan(data))
            validated_data[curve_name] = data.copy()
            
            try:
                # Apply range validation
                if self.range_validation_var.get():
                    validated_data[curve_name] = self.apply_range_validation(curve_name, validated_data[curve_name])
                
                # Apply outlier detection
                if self.outlier_detection_var.get():
                    outlier_mask = self.detect_outliers_iqr(validated_data[curve_name])
                    if np.any(outlier_mask):
                        validated_data[curve_name][outlier_mask] = np.nan
                
                # Calculate validation results
                final_count = np.sum(~np.isnan(validated_data[curve_name]))
                removed_count = original_count - final_count
                quality_percentage = (final_count / len(data)) * 100 if len(data) > 0 else 0
                
                validation_summary[curve_name] = {
                    'original_points': original_count,
                    'final_points': final_count,
                    'removed_points': removed_count,
                    'quality_percentage': quality_percentage
                }
                
                if removed_count > 0:
                    self.log_processing(f"Validation summary for {curve_name}: {removed_count} points removed, {final_count} remaining ({quality_percentage:.1f}% quality)")
                else:
                    self.log_processing(f"Validation summary for {curve_name}: No points removed, {final_count} points ({quality_percentage:.1f}% quality)")
                    
            except Exception as e:
                error_category = self.categorize_error(e, "comprehensive_validation")
                error_msg = f"[{error_category}] Validation failed for {curve_name}: {e}"
                self.log_processing(f"ERROR: {error_msg}")
                
                # Keep original data if validation fails
                validated_data[curve_name] = data
                validation_summary[curve_name] = {
                    'original_points': original_count,
                    'final_points': original_count,
                    'removed_points': 0,
                    'quality_percentage': 100.0,
                    'validation_failed': True,
                    'error': str(e)
                }
        
        # Log overall validation summary
        total_original = sum(summary['original_points'] for summary in validation_summary.values())
        total_final = sum(summary['final_points'] for summary in validation_summary.values())
        total_removed = total_original - total_final
        overall_quality = (total_final / total_original) * 100 if total_original > 0 else 0
        
        self.log_processing("=" * 50)
        self.log_processing("COMPREHENSIVE DATA QUALITY VALIDATION SUMMARY")
        self.log_processing("=" * 50)
        self.log_processing(f"Total curves processed: {total_curves}")
        self.log_processing(f"Total original points: {total_original:,}")
        self.log_processing(f"Total final points: {total_final:,}")
        self.log_processing(f"Total points removed: {total_removed:,}")
        self.log_processing(f"Overall data quality: {overall_quality:.1f}%")
        self.log_processing("=" * 50)
        
        return validated_data

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if 'psutil' in globals():
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _count_consecutive_missing(self, data: pd.Series) -> list:
        """Count consecutive missing (NaN) values in a series.
        
        Returns a list of gap sizes (consecutive missing values).
        Used to distinguish data errors from geological/logging gaps.
        
        Args:
            data: Pandas Series of curve data
            
        Returns:
            List of integers representing consecutive missing data lengths
        """
        try:
            if data is None or len(data) == 0:
                return []
            
            gap_sizes = []
            current_gap = 0
            
            for value in data:
                if pd.isna(value):
                    current_gap += 1
                else:
                    if current_gap > 0:
                        gap_sizes.append(current_gap)
                        current_gap = 0
            
            # Don't forget trailing gap
            if current_gap > 0:
                gap_sizes.append(current_gap)
            
            return gap_sizes
            
        except Exception as e:
            self.log_processing(f"Error counting consecutive missing: {e}")
            return []
    
    def get_depth_aware_parameters(self) -> dict:
        """Calculate depth-aware parameters based on current depth spacing.
        
        Adjusts gap thresholds, filter windows, and geological thresholds to account
        for actual depth spacing. This ensures that thresholds represent physical 
        distances rather than just point counts.
        
        Returns:
            Dictionary with adjusted parameters
        """
        try:
            depth_spacing = self.depth_spacing_var.get()
            
            # Reference spacing (0.5m) - all default thresholds assume this
            reference_spacing = 0.5
            
            # Scaling factor
            spacing_ratio = reference_spacing / depth_spacing if depth_spacing > 0 else 1.0
            
            # Scale thresholds to maintain physical distances
            raw_geological = self.geological_gap_threshold_var.get()
            raw_large = self.large_gap_threshold_var.get()
            raw_max = self.max_gap_var.get()
            
            geological_points = max(1, int(np.ceil(raw_geological * spacing_ratio)))
            large_gap_points = max(1, int(np.ceil(raw_large * spacing_ratio)))
            max_gap_points = max(1, int(np.ceil(raw_max * spacing_ratio)))
            
            # Adjusted parameters
            adjusted = {
                'depth_spacing': depth_spacing,
                'spacing_ratio': spacing_ratio,
                
                # Gap thresholds (scale to maintain same physical distance)
                'geological_gap_threshold': geological_points,
                'large_gap_threshold': large_gap_points,
                'max_gap_size': max_gap_points,
                
                # Filter windows (scale to maintain same physical smoothing distance)
                'savgol_window': max(5, int(np.ceil(11 * spacing_ratio))),
                'median_window': max(3, int(np.ceil(5 * spacing_ratio))),
                'bilateral_window': max(5, int(np.ceil(10 * spacing_ratio))),
                
                # Physical interpretation
                'geological_gap_meters': geological_points * depth_spacing,
                'large_gap_meters': large_gap_points * depth_spacing,
                'max_gap_meters': max_gap_points * depth_spacing
            }
            
            return adjusted
            
        except Exception as e:
            self.log_processing(f"Error calculating depth-aware parameters: {e}")
            # Return defaults
            return {
                'depth_spacing': 0.5,
                'spacing_ratio': 1.0,
                'geological_gap_threshold': 200,
                'large_gap_threshold': 500,
                'max_gap_size': 500
            }

    def detect_curve_category(self, curve_name: str, curve_data: np.ndarray) -> str:
        """Detect the category of a curve based on name and data characteristics."""
        try:
            curve_name_upper = curve_name.upper()
            
            # Check for depth curves
            if any(keyword in curve_name_upper for keyword in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']):
                return 'DEPTH'
            
            # Check for essential curves
            if any(keyword in curve_name_upper for keyword in ['GR', 'GAMMA', 'GAMMA_RAY']):
                return 'ESSENTIAL'
            
            # Check for resistivity curves
            if any(keyword in curve_name_upper for keyword in ['RT', 'RM', 'RS', 'RXO', 'RESISTIVITY']):
                return 'RESISTIVITY'
            
            # Check for porosity curves
            if any(keyword in curve_name_upper for keyword in ['NPHI', 'NEUTRON', 'TNPH']):
                return 'POROSITY'
            
            # Check for density curves
            if any(keyword in curve_name_upper for keyword in ['RHOB', 'DENSITY', 'RHOZ']):
                return 'DENSITY'
            
            # Check for sonic curves
            if any(keyword in curve_name_upper for keyword in ['DT', 'SONIC', 'DTCO']):
                return 'SONIC'
            
            # Check for caliper curves
            if any(keyword in curve_name_upper for keyword in ['CALI', 'CALIPER']):
                return 'CALIPER'
            
            # Check for photoelectric curves
            if any(keyword in curve_name_upper for keyword in ['PE', 'PHOTOELECTRIC']):
                return 'PHOTOELECTRIC'
            
            # Default category
            return 'UNKNOWN'
            
        except Exception:
            return 'UNKNOWN'

    def get_curve_info(self, curve_name: str):
        """Get curve info from the curve identifier, creating default if not exists."""
        try:
            if hasattr(self, 'curve_identifier') and self.curve_identifier:
                return self.curve_identifier.get_curve_info(curve_name)
            else:
                # Fallback to direct curve_info access
                return self.curve_info.get(curve_name, {
                    'curve_type': 'UNKNOWN',
                    'unit': '',
                    'description': f'Curve {curve_name}',
                    'typical_range': (0.0, 1.0),
                    'type_confidence': 0.0
                })
        except Exception:
            # Return default curve info if anything fails
            return {
                'curve_type': 'UNKNOWN',
                'unit': '',
                'description': f'Curve {curve_name}',
                'typical_range': (0.0, 1.0),
                'type_confidence': 0.0
            }

    def apply_range_validation(self, curve_name: str, data: np.ndarray) -> np.ndarray:
        """Alias for validate_curve_range - applies range validation and returns cleaned data.
        
        This method exists to maintain compatibility with existing code that calls
        apply_range_validation. It uses the existing validate_curve_range method
        and applies the validation results to clean the data.
        """
        try:
            # Use existing validation method
            validation_result = self.curve_identifier.validate_curve_range(curve_name, data)
            
            if not validation_result['valid']:
                # If range is invalid, apply tolerance-based cleaning
                curve_info = self.get_curve_info(curve_name)
                min_expected, max_expected = curve_info.typical_range
                
                # Apply tolerance factor (same as in validate_curve_range)
                tolerance_factor = 2.0
                min_allowed = min_expected / tolerance_factor
                max_allowed = max_expected * tolerance_factor
                
                # Create mask for valid data within tolerance
                valid_mask = (data >= min_allowed) & (data <= max_allowed)
                
                # Replace out-of-range values with NaN
                cleaned_data = data.copy()
                cleaned_data[~valid_mask] = np.nan
                
                return cleaned_data
            
            # If validation passed, return original data
            return data
            
        except Exception as e:
            # If validation fails, return original data unchanged
            try:
                self.log_processing(f"Range validation failed for {curve_name}: {e}")
            except Exception:
                pass
            return data

    def uniformize_curves(self) -> None:
        """Standardize curve names and units in `processed_data`.

        - If `rename_curves_var` is enabled, renames columns to standard
          mnemonics using the mnemonic library's canonical entries while
          preserving `curve_info` mappings.
        - If `standardize_units_var` is enabled, applies the unit
          standardization pipeline to `processed_data`.
        """
        try:
            if self.processed_data is None or self.processed_data.empty:
                return

            # Rename curves to standard mnemonics if requested
            if hasattr(self, 'rename_curves_var') and self.rename_curves_var.get():
                rename_map = {}
                used_names = set(self.processed_data.columns)
                for col in list(self.processed_data.columns):
                    try:
                        unit = self.curve_info.get(col, {}).get('unit', '')
                        desc = self.curve_info.get(col, {}).get('description', '')
                        curve_type, confidence, info = self.curve_identifier.identify_curve(col, unit, desc)
                        mnemonics = info.get('mnemonics', []) if isinstance(info, dict) else []
                        if confidence >= 0.5 and mnemonics:
                            standard_name = mnemonics[0]
                            if standard_name != col and standard_name not in used_names:
                                rename_map[col] = standard_name
                                used_names.add(standard_name)
                    except Exception:
                        continue

                if rename_map:
                    # CRITICAL: Record all curve renames for audit trail
                    if hasattr(self, 'standardization_reporter'):
                        for old_name, new_name in rename_map.items():
                            self.standardization_reporter.record_curve_rename(
                                original_name=old_name,
                                standardized_name=new_name,
                                reason='standardization',
                                confidence=0.9  # High confidence for mnemonic library matches
                            )
                    
                    self.processed_data.rename(columns=rename_map, inplace=True)
                    updated_curve_info = {}
                    for old_name, info in self.curve_info.items():
                        new_name = rename_map.get(old_name, old_name)
                        updated_curve_info[new_name] = info
                    self.curve_info = updated_curve_info
                    try:
                        summary = ", ".join([f"{k}→{v}" for k, v in rename_map.items()])
                        self.log_processing(f"Renamed curves to standard mnemonics: {summary}")
                    except Exception:
                        pass

            # Apply unit standardization to processed_data if enabled
            if hasattr(self, 'standardize_units_var') and self.standardize_units_var.get():
                original_current_data = self.current_data
                try:
                    self.current_data = self.processed_data
                    self.unit_standardizer.set_application_reference(self)
                    self.unit_standardizer.apply_unit_standardization()
                    self.processed_data = self.current_data
                finally:
                    self.current_data = original_current_data
            else:
                # Skipping is a logged, visible outcome, not a silent pass. State the
                # units the data is actually carrying forward, because every later
                # stage that compares against a reference range assumes some unit and
                # currently has no way to declare which (C1 is not implemented yet).
                try:
                    declared = sorted({
                        str(self.curve_info.get(col, {}).get('unit', '')).strip()
                        for col in self.processed_data.columns
                    } - {''})
                    self.log_processing(
                        "Unit standardization is OFF (default). Values and declared "
                        "units are carried through as loaded; no conversion applied.")
                    if declared:
                        self.log_processing(
                            f"  Units in play: {', '.join(declared)}")
                except Exception:
                    pass
        except Exception as e:
            try:
                self.log_processing(f"ERROR: Uniformization failed: {e}")
            except Exception:
                pass

    def resample_to_standard_spacing(self, depth_column: str, target_spacing: float) -> None:
        """Resample `processed_data` onto a uniform depth grid using index interpolation."""
        try:
            if (self.processed_data is None or
                depth_column not in self.processed_data.columns or
                target_spacing is None or target_spacing <= 0):
                return

            depth_series = pd.to_numeric(self.processed_data[depth_column], errors='coerce').dropna()
            if len(depth_series) < 2:
                return

            dmin = float(depth_series.min())
            dmax = float(depth_series.max())
            if dmax <= dmin:
                return

            new_depth = np.arange(dmin, dmax + target_spacing / 2.0, target_spacing)
            resampled_df = pd.DataFrame({depth_column: new_depth})

            for col in self.processed_data.columns:
                if col == depth_column:
                    continue
                series = pd.to_numeric(self.processed_data[col], errors='coerce')
                idx = pd.to_numeric(self.processed_data[depth_column], errors='coerce')
                valid = (~series.isna()) & (~idx.isna())
                if valid.sum() < 2:
                    resampled_df[col] = np.nan
                    continue

                s = pd.Series(series[valid].values, index=idx[valid].values)
                s = s.groupby(level=0).mean().sort_index()
                # limit_area='inside' confines interpolation to gaps bracketed by real
                # samples. Without it, a curve logged over only part of the well (e.g.
                # a density tool run 4250-5326 ft in a 0-5359 ft hole) gets extrapolated
                # to a fabricated value at every depth in the grid.
                s_interp = s.reindex(s.index.union(new_depth)).interpolate(method='index', limit_area='inside')
                resampled_df[col] = s_interp.reindex(new_depth).values

            self.processed_data = resampled_df
        except Exception as e:
            try:
                self.log_processing(f"ERROR: Resampling failed: {e}")
            except Exception:
                pass

    def _sync_depth_spacing_default(self) -> None:
        """Set depth resampling default to 0.1 m or 0.5 ft based on current depth units."""
        try:
            # Determine current depth unit from curve_info
            depth_col = None
            for col in (self.processed_data.columns if self.processed_data is not None else []):
                ctype = str(self.curve_info.get(col, {}).get('curve_type', '')).upper()
                if 'DEPTH' in ctype or col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']:
                    depth_col = col
                    break
            if not depth_col:
                return
            unit = str(self.curve_info.get(depth_col, {}).get('unit', 'M')).upper()
            if unit in ['FT', 'FEET']:
                # 0.5 ft default
                if abs(self.depth_spacing_var.get() - 0.5) > 1e-9:
                    self.depth_spacing_var.set(0.5)
                    self.log_processing("Depth spacing default set to 0.5 ft based on depth units")
            else:
                # 0.1 m default
                if abs(self.depth_spacing_var.get() - 0.1) > 1e-9:
                    self.depth_spacing_var.set(0.1)
                    self.log_processing("Depth spacing default set to 0.1 m based on depth units")
        except Exception:
            pass

    def finalize_uniformization(self):
        """Apply final uniformization steps.

        NaN is the single internal representation for missing data. This step
        normalises every null sentinel found in the working frame to NaN,
        including the sentinel the file's own header declares, so that
        `processed_data` and the per-curve arrays in `processing_results`
        express missingness the same way. Consumers that count gaps, compute
        correlations or plot can therefore trust `isna`/`isnan` without each
        having to know the session's null convention.

        The sentinel is re-emitted only at the export boundary, where the LAS
        and CSV formats require a numeric placeholder.
        """
        try:
            self.log_processing("Applying final uniformization...")
            
            if self.processed_data is None:
                return
            
            # Retained only for the log message; the declared convention no
            # longer changes how missing data is stored internally.
            session_null_label = (
                self.null_value_var.get()
                if hasattr(self, 'null_value_var') and self.null_value_var.get()
                else '-999.25'
            )
            null_value = self._get_null_value()
            
            # The declared sentinel is normalised alongside the common
            # alternates, because a curve may carry a sentinel that its own
            # header never declared. Hits are logged, not silent.
            null_patterns = [-999.25, -999, -9999, 99999, -99999]
            if np.isfinite(null_value) and not any(
                abs(float(pattern) - float(null_value)) < 1e-9
                for pattern in null_patterns
            ):
                null_patterns.append(float(null_value))
            
            for curve in self.processed_data.columns:
                data = self.processed_data[curve]
                numeric = pd.to_numeric(data, errors='coerce')
                
                for pattern in null_patterns:
                    try:
                        hit_count = int((numeric == pattern).sum())
                    except Exception:
                        hit_count = 0
                    if hit_count > 0:
                        self.log_processing(
                            f"NULL normalisation: {curve} had {hit_count} value(s) equal to "
                            f"{pattern} (session NULL is {session_null_label}); "
                            f"converting to NaN"
                        )
                    data = data.replace(pattern, np.nan)
                    numeric = pd.to_numeric(data, errors='coerce')
                
                self.processed_data[curve] = data
            
            # Ensure consistent data types
            for curve in self.processed_data.columns:
                try:
                    self.processed_data[curve] = pd.to_numeric(self.processed_data[curve], errors='coerce')
                except Exception as e:
                    self.log_processing(f"Warning: Could not convert {curve} to numeric: {e}")
            
            self.log_processing("Final uniformization completed")
            
        except Exception as e:
            self.log_processing(f"Error in final uniformization: {e}")

    def _calculate_total_gaps_filled(self) -> int:
        """Calculate total number of gaps filled across all curves"""
        try:
            total_gaps = 0
            
            if not self.processing_results:
                return 0
            
            for curve, result in self.processing_results.items():
                gap_filling = result.get('gap_filling', {})
                quality_metrics = gap_filling.get('quality_metrics', {})
                total_gaps += quality_metrics.get('total_points_filled', 0)
            
            return total_gaps
            
        except Exception as e:
            self.log_processing(f"Error calculating total gaps filled: {e}")
            return 0

    def plot_uncertainty(self, curve: str):
        """Plot uncertainty visualization for processed data"""
        # Check if curve has been processed
        if curve in self.processing_results:
            processed = self.processing_results[curve]['final_data']
            has_processed = True
        elif self.current_data is not None and curve in self.current_data.columns:
            # Show only original data if not processed
            processed = self.current_data[curve].values
            has_processed = False
        else:
            messagebox.showwarning("Warning", f"Curve '{curve}' not found in data")
            return
        
        try:
            self.ensure_figure_exists()
            self.fig.set_size_inches(12, 9)
            ax = self.fig.add_subplot(111)
            
            # Data already retrieved above
            
            # Calculate uncertainty from gap filling results if available
            if has_processed:
                gap_result = self.processing_results[curve].get('gap_filling', {})
                uncertainty = gap_result.get('uncertainty', np.zeros_like(processed))
                confidence = gap_result.get('confidence', np.ones_like(processed))
            else:
                # For unprocessed curves, use default uncertainty
                uncertainty = np.full_like(processed, 0.1)  # 10% default uncertainty
                confidence = np.full_like(processed, 0.5)   # 50% default confidence
            
            # Find depth curve
            depth_curve = None
            for col in self.processed_data.columns:
                curve_type = self.curve_info.get(col, {}).get('curve_type', '')
                if 'DEPTH' in curve_type:
                    depth_curve = col
                    break
            
            if depth_curve:
                depth = self.processed_data[depth_curve].values
                depth_unit = self.curve_info.get(depth_curve, {}).get('unit', 'm')
                y_label = f'Depth ({depth_unit})'
                # Get actual depth range for proper axis limits
                depth_min, depth_max = self._get_depth_limits(depth)
            else:
                depth = np.arange(len(processed))
                y_label = 'Depth (index)'
                depth_min, depth_max = self._get_depth_limits(depth)
            
            # Convert null values to NaN for proper line breaking (for visualization only)
            processed_plot = self._convert_nulls_to_nan(processed)
            
            # Plot main curve
            ax.plot(processed_plot, depth, 'b-', linewidth=2, label=LABEL_PROCESSED_DATA)
            
            # CRITICAL: Set axis limits to ACTUAL data range before fills/scatters.
            # set_ylim also disables y autoscaling, so the fill and scatter added
            # below cannot widen these limits.
            self.apply_depth_axis(ax, depth, label=y_label)
            
            # Plot uncertainty bands (also convert nulls in bounds)
            upper_bound = processed_plot + uncertainty
            lower_bound = processed_plot - uncertainty
            
            ax.fill_betweenx(depth, lower_bound, upper_bound, alpha=0.3, color='lightblue', 
                            label='Uncertainty Band')
            
            # Color code by confidence using industry-standard uncertainty colormap
            confidence_colors = confidence.copy()
            uncertainty_cmap = PHYSICAL_CONSTANTS.COLORMAP_STANDARDS["uncertainty"]
            scatter = ax.scatter(processed_plot, depth, c=confidence_colors, cmap=uncertainty_cmap, 
                               s=20, alpha=0.7, label='Confidence', vmin=0, vmax=1)
            
            # Add professional colorbar for confidence
            cbar = self.fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Confidence Level (0-1)', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # Set title with processing status
            if has_processed:
                ax.set_title(f'Uncertainty Analysis: {curve} (Processed)', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Uncertainty Analysis: {curve} (Not Yet Processed)', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{curve} ({self.curve_info[curve]["unit"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            
        except Exception as e:
            messagebox.showerror("Uncertainty Plot Error", f"Failed to create uncertainty plot: {str(e)}")

    def plot_quality_metrics(self, curve: str):
        """Plot quality metrics for processed data"""
        # Check if curve has been processed
        if curve in self.processing_results:
            result = self.processing_results[curve]
            has_processed = True
        elif self.current_data is not None and curve in self.current_data.columns:
            # For unprocessed curves, create basic quality metrics
            has_processed = False
            # Calculate basic statistics from original data
            data = self.current_data[curve].values
            valid_mask = ~np.isnan(data)
            completeness = np.sum(valid_mask) / len(data) * 100 if len(data) > 0 else 0
        else:
            messagebox.showwarning("Warning", f"Curve '{curve}' not found in data")
            return
        
        try:
            # Use proper figure management with larger size
            self.ensure_figure_exists()
            self.fig.set_size_inches(14, 10)
            
            # Create subplots for different quality metrics
            axes = self.fig.subplots(2, 2)
            
            # Set overall title with processing status
            if has_processed:
                self.fig.suptitle(f'Quality Metrics: {curve} (Processed)', fontsize=16, fontweight='bold')
            else:
                self.fig.suptitle(f'Quality Metrics: {curve} (Not Yet Processed)', fontsize=16, fontweight='bold')
            
            # Get metrics based on processing status
            if has_processed:
                gap_metrics = result.get('gap_filling', {}).get('quality_metrics', {})
                denoise_metrics = result.get('denoising', {})
            else:
                # For unprocessed curves, create basic metrics
                gap_metrics = {'data_completeness': completeness, 'total_gaps_filled': 0, 'total_points_filled': 0, 'methods_used': [], 'average_confidence': 0.5, 'average_uncertainty': 0.5}
                denoise_metrics = {'quality': 0.5, 'noise_reduction_db': 0, 'signal_preservation': 0.5}
            
            # Use industry-standard quality colormap
            quality_cmap = PHYSICAL_CONSTANTS.COLORMAP_STANDARDS["quality"]
            
            # Plot 1: Data Completeness (Industry Standard: Green=Good, Red=Poor)
            ax1 = axes[0, 0]
            completeness = gap_metrics.get('data_completeness', 0)
            colors = ['#00FF00' if completeness > 80 else '#FFA500' if completeness > 60 else '#FF0000', '#FF0000']
            ax1.pie([completeness, 100-completeness], labels=['Valid Data', LABEL_MISSING_DATA],
                    colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Data Completeness', fontsize=12, fontweight='bold')
            
            # Plot 2: Gap Filling Quality (Industry Standard: Blue=Confidence, Orange=Uncertainty)
            ax2 = axes[0, 1]
            confidence_values = gap_metrics.get('average_confidence', 0)
            uncertainty_values = gap_metrics.get('average_uncertainty', 0)
            
            metrics = ['Confidence', 'Uncertainty (inv)']
            values = [confidence_values, 1.0 - uncertainty_values]
            colors = ['#0000FF', '#FFA500']  # Blue for confidence, Orange for uncertainty
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Gap Filling Quality', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Quality Score (0-1)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Denoising Performance (Industry Standard: Green=Good, Blue=Noise, Red=Signal)
            ax3 = axes[1, 0]
            denoise_quality = denoise_metrics.get('quality', 0)
            noise_reduction = denoise_metrics.get('noise_reduction_db', 0) / 20.0  # Normalize
            signal_preservation = denoise_metrics.get('signal_preservation', 0)
            
            categories = ['Overall Quality', 'Noise Reduction', 'Signal Preservation']
            values = [denoise_quality, min(1.0, noise_reduction), signal_preservation]
            colors = ['#00FF00', '#0000FF', '#FF0000']  # Green, Blue, Red
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax3.set_ylim(0, 1)
            ax3.set_title('Denoising Performance', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Quality Score (0-1)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 4: Processing Summary (Industry Standard: Purple=Gaps, Cyan=Points, Yellow=Methods)
            ax4 = axes[1, 1]
            gaps_filled = gap_metrics.get('total_gaps_filled', 0)
            points_filled = gap_metrics.get('total_points_filled', 0)
            methods_used = len(gap_metrics.get('methods_used', []))
            
            summary_data = [gaps_filled, points_filled, methods_used]
            summary_labels = ['Gaps Filled', 'Points Filled', 'Methods Used']
            colors = ['#800080', '#00FFFF', '#FFFF00']  # Purple, Cyan, Yellow
            
            bars = ax4.bar(summary_labels, summary_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax4.set_title('Processing Summary', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Count', fontsize=10)
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            # Apply proper spacing for quality metrics display
            self.fig.tight_layout()
            
            # Create canvas if not embedded
            if hasattr(self, 'viz_content') and self.viz_content:
                self.cleanup_visualization()
                self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
                self.canvas.draw()
                
                if NavigationToolbar2Tk:
                    toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                    toolbar.update()
                    toolbar.pack(side='top', fill='x')
                
                self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
            
            self.log_processing(f"Quality metrics visualization created: {curve}")
            
        except Exception as e:
            self.log_processing(f"Error in quality metrics visualization: {e}")
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, f"Failed to create quality metrics plot:\n{e}")
    
    def plot_histogram(self, curve: str):
        """Create a professional histogram/distribution plot for data quality control.
        
        Industry-standard histogram visualization for petrophysical data analysis.
        Shows data distribution, identifies outliers, bimodality, and data quality issues.
        
        Features:
        - Original vs processed comparison (if available)
        - Statistical annotations (mean, median, std dev)
        - Outlier detection visualization
        - Normal distribution overlay (if applicable)
        - Professional styling with industry-standard colors
        """
        # Check if curve has been processed, otherwise use original data
        if curve in self.processing_results:
            processed = self.processing_results[curve]['final_data']
            original = self.processing_results[curve].get('original_data', None)
            has_processed = True
        elif self.current_data is not None and curve in self.current_data.columns:
            processed = self.current_data[curve].values
            original = None
            has_processed = False
        else:
            messagebox.showwarning("Warning", f"Curve '{curve}' not found in data")
            return
        
        try:
            self.cleanup_visualization()
            self.ensure_figure_exists()
            self.fig.set_size_inches(12, 9)
            
            # Convert null values to NaN for proper filtering
            processed_clean = self._convert_nulls_to_nan(processed)
            valid_processed = processed_clean[~np.isnan(processed_clean) & np.isfinite(processed_clean)]
            
            if len(valid_processed) == 0:
                messagebox.showwarning("Warning", f"No valid data points for curve '{curve}'")
                return
            
            # Create main histogram plot
            ax = self.fig.add_subplot(111)
            
            # Calculate optimal number of bins (Freedman-Diaconis rule for petrophysical data)
            iqr = np.percentile(valid_processed, 75) - np.percentile(valid_processed, 25)
            bin_width = 2 * iqr / (len(valid_processed) ** (1/3)) if iqr > 0 else (valid_processed.max() - valid_processed.min()) / 30
            num_bins = max(20, min(50, int((valid_processed.max() - valid_processed.min()) / bin_width))) if bin_width > 0 else 30
            
            # Plot processed data histogram
            n, bins, patches = ax.hist(valid_processed, bins=num_bins, alpha=0.7, color='blue', 
                                      edgecolor='black', linewidth=1.2, label=LABEL_PROCESSED_DATA if has_processed else 'Data')
            
            # Color-code bins by frequency (darker = higher frequency) for better visualization
            max_freq = n.max() if len(n) > 0 else 1
            for i, (patch, freq) in enumerate(zip(patches, n)):
                intensity = 0.3 + 0.7 * (freq / max_freq) if max_freq > 0 else 0.5
                patch.set_facecolor(plt.cm.Blues(intensity))
            
            # Plot original data histogram if available (overlay)
            if original is not None:
                original_clean = self._convert_nulls_to_nan(original)
                valid_original = original_clean[~np.isnan(original_clean) & np.isfinite(original_clean)]
                
                if len(valid_original) > 0:
                    ax.hist(valid_original, bins=bins, alpha=0.4, color='red', 
                           edgecolor='darkred', linestyle='--', 
                           label=LABEL_ORIGINAL_DATA, histtype='step', linewidth=2)
            
            # Calculate and display statistics
            mean_val = np.mean(valid_processed)
            median_val = np.median(valid_processed)
            std_val = np.std(valid_processed)
            min_val = np.min(valid_processed)
            max_val = np.max(valid_processed)
            
            # Add vertical lines for mean and median
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # Add normal distribution overlay if data appears normally distributed
            if len(valid_processed) > 30:  # Only for sufficient data points
                from scipy import stats
                try:
                    # Test for normality (Shapiro-Wilk test)
                    if len(valid_processed) <= 5000:  # Test limited to reasonable size
                        _, p_value = stats.shapiro(valid_processed[:5000])
                        if p_value > 0.05:  # Data appears normal
                            # Overlay normal distribution
                            x_norm = np.linspace(valid_processed.min(), valid_processed.max(), 100)
                            y_norm = stats.norm.pdf(x_norm, mean_val, std_val) * len(valid_processed) * (bins[1] - bins[0])
                            ax.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.6, label='Normal Distribution Fit')
                except:
                    pass  # Skip normal overlay if scipy not available or test fails
            
            # Add outlier detection visualization (IQR method)
            q1 = np.percentile(valid_processed, 25)
            q3 = np.percentile(valid_processed, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = valid_processed[(valid_processed < lower_bound) | (valid_processed > upper_bound)]
            
            if len(outliers) > 0:
                ax.axvspan(lower_bound, upper_bound, alpha=0.1, color='green', label='Normal Range (IQR)')
                ax.scatter(outliers, np.zeros_like(outliers) + max(n) * 0.05, 
                          color='red', marker='x', s=50, alpha=0.7, zorder=5, label=f'Outliers ({len(outliers)})')
            
            # Professional styling
            curve_unit = self.curve_info.get(curve, {}).get('unit', '')
            ax.set_xlabel(f'{curve} ({curve_unit})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            
            if has_processed:
                ax.set_title(f'Distribution Analysis: {curve} (Processed)', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Distribution Analysis: {curve} (Original)', fontsize=14, fontweight='bold')
            
            # Add statistics text box
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {mean_val:.3f}\n'
            stats_text += f'Median: {median_val:.3f}\n'
            stats_text += f'Std Dev: {std_val:.3f}\n'
            stats_text += f'Min: {min_val:.3f}\n'
            stats_text += f'Max: {max_val:.3f}\n'
            stats_text += f'Count: {len(valid_processed):,}\n'
            if len(outliers) > 0:
                stats_text += f'Outliers: {len(outliers)} ({len(outliers)/len(valid_processed)*100:.1f}%)'
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, family='monospace')
            
            ax.legend(loc=LABEL_UPPER_LEFT, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            
            # Create canvas if embedded
            if hasattr(self, 'viz_content') and self.viz_content:
                self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
                self.canvas.draw()
                
                if NavigationToolbar2Tk:
                    toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                    toolbar.update()
                    toolbar.pack(side='top', fill='x')
                
                self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
            
            self.log_processing(f"Histogram visualization created: {curve}")
            
        except Exception as e:
            self.log_processing(f"Error in histogram visualization: {e}")
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, f"Failed to create histogram plot:\n{e}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix for all processed curves"""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "No processed data available")
            return
        
        try:
            self.ensure_figure_exists()
            
            # Calculate correlation matrix
            numeric_data = self.processed_data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()
            
            # Set larger figure size for better readability
            num_curves = len(correlation_matrix.columns)
            fig_size = max(10, min(16, num_curves * 0.8))  # Scale with number of curves
            self.fig.set_size_inches(fig_size, fig_size)
            
            # Create heatmap
            ax = self.fig.add_subplot(111)
            
            # Use seaborn for better visualization if available
            try:
                import seaborn as sns
                # Use industry-standard correlation colormap
                correlation_cmap = PHYSICAL_CONSTANTS.COLORMAP_STANDARDS["correlation"]
                sns.heatmap(correlation_matrix, annot=True, cmap=correlation_cmap, center=0,
                           square=True, ax=ax, fmt='.2f', 
                           cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
            except ImportError:
                # Fallback to matplotlib with industry standards
                correlation_cmap = PHYSICAL_CONSTANTS.COLORMAP_STANDARDS["correlation"]
                im = ax.imshow(correlation_matrix, cmap=correlation_cmap, aspect='auto', vmin=-1, vmax=1)
                
                # Add professional colorbar
                cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
                
                # Add labels
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix.columns)))
                ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(correlation_matrix.columns)
                
                # Add correlation values as text
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        value = correlation_matrix.iloc[i, j]
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center')
            
            ax.set_title('Curve Correlation Matrix', fontsize=14, fontweight='bold')
            
            self.fig.tight_layout()
            
        except Exception as e:
            messagebox.showerror("Correlation Matrix Error", f"Failed to create correlation matrix: {str(e)}")

    # ============================================================================
    # ENHANCED GRAPHING FOR UNPROCESSED CURVES
    # ============================================================================
    
    def plot_unprocessed_curves(self, curve_names=None):
        """Plot curves that didn't get processed due to insufficient data or quality issues"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded for visualization")
            return
        
        try:
            # If no specific curves specified, plot all available curves
            if curve_names is None:
                curve_names = list(self.current_data.columns)
            
            # Filter out depth curves for plotting (they'll be used as Y-axis)
            depth_curves = []
            plot_curves = []
            for curve in curve_names:
                curve_type = self.curve_info.get(curve, {}).get('curve_type', '')
                if 'DEPTH' in curve_type:
                    depth_curves.append(curve)
                else:
                    plot_curves.append(curve)
            
            if not plot_curves:
                messagebox.showwarning("Warning", "No non-depth curves available for plotting")
                return
            
            # Use first available depth curve, or create index-based depth
            if depth_curves:
                depth_data = self.current_data[depth_curves[0]].values
                depth_unit = self.curve_info.get(depth_curves[0], {}).get('unit', 'm')
                y_label = f'Depth ({depth_unit})'
            else:
                depth_data = np.arange(len(self.current_data))
                y_label = 'Depth (index)'
            
            # Clean up previous visualization
            self.cleanup_visualization()
            self.ensure_figure_exists()
            
            # Create multi-track layout for better organization
            num_curves = len(plot_curves)
            num_tracks = min(3, (num_curves + 2) // 3)  # Maximum 3 tracks
            
            if num_tracks == 1:
                # Single track for 1-3 curves
                ax = self.fig.add_subplot(111)
                self._plot_unprocessed_curves_on_axis(ax, plot_curves, depth_data, y_label)
            else:
                # Multiple tracks for better organization
                axes = []
                for i in range(num_tracks):
                    if i == 0:
                        ax = self.fig.add_subplot(1, num_tracks, i+1)
                        axes.append(ax)
                    else:
                        ax = self.fig.add_subplot(1, num_tracks, i+1, sharey=axes[0])
                        axes.append(ax)
                
                # Distribute curves among tracks
                for i, track_ax in enumerate(axes):
                    start_idx = i * 3
                    end_idx = min((i + 1) * 3, num_curves)
                    track_curves = plot_curves[start_idx:end_idx]
                    
                    self._plot_unprocessed_curves_on_axis(track_ax, track_curves, depth_data, y_label)
                    
                    # Only show depth labels on first track
                    if i > 0:
                        track_ax.set_ylabel('')
                    
                    # Add track title
                    track_ax.set_title(f'Track {i+1}', fontsize=12, fontweight='bold')
            
            self.fig.suptitle('Unprocessed Curves Visualization', fontsize=16, fontweight='bold')
            self.fig.tight_layout()
            
            # Create canvas display for embedded visualization (same pattern as update_visualization)
            if hasattr(self, 'viz_content') and self.viz_content:
                # Clean up any existing widgets in viz_content
                for widget in self.viz_content.winfo_children():
                    widget.destroy()
                self.canvas = None
                
                # Create canvas using existing professional pattern
                self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
                self.canvas.draw()
                
                # Create navigation toolbar for professional interaction
                if NavigationToolbar2Tk:
                    toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                    toolbar.update()
                    toolbar.pack(side='top', fill='x')
                
                # Pack canvas below toolbar
                self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Unprocessed Curves Plot Error", f"Failed to create unprocessed curves plot: {str(e)}")
            self.log_processing(f"Error in plot_unprocessed_curves: {e}")

    def _plot_unprocessed_curves_on_axis(self, ax, curves, depth_data, y_label):
        """Helper function to plot unprocessed curves on a specific axis"""
        # Use industry-standard colors
        industry_colors = PHYSICAL_CONSTANTS.LOG_COLORS
        
        # Create twin axes for different scales
        twin_axes = []
        current_ax = ax
        
        for i, curve in enumerate(curves):
            if curve not in self.current_data.columns:
                continue
            
            curve_data = self.current_data[curve].values
            
            # CRITICAL: Convert null values to NaN for proper line breaking
            curve_data = self._convert_nulls_to_nan(curve_data)
            curve_type = self.curve_info.get(curve, {}).get('curve_type', 'UNKNOWN')
            curve_family = curve_type.split('_')[0] if '_' in curve_type else 'UNKNOWN'
            
            # Determine curve characteristics
            use_log_scale = curve_family in ['RESISTIVITY', 'PERMEABILITY']
            curve_quality = self.curve_info.get(curve, {}).get('quality', 'UNKNOWN')
            
            # Determine color and styling based on curve family and quality
            if curve_family in industry_colors:
                color = industry_colors[curve_family]
            else:
                color = plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)]
            
            # Adjust line style based on quality
            if curve_quality == 'Poor':
                line_style = '--'  # Dashed for poor quality
                line_width = 1.0
                alpha = 0.6
            elif curve_quality == 'Fair':
                line_style = '-.'  # Dash-dot for fair quality
                line_width = 1.2
                alpha = 0.8
            else:
                line_style = '-'   # Solid for good/excellent quality
                line_width = 1.5
                alpha = 1.0
            
            # Create twin axis if needed for different scales
            if i > 0 and use_log_scale != (current_ax.get_xscale() == 'log'):
                twin_ax = current_ax.twiny()
                twin_axes.append(twin_ax)
                current_ax = twin_ax
                current_ax.xaxis.set_ticks_position('top')
                current_ax.xaxis.set_label_position('top')
            else:
                current_ax = ax
            
            # Skip if entire curve is NaN
            if np.all(np.isnan(curve_data)):
                continue
            
            # Handle missing data and create valid data mask
            valid_mask = ~np.isnan(curve_data) & np.isfinite(curve_data)
            valid_data = curve_data[valid_mask]
            valid_depth = depth_data[valid_mask]
            
            if len(valid_data) > 0:
                # Set appropriate scale
                if use_log_scale:
                    # Handle zeros and negatives for log scale
                    positive_mask = valid_data > 0
                    if np.any(positive_mask):
                        log_data = valid_data[positive_mask]
                        log_depth = valid_depth[positive_mask]
                        current_ax.set_xscale('log')
                        
                        # Set reasonable log bounds
                        min_val = np.min(log_data)
                        max_val = np.max(log_data)
                        current_ax.set_xlim([max(0.1, min_val * 0.5), max_val * 2])
                        
                        # Plot with log scale
                        current_ax.plot(log_data, log_depth, color=color, linestyle=line_style,
                                      linewidth=line_width, alpha=alpha, label=f'{curve} (log)')
                    else:
                        # No positive values for log scale, use linear
                        current_ax.plot(valid_data, valid_depth, color=color, linestyle=line_style,
                                      linewidth=line_width, alpha=alpha, label=curve)
                else:
                    # Linear scale
                    current_ax.plot(valid_data, valid_depth, color=color, linestyle=line_style,
                                  linewidth=line_width, alpha=alpha, label=curve)
                
                # Add data quality annotation
                missing_percent = self.curve_info.get(curve, {}).get('missing_percent', 0)
                if missing_percent > 50:
                    # Add warning annotation for high missing data
                    mid_point = len(valid_data) // 2
                    if mid_point < len(valid_data):
                        current_ax.annotate(f'{missing_percent:.1f}% missing',
                                          xy=(valid_data[mid_point], valid_depth[mid_point]),
                                          xytext=(10, 10), textcoords=LABEL_OFFSET_POINTS,
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                          fontsize=8, color='black')
            
            # Reset current_ax to main axis for next iteration
            current_ax = ax
        
        # CRITICAL: Set axis limits to ACTUAL data range (not default range).
        self.apply_depth_axis(ax, depth_data, label=y_label)
        
        # Set axis properties
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        # Also get handles from twin axes
        for twin_ax in twin_axes:
            twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
            handles.extend(twin_handles)
            labels.extend(twin_labels)
        
        if handles:
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=min(3, len(handles)), fontsize=10)
        
        # Set x-axis label
        ax.set_xlabel('Curve Values')

    def plot_curve_quality_overview(self):
        """Create a comprehensive overview of all curves showing quality and processing status"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded for quality overview")
            return
        
        try:
            self.cleanup_visualization()
            self.ensure_figure_exists()
            
            # Set larger figure size for comprehensive quality overview
            self.fig.set_size_inches(16, 12)
            
            # Create a comprehensive quality overview
            axes = self.fig.subplots(2, 2)
            
            # Get all curve information
            curves = list(self.current_data.columns)
            quality_data = []
            missing_data = []
            curve_types = []
            processing_status = []
            
            for curve in curves:
                curve_info = self.curve_info.get(curve, {})
                quality_data.append(curve_info.get('quality', 'UNKNOWN'))
                missing_data.append(curve_info.get('missing_percent', 0))
                curve_types.append(curve_info.get('curve_type', 'UNKNOWN'))
                
                # Determine processing status
                if hasattr(self, 'processing_results') and curve in self.processing_results:
                    processing_status.append('Processed')
                else:
                    processing_status.append('Unprocessed')
            
            # Plot 1: Quality Distribution
            ax1 = axes[0, 0]
            quality_counts = {}
            for quality in quality_data:
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            if quality_counts:
                colors = ['red', 'orange', 'yellow', 'green']
                wedges, texts, autotexts = ax1.pie(quality_counts.values(), labels=quality_counts.keys(),
                                                   colors=colors[:len(quality_counts)], autopct='%1.1f%%')
                ax1.set_title('Data Quality Distribution', fontweight='bold')
            
            # Plot 2: Missing Data vs Quality
            ax2 = axes[0, 1]
            quality_colors = {'Poor': 'red', 'Fair': 'orange', 'Good': 'yellow', 'Excellent': 'green'}
            for quality in set(quality_data):
                if quality != 'UNKNOWN':
                    mask = [q == quality for q in quality_data]
                    ax2.scatter([missing_data[i] for i in range(len(missing_data)) if mask[i]],
                               [i for i in range(len(missing_data)) if mask[i]],
                               c=quality_colors.get(quality, 'gray'), label=quality, s=50, alpha=0.7)
            
            ax2.set_xlabel('Missing Data (%)')
            ax2.set_ylabel('Curve Index')
            ax2.set_title('Missing Data vs Quality', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Processing Status
            ax3 = axes[1, 0]
            status_counts = {}
            for status in processing_status:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                bars = ax3.bar(status_counts.keys(), status_counts.values(), 
                              color=['lightblue', 'lightcoral'], alpha=0.7)
                ax3.set_title('Processing Status', fontweight='bold')
                ax3.set_ylabel(LABEL_NUMBER_OF_CURVES)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
            
            # Plot 4: Curve Type Distribution
            ax4 = axes[1, 1]
            type_counts = {}
            for curve_type in curve_types:
                if curve_type != 'UNKNOWN':
                    main_type = curve_type.split('_')[0]
                    type_counts[main_type] = type_counts.get(main_type, 0) + 1
            
            if type_counts:
                # Sort by count for better visualization
                sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                types, counts = zip(*sorted_types)
                
                bars = ax4.bar(range(len(types)), counts, color='lightgreen', alpha=0.7)
                ax4.set_title('Curve Type Distribution', fontweight='bold')
                ax4.set_ylabel(LABEL_NUMBER_OF_CURVES)
                ax4.set_xticks(range(len(types)))
                ax4.set_xticklabels(types, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
            
            # Apply proper spacing for quality overview display
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.94, bottom=0.10, wspace=0.25, hspace=0.35)
            
            # Create canvas display for embedded visualization
            if hasattr(self, 'viz_content') and self.viz_content:
                # Clean up any existing widgets in viz_content
                for widget in self.viz_content.winfo_children():
                    widget.destroy()
                self.canvas = None
                
                # Create canvas using existing professional pattern
                self.canvas = FigureCanvasTkAgg(self.fig, self.viz_content)
                self.canvas.draw()
                
                # Create navigation toolbar for professional interaction
                if NavigationToolbar2Tk:
                    toolbar = NavigationToolbar2Tk(self.canvas, self.viz_content)
                    toolbar.update()
                    toolbar.pack(side='top', fill='x')
                
                # Pack canvas below toolbar
                self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Quality Overview Error", f"Failed to create quality overview: {str(e)}")
            self.log_processing(f"Error in plot_curve_quality_overview: {e}")

    def plot_curve_comparison_all(self):
        """Plot all curves for comparison, including unprocessed ones
        
        DEPRECATED: This visualization has been replaced by 'multi_curve' (for selected curves)
        and 'quality_overview' (for comprehensive analysis). This function is kept for backward
        compatibility but will show a deprecation message.
        """
        messagebox.showinfo("Visualization Update", 
                          "The 'curve_comparison_all' visualization has been replaced.\n\n"
                          "Please use:\n"
                          "- 'multi_curve' for selected curves in organized tracks\n"
                          "- 'quality_overview' for comprehensive quality analysis\n\n"
                          "This function is deprecated and will be removed in a future version.")
        return

    # ============================================================================
    # NEW SINGLE CURVE VISUALIZATION METHODS
    # ============================================================================
    
    def plot_single_curve(self, curve: str):
        """Create a large, detailed view of a single curve with original vs processed comparison.
        
        Features:
        - Large figure (12x10) for excellent readability
        - Depth-based plotting (industry standard)
        - Original vs Processed comparison
        - Statistical annotations
        - Gap indicators
        - Quality metrics overlay
        - Processing status indication
        """
        try:
            # Check if curve exists in original data
            if self.current_data is None or curve not in self.current_data.columns:
                messagebox.showwarning("Warning", f"Curve '{curve}' not found in data")
                return
            
            # Clean up and create large figure
            self.cleanup_visualization()
            self.ensure_figure_exists()
            self.fig.set_size_inches(12, 10)
            
            # Create main axis
            ax = self.fig.add_subplot(111)
            
            # Get depth data
            depth_curve = None
            for col in self.current_data.columns:
                curve_type = self.curve_info.get(col, {}).get('curve_type', '')
                if 'DEPTH' in curve_type:
                    depth_curve = col
                    break
            
            if depth_curve:
                depth = self.current_data[depth_curve].values
                depth_unit = self.curve_info.get(depth_curve, {}).get('unit', 'm')
                y_label = f'Depth ({depth_unit})'
            else:
                depth = np.arange(len(self.current_data))
                y_label = 'Sample Index'
            
            # Plot original data
            original_data = self.current_data[curve].values
            ax.plot(original_data, depth, color='red', linewidth=1.5, 
                   alpha=0.7, label='Original', linestyle='-')
            
            # Plot processed data if available
            if (self.processed_data is not None and 
                curve in self.processed_data.columns):
                processed_data = self.processed_data[curve].values
                ax.plot(processed_data, depth, color='blue', linewidth=2.0, 
                       label='Processed', linestyle='-')
                
                # Add processing quality info if available
                if curve in self.processing_results:
                    quality = self.processing_results[curve].get('quality_score', 0)
                    methods = self.processing_results[curve].get('methods_used', [])
                    status = f'Processed (Quality: {quality:.2f})'
                    if methods:
                        status += f' - Methods: {", ".join(methods[:3])}'
                else:
                    status = 'Processed'
            else:
                status = 'Original (Not Yet Processed)'
            
            # Set proper axis limits based on data range
            all_data = [original_data]
            if (self.processed_data is not None and curve in self.processed_data.columns):
                all_data.append(self.processed_data[curve].values)
            
            combined_data = np.concatenate([d[~np.isnan(d)] for d in all_data])
            if len(combined_data) > 0:
                data_min, data_max = np.min(combined_data), np.max(combined_data)
                data_range = data_max - data_min
                if data_range > 0:
                    padding = data_range * 0.05
                    ax.set_xlim(data_min - padding, data_max + padding)
                else:
                    ax.set_xlim(data_min - 1, data_min + 1)
            
            # Highlight gaps in original data
            gap_mask = np.isnan(original_data)
            if np.any(gap_mask):
                gap_indices = np.where(gap_mask)[0]
                if len(gap_indices) > 0:
                    ax.scatter(np.full(len(gap_indices), data_min if len(combined_data) > 0 else 0), 
                             depth[gap_indices], color='orange', s=10, alpha=0.5, 
                             label=LABEL_MISSING_DATA, zorder=1)
            
            # Set title and labels
            curve_info = self.curve_info.get(curve, {})
            curve_type = curve_info.get('curve_type', 'UNKNOWN')
            unit = curve_info.get('unit', '')
            
            ax.set_title(f'{curve} - {curve_type}\nOriginal vs Processed Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{curve} ({unit})', fontsize=12)
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            self.apply_depth_axis(ax, depth, label=y_label)
            
            # Add processing statistics if available
            if curve in self.processing_results:
                result = self.processing_results[curve]
                quality = result.get('quality_score', 0)
                methods = result.get('methods_used', [])
                
                stats_text = f'Quality Score: {quality:.2f}'
                if methods:
                    stats_text += f'\nMethods: {", ".join(methods[:2])}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightblue', alpha=0.8), fontsize=9)
            
            # Update display
            self.canvas.draw()
            self.log_processing(f"Single curve visualization updated: {curve} - {status}")
            
        except Exception as e:
            self.log_processing(f"Error in single curve visualization: {e}")
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, f"Failed to create single curve plot:\n{e}")
    
    def plot_single_curve_comparison(self):
        """Create a side-by-side comparison of two single curves for easy visual comparison.
        
        Features:
        - Two panels side-by-side (14x8 figure)
        - Independent axis scaling per curve
        - Aligned depth axes for easy correlation
        - Statistical overlays on both curves
        - Clear labeling and professional appearance
        """
        try:
            # Get both curves
            curve1 = self.viz_curve_var.get()
            curve2 = self.viz_curve2_var.get()
            
            if not curve1 or curve1 not in self.current_data.columns:
                messagebox.showwarning("Warning", "Please select a valid primary curve")
                return
            
            if not curve2 or curve2 not in self.current_data.columns:
                messagebox.showwarning("Warning", "Please select a valid secondary curve for comparison")
                return
            
            # Clean up and create figure
            self.cleanup_visualization()
            self.ensure_figure_exists()
            self.fig.set_size_inches(14, 9)
            
            # Create two side-by-side subplots with shared Y-axis
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122, sharey=ax1)
            
            # Get depth data
            depth_curve = None
            for col in self.current_data.columns:
                curve_type = self.curve_info.get(col, {}).get('curve_type', '')
                if 'DEPTH' in curve_type:
                    depth_curve = col
                    break
            
            if depth_curve:
                depth = self.current_data[depth_curve].values
                depth_unit = self.curve_info.get(depth_curve, {}).get('unit', 'm')
                y_label = f'Depth ({depth_unit})'
            else:
                depth = np.arange(len(self.current_data))
                y_label = 'Depth (index)'
            
            # Plot Curve 1
            if curve1 in self.processing_results:
                data1 = self.processing_results[curve1]['final_data']
                status1 = 'Processed'
                color1 = 'blue'
            else:
                data1 = self.current_data[curve1].values
                status1 = 'Original'
                color1 = 'red'
            
            ax1.plot(data1, depth, color=color1, linewidth=2, label=status1)
            ax1.set_title(f'{curve1}\n({status1})', fontsize=12, fontweight='bold')
            ax1.set_xlabel(f'{curve1} ({self.curve_info.get(curve1, {}).get("unit", "")})', fontsize=11)
            ax1.grid(True, alpha=0.3)
            self.apply_depth_axis(ax1, depth, label=y_label)
            ax1.legend(loc='best')
            
            # Add statistics for curve 1
            valid1 = data1[~np.isnan(data1)]
            if len(valid1) > 0:
                stats1_text = (
                    f"Min: {np.min(valid1):.2f}\n"
                    f"Max: {np.max(valid1):.2f}\n"
                    f"Mean: {np.mean(valid1):.2f}\n"
                    f"Missing: {(np.sum(np.isnan(data1))/len(data1)*100):.1f}%"
                )
                ax1.text(0.02, 0.98, stats1_text, transform=ax1.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
            
            # Plot Curve 2
            if curve2 in self.processing_results:
                data2 = self.processing_results[curve2]['final_data']
                status2 = 'Processed'
                color2 = 'blue'
            else:
                data2 = self.current_data[curve2].values
                status2 = 'Original'
                color2 = 'red'
            
            ax2.plot(data2, depth, color=color2, linewidth=2, label=status2)
            ax2.set_title(f'{curve2}\n({status2})', fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'{curve2} ({self.curve_info.get(curve2, {}).get("unit", "")})', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
            
            # Add statistics for curve 2
            valid2 = data2[~np.isnan(data2)]
            if len(valid2) > 0:
                stats2_text = (
                    f"Min: {np.min(valid2):.2f}\n"
                    f"Max: {np.max(valid2):.2f}\n"
                    f"Mean: {np.mean(valid2):.2f}\n"
                    f"Missing: {(np.sum(np.isnan(data2))/len(data2)*100):.1f}%"
                )
                ax2.text(0.02, 0.98, stats2_text, transform=ax2.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))
            
            # Overall title
            self.fig.suptitle(f'Side-by-Side Comparison: {curve1} vs {curve2}', 
                            fontsize=14, fontweight='bold')
            
            # Apply proper spacing
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.93, wspace=0.30)
            
        except Exception as e:
            messagebox.showerror("Single Curve Comparison Error", f"Failed to create comparison plot: {str(e)}")
            self.log_processing(f"Error in plot_single_curve_comparison: {e}")
    
    # ============================================================================
    # POPUP VISUALIZATION SYSTEM (Professional Workflow)
    # ============================================================================
    
    def _create_popup_visualization(self, viz_type, curve):
        """Open visualization in separate Toplevel window with proper memory management.
        
        Professional workflow: Separate windows allow resizing, zooming, dual monitors,
        and keeping multiple plots open simultaneously - industry standard practice.
        
        PROPER IMPLEMENTATION: Uses Toplevel windows with embedded FigureCanvasTkAgg
        instead of plt.show(block=False) to prevent memory leaks and event loop conflicts.
        """
        try:
            # Validate data is available
            if self.processed_data is None and self.current_data is None:
                messagebox.showwarning("No Data", "Please load and process data before creating visualizations.")
                return
            
            # Validate curve exists if needed
            if curve:
                data_available = False
                if self.processed_data is not None and curve in self.processed_data.columns:
                    data_available = True
                elif self.current_data is not None and curve in self.current_data.columns:
                    data_available = True
                
                if not data_available:
                    messagebox.showwarning("Curve Not Found", 
                                         f"Selected curve '{curve}' not found in available data.")
                    return
            
            # Determine appropriate figure size for viz type
            size_map = {
                'single_curve': (12, 10),
                'single_curve_comparison': (14, 9),
                'comparison': (12, 9),
                'log_display': (18, 10),
                'quality_overview': (16, 12),
                'quality_metrics': (14, 10),
                'correlation_matrix': (12, 12),
                'scatter_plot': (12, 10),
                '3d_visualization': (12, 10),
                'multi_curve': (16, 10),
                'unprocessed_curves': (14, 10),
                'histogram': (12, 9),
                'uncertainty': (12, 9)
            }
            
            figsize = size_map.get(viz_type, (12, 9))
            
            # Create Toplevel window (proper approach - no event loop conflicts)
            popup = tk.Toplevel(self.root)
            popup.title(f"{viz_type.replace('_', ' ').title()} - {curve if curve else 'Multiple Curves'}")
            
            # Add well identification to window title for safety
            if hasattr(self, 'well_info') and self.well_info:
                well_name = self.well_info.get('well_name', '')
                if well_name and well_name != 'UNKNOWN':
                    current_title = popup.title()
                    popup.title(f"{current_title} (Well: {well_name})")
            
            # Set window size based on figure size (convert inches to pixels roughly)
            window_width = int(figsize[0] * 80)
            window_height = int(figsize[1] * 80) + 100  # Extra for toolbar
            popup.geometry(f"{window_width}x{window_height}")
            
            # Create matplotlib figure (NOT using plt.figure - use Figure class)
            from matplotlib.figure import Figure
            fig = Figure(figsize=figsize, dpi=100)
            fig.patch.set_facecolor('white')

            # Route to appropriate plotting method on the figure
            if viz_type == "single_curve":
                self._plot_single_curve_popup(fig, curve)
            elif viz_type == "single_curve_comparison":
                self._plot_single_curve_comparison_popup(fig)
            elif viz_type == "comparison":
                self._plot_comparison_popup(fig, curve)
            elif viz_type == "multi_curve":
                self._plot_multi_curve_popup(fig)
            elif viz_type == "log_display":
                self._plot_log_display_popup(fig)
            elif viz_type == "quality_overview":
                self._plot_quality_overview_popup(fig)
            elif viz_type == "unprocessed_curves":
                self._plot_unprocessed_curves_popup(fig)
            elif viz_type == "correlation_matrix":
                self._plot_correlation_matrix_popup(fig)
            elif viz_type == "scatter_plot":
                self._plot_scatter_plot_popup(fig, curve)
            elif viz_type == "3d_visualization":
                self._plot_3d_visualization_popup(fig, curve)
            elif viz_type == "quality_metrics":
                self._plot_quality_metrics_popup(fig)
            elif viz_type == "uncertainty":
                self._plot_uncertainty_popup(fig, curve)
            elif viz_type == "histogram":
                self._plot_histogram_popup(fig, curve)
            else:
                # For other types, show message
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Popup visualization for '{viz_type}' not yet implemented.\nUse embedded mode.",
                       ha='center', va='center', fontsize=12)

            # Embed figure in Toplevel window using FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            
            # Add matplotlib navigation toolbar
            toolbar_frame = ttk.Frame(popup)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Pack canvas
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Register popup for cleanup tracking
            self.popup_windows.append(popup)
            self.popup_figures.append(fig)
            
            # Add close callback for proper cleanup
            def on_close():
                try:
                    # Remove from registries
                    if popup in self.popup_windows:
                        self.popup_windows.remove(popup)
                    if fig in self.popup_figures:
                        self.popup_figures.remove(fig)
                    # Clean up canvas and toolbar
                    try:
                        toolbar.destroy()
                    except (tk.TclError, AttributeError) as toolbar_error:
                        # Toolbar may already be destroyed or accessed incorrectly
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Toolbar cleanup failed: {type(toolbar_error).__name__}: {str(toolbar_error)}")
                    except Exception as toolbar_error:
                        # Unexpected error - log for debugging
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Unexpected error cleaning up toolbar: {type(toolbar_error).__name__}: {str(toolbar_error)}")
                    
                    try:
                        canvas.get_tk_widget().destroy()
                    except (tk.TclError, AttributeError) as canvas_error:
                        # Canvas widget may already be destroyed or accessed incorrectly
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Canvas cleanup failed: {type(canvas_error).__name__}: {str(canvas_error)}")
                    except Exception as canvas_error:
                        # Unexpected error - log for debugging
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Unexpected error cleaning up canvas: {type(canvas_error).__name__}: {str(canvas_error)}")
                    
                    # Close figure properly (import plt here to ensure availability)
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except (AttributeError, ImportError) as plt_error:
                        # Matplotlib may not be available or figure already closed
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Figure cleanup failed: {type(plt_error).__name__}: {str(plt_error)}")
                    except Exception as plt_error:
                        # Unexpected error - log for debugging
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Unexpected error closing figure: {type(plt_error).__name__}: {str(plt_error)}")
                    
                    # Destroy window
                    popup.destroy()
                    # Garbage collection
                    gc.collect()
                    self.log_processing(f"Closed popup visualization: {viz_type}")
                except Exception as cleanup_error:
                    self.log_processing(f"Warning during popup cleanup: {type(cleanup_error).__name__}: {str(cleanup_error)}")
                    # Force destroy even if cleanup fails
                    try:
                        popup.destroy()
                    except (tk.TclError, AttributeError) as destroy_error:
                        # Window may already be destroyed
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Popup window destruction failed: {type(destroy_error).__name__}: {str(destroy_error)}")
                    except Exception as destroy_error:
                        # Unexpected error - log for debugging
                        if hasattr(self, 'log_processing'):
                            self.log_processing(f"Warning: Unexpected error destroying popup window: {type(destroy_error).__name__}: {str(destroy_error)}")
            
            popup.protocol("WM_DELETE_WINDOW", on_close)
            
            self.log_processing(f"Opened {viz_type} visualization in Toplevel window (proper memory management)")
            
        except Exception as e:
            # Build the message now and pass it by default argument. Python
            # unbinds `e` when the except block exits, while root.after defers
            # the callback into the event loop, so a lambda closing over `e`
            # would raise NameError there and the dialog would never appear.
            error_message = (
                f"Failed to open visualization in new window:\n{str(e)}\n\n"
                f"Details: Check that data is loaded and processed.\n"
                f"Try unchecking 'Open in new window' to use embedded mode."
            )
            try:
                self.root.after(0, lambda msg=error_message: messagebox.showerror(
                    "Popup Visualization Error", msg))
            except Exception:
                messagebox.showerror("Popup Visualization Error", error_message)
            self.log_processing(f"Error creating popup visualization: {e}")
            import traceback
            self.log_processing(f"Traceback: {traceback.format_exc()}")
    
    # Simplified popup plotting methods (delegate to matplotlib's popup system)
    def _plot_single_curve_popup(self, fig, curve):
        """Plot single curve in popup window with original vs processed comparison"""
        ax = fig.add_subplot(111)
        
        # Check if curve exists in original data
        if self.current_data is None or curve not in self.current_data.columns:
            ax.text(0.5, 0.5, f"Curve '{curve}' not found in data", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Each trace is drawn against the depth channel of its own frame, because
        # resampling can leave processed_data on a different grid to current_data.
        original_depth = self._get_depth_for_frame(self.current_data)
        
        # Plot original data
        original_data = self.current_data[curve].values

        ax.plot(original_data, original_depth, color='red', linewidth=1.5, 
               alpha=0.7, label='Original', linestyle='-')
        
        # Plot processed data if available
        if (self.processed_data is not None and curve in self.processed_data.columns):
            processed_depth = self._get_depth_for_frame(self.processed_data)
            processed_data = self.processed_data[curve].values

            ax.plot(processed_data, processed_depth, color='blue', linewidth=2.0, 
                   label='Processed', linestyle='-')
            
            # Add processing quality info if available
            if curve in self.processing_results:
                quality = self.processing_results[curve].get('quality_score', 0)
                methods = self.processing_results[curve].get('methods_used', [])
                status = f'Processed (Quality: {quality:.2f})'
                if methods:
                    status += f' - Methods: {", ".join(methods[:3])}'
            else:
                status = 'Processed'
        else:
            status = 'Original (Not Yet Processed)'
        
        # Set proper axis limits based on data range
        all_data = [original_data]
        if (self.processed_data is not None and curve in self.processed_data.columns):
            all_data.append(self.processed_data[curve].values)
        
        combined_data = np.concatenate([d[~np.isnan(d)] for d in all_data])
        if len(combined_data) > 0:
            data_min, data_max = np.min(combined_data), np.max(combined_data)
            data_range = data_max - data_min
            if data_range > 0:
                padding = data_range * 0.05
                ax.set_xlim(data_min - padding, data_max + padding)
            else:
                ax.set_xlim(data_min - 1, data_min + 1)
        
        # Highlight gaps in original data
        gap_mask = np.isnan(original_data)
        if np.any(gap_mask):
            gap_indices = np.where(gap_mask)[0]
            if len(gap_indices) > 0:
                data_min = np.min(combined_data) if len(combined_data) > 0 else 0

                # Gaps are detected in the original trace, so they are marked
                # against the original frame's depth.
                ax.scatter(np.full(len(gap_indices), data_min), 
                         original_depth[gap_indices], color='orange', s=10, alpha=0.5, 
                         label=LABEL_MISSING_DATA, zorder=1)
        
        # Set title and labels
        curve_info = self.curve_info.get(curve, {})
        curve_type = curve_info.get('curve_type', 'UNKNOWN')
        unit = curve_info.get('unit', '')
        
        ax.set_title(f'{curve} - {curve_type}\nOriginal vs Processed Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{curve} ({unit})', fontsize=12)

        # Add legend
        ax.legend(loc='best', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Span every frame that contributed a trace so a resampled processed
        # grid cannot clip the original (or the reverse).
        depth_for_axis = original_depth
        if (self.processed_data is not None and curve in self.processed_data.columns):
            depth_for_axis = np.concatenate([
                original_depth, self._get_depth_for_frame(self.processed_data)])
        self.apply_depth_axis(ax, depth_for_axis, label=LABEL_DEPTH_M)
        
        # Add processing statistics if available
        if curve in self.processing_results:
            result = self.processing_results[curve]
            quality = result.get('quality_score', 0)
            methods = result.get('methods_used', [])
            
            stats_text = f'Quality Score: {quality:.2f}'
            if methods:
                stats_text += f'\nMethods: {", ".join(methods[:2])}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='lightblue', alpha=0.8), fontsize=9)
        
        fig.tight_layout()
    
    def _plot_single_curve_comparison_popup(self, fig):
        """Plot side-by-side comparison in popup"""
        curve1 = self.viz_curve_var.get()
        curve2 = self.viz_curve2_var.get()
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)
        
        # processing_results arrays are produced from processed_data and share its
        # grid, while the fallback reads current_data. Depth follows the source
        # actually used, since the two frames can differ after resampling.
        processed_depth = self._get_depth_for_frame(self.processed_data)
        original_depth = self._get_depth_for_frame(self.current_data)
        
        # Plot curve 1
        if curve1 in self.processing_results and 'final_data' in self.processing_results[curve1]:
            data1, depth1 = self.processing_results[curve1]['final_data'], processed_depth
        else:
            data1, depth1 = self.current_data[curve1].values, original_depth
        ax1.plot(data1, depth1, 'b-', linewidth=2)
        ax1.set_title(curve1, fontsize=12, fontweight='bold')
        ax1.set_xlabel(f"{curve1}")
        ax1.grid(True, alpha=0.3)
        
        # Plot curve 2
        if curve2 in self.processing_results and 'final_data' in self.processing_results[curve2]:
            data2, depth2 = self.processing_results[curve2]['final_data'], processed_depth
        else:
            data2, depth2 = self.current_data[curve2].values, original_depth
        ax2.plot(data2, depth2, 'r-', linewidth=2)
        ax2.set_title(curve2, fontsize=12, fontweight='bold')
        ax2.set_xlabel(f"{curve2}")
        ax2.grid(True, alpha=0.3)

        # sharey=ax1 means explicit ylim from depth1 alone would freeze ax2 as
        # well, silently truncating a coarser/shorter/offset second grid.
        # Match _plot_single_curve_popup: span every frame that contributed a
        # trace before apply_depth_axis. Limits still go on ax1 only so the
        # shared axis stays inverted via set_ylim, not invert_yaxis.
        depth_for_axis = np.concatenate([
            np.asarray(depth1, dtype=float),
            np.asarray(depth2, dtype=float),
        ])
        self.apply_depth_axis(ax1, depth_for_axis, label=LABEL_DEPTH_M)
        
        fig.suptitle(f"Comparison: {curve1} vs {curve2}", fontsize=14, fontweight='bold')
        fig.tight_layout()
    
    def _plot_comparison_popup(self, fig, curve):
        """Plot original vs processed comparison in popup"""
        ax = fig.add_subplot(111)
        
        if curve in self.processing_results:
            # Both arrays were captured from processed_data and share its grid.
            depth = self._get_depth_for_frame(self.processed_data)
            # Null sentinels are converted to NaN so matplotlib breaks the line at
            # gaps rather than drawing a spike to -999.25, which would also drag
            # the value-axis autoscale far outside the real measurement range.
            # to_numeric coerces non-numeric entries to NaN rather than raising,
            # and yields the float dtype _convert_nulls_to_nan needs to assign NaN.
            # The Series wrapper is required because to_numeric returns a bare
            # ndarray for ndarray input, which has no to_numpy method.
            original = self._convert_nulls_to_nan(
                pd.to_numeric(pd.Series(self.processing_results[curve]['original_data']),
                              errors='coerce').to_numpy(dtype=float))
            processed = self._convert_nulls_to_nan(
                pd.to_numeric(pd.Series(self.processing_results[curve]['final_data']),
                              errors='coerce').to_numpy(dtype=float))

            ax.plot(original, depth, 'r-', alpha=0.7, label='Original', linewidth=1)
            ax.plot(processed, depth, 'b-', alpha=0.9, label='Processed', linewidth=2)
        else:
            depth = self._get_depth_for_frame(self.current_data)
            data = self._convert_nulls_to_nan(
                pd.to_numeric(self.current_data[curve],
                              errors='coerce').to_numpy(dtype=float))
            ax.plot(data, depth, 'r-', label='Original', linewidth=1.5)

        # The depth axis spans the full grid of the frame being plotted. Without
        # this, autoscale collapses onto the interval where the curve happens to
        # hold finite values, hiding where that interval sits in the well.
        self.apply_depth_axis(ax, depth, label=LABEL_DEPTH_M)

        ax.set_title(f"Comparison: {curve}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{curve}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
    
    def _plot_multi_curve_popup(self, fig):
        """Plot multiple curves in popup window"""
        selected_indices = self.curve_listbox.curselection()
        if not selected_indices:
            return
        
        selected_curves = [self.curve_listbox.get(i) for i in selected_indices]
        # Values are read from current_data below, so depth comes from the same frame.
        depth = self._get_depth_for_frame(self.current_data)
        
        ax = fig.add_subplot(111)
        for i, curve in enumerate(selected_curves[:10]):  # Limit to 10 curves
            if curve in self.current_data.columns:
                data = self.current_data[curve].values
                ax.plot(data, depth, label=curve, linewidth=1.5, alpha=0.8)
        
        ax.set_title("Multi-Curve Display", fontsize=14, fontweight='bold')
        self.apply_depth_axis(ax, depth, label=LABEL_DEPTH_M)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=LABEL_UPPER_LEFT)
        fig.tight_layout()
    
    def _plot_log_display_popup(self, fig):
        """Plot industry log display in popup"""
        # Similar to embedded but uses popup fig
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Log display in popup - use embedded mode for full features",
               ha='center', va='center')
        fig.tight_layout()
    
    def _plot_quality_overview_popup(self, fig):
        """Plot quality overview in popup"""
        # Similar to embedded but uses popup fig
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Quality overview - use embedded mode for full dashboard",
               ha='center', va='center')
        fig.tight_layout()
    
    def _plot_unprocessed_curves_popup(self, fig):
        """Plot unprocessed curves in popup with proper null handling and depth range"""
        ax = fig.add_subplot(111)
        
        # Get depth array and actual range. Every curve below is read from
        # current_data, so depth is taken from that same frame.
        depth = self._get_depth_for_frame(self.current_data)
        depth_min, depth_max = self._get_depth_limits(depth)
        
        # Get depth column name for detection
        depth_column = None
        for col in self.current_data.columns:
            curve_type = self.curve_info.get(col, {}).get('curve_type', '')
            if 'DEPTH' in curve_type.upper() or 'DEPT' in col.upper():
                depth_column = col
                break
        
        # Determine depth unit from data or info
        depth_unit = 'ft'
        if depth_column and depth_column in self.curve_info:
            unit_info = self.curve_info[depth_column].get('unit', '').upper()
            if 'M' in unit_info or 'MET' in unit_info:
                depth_unit = 'm'
        
        # Count curves and gaps
        total_curves = 0
        curves_plotted = 0
        
        # Plot each curve (limit to first 10 for readability)
        for curve in list(self.current_data.columns)[:10]:
            if curve == depth_column:
                continue
            
            if curve not in self.current_data.columns:
                continue
            
            total_curves += 1
            curve_data = self.current_data[curve].values
            
            # CRITICAL: Convert null values to NaN for proper line breaking
            # This creates gaps where data is missing instead of drawing lines
            if hasattr(self, '_convert_nulls_to_nan'):
                curve_data = self._convert_nulls_to_nan(curve_data)
            else:
                # Fallback: replace common null values with NaN
                null_value = -999.25
                curve_data = np.where(curve_data == null_value, np.nan, curve_data)
            
            # Skip if entire curve is NaN
            if np.all(np.isnan(curve_data)):
                continue
            
            # Plot with proper NaN handling (matplotlib breaks lines at NaN)
            ax.plot(curve_data, depth, label=curve, alpha=0.7, linewidth=1.0)
            curves_plotted += 1
        
        # CRITICAL: Set axis limits to ACTUAL data range (not 0-5000 default)
        self.apply_depth_axis(ax, depth, label=f'Depth ({depth_unit})')
        
        # Labels and formatting
        ax.set_xlabel('Curve Values', fontsize=12)
        ax.set_title("Unprocessed Curves - Gaps Indicate Missing Data", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        
        # Legend
        if curves_plotted > 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        # Add info text
        null_value = -999.25
        info_text = (
            f"Depth Range: {depth_min:.1f} - {depth_max:.1f} {depth_unit}\n"
            f"Total Depth Points: {len(depth)}\n"
            f"Curves Displayed: {curves_plotted}/{total_curves}\n"
            f"Null Value: {null_value} (shown as gaps)"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
    
    def _plot_correlation_matrix_popup(self, fig):
        """Plot correlation matrix with professional styling"""
        ax = fig.add_subplot(111)
        
        if self.processed_data is None or len(self.processed_data.columns) < 2:
            ax.text(0.5, 0.5, "Need at least 2 curves for correlation matrix", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Calculate correlation matrix
        numeric_data = self.processed_data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Use seaborn for professional heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8},
                   fmt='.2f', annot_kws={'size': 8})
        
        ax.set_title('Curve Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        fig.tight_layout()
    
    def _plot_scatter_plot_popup(self, fig, curve):
        """Plot scatter plot with proper industry styling"""
        ax = fig.add_subplot(111)
        
        if self.processed_data is None or curve not in self.processed_data.columns:
            ax.text(0.5, 0.5, f"Curve '{curve}' not available for scatter plot", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get depth and curve data
        depth = self._get_depth_array()
        data = self.processed_data[curve].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(data) | np.isnan(depth))
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, "No valid data points for scatter plot", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Create scatter plot with color gradient by depth (industry standard)
        scatter = ax.scatter(data[valid_mask], depth[valid_mask], 
                           c=depth[valid_mask], cmap='viridis', 
                           alpha=0.7, s=20, edgecolors='none')
        
        # Add colorbar for depth reference. Use the figure's own method rather
        # than plt.colorbar: this figure is a bare Figure that pyplot does not
        # track, so plt.colorbar would fall back to gcf() and create a stray
        # pyplot figure that is never released.
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(LABEL_DEPTH_M, rotation=270, labelpad=20)
        
        # Set proper axis limits
        data_min, data_max = np.nanmin(data[valid_mask]), np.nanmax(data[valid_mask])
        if data_max > data_min:
            padding = (data_max - data_min) * 0.05
            ax.set_xlim(data_min - padding, data_max + padding)
        
        ax.set_xlabel(f"{curve} ({self.curve_info.get(curve, {}).get('unit', '')})")
        ax.set_title(f'{curve} vs Depth Scatter Plot', fontsize=14, fontweight='bold')
        self.apply_depth_axis(ax, depth, label=LABEL_DEPTH_M)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    
    def _plot_3d_visualization_popup(self, fig, curve):
        """Plot 3D visualization (depth vs curve vs another curve)"""
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '3D toolkit unavailable (matplotlib.mplot3d). Use embedded 2D plots.',
                    ha='center', va='center')
            ax.axis('off')
            fig.tight_layout()
            return
        
        ax = fig.add_subplot(111, projection='3d')
        
        if self.processed_data is None or len(self.processed_data.columns) < 2:
            ax.text(0.5, 0.5, 0.5, "Need at least 2 curves for 3D visualization", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get first two numeric curves for 3D plot
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, 0.5, "Need numeric curves for 3D plot", 
                   ha='center', va='center', fontsize=12)
            return
        
        curve1, curve2 = numeric_cols[0], numeric_cols[1]
        if curve in numeric_cols:
            curve1 = curve
        
        depth = self._get_depth_array()
        data1 = self.processed_data[curve1].values
        data2 = self.processed_data[curve2].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(data1) | np.isnan(data2) | np.isnan(depth))
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 0.5, "No valid data for 3D plot", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Create 3D scatter plot
        ax.scatter(data1[valid_mask], data2[valid_mask], depth[valid_mask],
                  c=depth[valid_mask], cmap='viridis', alpha=0.7, s=10)
        
        ax.set_xlabel(f"{curve1} ({self.curve_info.get(curve1, {}).get('unit', '')})")
        ax.set_ylabel(f"{curve2} ({self.curve_info.get(curve2, {}).get('unit', '')})")
        ax.set_zlabel(LABEL_DEPTH_M)
        ax.set_title(f'3D Plot: {curve1} vs {curve2} vs Depth', fontsize=14, fontweight='bold')
        ax.invert_zaxis()  # Industry standard: depth downward
        fig.tight_layout()
    
    def _plot_quality_metrics_popup(self, fig):
        """Plot quality metrics dashboard"""
        if not hasattr(self, 'processing_results') or not self.processing_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No processing results available for quality metrics", 
                   ha='center', va='center', fontsize=12)
            return
        
        # Create subplots for different quality metrics
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Quality scores bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        curves = list(self.processing_results.keys())
        scores = [self.processing_results[curve].get('quality_score', 0) for curve in curves]
        
        bars = ax1.bar(range(len(curves)), scores, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(curves)))
        ax1.set_xticklabels(curves, rotation=45, ha='right')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Processing Quality Scores', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Processing methods used
        ax2 = fig.add_subplot(gs[0, 1])
        methods_used = {}
        for curve, result in self.processing_results.items():
            for method in result.get('methods_used', []):
                methods_used[method] = methods_used.get(method, 0) + 1
        
        if methods_used:
            method_names = list(methods_used.keys())
            method_counts = list(methods_used.values())
            ax2.pie(method_counts, labels=method_names, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Processing Methods Used', fontweight='bold')
        
        # Data completeness
        ax3 = fig.add_subplot(gs[1, :])
        completeness = []
        for curve in curves:
            if curve in self.processed_data.columns:
                total_points = len(self.processed_data[curve])
                valid_points = self.processed_data[curve].count()
                completeness.append(valid_points / total_points * 100)
            else:
                completeness.append(0)
        
        bars = ax3.bar(range(len(curves)), completeness, color='forestgreen', alpha=0.7)
        ax3.set_xticks(range(len(curves)))
        ax3.set_xticklabels(curves, rotation=45, ha='right')
        ax3.set_ylabel('Data Completeness (%)')
        ax3.set_title('Data Completeness by Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, comp in zip(bars, completeness):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{comp:.1f}%', ha='center', va='bottom', fontsize=8)
        
        fig.suptitle('Quality Metrics Dashboard', fontsize=16, fontweight='bold')
        fig.tight_layout()
    
    def _plot_uncertainty_popup(self, fig, curve):
        """Plot uncertainty analysis for a curve"""
        ax = fig.add_subplot(111)
        
        if (not hasattr(self, 'processing_results') or 
            not self.processing_results or 
            curve not in self.processing_results):
            ax.text(0.5, 0.5, f"No uncertainty data available for '{curve}'", 
                   ha='center', va='center', fontsize=12)
            return
        
        result = self.processing_results[curve]
        # The processed trace and its uncertainty band come from processing_results,
        # which shares the processed_data grid.
        depth = self._get_depth_for_frame(self.processed_data)
        
        # Get processed data and uncertainty estimates
        if 'final_data' in result:
            data = result['final_data']
            
            # Plot main curve
            ax.plot(data, depth, color='blue', linewidth=2, label=f'{curve} (processed)')
            
            # Add uncertainty bands if available
            if 'uncertainty' in result:
                uncertainty = result['uncertainty']
                ax.fill_betweenx(depth, data - uncertainty, data + uncertainty,
                               alpha=0.3, color='blue', label='±1σ Uncertainty')
            
            # Add original data for comparison if available. This trace belongs
            # to current_data, so it needs that frame's own depth reference.
            if (self.current_data is not None and 
                curve in self.current_data.columns):
                original_depth = self._get_depth_for_frame(self.current_data)
                original_data = self.current_data[curve].values
                ax.plot(original_data, original_depth, color='red', alpha=0.7, 
                       linewidth=1, label=f'{curve} (original)')
            
            # Set proper axis limits
            data_min, data_max = np.nanmin(data), np.nanmax(data)
            if data_max > data_min:
                padding = (data_max - data_min) * 0.05
                ax.set_xlim(data_min - padding, data_max + padding)
            
            ax.set_xlabel(f"{curve} ({self.curve_info.get(curve, {}).get('unit', '')})")
            ax.set_title(f'{curve} - Uncertainty Analysis', fontsize=14, fontweight='bold')
            depth_for_axis = depth
            if (self.current_data is not None and curve in self.current_data.columns):
                depth_for_axis = np.concatenate([
                    depth, self._get_depth_for_frame(self.current_data)])
            self.apply_depth_axis(ax, depth_for_axis, label=LABEL_DEPTH_M)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No processed data available for '{curve}'", 
                   ha='center', va='center', fontsize=12)
        
        fig.tight_layout()
    
    def _plot_histogram_popup(self, fig, curve: str):
        """Plot histogram in popup window (reuses plot_histogram logic)"""
        # Check if curve has been processed, otherwise use original data
        if curve in self.processing_results:
            processed = self.processing_results[curve]['final_data']
            original = self.processing_results[curve].get('original_data', None)
            has_processed = True
        elif self.current_data is not None and curve in self.current_data.columns:
            processed = self.current_data[curve].values
            original = None
            has_processed = False
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Curve '{curve}' not found in data", 
                   ha='center', va='center', fontsize=12)
            return
        
        try:
            # Convert null values to NaN for proper filtering
            processed_clean = self._convert_nulls_to_nan(processed)
            valid_processed = processed_clean[~np.isnan(processed_clean) & np.isfinite(processed_clean)]
            
            if len(valid_processed) == 0:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"No valid data points for curve '{curve}'", 
                       ha='center', va='center', fontsize=12)
                return
            
            # Create main histogram plot
            ax = fig.add_subplot(111)
            
            # Calculate optimal number of bins (Freedman-Diaconis rule)
            iqr = np.percentile(valid_processed, 75) - np.percentile(valid_processed, 25)
            bin_width = 2 * iqr / (len(valid_processed) ** (1/3)) if iqr > 0 else (valid_processed.max() - valid_processed.min()) / 30
            num_bins = max(20, min(50, int((valid_processed.max() - valid_processed.min()) / bin_width))) if bin_width > 0 else 30
            
            # Plot processed data histogram
            n, bins, patches = ax.hist(valid_processed, bins=num_bins, alpha=0.7, color='blue', 
                                      edgecolor='black', linewidth=1.2, label=LABEL_PROCESSED_DATA if has_processed else 'Data')
            
            # Color-code bins by frequency
            max_freq = n.max() if len(n) > 0 else 1
            for i, (patch, freq) in enumerate(zip(patches, n)):
                intensity = 0.3 + 0.7 * (freq / max_freq) if max_freq > 0 else 0.5
                patch.set_facecolor(plt.cm.Blues(intensity))
            
            # Plot original data histogram if available (overlay)
            if original is not None:
                original_clean = self._convert_nulls_to_nan(original)
                valid_original = original_clean[~np.isnan(original_clean) & np.isfinite(original_clean)]
                
                if len(valid_original) > 0:
                    ax.hist(valid_original, bins=bins, alpha=0.4, color='red', 
                           edgecolor='darkred', linestyle='--', 
                           label=LABEL_ORIGINAL_DATA, histtype='step', linewidth=2)
            
            # Calculate and display statistics
            mean_val = np.mean(valid_processed)
            median_val = np.median(valid_processed)
            std_val = np.std(valid_processed)
            min_val = np.min(valid_processed)
            max_val = np.max(valid_processed)
            
            # Add vertical lines for mean and median
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # Add normal distribution overlay if data appears normally distributed
            if len(valid_processed) > 30:
                from scipy import stats
                try:
                    if len(valid_processed) <= 5000:
                        _, p_value = stats.shapiro(valid_processed[:5000])
                        if p_value > 0.05:
                            x_norm = np.linspace(valid_processed.min(), valid_processed.max(), 100)
                            y_norm = stats.norm.pdf(x_norm, mean_val, std_val) * len(valid_processed) * (bins[1] - bins[0])
                            ax.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.6, label='Normal Distribution Fit')
                except:
                    pass
            
            # Add outlier detection visualization (IQR method)
            q1 = np.percentile(valid_processed, 25)
            q3 = np.percentile(valid_processed, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = valid_processed[(valid_processed < lower_bound) | (valid_processed > upper_bound)]
            
            if len(outliers) > 0:
                ax.axvspan(lower_bound, upper_bound, alpha=0.1, color='green', label='Normal Range (IQR)')
                ax.scatter(outliers, np.zeros_like(outliers) + max(n) * 0.05, 
                          color='red', marker='x', s=50, alpha=0.7, zorder=5, label=f'Outliers ({len(outliers)})')
            
            # Professional styling
            curve_unit = self.curve_info.get(curve, {}).get('unit', '')
            ax.set_xlabel(f'{curve} ({curve_unit})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            
            if has_processed:
                ax.set_title(f'Distribution Analysis: {curve} (Processed)', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Distribution Analysis: {curve} (Original)', fontsize=14, fontweight='bold')
            
            # Add statistics text box
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {mean_val:.3f}\n'
            stats_text += f'Median: {median_val:.3f}\n'
            stats_text += f'Std Dev: {std_val:.3f}\n'
            stats_text += f'Min: {min_val:.3f}\n'
            stats_text += f'Max: {max_val:.3f}\n'
            stats_text += f'Count: {len(valid_processed):,}\n'
            if len(outliers) > 0:
                stats_text += f'Outliers: {len(outliers)} ({len(outliers)/len(valid_processed)*100:.1f}%)'
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, family='monospace')
            
            ax.legend(loc=LABEL_UPPER_LEFT, fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating histogram: {str(e)}", 
                   ha='center', va='center', fontsize=12)
    
    def _get_depth_for_frame(self, data_source):
        """Return the depth channel belonging to one specific DataFrame.

        Each frame carries its own depth reference. `current_data` holds the file
        as loaded, while `processed_data` may have been placed on a uniform grid
        by resample_to_standard_spacing, so the two frames can differ in both
        sample count and sample positions.

        Depth must therefore be taken from the same frame as the values being
        plotted. Sharing a single axis across frames raises a shape error when
        the lengths differ and, worse, plots every sample at the wrong depth when
        the lengths happen to coincide. Depth placement is the core correctness
        guarantee of a log plot, so this is resolved per frame rather than once.
        """
        try:
            if data_source is None:
                return np.arange(100)  # Fallback if no data

            # Look for depth column
            for col in data_source.columns:
                curve_type = self.curve_info.get(col, {}).get('curve_type', '')
                if 'DEPTH' in curve_type or col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD']:
                    return data_source[col].values

            # No depth column found, use index
            return np.arange(len(data_source))
        except Exception as e:
            # Fallback with logging
            import warnings
            warnings.warn(f"Could not get depth array: {str(e)}. Using index as depth.", UserWarning)
            return np.arange(100)

    def _get_depth_array(self):
        """Helper to get depth array for plotting (works with processed or current data)"""
        # Preferring processed data preserves the behaviour of every existing
        # caller; helpers that mix frames resolve depth per frame instead.
        data_source = self.processed_data if self.processed_data is not None else self.current_data
        return self._get_depth_for_frame(data_source)
    
    def _get_depth_limits(self, depth: np.ndarray) -> Tuple[float, float]:
        """Get depth axis limits from depth array.
        
        Args:
            depth: Depth array
            
        Returns:
            Tuple[float, float]: (depth_min, depth_max) for axis limits
        """
        try:
            if len(depth) == 0:
                return (0.0, 100.0)  # Safe default
            
            valid_depth = depth[np.isfinite(depth)]
            if len(valid_depth) == 0:
                return (0.0, 100.0)  # Safe default
            
            depth_min = float(np.min(valid_depth))
            depth_max = float(np.max(valid_depth))
            
            # Ensure min < max (in case of single value)
            if depth_min >= depth_max:
                depth_max = depth_min + 1.0
            
            return (depth_min, depth_max)
        except Exception as e:
            self.log_processing(f"Warning: Error calculating depth limits: {e}")
            return (0.0, 100.0)  # Safe fallback

    def apply_depth_axis(self, ax, depth, *, label: str) -> Tuple[float, float]:
        """Apply the wireline depth convention to one axis and return limits.

        Depth increases downward. That is expressed solely as
        ``set_ylim(depth_max, depth_min)``. This method never calls
        ``invert_yaxis``: mixing the two is a double flip that silently
        renders depth upward.

        ``label`` is required because the smoke harness matches depth axes on
        the Y label; an optional label would let a caller silently opt out of
        the orientation guard.
        """
        depth_min, depth_max = self._get_depth_limits(depth)
        ax.set_ylim(depth_max, depth_min)
        ax.set_ylabel(label)
        return depth_min, depth_max

    def apply_depth_axis_shared(self, axes, depth, *, label: str) -> Tuple[float, float]:
        """Apply depth convention once across a sharey axis group.

        Limits are set on the first axis and propagate through sharey. Only the
        first axis receives the depth label. Calling invert_yaxis on every
        sharey track is a no-op only at even track counts, so the four-track
        log display was previously correct by accident under that pattern.
        """
        axis_list = list(axes)
        if not axis_list:
            raise ValueError("apply_depth_axis_shared requires at least one axis")
        depth_min, depth_max = self.apply_depth_axis(axis_list[0], depth, label=label)
        for sibling in axis_list[1:]:
            sibling.set_ylabel('')
        return depth_min, depth_max
    
    # ============================================================================
    # ENHANCED VISUALIZATION CONTROLLER
    # ============================================================================
    
    def update_visualization_enhanced(self):
        """Enhanced visualization update with support for popup and embedded displays"""
        try:
            # Pre-flight validation
            validation_result = self._validate_visualization_prerequisites()
            if not validation_result['valid']:
                messagebox.showwarning("Visualization Warning", validation_result['message'])
                return
            
            viz_type = self.viz_type_var.get()
            curve = self.viz_curve_var.get()
            
            # Check if user wants popup window (professional workflow)
            if self.plot_in_new_window_var.get():
                # Open in new Toplevel popup window
                self.log_processing(f"Creating popup visualization: type={viz_type}, curve={curve}")
                self._create_popup_visualization(viz_type, curve)
            else:
                # Embedded display (original behavior)
                # Add new visualization types
                if viz_type == "single_curve":
                    self.plot_single_curve(curve)
                elif viz_type == "single_curve_comparison":
                    self.plot_single_curve_comparison()
                elif viz_type == "unprocessed_curves":
                    self.plot_unprocessed_curves()  # Now includes canvas creation
                elif viz_type == "quality_overview":
                    self.plot_curve_quality_overview()  # Verify this also creates canvas
                elif viz_type == "histogram":
                    self.plot_histogram(curve)  # Histogram requires curve parameter
                elif viz_type == "curve_comparison_all":
                    # Deprecated: Use multi_curve or quality_overview instead
                    messagebox.showinfo("Visualization Update", 
                                      "The 'curve_comparison_all' visualization has been replaced.\n"
                                      "Please use 'multi_curve' for selected curves or 'quality_overview' for comprehensive analysis.")
                    return
                else:
                    # Use existing visualization methods
                    self.update_visualization()
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, 
                               f"Failed to update visualization:\n{str(e)}")
            self.log_processing(f"Error in update_visualization_enhanced: {e}")
    
    def on_viz_type_change_enhanced(self, event=None):
        """Enhanced visualization type change handler"""
        viz_type = self.viz_type_var.get()
        
        # Show/hide appropriate controls based on viz type
        if viz_type == "multi_curve":
            self.multi_curve_frame.pack(fill='x', pady=5)
            self.third_curve_frame.pack_forget()
        elif viz_type == "3d_visualization":
            self.multi_curve_frame.pack_forget()
            self.third_curve_frame.pack(fill='x', pady=5)
        elif viz_type in ["unprocessed_curves", "quality_overview"]:
            # Hide multi-curve frame for these new types
            self.multi_curve_frame.pack_forget()
            self.third_curve_frame.pack_forget()
        else:
            self.multi_curve_frame.pack_forget()
            self.third_curve_frame.pack_forget()
        
        # Enable/disable secondary curve combobox based on viz type
        if viz_type in ["3d_visualization", "single_curve_comparison"]:
            self.viz_curve2_combo['state'] = 'readonly'
        else:
            self.viz_curve2_combo['state'] = 'disabled'
            
        # Enable/disable third curve combobox for 3D visualization
        if viz_type == "3d_visualization":
            self.viz_curve3_combo['state'] = 'readonly'
        else:
            self.viz_curve3_combo['state'] = 'disabled'

    # ============================================================================
    # QUICK VISUALIZATION METHODS
    # ============================================================================
    
    def quick_view_unprocessed(self):
        """Quick access to view unprocessed curves from the data loading tab"""
        try:
            if self.current_data is None:
                messagebox.showwarning("Warning", "No data loaded. Please load a file first.")
                return
            
            # Switch to visualization tab and set the visualization type
            self.notebook.select(2)  # Visualization tab (0-indexed)
            self.viz_type_var.set("unprocessed_curves")
            
            # Update the visualization
            self.update_visualization_enhanced()
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, 
                               f"Failed to display unprocessed curves:\n{str(e)}")
            self.log_processing(f"Error in quick_view_unprocessed: {e}")
    
    def quick_quality_overview(self):
        """Quick access to quality overview from the data loading tab"""
        try:
            if self.current_data is None:
                messagebox.showwarning("Warning", "No data loaded. Please load a file first.")
                return
            
            # Switch to visualization tab and set the visualization type
            self.notebook.select(2)  # Visualization tab (0-indexed)
            self.viz_type_var.set("quality_overview")
            
            # Update the visualization
            self.update_visualization_enhanced()
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, 
                               f"Failed to display quality overview:\n{str(e)}")
            self.log_processing(f"Error in quick_quality_overview: {e}")
    
    def quick_compare_all(self):
        """Quick access to compare all curves from the data loading tab"""
        try:
            if self.current_data is None:
                messagebox.showwarning("Warning", "No data loaded. Please load a file first.")
                return
            
            # Switch to visualization tab and set the visualization type
            self.notebook.select(2)  # Visualization tab (0-indexed)
            self.viz_type_var.set("histogram")
            
            # Update the visualization
            self.update_visualization_enhanced()
        except Exception as e:
            messagebox.showerror(ERROR_TITLE_VISUALIZATION, 
                               f"Failed to compare curves:\n{str(e)}")
            self.log_processing(f"Error in quick_compare_all: {e}")

    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):

            self.root.destroy()

def main():
    """Main application entry point.

    Diagnostics are installed before anything else so that a failure during
    application construction still produces a readable local report. A packaged
    windowed build has no console, so without this an early failure is visible
    to the user only as a generic dialog or as nothing happening at all.
    """
    # The launch is already recorded at module import time, which is earlier
    # than this point and therefore survives an import-time failure. Only the
    # handlers are refreshed here.
    crash_report.install_global_handlers()

    try:
        # Check for advanced libraries - log to console instead of popup
        if not ADVANCED_LIBS:
            print("INFO: Some advanced features may not be available due to missing libraries.")
            print("      For full functionality, install: scipy, scikit-learn, pywavelets")

        # Create and run application
        app = AdvancedPreprocessingApplication()

        # Tk swallows exceptions raised inside widget callbacks, so the handler
        # can only be attached once the root window exists.
        crash_report.install_global_handlers(tk_root=getattr(app, 'root', None))

        app.run()

    except Exception as e:
        report_path = crash_report.write_crash_report(
            type(e), e, e.__traceback__, context="startup")

        message = f"Failed to start application:\n{str(e)}"
        if report_path:
            # Naming the file lets the user attach it to a support request
            # without needing to reproduce the fault or send a screenshot.
            message += (f"\n\nA diagnostic report was saved to:\n{report_path}"
                        "\n\nIt contains technical details only, no well data, "
                        "and was not sent anywhere.")
        try:
            messagebox.showerror("Startup Error", message)
        except Exception:
            # Tk itself may be what failed, so fall back to standard error.
            print(message, file=sys.stderr)

if __name__ == "__main__":
    main()
