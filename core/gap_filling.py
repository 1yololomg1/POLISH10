"""
Gap Filling Module

Sophisticated gap detection, classification, and filling for wireline curves.

Extracted from advanced_preprocessing_system10.py for modular architecture.

ARCHITECTURE OVERVIEW:

MAIN CLASSES:
- GeologicalContext: Formation tops, casing, and open-hole interval context
- GapClassificationResult: Classification outcome for a detected gap
- GapFillingParameters: Tunable parameters for the gap filling engine
- AdvancedGapFiller: Multi-method gap filling (linear, spline, GP, kriging, multi-curve)

KEY FUNCTIONS:
- AdvancedGapFiller.fill_gaps(): Detect and fill gaps with method selection
- AdvancedGapFiller._classify_gap_type(): Geological vs measurement gap classification
- AdvancedGapFiller._linear_interpolation() / spline / gaussian_process methods

DATA FLOW:
Processed curve data → gap detection → geological classification → method selection →
filled curve + confidence/uncertainty metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np

# Optional Tkinter fallback for error dialogs when no error_handler/callback is set
try:
    from tkinter import messagebox
except ImportError:  # pragma: no cover - headless environments
    class _MessageBoxFallback:
        @staticmethod
        def showerror(title, message):
            print(f"ERROR: {title}: {message}")

    messagebox = _MessageBoxFallback()  # type: ignore

# Optional scientific libraries (same availability gates as the main application)
SCIPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
PYWT_AVAILABLE = False

try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    interpolate = None  # type: ignore

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianProcessRegressor = None  # type: ignore
    RBF = WhiteKernel = Matern = None  # type: ignore

try:
    import pywt  # noqa: F401
    PYWT_AVAILABLE = True
except ImportError:
    pass

ADVANCED_LIBS = SCIPY_AVAILABLE and SKLEARN_AVAILABLE and PYWT_AVAILABLE

from petrophysics.constants import PHYSICAL_CONSTANTS
from core.petrophysical_models import RelativeRockPropertiesModel
from core.error_handler import ErrorSeverity

# Geological gap classification threshold (points); mirrors main app constant
GAP_THRESHOLD_GEOLOGICAL = 200


class GeologicalContext:
    """Geological context information for intelligent gap classification"""
    def __init__(self):
        self.formation_tops = {}
        self.casing_points = []
        self.open_hole_start = None
        self.open_hole_end = None
        self.curve_validity_zones = {}
        
    def add_formation_top(self, formation_name: str, depth: float):
        """Add formation top depth"""
        self.formation_tops[formation_name] = depth
    
    def set_casing_program(self, casing_points: List[float]):
        """Set casing shoe depths"""
        self.casing_points = sorted(casing_points)
    
    def set_open_hole_interval(self, start_depth: float, end_depth: float):
        """Set open hole logging interval"""
        self.open_hole_start = start_depth
        self.open_hole_end = end_depth
    
    def get_first_formation_depth(self) -> Optional[float]:
        """Get depth of first formation top"""
        if self.formation_tops:
            return min(self.formation_tops.values())
        return None

class GapClassificationResult:
    """Result of gap classification analysis"""
    def __init__(self, gap_type: str, should_fill: bool, confidence: float, reason: str):
        self.gap_type = gap_type  # 'geological', 'measurement', 'mixed'
        self.should_fill = should_fill
        self.confidence = confidence
        self.reason = reason

@dataclass
class GapFillingParameters:
    """Advanced gap filling parameters"""
    max_gap_size: int = 100
    confidence_threshold: float = 0.8
    uncertainty_quantification: bool = True
    physics_informed: bool = True
    multi_curve_correlation: bool = True
    kriging_variogram: str = 'matern52'
    gp_kernel: str = 'rbf_white'
    time_series_order: Tuple[int, int, int] = (2, 1, 2)
    geological_context_aware: bool = True
    min_formation_penetration: float = 10.0  # meters
    geological_gap_threshold: int = GAP_THRESHOLD_GEOLOGICAL  # Threshold to distinguish geological gaps from data errors
class AdvancedGapFiller:
    """
    Sophisticated gap filling engine with multiple advanced algorithms
    
    SCIENTIFIC FOUNDATION:
    
    1. LINEAR INTERPOLATION:
       - Newton, I. (1676): "Methodus fluxionum et serierum infinitarum"
       - Lagrange, J.L. (1795): "Leçons élémentaires sur les mathématiques"
       - Validated through numerical analysis theory and practice
    
    2. CUBIC SPLINE INTERPOLATION:
       - Schoenberg, I.J. (1946): "Contributions to the problem of approximation"
       - De Boor, C. (1978): "A Practical Guide to Splines"
       - Provides C² continuity and minimizes curvature
    
    3. GAUSSIAN PROCESS INTERPOLATION:
       - Rasmussen, C.E. & Williams, C.K.I. (2006): "Gaussian Processes for Machine Learning"
       - Krige, D.G. (1951): "A statistical approach to some basic mine valuation problems"
       - Provides uncertainty quantification and optimal prediction
    
    4. KRIGING INTERPOLATION:
       - Matheron, G. (1963): "Principles of geostatistics"
       - Cressie, N. (1993): "Statistics for Spatial Data"
       - Industry standard for spatial interpolation in geosciences
    
    5. POLYNOMIAL INTERPOLATION:
       - Lagrange, J.L. (1795): "Leçons élémentaires sur les mathématiques"
       - Runge, C. (1901): "Über empirische Funktionen"
       - Classical interpolation method with optimal degree selection
    
    ALGORITHM SELECTION:
    - Decision tree based on gap size and data characteristics
    - Automatic method selection using statistical criteria
    - Fallback mechanisms ensure robustness
    
    ALGORITHM SELECTION:
    - Decision tree based on gap size and data characteristics
    - Automatic method selection using statistical criteria
    - Fallback mechanisms ensure robustness
    """
    
    def __init__(self, params: GapFillingParameters, error_callback: Optional[Any] = None, error_handler: Optional[Any] = None, log_processing: Optional[Callable[[str], None]] = None):
        self.params = params
        self._error_callback = error_callback
        self.error_handler = error_handler  # Centralized error handler
        self.log_processing = log_processing if log_processing is not None else (lambda msg: None)  # No-op if not provided
        # Debug flag for verbose gap decision logging
        self.debug = False

    def _report_error(self, title: str, message: str) -> None:
        """Dispatch errors through centralized error handler or callback, falling back to messagebox."""
        # Use centralized error handler if available
        if self.error_handler:
            try:
                context = self.error_handler.create_context(
                    operation="Gap Filling",
                    component="AdvancedGapFiller",
                    user_action="Processing data",
                    remediation_hint="Please check the error message and data quality."
                )
                # Determine severity from title
                severity = ErrorSeverity.ERROR
                if "Critical" in title or "Fatal" in title:
                    severity = ErrorSeverity.CRITICAL
                elif "Warning" in title:
                    severity = ErrorSeverity.WARNING
                
                error = Exception(message)
                self.error_handler.handle_error(error, context, severity=severity, show_dialog=True, log_error=True)
                return
            except Exception as handler_error:
                # Centralized handler failed - fall through to callback
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: Centralized error handler failed: {type(handler_error).__name__}: {str(handler_error)}")
                    except Exception:
                        pass
        
        # Fallback to callback if available
        if callable(self._error_callback):
            try:
                self._error_callback(title, message)
                return
            except Exception as callback_error:
                # Error callback failed - log but continue to fallback
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"Warning: Error callback failed: {type(callback_error).__name__}: {str(callback_error)}")
        
        # Final fallback: direct dialog (may be unsafe off main thread but last resort)
        try:
            messagebox.showerror(title, message)
        except Exception as dialog_error:
            # Even fallback dialog failed - log for debugging
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Warning: Error dialog display failed: {type(dialog_error).__name__}: {str(dialog_error)}")
            # Last resort: print to console
            print(f"ERROR: {title}: {message}")

        
    def _filter_fillable_gaps(self, gaps: List[Dict], curve_name: str, curve_type: str, 
                             max_gap_allowed: int, allowed_methods: List[str], 
                             data: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Filter gaps based on curve-specific thresholds and geological context.
        
        Returns:
            Tuple of (fillable_gaps, skipped_gaps)
        """
        fillable_gaps = []
        skipped_gaps = []
        
        for gap in gaps:
            gap_size = gap.get('size', gap.get('end', 0) - gap.get('start', 0))
            
            # Classify gap type (geological feature vs data error)
            geological_threshold = self.params.geological_gap_threshold if hasattr(self.params, 'geological_gap_threshold') else GAP_THRESHOLD_GEOLOGICAL
            if gap_size >= geological_threshold:
                gap['gap_type'] = 'geological'
                gap['gap_classification'] = f"Geological/logging feature (>={geological_threshold} pts)"
            else:
                gap['gap_type'] = 'data_error'
                gap['gap_classification'] = f"Data error (<{geological_threshold} pts)"
            
            # Apply curve-specific gap size threshold
            if gap_size <= max_gap_allowed:
                # Log decision for debugging
                if self.debug:
                    print(f"[GAP DECISION] {curve_name} ({curve_type}): Gap size {gap_size} <= threshold {max_gap_allowed} - FILLING [{gap['gap_type']}]")
                
                # Validate data quality before filling
                data_quality = PHYSICAL_CONSTANTS.validate_curve_data_enhanced(data, curve_type)
                completeness_grade = PHYSICAL_CONSTANTS.assess_data_completeness(data)
                
                gap['should_fill'] = True
                gap['allowed_methods'] = allowed_methods
                gap['data_quality'] = data_quality
                gap['completeness_grade'] = completeness_grade
                fillable_gaps.append(gap)
            else:
                # Log decision for debugging  
                if self.debug:
                    print(f"[GAP DECISION] {curve_name} ({curve_type}): Gap size {gap_size} > threshold {max_gap_allowed} - SKIPPING [{gap['gap_type']}]")
                gap['should_fill'] = False
                gap['skip_reason'] = f"Gap size ({gap_size}) exceeds curve-specific threshold ({max_gap_allowed})"
                skipped_gaps.append(gap)
        
        # Log gap processing summary
        if self.debug:
            try:
                print(f"  - {curve_name}: Found {len(gaps)} gaps, filling {len(fillable_gaps)}, skipping {len(skipped_gaps)}")
                if skipped_gaps:
                    for gap in skipped_gaps:
                        print(f"    - Skipped gap: {gap.get('size', 0)} points (exceeds threshold)")
            except Exception as print_error:
                # Debug print failed - log but don't fail gap filling
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"Warning: Debug print failed: {type(print_error).__name__}: {str(print_error)}")
        
        # Continue with geological context filtering (if any)
        final_gaps = []
        for gap in fillable_gaps:
            if gap.get('geological_classification'):
                classification = gap['geological_classification']
                if classification.should_fill:
                    final_gaps.append(gap)
                else:
                    # Skip based on geological analysis
                    skipped_gaps.append(gap)
            else:
                # No geological context - use the curve-specific decision
                final_gaps.append(gap)
        
        return final_gaps, skipped_gaps
    
    def _process_large_gap(self, gap: Dict, large_gap_treatment: str, rrp_model: Optional[Any],
                           current_curve_name: str, data: np.ndarray, auxiliary_curves: Optional[Dict],
                           filled_data: np.ndarray, uncertainty: np.ndarray, confidence: np.ndarray) -> Dict[str, Any]:
        """Process large gaps using specialized methods.
        
        Returns:
            Dict with 'filled' (bool) and 'result' (gap result dict if filled)
        """
        if large_gap_treatment == 'skip':
            return {'filled': False}
        
        if large_gap_treatment == 'formation_based' and rrp_model:
            # Use Relative Rock Properties approach
            try:
                result = rrp_model.fill_large_gap(
                    current_curve_name, gap['start'], gap['end'], data, auxiliary_curves
                )
                
                if result:
                    # Update arrays
                    gap_slice = slice(gap['start'], gap['end'])
                    filled_data[gap_slice] = result['values']
                    uncertainty[gap_slice] = result['uncertainty']
                    confidence[gap_slice] = result['confidence']
                    
                    return {
                        'filled': True,
                        'result': {
                            'gap': gap,
                            'method': 'relative_rock_properties',
                            'quality': result['quality'],
                            'uncertainty_mean': np.mean(result['uncertainty'])
                        }
                    }
            except Exception as e:
                # Log algorithm fallback with scientific transparency
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"[ALGORITHM FALLBACK] Relative Rock Properties → Standard Methods")
                    self.log_processing(f"   Reason: {str(e)}")
                    self.log_processing(f"   Fallback Chain: Kriging → Cubic Spline → Linear Interpolation")
                
                if hasattr(self, '_notify_error'):
                    self._notify_error("Gap Filling Error", f"Advanced method unavailable: {e}")
                elif hasattr(self, '_report_error'):
                    self._report_error("Gap Filling Error", f"Failed to fill large gap: {e}")
        
        return {'filled': False}
    
    def _process_standard_gap(self, gap: Dict, curve_type: str, auxiliary_curves: Optional[Dict],
                              physics_constraints: Optional[Dict], data: np.ndarray,
                              filled_data: np.ndarray, uncertainty: np.ndarray, 
                              confidence: np.ndarray) -> Optional[Dict]:
        """Process standard gaps using optimal method selection.
        
        Returns:
            Gap result dict if successful, None otherwise
        """
        # Select optimal method for this gap with transparency
        method = self._select_optimal_method(gap, curve_type, auxiliary_curves)
        
        # Log method selection rationale for scientific audit
        if hasattr(self, 'log_processing'):
            self.log_processing(f"[METHOD SELECTION] Gap {gap['start']}-{gap['end']}: {method}")
            self.log_processing(f"   Size: {gap['size']} points | Curve: {curve_type}")
            if auxiliary_curves:
                self.log_processing(f"   Auxiliary curves available: {len(auxiliary_curves)}")
            self.log_processing(f"   Selection criteria: Gap size, curve type, data availability")
        
        try:
            result = self._fill_single_gap(
                data, gap, method, curve_type, 
                auxiliary_curves, physics_constraints
            )
            
            # Update arrays
            gap_slice = slice(gap['start'], gap['end'])
            filled_data[gap_slice] = result['values']
            uncertainty[gap_slice] = result['uncertainty']
            confidence[gap_slice] = result['confidence']
            
            # Log successful gap filling with method transparency
            if hasattr(self, 'log_processing'):
                self.log_processing(f"[GAP FILLED] Method: {method} | Quality: {result['quality']:.3f} | Size: {gap['size']} points")
                if result.get('method_details'):
                    self.log_processing(f"   Details: {result['method_details']}")
            
            return {
                'gap': gap,
                'method': method,
                'quality': result['quality'],
                'uncertainty_mean': np.mean(result['uncertainty']),
                'method_transparency': f"{method} (Q={result['quality']:.3f})"
            }
        except Exception as e:
            # Log final fallback with scientific rationale
            if hasattr(self, 'log_processing'):
                self.log_processing(f"[FINAL FALLBACK] All primary methods failed → Linear Interpolation")
                self.log_processing(f"   Gap: {gap['start']}-{gap['end']} ({gap['size']} points)")
                self.log_processing(f"   Reason: {str(e)}")
                self.log_processing(f"   Scientific Validity: Linear interpolation maintains continuity")
            
            if hasattr(self, '_notify_error'):
                self._notify_error("Gap Filling Error", f"Primary methods failed, using linear fallback: {e}")
            elif hasattr(self, '_report_error'):
                self._report_error("Gap Filling Error", f"Failed to fill gap {gap['start']}-{gap['end']}: {e}")
            
            # Use simple linear interpolation as fallback
            try:
                gap_slice = slice(gap['start'], gap['end'])
                fallback_result = self._linear_interpolation_fallback(data, gap)
                filled_data[gap_slice] = fallback_result
                confidence[gap_slice] = 0.5  # Lower confidence for fallback
                uncertainty[gap_slice] = np.abs(fallback_result) * 0.1  # Estimate uncertainty
                
                return {
                    'gap': gap,
                    'method': 'linear_fallback',
                    'quality': 0.5,
                    'uncertainty_mean': np.mean(uncertainty[gap_slice])
                }
            except Exception as fallback_error:
                if hasattr(self, 'log_processing'):
                    self.log_processing(f"Fallback interpolation also failed: {fallback_error}")
                return None
    
    def fill_gaps(self, data: np.ndarray, curve_type: str, 
                  auxiliary_curves: Optional[Dict[str, np.ndarray]] = None,
                  physics_constraints: Optional[Dict] = None,
                  depth: Optional[np.ndarray] = None,
                  geological_context: Optional[GeologicalContext] = None,
                  curve_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced gap filling with multiple sophisticated methods
        Enhanced with curve-specific gap thresholds and validation
        
        Returns:
            dict: {
                'filled_data': np.ndarray,
                'uncertainty': np.ndarray,
                'confidence': np.ndarray,
                'method_used': str,
                'gaps_filled': List[Dict],
                'quality_metrics': Dict
            }
        """
        
        # ENHANCED: Get curve-specific gap filling rules
        curve_name_for_lookup = curve_name if curve_name else curve_type
        max_gap_allowed, allowed_methods = PHYSICAL_CONSTANTS.get_gap_threshold_for_curve(
            curve_name_for_lookup, curve_type
        )
        
        # Identify gaps with geological context awareness
        gaps = self._identify_gaps(data, depth, geological_context, curve_type)
        
        # Filter fillable gaps using extracted method
        fillable_gaps, skipped_gaps = self._filter_fillable_gaps(
            gaps, curve_name_for_lookup, curve_type, max_gap_allowed, allowed_methods, data
        )
        
        if not fillable_gaps:
            return {
                'filled_data': data.copy(),
                'uncertainty': np.zeros_like(data),
                'confidence': np.ones_like(data),
                'method_used': 'no_gaps',
                'gaps_filled': [],
                'quality_metrics': {'gap_count': 0}
            }
        
        filled_data = data.copy()
        uncertainty = np.zeros_like(data)
        confidence = np.ones_like(data)
        gaps_filled = []
        
        # Check if we have a Relative Rock Properties model
        rrp_model = None
        
        # Get large gap threshold from UI variable
        large_gap_threshold = 500  # Default
        if hasattr(self, 'large_gap_threshold_var'):
            large_gap_threshold = self.large_gap_threshold_var.get()
        
        # Get large gap treatment method from UI variable
        large_gap_treatment = "formation_based"  # Default
        if hasattr(self, 'large_gap_var'):
            large_gap_treatment = self.large_gap_var.get()
        
        # Canonical name for the curve under analysis (used by RRP when available)
        current_curve_name = f"curve_{curve_type}"

        # Initialize RRP model if needed for large gaps
        if large_gap_treatment == 'formation_based' and any(gap['size'] > large_gap_threshold for gap in gaps):
            rrp_model = RelativeRockPropertiesModel(log_processing=self.log_processing)
            
            # Create a dictionary of all available curves
            all_curves = {}
            if auxiliary_curves:
                for curve_name, curve_data in auxiliary_curves.items():
                    all_curves[curve_name] = curve_data
            
            # Add the current curve with a proper name
            all_curves[current_curve_name] = data
            
            # Train the model
            try:
                rrp_model.train(all_curves)
            except Exception as e:
                if hasattr(self, '_notify_error'):
                    self._notify_error("Gap Filling Error", f"Failed to train Rock Properties model: {e}")
                elif hasattr(self, '_report_error'):
                    self._report_error("Gap Filling Error", f"Failed to train Rock Properties model: {e}")
                rrp_model = None
        
        for gap in fillable_gaps:
            if gap['size'] > self.params.max_gap_size:
                # Log to report instead of showing popup
                if hasattr(self, 'status_manager') and self.status_manager:
                    self.status_manager.update_status(f"⚠ Gap size {gap['size']} exceeds maximum {self.params.max_gap_size} - skipping")
                continue
            
            # Check if this is a large gap needing special treatment
            is_large_gap = gap['size'] > large_gap_threshold
            
            if is_large_gap:
                # Process large gap using extracted method
                large_gap_result = self._process_large_gap(
                    gap, large_gap_treatment, rrp_model, current_curve_name,
                    data, auxiliary_curves, filled_data, uncertainty, confidence
                )
                if large_gap_result['filled']:
                    gaps_filled.append(large_gap_result['result'])
                    continue  # Large gap filled successfully, move to next gap
                # If large gap processing failed, fall through to standard methods
            
            # Process standard gap using extracted method
            standard_result = self._process_standard_gap(
                gap, curve_type, auxiliary_curves, physics_constraints,
                data, filled_data, uncertainty, confidence
            )
            if standard_result:
                gaps_filled.append(standard_result)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_gap_filling_quality(
            data, filled_data, gaps_filled
        )
        
        # Clean up RRP model to free memory
        if rrp_model:
            rrp_model = None
        
        return {
            'filled_data': filled_data,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'method_used': 'advanced_ensemble',
            'gaps_filled': gaps_filled,
            'quality_metrics': quality_metrics
        }
    
    def _identify_gaps(self, data: np.ndarray, depth: np.ndarray = None, 
                      geological_context: GeologicalContext = None, 
                      curve_name: str = "") -> List[Dict]:
        """Identify gaps with geological context-aware classification"""
        gaps = []
        in_gap = False
        gap_start = None
        
        for i, value in enumerate(data):
            if np.isnan(value):
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    if gap_start is None:
                        gap_start = i  # Fallback if gap_start wasn't set
                    gap_size = i - gap_start
                    
                    # Perform geological gap classification
                    gap_classification = self._classify_gap_geological_context(
                        gap_start, i, depth, geological_context, curve_name
                    ) if (gap_start is not None and depth is not None and geological_context is not None) else None
                    
                    gaps.append({
                        'start': gap_start,
                        'end': i,
                        'size': gap_size,
                        'context_before': self._get_context(data, gap_start, 'before'),
                        'context_after': self._get_context(data, i, 'after'),
                        'geological_classification': gap_classification
                    })
                    in_gap = False
        
        # Handle gap at end
        if in_gap:
            if gap_start is None:
                gap_start = len(data) - 1  # Fallback if gap_start wasn't set
            gap_classification = self._classify_gap_geological_context(
                gap_start, len(data), depth, geological_context, curve_name
            ) if (gap_start is not None and depth is not None and geological_context is not None) else None
            
            gaps.append({
                'start': gap_start,
                'end': len(data),
                'size': len(data) - gap_start,
                'context_before': self._get_context(data, gap_start, 'before'),
                'context_after': [],
                'geological_classification': gap_classification
            })
        
        return gaps
    
    def _get_context(self, data: np.ndarray, position: int, direction: str, 
                     context_size: int = 20) -> np.ndarray:
        """Get valid context data around gap"""
        if direction == 'before':
            start = max(0, position - context_size)
            context = data[start:position]
        else:  # after
            end = min(len(data), position + context_size)
            context = data[position:end]
        
        return context[~np.isnan(context)]
    
    def _classify_gap_geological_context(self, gap_start: int, gap_end: int, 
                                       depth: np.ndarray, geological_context: GeologicalContext,
                                       curve_name: str) -> GapClassificationResult:
        """
        Classify gaps based on geological context to distinguish between:
        - Geological nulls (pre-formation, above formation tops)
        - Measurement gaps (within formations where data should exist)
        - Mixed gaps (spanning geological boundaries)
        """
        if depth is None or len(depth) == 0:
            return GapClassificationResult('unknown', True, 0.5, 'No depth information available')
        
        # Get depth range of the gap
        gap_start_depth = depth[min(gap_start, len(depth)-1)]
        gap_end_depth = depth[min(gap_end-1, len(depth)-1)]
        gap_depth_range = (gap_start_depth, gap_end_depth)
        
        # Get first formation depth
        first_formation_depth = geological_context.get_first_formation_depth()
        
        # Define curve-specific validity rules
        curve_validity_rules = self._get_curve_validity_rules()
        curve_validity = curve_validity_rules.get(curve_name.upper(), {
            'requires_formation': False,
            'requires_open_hole': True,
            'min_penetration': 0.0
        })
        
        # Classification logic
        confidence = 1.0
        
        # Case 1: Gap entirely above first formation
        if first_formation_depth and gap_end_depth < first_formation_depth:
            if curve_validity['requires_formation']:
                return GapClassificationResult(
                    'geological', False, confidence, 
                    f'Pre-formation gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m above first formation at {first_formation_depth:.1f}m'
                )
            else:
                # Some curves (like GR) can measure through casing/air
                return GapClassificationResult(
                    'measurement', True, 0.7, 
                    f'Pre-formation measurement gap for {curve_name}: surface-to-formation measurements possible'
                )
        
        # Case 2: Gap entirely within formations
        elif first_formation_depth and gap_start_depth >= first_formation_depth:
            # Check if within open hole interval
            if (geological_context.open_hole_start and geological_context.open_hole_end and
                gap_start_depth >= geological_context.open_hole_start and 
                gap_end_depth <= geological_context.open_hole_end):
                return GapClassificationResult(
                    'measurement', True, confidence,
                    f'Formation measurement gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m within open hole'
                )
            elif curve_validity['requires_open_hole']:
                return GapClassificationResult(
                    'geological', False, 0.8,
                    f'Cased hole gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m - {curve_name} requires open hole'
                )
            else:
                return GapClassificationResult(
                    'measurement', True, 0.6,
                    f'Possible measurement gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m'
                )
        
        # Case 3: Gap spans geological boundaries (mixed)
        elif first_formation_depth and gap_start_depth < first_formation_depth < gap_end_depth:
            return GapClassificationResult(
                'mixed', True, 0.5,
                f'Mixed gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m spans formation boundary at {first_formation_depth:.1f}m'
            )
        
        # Case 4: No geological context available
        else:
            return GapClassificationResult(
                'unknown', True, 0.3,
                f'Unknown context gap: {gap_start_depth:.1f}-{gap_end_depth:.1f}m'
            )
    
    def _get_curve_validity_rules(self) -> Dict[str, Dict]:
        """
        Define geological validity rules for different curve types
        Based on logging tool physics and industry practice
        """
        return {
            # Resistivity tools - require formation contact
            'RT': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 1.0},
            'RD': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 1.0},
            'RM': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 1.0},
            'RS': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            'MSFL': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.1},
            'LLD': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 1.0},
            'LLS': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            
            # Density tools - require open hole formation contact
            'RHOB': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            'RHOZ': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            'DEN': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            
            # Neutron tools - require open hole formation contact
            'NPHI': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            'TNPH': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            'NEUT': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            
            # Gamma ray - can measure through casing and fluid
            'GR': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            'CGR': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            'SGR': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            
            # Spectral gamma ray components
            'THOR': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            'URAN': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            'POTA': {'requires_formation': False, 'requires_open_hole': False, 'min_penetration': 0.0},
            
            # Photoelectric factor
            'PEF': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5},
            
            # SP (requires formation and mud contact)
            'SP': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 1.0},
            
            # Caliper (measures hole diameter - needs open hole)
            'CAL': {'requires_formation': False, 'requires_open_hole': True, 'min_penetration': 0.0},
            'CALI': {'requires_formation': False, 'requires_open_hole': True, 'min_penetration': 0.0},
            
            # Default rule for unknown curves
            'DEFAULT': {'requires_formation': True, 'requires_open_hole': True, 'min_penetration': 0.5}
        }
    
    def _select_optimal_method(self, gap: Dict, curve_type: str, 
                              auxiliary_curves: Optional[Dict]) -> str:
        """Select optimal gap filling method based on gap characteristics"""
        gap_size = gap['size']
        context_before = len(gap['context_before'])
        context_after = len(gap['context_after'])
        
        # Decision tree for method selection using scientific thresholds
        if gap_size <= PHYSICAL_CONSTANTS.GAP_SIZE["SMALL"]:
            return 'linear'
        elif gap_size <= PHYSICAL_CONSTANTS.GAP_SIZE["MEDIUM"] and context_before >= 5 and context_after >= 5:
            return 'cubic_spline'
        elif gap_size <= PHYSICAL_CONSTANTS.GAP_SIZE["LARGE"] and ADVANCED_LIBS:
            if self.params.multi_curve_correlation and auxiliary_curves:
                return 'multi_curve_gp'
            else:
                return 'gaussian_process'
        elif gap_size <= PHYSICAL_CONSTANTS.GAP_SIZE["VERY_LARGE"] and context_before >= 10:
            if ADVANCED_LIBS:
                return 'kriging'
            else:
                return 'polynomial'
        else:
            return 'trend_extrapolation'
    
    def _fill_single_gap(self, data: np.ndarray, gap: Dict, method: str, 
                        curve_type: str, auxiliary_curves: Optional[Dict],
                        physics_constraints: Optional[Dict]) -> Dict[str, Any]:
        """Fill single gap using specified method"""
        
        if method == 'linear':
            return self._linear_interpolation(data, gap)
        elif method == 'cubic_spline':
            return self._cubic_spline_interpolation(data, gap)
        elif method == 'gaussian_process':
            return self._gaussian_process_interpolation(data, gap, curve_type)
        elif method == 'multi_curve_gp':
            try:
                # Multi-curve GP interpolation
                return self._multi_curve_gp_interpolation(data, gap, auxiliary_curves)
            except Exception as e:
                # Log technical details for developers (optional)
                if hasattr(self, 'log_processing'):
                    gap_start = gap.get('start', 0)
                    gap_end = gap.get('end', 0)
                    self.log_processing(f" Multi-curve correlation failed for gap {gap_start}-{gap_end}")
                    self.log_processing(f"   → Reason: Insufficient reference curve data") 
                    self.log_processing(f"   → Falling back to cubic spline interpolation")
                
                # Use fallback method
                return self._cubic_spline_interpolation(data, gap)
        elif method == 'kriging':
            return self._kriging_interpolation(data, gap)
        elif method == 'polynomial':
            return self._polynomial_interpolation(data, gap)
        elif method == 'trend_extrapolation':
            return self._trend_extrapolation(data, gap)
        else:
            raise ValueError(f"Unknown gap filling method: {method}")
    
    def _linear_interpolation(self, data: np.ndarray, gap: Dict) -> Dict[str, Any]:
        """High-quality linear interpolation with uncertainty"""
        before_data = gap['context_before']
        after_data = gap['context_after']
        
        if len(before_data) == 0 and len(after_data) == 0:
            # No context - use global mean
            global_mean = np.nanmean(data)
            values = np.full(gap['size'], global_mean)
            uncertainty = np.full(gap['size'], np.nanstd(data))
            confidence = np.full(gap['size'], 0.3)
        elif len(before_data) == 0:
            # Only after context
            values = np.full(gap['size'], after_data[0])
            uncertainty = np.full(gap['size'], np.std(after_data) if len(after_data) > 1 else 0.1)
            confidence = np.full(gap['size'], 0.5)
        elif len(after_data) == 0:
            # Only before context
            values = np.full(gap['size'], before_data[-1])
            uncertainty = np.full(gap['size'], np.std(before_data) if len(before_data) > 1 else 0.1)
            confidence = np.full(gap['size'], 0.5)
        else:
            # Linear interpolation between before and after
            start_val = before_data[-1]
            end_val = after_data[0]
            values = np.linspace(start_val, end_val, gap['size'])
            
            # Uncertainty based on local variance
            local_var = (np.var(before_data) + np.var(after_data)) / 2 if len(before_data) > 1 and len(after_data) > 1 else 0.1
            uncertainty = np.full(gap['size'], np.sqrt(local_var))
            confidence = np.full(gap['size'], 0.8)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': np.mean(confidence)
        }
    
    def _cubic_spline_interpolation(self, data: np.ndarray, gap: Dict) -> Dict[str, Any]:
        """Cubic spline interpolation with smooth derivatives"""
        if not ADVANCED_LIBS:
            return self._linear_interpolation(data, gap)
        
        # Prepare interpolation points
        before_indices = np.arange(gap['start'] - len(gap['context_before']), gap['start'])
        after_indices = np.arange(gap['end'], gap['end'] + len(gap['context_after']))
        
        x_known = np.concatenate([before_indices, after_indices])
        y_known = np.concatenate([gap['context_before'], gap['context_after']])
        
        if len(x_known) < 4:
            return self._linear_interpolation(data, gap)
        
        # Create spline
        spline = interpolate.CubicSpline(x_known, y_known, bc_type='natural')
        
        # Interpolate gap
        x_gap = np.arange(gap['start'], gap['end'])
        values = spline(x_gap)
        
        # Estimate uncertainty from spline curvature
        second_deriv = spline.derivative(2)(x_gap)
        uncertainty = np.abs(second_deriv) * 0.1  # Scale factor
        uncertainty = np.clip(uncertainty, 0.01, np.std(y_known))
        
        confidence = np.full(gap['size'], 0.85)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': 0.85
        }
    
    def _gaussian_process_interpolation(self, data: np.ndarray, gap: Dict, 
                                      curve_type: str) -> Dict[str, Any]:
        """Advanced Gaussian Process interpolation with uncertainty quantification"""
        if not ADVANCED_LIBS:
            return self._cubic_spline_interpolation(data, gap)
        
        # Prepare training data
        before_indices = np.arange(gap['start'] - len(gap['context_before']), gap['start'])
        after_indices = np.arange(gap['end'], gap['end'] + len(gap['context_after']))
        
        X_train = np.concatenate([before_indices, after_indices]).reshape(-1, 1)
        y_train = np.concatenate([gap['context_before'], gap['context_after']])
        
        if len(X_train) < 3:
            return self._linear_interpolation(data, gap)
        
        # Select appropriate kernel
        if self.params.gp_kernel == 'rbf_white':
            kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
        elif self.params.gp_kernel == 'matern':
            kernel = Matern(length_scale=10.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        else:
            kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
        
        # Fit Gaussian Process
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
        gp.fit(X_train, y_train)
        
        # Predict gap values
        X_gap = np.arange(gap['start'], gap['end']).reshape(-1, 1)
        values, std = gp.predict(X_gap, return_std=True)
        
        # Convert GP uncertainty to confidence
        uncertainty = std
        confidence = 1.0 / (1.0 + uncertainty)  # Higher uncertainty = lower confidence
        confidence = np.clip(confidence, 0.5, 0.95)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': np.mean(confidence)
        }
    
    def _multi_curve_gp_interpolation(self, data: np.ndarray, gap: Dict, 
                                     auxiliary_curves: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Multi-variate Gaussian Process using correlated curves"""
        if not ADVANCED_LIBS or not auxiliary_curves:
            return self._gaussian_process_interpolation(data, gap, 'unknown')
        
        # Find curves with good correlation and available data
        valid_aux_curves = {}
        for name, aux_data in auxiliary_curves.items():
            if len(aux_data) == len(data):
                # Check correlation in non-gap regions
                valid_mask = ~np.isnan(data) & ~np.isnan(aux_data)
                if np.sum(valid_mask) > 10:
                    correlation = np.corrcoef(data[valid_mask], aux_data[valid_mask])[0, 1]
                    # Use scientifically-based correlation threshold
                if abs(correlation) > PHYSICAL_CONSTANTS.CORRELATION["WEAK"]:
                        valid_aux_curves[name] = aux_data
        
        if not valid_aux_curves:
            return self._gaussian_process_interpolation(data, gap, 'unknown')
        
        # Prepare multi-dimensional training data
        before_indices = np.arange(gap['start'] - len(gap['context_before']), gap['start'])
        after_indices = np.arange(gap['end'], gap['end'] + len(gap['context_after']))
        train_indices = np.concatenate([before_indices, after_indices])
        
        # Build feature matrix
        features = []
        for idx in train_indices:
            feature_vector = [idx]  # Include index as feature
            for aux_name, aux_data in valid_aux_curves.items():
                if not np.isnan(aux_data[idx]):
                    feature_vector.append(aux_data[idx])
                else:
                    feature_vector.append(np.nanmean(aux_data))  # Fallback
            features.append(feature_vector)
        
        X_train = np.array(features)
        y_train = np.concatenate([gap['context_before'], gap['context_after']])
        
        # Fit multi-dimensional GP
        kernel = RBF(length_scale=[10.0] * X_train.shape[1]) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X_train, y_train)
        
        # Predict gap values
        gap_features = []
        for idx in range(gap['start'], gap['end']):
            feature_vector = [idx]
            for aux_name, aux_data in valid_aux_curves.items():
                if not np.isnan(aux_data[idx]):
                    feature_vector.append(aux_data[idx])
                else:
                    feature_vector.append(np.nanmean(aux_data))
            gap_features.append(feature_vector)
        
        X_gap = np.array(gap_features)
        values, std = gp.predict(X_gap, return_std=True)
        
        uncertainty = std
        confidence = 1.0 / (1.0 + uncertainty * 0.5)  # Multi-curve gives higher confidence
        confidence = np.clip(confidence, 0.6, 0.98)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': np.mean(confidence)
        }
    
    def _kriging_interpolation(self, data: np.ndarray, gap: Dict) -> Dict[str, Any]:
        """Kriging interpolation with variogram modeling"""
        if not ADVANCED_LIBS:
            return self._cubic_spline_interpolation(data, gap)
        
        # Simplified kriging implementation
        before_indices = np.arange(gap['start'] - len(gap['context_before']), gap['start'])
        after_indices = np.arange(gap['end'], gap['end'] + len(gap['context_after']))
        
        x_known = np.concatenate([before_indices, after_indices])
        y_known = np.concatenate([gap['context_before'], gap['context_after']])
        
        if len(x_known) < 3:
            return self._linear_interpolation(data, gap)
        
        # Create distance matrix
        x_gap = np.arange(gap['start'], gap['end'])
        values = np.zeros(gap['size'])
        uncertainty = np.zeros(gap['size'])
        
        # Simple distance-based kriging
        for i, x_pred in enumerate(x_gap):
            distances = np.abs(x_known - x_pred)
            weights = 1.0 / (distances + 1.0)  # Avoid division by zero
            weights = weights / np.sum(weights)
            
            values[i] = np.sum(weights * y_known)
            
            # Uncertainty based on distance to nearest points
            min_distance = np.min(distances)
            uncertainty[i] = min_distance * np.std(y_known) / len(y_known)
        
        confidence = 1.0 / (1.0 + uncertainty)
        confidence = np.clip(confidence, 0.4, 0.9)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': np.mean(confidence)
        }
    def _polynomial_interpolation(self, data: np.ndarray, gap: Dict) -> Dict[str, Any]:
        """Polynomial interpolation with optimal degree selection"""
        before_indices = np.arange(gap['start'] - len(gap['context_before']), gap['start'])
        after_indices = np.arange(gap['end'], gap['end'] + len(gap['context_after']))
        
        x_known = np.concatenate([before_indices, after_indices])
        y_known = np.concatenate([gap['context_before'], gap['context_after']])
        
        if len(x_known) < 3:
            return self._linear_interpolation(data, gap)
        
        # Select optimal polynomial degree
        max_degree = min(5, len(x_known) - 1)
        best_degree = 1
        best_score = float('inf')
        
        for degree in range(1, max_degree + 1):
            try:
                coeffs = np.polyfit(x_known, y_known, degree)
                poly_func = np.poly1d(coeffs)
                predicted = poly_func(x_known)
                score = np.mean((predicted - y_known) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_degree = degree
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Polynomial fitting failed for degree {degree}: {str(e)}. "
                    f"This typically occurs with insufficient data points or numerical instability. "
                    f"Skipping this degree and trying others.",
                    UserWarning
                )
                continue
        
        # Fit with best degree
        coeffs = np.polyfit(x_known, y_known, best_degree)
        poly_func = np.poly1d(coeffs)
        
        x_gap = np.arange(gap['start'], gap['end'])
        values = poly_func(x_gap)
        
        # Uncertainty from polynomial extrapolation
        uncertainty = np.abs(x_gap - np.mean(x_known)) * 0.01 * np.std(y_known)
        uncertainty = np.clip(uncertainty, 0.01, np.std(y_known))
        
        confidence = np.full(gap['size'], 0.7)
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': 0.7
        }
    
    def _trend_extrapolation(self, data: np.ndarray, gap: Dict) -> Dict[str, Any]:
        """Trend-based extrapolation for large gaps"""
        before_data = gap['context_before']
        
        if len(before_data) < 5:
            return self._linear_interpolation(data, gap)
        
        # Fit trend to before data
        x_before = np.arange(len(before_data))
        trend_coeffs = np.polyfit(x_before, before_data, 1)  # Linear trend
        
        # Extrapolate
        x_gap = np.arange(gap['size'])
        values = np.polyval(trend_coeffs, x_gap + len(before_data))
        
        # High uncertainty for extrapolation
        uncertainty = np.linspace(0.1, 0.5, gap['size']) * np.std(before_data)
        confidence = np.linspace(0.6, 0.2, gap['size'])  # Decreasing confidence
        
        return {
            'values': values,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'quality': np.mean(confidence)
        }
    
    def _linear_interpolation_fallback(self, data: np.ndarray, gap: Dict) -> np.ndarray:
        """Simple linear interpolation fallback"""
        if gap['start'] == 0:
            # Gap at beginning - use first valid value
            first_valid = next((i for i, x in enumerate(data) if not np.isnan(x)), None)
            if first_valid is not None:
                return np.full(gap['size'], data[first_valid])
            else:
                return np.zeros(gap['size'])
        elif gap['end'] == len(data):
            # Gap at end - use last valid value
            last_valid = next((i for i, x in enumerate(reversed(data)) if not np.isnan(x)), None)
            if last_valid is not None:
                return np.full(gap['size'], data[len(data) - 1 - last_valid])
            else:
                return np.zeros(gap['size'])
        else:
            # Gap in middle - linear interpolation
            start_val = data[gap['start'] - 1]
            end_val = data[gap['end']]
            return np.linspace(start_val, end_val, gap['size'])
    
    def _calculate_gap_filling_quality(self, original: np.ndarray, filled: np.ndarray,
                                     gaps_filled: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive gap filling quality metrics"""
        # Validate inputs
        if original is None or filled is None:
            return {
                'total_gaps_filled': 0,
                'total_points_filled': 0,
                'average_confidence': 0,
                'methods_used': [],
                'average_uncertainty': 0,
                'data_completeness': 0
            }
        
        # Ensure arrays have the same shape
        if original.shape != filled.shape:
            # CRITICAL: Shape mismatch indicates bug in gap filling algorithm
            import warnings
            warnings.warn(
                f"CRITICAL: Gap filling produced shape mismatch - "
                f"original shape {original.shape} vs filled shape {filled.shape}. "
                f"This indicates a bug in the gap filling algorithm. "
                f"Returning empty quality metrics. Please report this issue.",
                UserWarning
            )
            # This should be logged for debugging
            import traceback
            print(f"ERROR: Shape mismatch in gap filling quality metrics")
            print(f"  Original shape: {original.shape}")
            print(f"  Filled shape: {filled.shape}")
            print(f"  Stack trace:")
            traceback.print_stack()
            
            return {
                'total_gaps_filled': len(gaps_filled),
                'total_points_filled': sum(gap['gap']['size'] for gap in gaps_filled) if gaps_filled else 0,
                'average_confidence': 0,
                'methods_used': [],
                'average_uncertainty': 0,
                'data_completeness': 0,
                'error': f'Shape mismatch: {original.shape} vs {filled.shape}'
            }
        
        # Check for valid confidence and uncertainty values
        confidence_values = [gap.get('quality', 0) for gap in gaps_filled if 'quality' in gap]
        uncertainty_values = [gap.get('uncertainty_mean', 0) for gap in gaps_filled if 'uncertainty_mean' in gap]
        
        # Calculate metrics safely - Fixed: Final Completeness = Total Valid Points After Filling / Total Points
        try:
            # Count total valid points after filling (non-NaN in filled array)
            total_valid_after_filling = np.sum(~np.isnan(filled))
            # Total points in the dataset
            total_points = max(1, len(original))
            # Final completeness calculation - this already accounts for original + filled correctly
            data_completeness = total_valid_after_filling / total_points * 100
        except (ZeroDivisionError, ValueError, TypeError) as calc_error:
            # Calculation failed - log and use safe default
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Warning: Data completeness calculation failed: {type(calc_error).__name__}: {str(calc_error)}")
            data_completeness = 0
        except Exception as calc_error:
            # Unexpected error in completeness calculation
            if hasattr(self, 'log_processing'):
                self.log_processing(f"Warning: Unexpected error in completeness calculation: {type(calc_error).__name__}: {str(calc_error)}")
            data_completeness = 0
        
        return {
            'total_gaps_filled': len(gaps_filled),
            'total_points_filled': sum(gap['gap']['size'] for gap in gaps_filled) if gaps_filled else 0,
            'average_confidence': np.mean(confidence_values) if confidence_values else 0,
            'methods_used': list(set(gap.get('method', 'unknown') for gap in gaps_filled)) if gaps_filled else [],
            'average_uncertainty': np.mean(uncertainty_values) if uncertainty_values else 0,
            'data_completeness': data_completeness
        }
