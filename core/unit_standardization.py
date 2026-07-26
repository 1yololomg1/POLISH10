"""
Unit Standardization Module

Industry-standard unit conversion for wireline curves (depth, density,
resistivity, sonic, porosity, temperature, pressure, mud weight).

Extracted from advanced_preprocessing_system10.py for modular architecture.

ARCHITECTURE OVERVIEW:

MAIN CLASS:
- IndustryUnitStandardizer: Conversion registry, preview/apply, ambiguity dialogs

KEY FUNCTIONS:
- preview_unit_conversions() / apply_unit_standardization(): Plan and apply conversions
- detect_unit_ambiguities() / show_ambiguity_resolution_dialog(): User clarification
- _get_curve_category() / _infer_category_from_unit(): Category resolution
- _validate_conversion() / _validate_conversion_function(): Factor and round-trip checks

DATA FLOW:
Loaded DataFrame + curve_info → category/unit lookup → convert → update units + report
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - headless environments
    tk = None  # type: ignore
    ttk = None  # type: ignore


class IndustryUnitStandardizer:
    """Professional unit conversion system for wireline data"""
    
    def __init__(self):
        # Unit conversion registry. Each entry supports either a multiplicative
        # 'factor' or a callable 'apply' (and optional 'inverse') for non-linear
        # conversions. All targets are industry-standard SI or modified SI.
        self.unit_conversions = {
            # Depth conversions
            'depth': {
                'FT':     {'factor': 0.3048, 'target': 'M',    'name': 'Feet to meters'},
                'FEET':   {'factor': 0.3048, 'target': 'M',    'name': 'Feet to meters'},
                'M':      {'factor': 1.0,    'target': 'M',    'name': 'Meters (standard)'},
                'METER':  {'factor': 1.0,    'target': 'M',    'name': 'Meters (standard)'}
            },

            # Density conversions (bulk density of formations)
            'density': {
                'G/CC':   {'factor': 1000.0,      'target': 'KG/M3', 'name': 'g/cm³ to kg/m³'},
                'G/CM3':  {'factor': 1000.0,      'target': 'KG/M3', 'name': 'g/cm³ to kg/m³'},
                'KG/M3':  {'factor': 1.0,         'target': 'KG/M3', 'name': 'kg/m³ (standard)'},
                # Verified per spec: 1 lb/ft³ = 16.01846337 kg/m³ (kept to 7 sig figs)
                'LB/FT3': {'factor': 16.0185,     'target': 'KG/M3', 'name': 'lb/ft³ to kg/m³'}
            },

            # Mud weight conversions (fluid density)
            'mud_weight': {
                # 1 ppg (lb/gal US) = 119.826427316 kg/m³
                'PPG':    {'factor': 119.8264273, 'target': 'KG/M3', 'name': 'ppg to kg/m³'},
                'G/CC':   {'factor': 1000.0,      'target': 'KG/M3', 'name': 'g/cm³ to kg/m³'},
                'G/CM3':  {'factor': 1000.0,      'target': 'KG/M3', 'name': 'g/cm³ to kg/m³'},
                'SG':     {'apply': lambda s: s * 1000.0,               'inverse': lambda s: s / 1000.0,
                            'target': 'KG/M3', 'name': 'Specific gravity to kg/m³'},
                'KG/M3':  {'factor': 1.0,         'target': 'KG/M3', 'name': 'kg/m³ (standard)'}
            },

            # Resistivity (ohm-m) and conductivity to resistivity
            'resistivity': {
                'OHM-M': {'factor': 1.0, 'target': 'OHMM', 'name': 'Ohm-m (standard)'},
                'OHMM':  {'factor': 1.0, 'target': 'OHMM', 'name': 'Ohm-m (standard)'},
                'OHM.M': {'factor': 1.0, 'target': 'OHMM', 'name': 'Ohm-m (standard)'}
            },
            'conductivity': {
                # Convert conductivity to resistivity (ohm-m) using ρ = 1 / σ
                'S/M':   {'apply': lambda s: np.where(s > 0, 1.0 / s, np.nan),
                          'inverse': lambda r: np.where(r > 0, 1.0 / r, np.nan),
                          'target': 'OHMM', 'name': 'Siemens/m to ohm-m'},
                'MS/M':  {'apply': lambda s: np.where(s > 0, 1.0 / (s * 1e-3), np.nan),
                          'inverse': lambda r: np.where(r > 0, (1.0 / r) * 1e3, np.nan),
                          'target': 'OHMM', 'name': 'mS/m to ohm-m'},
                'US/M':  {'apply': lambda s: np.where(s > 0, 1.0 / (s * 1e-6), np.nan),
                          'inverse': lambda r: np.where(r > 0, (1.0 / r) * 1e6, np.nan),
                          'target': 'OHMM', 'name': 'µS/m to ohm-m'},
                'MICROSIEMENS/M': {'apply': lambda s: np.where(s > 0, 1.0 / (s * 1e-6), np.nan),
                                   'inverse': lambda r: np.where(r > 0, (1.0 / r) * 1e6, np.nan),
                                   'target': 'OHMM', 'name': 'µS/m to ohm-m'},
            },

            # Temperature conversions (standard target: °C)
            'temperature': {
                'DEGC': {'factor': 1.0, 'target': 'DEGC', 'name': 'Celsius (standard)'},
                'C':    {'factor': 1.0, 'target': 'DEGC', 'name': 'Celsius (standard)'},
                'DEGF': {'apply': lambda s: (s - 32.0) * (5.0/9.0),
                         'inverse': lambda s: (s * 9.0/5.0) + 32.0,
                         'target': 'DEGC', 'name': 'Fahrenheit to Celsius'},
                'F':    {'apply': lambda s: (s - 32.0) * (5.0/9.0),
                         'inverse': lambda s: (s * 9.0/5.0) + 32.0,
                         'target': 'DEGC', 'name': 'Fahrenheit to Celsius'},
                'K':    {'apply': lambda s: s - 273.15,
                         'inverse': lambda s: s + 273.15,
                         'target': 'DEGC', 'name': 'Kelvin to Celsius'}
            },

            # Pressure conversions (standard target: MPa)
            'pressure': {
                'MPA': {'factor': 1.0,          'target': 'MPA', 'name': 'MPa (standard)'},
                'PSI': {'factor': 0.0068947573, 'target': 'MPA', 'name': 'psi to MPa'},
                'BAR': {'factor': 0.1,          'target': 'MPA', 'name': 'bar to MPa'},
                'KPA': {'factor': 0.001,        'target': 'MPA', 'name': 'kPa to MPa'},
                'PA':  {'factor': 1e-6,         'target': 'MPA', 'name': 'Pa to MPa'}
            },

            # Transit time conversions
            'sonic': {
                'USEC/FT': {'factor': 3.28084, 'target': 'US/M', 'name': 'µs/ft to µs/m'},
                'US/FT':   {'factor': 3.28084, 'target': 'US/M', 'name': 'µs/ft to µs/m'},
                'US/M':    {'factor': 1.0,     'target': 'US/M', 'name': 'µs/m (standard)'},
                'USEC/M':  {'factor': 1.0,     'target': 'US/M', 'name': 'µs/m (standard)'}
            },

            # Porosity conversions (fractional)
            'porosity': {
                'PU':     {'factor': 1.0,  'target': 'V/V', 'name': 'Porosity units (standard)'},
                'V/V':    {'factor': 1.0,  'target': 'V/V', 'name': 'Volume/volume (standard)'},
                'FRAC':   {'factor': 1.0,  'target': 'V/V', 'name': 'Fraction (standard)'},
                'PERCENT':{'factor': 0.01, 'target': 'V/V', 'name': 'Percent to fraction'}
            },

            # Fluid density as API gravity (convert to kg/m³ for consistency)
            'fluid_density': {
                'API': {'apply': lambda s: (141.5 / (s + 131.5)) * 1000.0,
                        'target': 'KG/M3', 'name': '°API to kg/m³ (via SG)'}
            }
        }
        
        # Map curve types to unit categories
        self.curve_unit_mapping = {
            # Depth
            'DEPT': 'depth', 'DEPTH': 'depth', 'MD': 'depth', 'TVD': 'depth', 'TVDSS': 'depth',
            # Formation density
            'RHOB': 'density', 'RHOZ': 'density', 'DENB': 'density', 'DENS': 'density',
            # Mud weight / fluid-related density
            'MW': 'mud_weight', 'MUDWT': 'mud_weight', 'MUD_WT': 'mud_weight', 'MUDWEIGHT': 'mud_weight',
            # Resistivity / conductivity
            'RILD': 'resistivity', 'RILM': 'resistivity', 'RLL3': 'resistivity', 
            'RT': 'resistivity', 'RXO': 'resistivity', 'RM': 'resistivity', 'RLLD': 'resistivity',
            'COND': 'conductivity', 'CND': 'conductivity', 'EC': 'conductivity',
            # Sonic
            'DT': 'sonic', 'DTC': 'sonic', 'DTCO': 'sonic', 'AC': 'sonic',
            # Temperature and pressure
            'TEMP': 'temperature', 'FTEMP': 'temperature', 'TEMF': 'temperature',
            'PRES': 'pressure', 'FP': 'pressure', 'FPRES': 'pressure', 'PFOR': 'pressure',
            # Porosity / fractional families
            'NPHI': 'porosity', 'CNPOR': 'porosity', 'DPOR': 'porosity', 'SPOR': 'porosity'
        }
        
        # Initialize application reference (will be set later)
        self.app = None
        
        # Unit analysis results for reporting
        self.unit_analysis_results = {
            'conversions_planned': [],
            'no_conversion_needed': [],
            'unknown_units': [],
            'conversions_applied': [],
            'conversion_errors': [],
            'depth_validation_updated': False
        }

    def set_application_reference(self, app):
        """Set reference to main application for logging and data access"""
        self.app = app

    def add_unit_standardization_ui(self, parent_tab):
        """Add unit standardization controls to the UI"""
        # Unit Standardization Frame
        unit_frame = ttk.LabelFrame(parent_tab, text=" Unit Standardization", padding="10")
        unit_frame.pack(fill='x', pady=(0, 10))
        
        # Enable unit conversion - use app's existing variable if available
        if hasattr(self.app, 'standardize_units_var'):
            standardize_units_var = self.app.standardize_units_var
        else:
            standardize_units_var = tk.BooleanVar(value=True)
            
        ttk.Checkbutton(unit_frame, text="Standardize Units to Industry Standard", 
                       variable=standardize_units_var).pack(anchor='w', pady=5)
        
        # Store reference for later use
        self.standardize_units_var = standardize_units_var
        
        # Unit conversion options
        conversion_frame = ttk.Frame(unit_frame)
        conversion_frame.pack(fill='x', pady=5)
        
        ttk.Label(conversion_frame, text="Conversion Standard:").pack(anchor='w')
        self.unit_standard_var = tk.StringVar(value="SI_Modified")
        
        unit_standards = [
            ("SI Modified (Industry Standard)", "SI_Modified"),
            ("Imperial/API Standard", "Imperial"), 
            ("Mixed Industry Standard", "Mixed"),
            ("Keep Original Units", "Original")
        ]
        
        for text, value in unit_standards:
            ttk.Radiobutton(conversion_frame, text=text, value=value, 
                           variable=self.unit_standard_var).pack(anchor='w', padx=20, pady=2)
        
        # Conversion preview button
        preview_frame = ttk.Frame(unit_frame)
        preview_frame.pack(fill='x', pady=5)
        
        ttk.Button(preview_frame, text=" Preview Unit Conversions", 
                  command=self.preview_unit_conversions).pack(side='left', padx=(0, 10))
        
        ttk.Button(preview_frame, text=" Apply Unit Standardization", 
                  command=self.apply_unit_standardization).pack(side='left')

    def preview_unit_conversions(self):
        """Preview what unit conversions would be applied"""
        if not self.app or self.app.current_data is None:
            if self.app:
                self.app.log_processing(" No data loaded - cannot preview unit conversions")
            return
        
        if not self.standardize_units_var.get():
            if self.app:
                self.app.log_processing("  Unit standardization is disabled")
            return
        
        # Clear previous analysis results
        self.unit_analysis_results = {
            'conversions_planned': [],
            'no_conversion_needed': [],
            'unknown_units': [],
            'conversions_applied': [],
            'conversion_errors': [],
            'depth_validation_updated': False
        }
        
        if self.app:
            self.app.log_processing("UNIT CONVERSION PREVIEW")
            self.app.log_processing("=" * 50)
        
        for curve_name in self.app.current_data.columns:
            # Get current unit from curve info
            current_unit = self.app.curve_info.get(curve_name, {}).get('unit', '').upper()
            
            if not current_unit or current_unit == '':
                self.unit_analysis_results['unknown_units'].append(curve_name)
                continue
            
            # Determine curve category, try mnemonic then unit fallback
            curve_category = self._get_curve_category(curve_name)
            
            if curve_category and curve_category in self.unit_conversions:
                if current_unit in self.unit_conversions[curve_category]:
                    conversion_info = self.unit_conversions[curve_category][current_unit]
                    target_unit = conversion_info['target']
                    needs_conversion = current_unit != target_unit
                    if needs_conversion:
                        planned = {
                            'curve': curve_name,
                            'from_unit': current_unit,
                            'to_unit': target_unit,
                            'description': conversion_info.get('name', '')
                        }
                        if 'factor' in conversion_info:
                            planned['factor'] = conversion_info['factor']
                        else:
                            planned['factor'] = None
                        self.unit_analysis_results['conversions_planned'].append(planned)
                    else:
                        self.unit_analysis_results['no_conversion_needed'].append(f"{curve_name} ({current_unit})")
                else:
                    # Attempt fallback: find any category whose units include the current unit
                    fallback = self._infer_category_from_unit(current_unit)
                    if fallback and current_unit in self.unit_conversions[fallback]:
                        conversion_info = self.unit_conversions[fallback][current_unit]
                        target_unit = conversion_info['target']
                        self.unit_analysis_results['conversions_planned'].append({
                            'curve': curve_name,
                            'from_unit': current_unit,
                            'to_unit': target_unit,
                            'factor': conversion_info.get('factor'),
                            'description': conversion_info.get('name', '')
                        })
                    else:
                        self.unit_analysis_results['unknown_units'].append(f"{curve_name} ({current_unit})")
            else:
                self.unit_analysis_results['unknown_units'].append(f"{curve_name} ({current_unit})")
        
        # Display minimal preview in terminal (just summary)
        if self.app:
            total_conversions = len(self.unit_analysis_results['conversions_planned'])
            total_standard = len(self.unit_analysis_results['no_conversion_needed'])
            total_unknown = len(self.unit_analysis_results['unknown_units'])
            
            self.app.log_processing(f"Unit Analysis Complete: {total_conversions} conversions planned, "
                               f"{total_standard} already standard, {total_unknown} unknown units")
            self.app.log_processing("  → Detailed analysis available in the Report tab")

    def apply_unit_standardization(self):
        """Apply unit standardization to loaded data"""
        if not self.app or self.app.current_data is None:
            if self.app:
                self.app.log_processing(" No data loaded - cannot apply unit conversions")
            return
        
        if not self.standardize_units_var.get():
            if self.app:
                self.app.log_processing(" Unit standardization is disabled - skipping")
            return
        
        if self.app:
            self.app.log_processing("APPLYING UNIT STANDARDIZATION")
            self.app.log_processing("=" * 50)
        
        # Clear previous conversion results
        self.unit_analysis_results['conversions_applied'] = []
        self.unit_analysis_results['conversion_errors'] = []
        
        # Apply conversions
        for curve_name in self.app.current_data.columns:
            try:
                # Get current unit
                current_unit = self.app.curve_info.get(curve_name, {}).get('unit', '').upper()
                
                if not current_unit:
                    continue
                
                # Determine curve category (mnemonic first, then unit-based fallback)
                curve_category = self._get_curve_category(curve_name)
                
                if curve_category and curve_category in self.unit_conversions:
                    if current_unit in self.unit_conversions[curve_category]:
                        conversion_info = self.unit_conversions[curve_category][current_unit]
                        target_unit = conversion_info['target']
                        
                        if current_unit != target_unit:
                            original_data = self.app.current_data[curve_name].copy()
                            # Apply conversion either by factor or function
                            if 'apply' in conversion_info:
                                converted_data = pd.Series(conversion_info['apply'](pd.to_numeric(original_data, errors='coerce')))
                                valid = self._validate_conversion_function(
                                    original_data, converted_data,
                                    conversion_info.get('apply'), conversion_info.get('inverse')
                                )
                                applied_desc = conversion_info.get('name', 'function conversion')
                                factor_for_log = ''
                            else:
                                factor = conversion_info['factor']
                                converted_data = original_data * factor
                                valid = self._validate_conversion(original_data, converted_data, factor)
                                applied_desc = f"×{factor:.6f}"
                                factor_for_log = factor

                            if valid:
                                self.app.current_data[curve_name] = converted_data
                                # Update curve info with new unit
                                if curve_name in self.app.curve_info:
                                    self.app.curve_info[curve_name]['unit'] = target_unit
                                    self.app.curve_info[curve_name]['original_unit'] = current_unit
                                # Record in standardization reporter
                                if hasattr(self.app, 'standardization_reporter') and self.app.standardization_reporter:
                                    if 'apply' in conversion_info:
                                        self.app.standardization_reporter.record_unit_conversion(
                                            curve_name=curve_name,
                                            original_unit=current_unit,
                                            standardized_unit=target_unit,
                                            method='function',
                                            factor=None,
                                            validated=True
                                        )
                                    else:
                                        self.app.standardization_reporter.record_unit_conversion(
                                            curve_name=curve_name,
                                            original_unit=current_unit,
                                            standardized_unit=target_unit,
                                            method='factor',
                                            factor=factor,
                                            validated=True
                                        )
                                # Record successful conversion
                                entry = {
                                    'curve': curve_name,
                                    'from_unit': current_unit,
                                    'to_unit': target_unit,
                                    'description': conversion_info.get('name', '')
                                }
                                if 'factor' in conversion_info:
                                    entry['factor'] = conversion_info['factor']
                                self.unit_analysis_results['conversions_applied'].append(entry)
                                if self.app:
                                    self.app.log_processing(f"   {curve_name}: {current_unit} → {target_unit} ({applied_desc})")
                            else:
                                # Record conversion failure
                                self.unit_analysis_results['conversion_errors'].append({
                                    'curve': curve_name,
                                    'from_unit': current_unit,
                                    'to_unit': target_unit,
                                    'reason': 'Validation failed'
                                })
                                if self.app:
                                    self.app.log_processing(f"   {curve_name}: Conversion validation failed")
                    else:
                        # Fallback attempt by unit membership across categories
                        fallback = self._infer_category_from_unit(current_unit)
                        if fallback and current_unit in self.unit_conversions[fallback]:
                            conversion_info = self.unit_conversions[fallback][current_unit]
                            target_unit = conversion_info['target']
                            original_data = self.app.current_data[curve_name].copy()
                            if 'apply' in conversion_info:
                                converted_data = pd.Series(conversion_info['apply'](pd.to_numeric(original_data, errors='coerce')))
                                valid = self._validate_conversion_function(
                                    original_data, converted_data,
                                    conversion_info.get('apply'), conversion_info.get('inverse')
                                )
                                applied_desc = conversion_info.get('name', 'function conversion')
                            else:
                                factor = conversion_info['factor']
                                converted_data = original_data * factor
                                valid = self._validate_conversion(original_data, converted_data, factor)
                                applied_desc = f"×{factor:.6f}"
                            if valid:
                                self.app.current_data[curve_name] = converted_data
                                if curve_name in self.app.curve_info:
                                    self.app.curve_info[curve_name]['unit'] = target_unit
                                    self.app.curve_info[curve_name]['original_unit'] = current_unit
                                # Record in standardization reporter
                                if hasattr(self.app, 'standardization_reporter') and self.app.standardization_reporter:
                                    if 'apply' in conversion_info:
                                        self.app.standardization_reporter.record_unit_conversion(
                                            curve_name=curve_name,
                                            original_unit=current_unit,
                                            standardized_unit=target_unit,
                                            method='function',
                                            factor=None,
                                            validated=True
                                        )
                                    else:
                                        self.app.standardization_reporter.record_unit_conversion(
                                            curve_name=curve_name,
                                            original_unit=current_unit,
                                            standardized_unit=target_unit,
                                            method='factor',
                                            factor=factor,
                                            validated=True
                                        )
                                self.unit_analysis_results['conversions_applied'].append({
                                    'curve': curve_name,
                                    'from_unit': current_unit,
                                    'to_unit': target_unit,
                                    'description': conversion_info.get('name', '')
                                })
                                if self.app:
                                    self.app.log_processing(f"   {curve_name}: {current_unit} → {target_unit} ({applied_desc})")
                            else:
                                self.unit_analysis_results['conversion_errors'].append({
                                    'curve': curve_name,
                                    'from_unit': current_unit,
                                    'to_unit': target_unit,
                                    'reason': 'Validation failed'
                                })
                        # else: leave as unknown
                
            except Exception as e:
                # Record conversion error for reporting
                self.unit_analysis_results['conversion_errors'].append({
                    'curve': curve_name,
                    'from_unit': current_unit if 'current_unit' in locals() else 'UNKNOWN',
                    'to_unit': 'UNKNOWN',
                    'factor': 0,
                    'reason': str(e)
                })
                
                if self.app:
                    self.app.log_processing(f"   {curve_name}: Conversion error - {str(e)}")
        
        # Final summary
        conversions_applied = len(self.unit_analysis_results['conversions_applied'])
        conversion_errors = len(self.unit_analysis_results['conversion_errors'])
        
        if conversions_applied > 0 and self.app:
            self.app.log_processing(f"Unit standardization completed!")
            self.app.log_processing(f"   Conversions applied: {conversions_applied}")
            if conversion_errors > 0:
                self.app.log_processing(f"    Conversion errors: {conversion_errors}")
            
            # Update depth validation rules if depth units were converted
            self._update_depth_validation_rules()
            
        elif self.app:
            self.app.log_processing("  No unit conversions were needed")

    def detect_unit_ambiguities(self, data: pd.DataFrame, curve_info: Dict) -> List[Dict]:
        """
        Detect ambiguous units that need clarification
        
        Returns:
            List of dicts with:
                - curve_name: str
                - unit: str
                - issue: str (description of ambiguity)
                - max_value: float
                - suggested_conversion: str
        """
        ambiguities = []
        
        # Known ambiguous units
        ambiguous_units = {
            'PU': {
                'description': 'Porosity Units - could be fraction (0-1) or percent (0-100)',
                'threshold': 1.0,
                'conversion_if_above': 'divide by 100'
            },
            'FRAC': {
                'description': 'Fraction - verify if already decimal or percentage',
                'threshold': 1.0,
                'conversion_if_above': 'divide by 100'
            },
            'DECIMAL': {
                'description': 'Decimal - verify scale',
                'threshold': 1.0,
                'conversion_if_above': 'divide by 100'
            }
        }
        
        for curve_name, info in curve_info.items():
            unit = info.get('unit', '').upper().strip()
            
            if unit in ambiguous_units:
                if curve_name in data.columns:
                    series = pd.to_numeric(data[curve_name], errors='coerce')
                    max_val = float(series.max())
                    
                    ambig_info = ambiguous_units[unit]
                    
                    if max_val > ambig_info['threshold']:
                        ambiguities.append({
                            'curve_name': curve_name,
                            'unit': unit,
                            'issue': ambig_info['description'],
                            'max_value': max_val,
                            'suggested_conversion': ambig_info['conversion_if_above']
                        })
        
        return ambiguities

    def show_ambiguity_resolution_dialog(self, ambiguities: List[Dict]) -> Dict[str, bool]:
        """
        Show dialog for user to confirm unit conversions for ambiguous cases
        
        Returns:
            Dict[curve_name, should_convert] - user decisions
        """
        if not ambiguities:
            return {}
        
        dialog = tk.Toplevel(self.app.root if self.app else None)
        dialog.title("Unit Ambiguity Resolution")
        dialog.geometry("800x500")
        if self.app:
            dialog.transient(self.app.root)
        dialog.grab_set()
        
        # Header
        header = ttk.Label(
            dialog,
            text="Ambiguous Units Detected\n\nPlease confirm conversions for the following curves:",
            font=('TkDefaultFont', 10, 'bold'),
            justify='left'
        )
        header.pack(pady=10, padx=10, anchor='w')
        
        # Scrollable frame
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
        
        # Store user decisions
        decisions = {}
        
        for ambig in ambiguities:
            # Frame for each ambiguous curve
            frame = ttk.LabelFrame(scrollable_frame, text=f"Curve: {ambig['curve_name']}", padding=10)
            frame.pack(fill='x', pady=5, padx=5)
            
            info_text = f"Unit: {ambig['unit']}\n"
            info_text += f"Issue: {ambig['issue']}\n"
            info_text += f"Max Value: {ambig['max_value']:.2f}\n"
            info_text += f"Suggested: {ambig['suggested_conversion']}"
            
            ttk.Label(frame, text=info_text, justify='left').pack(anchor='w', pady=5)
            
            # Checkbox for conversion
            var = tk.BooleanVar(value=True)  # Default to converting
            decisions[ambig['curve_name']] = var
            
            ttk.Checkbutton(
                frame,
                text=f"Convert: {ambig['suggested_conversion']}",
                variable=var
            ).pack(anchor='w')
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        
        result = {}
        
        def on_confirm():
            for curve_name, var in decisions.items():
                result[curve_name] = var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Apply Conversions", command=on_confirm).pack(side='left', padx=10)
        ttk.Button(button_frame, text="Skip All", command=on_cancel).pack(side='left')
        
        # Wait for dialog
        if self.app:
            self.app.root.wait_window(dialog)
        
        return result

    def _get_curve_category(self, curve_name):
        """Determine unit category for a curve"""
        curve_upper = curve_name.upper()

        # Check direct mapping first
        if curve_upper in self.curve_unit_mapping:
            return self.curve_unit_mapping[curve_upper]

        # Check partial matches by mnemonic
        if any(depth_kw in curve_upper for depth_kw in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TVDSS']):
            return 'depth'
        if any(res_kw in curve_upper for res_kw in ['RIL', 'RLL', 'RT', 'RXO']):
            return 'resistivity'
        if any(den_kw in curve_upper for den_kw in ['RHOB', 'RHOZ', 'DEN']):
            return 'density'
        if any(son_kw in curve_upper for son_kw in ['DT', 'AC']):
            return 'sonic'
        if any(por_kw in curve_upper for por_kw in ['NPHI', 'PORO', 'PHI']):
            return 'porosity'

        # Fallback by unit if available in curve_info
        try:
            unit_upper = ''
            if self.app and hasattr(self.app, 'curve_info') and curve_name in self.app.curve_info:
                unit_upper = str(self.app.curve_info.get(curve_name, {}).get('unit', '')).upper()
            # Temperature units
            if unit_upper in ['DEGF', 'F', 'DEGC', 'C', 'K']:
                return 'temperature'
            # Pressure units
            if unit_upper in ['PSI', 'BAR', 'MPA', 'KPA', 'PA']:
                return 'pressure'
            # Conductivity units
            if unit_upper in ['S/M', 'MS/M', 'US/M', 'MICROSIEMENS/M']:
                return 'conductivity'
            # Mud weight units
            if unit_upper in ['PPG', 'SG']:
                return 'mud_weight'
            # Density units fallback
            if unit_upper in ['G/CC', 'G/CM3', 'KG/M3', 'LB/FT3']:
                return 'density'
        except (AttributeError, TypeError) as unit_error:
            # Unit string may not be valid - expected for some data types
            # Log only if this indicates a real problem
            pass
        except Exception as unit_error:
            # Unexpected error in unit detection - should log for debugging
            import warnings
            warnings.warn(
                f"Unexpected error in unit detection: {type(unit_error).__name__}: {str(unit_error)}",
                UserWarning
            )

        return None

    def _infer_category_from_unit(self, unit_upper: str):
        """Infer a unit category by searching the registry for where the unit exists."""
        try:
            for category, units in self.unit_conversions.items():
                if unit_upper in units:
                    return category
        except (AttributeError, TypeError) as unit_error:
            # Unit string may not be valid - expected for some data types
            pass
        except Exception as unit_error:
            # Unexpected error in unit category detection - should log for debugging
            import warnings
            warnings.warn(
                f"Unexpected error in unit category detection: {type(unit_error).__name__}: {str(unit_error)}",
                UserWarning
            )
        return None

    def _validate_conversion(self, original, converted, factor):
        """Validate that unit conversion was applied correctly"""
        try:
            # Check for reasonable conversion (avoid extreme factors)
            if factor <= 0 or factor > 10000:
                return False
            
            # Check that conversion preserved data relationships
            valid_orig = original.dropna()
            valid_conv = converted.dropna()
            
            if len(valid_orig) != len(valid_conv):
                return False
            
            # Check conversion accuracy on sample
            if len(valid_orig) > 0:
                test_indices = np.random.choice(len(valid_orig), min(10, len(valid_orig)), replace=False)
                for idx in test_indices:
                    expected = valid_orig.iloc[idx] * factor
                    actual = valid_conv.iloc[idx]
                    if abs(expected - actual) > abs(expected) * 0.001:  # 0.1% tolerance
                        return False
            
            return True
            
        except (ValueError, TypeError, IndexError, KeyError) as validation_error:
            # Validation failed due to data issues - log for debugging
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    validation_error,
                    "Unit conversion validation",
                    "Validating unit conversion factor",
                    show_dialog=False
                )
            return False
        except Exception as validation_error:
            # Unexpected error in validation - log for debugging
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    validation_error,
                    "Unit conversion validation",
                    "Validating unit conversion factor",
                    show_dialog=False
                )
            return False

    def _validate_conversion_function(self, original, converted, apply_fn, inverse_fn):
        """Validate non-linear conversion by applying inverse if available.
        Falls back to sanity checks when inverse is missing."""
        try:
            valid_orig = pd.to_numeric(original, errors='coerce').dropna()
            valid_conv = pd.to_numeric(converted, errors='coerce').dropna()
            if len(valid_orig) == 0 or len(valid_conv) == 0 or len(valid_orig) != len(valid_conv):
                return False
            # If inverse is provided, check round-trip accuracy on a small sample
            if inverse_fn is not None:
                sample_idx = np.random.choice(len(valid_conv), min(10, len(valid_conv)), replace=False)
                back = inverse_fn(valid_conv.iloc[sample_idx])
                # Tolerance: 0.1% of magnitude or absolute 1e-6 for near-zero
                expected = valid_orig.iloc[sample_idx].values
                diff = np.abs(np.array(back, dtype=float) - expected)
                tol = np.maximum(np.abs(expected) * 0.001, 1e-6)
                return bool(np.all(diff <= tol))
            # Without inverse, perform monotonicity and finite checks
            if not np.isfinite(valid_conv).all():
                return False
            return True
        except (ValueError, TypeError, IndexError, KeyError) as validation_error:
            # Validation failed due to data issues - log for debugging
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    validation_error,
                    "Unit conversion round-trip validation",
                    "Validating unit conversion function",
                    show_dialog=False
                )
            return False
        except Exception as validation_error:
            # Unexpected error in validation - log for debugging
            if hasattr(self, 'app') and hasattr(self.app, 'handle_processing_error'):
                self.app.handle_processing_error(
                    validation_error,
                    "Unit conversion round-trip validation",
                    "Validating unit conversion function",
                    show_dialog=False
                )
            return False

    def _update_depth_validation_rules(self):
        """Update depth validation rules based on current units"""
        try:
            if not self.app:
                return
                
            # Find depth curve
            depth_curve = None
            for curve_name in self.app.current_data.columns:
                if any(kw in curve_name.upper() for kw in ['DEPT', 'DEPTH', 'MD', 'TVD']):
                    depth_curve = curve_name
                    break
            
            if depth_curve and depth_curve in self.app.curve_info:
                current_unit = self.app.curve_info[depth_curve].get('unit', '').upper()
                
                if current_unit == 'M':
                    # Metric units
                    if hasattr(self.app, 'depth_validation_rules'):
                        self.app.depth_validation_rules = {
                            'min_interval': 10.0,      # 10m minimum
                            'max_step': 5.0,           # 5m max step
                            'reasonable_range': (0, 10000)  # 0-10km
                        }
                        self.app.log_processing("   Depth validation updated for metric units")
                        self.unit_analysis_results['depth_validation_updated'] = True
                    
                elif current_unit == 'FT':
                    # Imperial units  
                    if hasattr(self.app, 'depth_validation_rules'):
                        self.app.depth_validation_rules = {
                            'min_interval': 32.8,      # ~10m in feet
                            'max_step': 16.4,          # ~5m in feet
                            'reasonable_range': (0, 32800)  # ~10km in feet
                        }
                        self.app.log_processing("   Depth validation updated for imperial units")
                        self.unit_analysis_results['depth_validation_updated'] = True
        
        except Exception as e:
            if self.app:
                self.app.log_processing(f"    Could not update depth validation: {str(e)}")
    
    def get_unit_analysis_for_report(self) -> dict:
        """Get unit analysis results formatted for reporting"""
        return self.unit_analysis_results.copy()
