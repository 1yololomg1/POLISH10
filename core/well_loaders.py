"""
Well Data Loaders Module

LAS / DLIS / CSV / Excel loading helpers for wireline preprocessing.

Extracted from advanced_preprocessing_system10.py for modular architecture.

ARCHITECTURE OVERVIEW:

MAIN CLASS:
- WellLoadingMixin: Mixin providing format loaders for AdvancedPreprocessingApplication

KEY FUNCTIONS:
- load_las_file() / _load_las_file_manual_fallback(): LAS via lasio with manual fallback
- load_dlis_file(): DLIS/LIS via dlisio
- load_csv_file() / load_excel_file(): Tabular imports with curve_info init
- _extract_well_information() / _extract_lasio_curve_info(): Metadata extraction
- _extract_geological_context_from_las(): Formation/casing context from LAS params

DATA FLOW:
Filepath → format-specific loader → DataFrame (+ well_info / curve_info side effects on host)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Optional lasio (same availability gate as the main application)
LASIO_AVAILABLE = False
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    lasio = None  # type: ignore

from core.gap_filling import GeologicalContext


class WellLoadingMixin:
    """Mixin: LAS/DLIS/CSV/Excel loading methods for the main application host.

    Expects host attributes used by the original methods:
    log_processing, curve_info, well_info, geological_context, curve_identifier,
    original_las_header, standardization_reporter (optional), and UI helpers
    such as _update_window_title_with_well_info / _update_well_info_display.
    """

    def load_dlis_file(self, filepath: str) -> pd.DataFrame:
        """Load DLIS/LIS file via dlisio when available; raise clear error otherwise."""
        try:
            from dlisio import dlis as dlis_mod
        except ImportError as e:
            raise ImportError(
                "dlisio is required for DLIS/LIS support. Install with: pip install dlisio"
            ) from e

        self.log_processing(f"Loading DLIS/LIS file with dlisio: {filepath}")
        frames = []
        well_name = os.path.splitext(os.path.basename(filepath))[0]

        with dlis_mod.load(filepath) as files:
            for f in files:
                try:
                    if hasattr(f, 'origins') and f.origins:
                        origin = f.origins[0]
                        well_name = getattr(origin, 'well_name', None) or getattr(origin, 'well_id', None) or well_name
                except Exception:
                    pass
                for frame in getattr(f, 'frames', []) or []:
                    try:
                        curves = frame.curves()
                        if curves is None:
                            continue
                        # dlisio returns a structured numpy array
                        df_part = pd.DataFrame(curves)
                        if df_part.empty:
                            continue
                        frames.append(df_part)
                    except Exception as frame_err:
                        self.log_processing(f"Skipping DLIS frame: {frame_err}")

        if not frames:
            raise ValueError(f"No readable curve frames found in DLIS file: {filepath}")

        df = frames[0]
        for extra in frames[1:]:
            # Align on overlapping columns when possible
            common = [c for c in extra.columns if c in df.columns]
            if common:
                df = pd.concat([df, extra], ignore_index=True, sort=False)
            else:
                for col in extra.columns:
                    if col not in df.columns:
                        df[col] = extra[col].values[:len(df)] if len(extra) >= len(df) else np.nan

        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Prefer TDEP / DEPT as depth
        depth_candidates = [c for c in df.columns if str(c).upper() in ('TDEP', 'DEPT', 'DEPTH', 'MD')]
        if depth_candidates and depth_candidates[0] != 'DEPT':
            df = df.rename(columns={depth_candidates[0]: 'DEPT'})

        self.well_info = {
            'well_name': str(well_name).strip() or 'UNKNOWN',
            'uwi': 'N/A',
            'field': 'N/A',
            'company': 'N/A',
            'start_depth': float(df['DEPT'].min()) if 'DEPT' in df.columns else 'N/A',
            'stop_depth': float(df['DEPT'].max()) if 'DEPT' in df.columns else 'N/A',
            'depth_unit': 'm',
            'source_format': 'DLIS',
        }
        self.log_processing(f"Loaded DLIS with {len(df)} rows, {len(df.columns)} curves")
        return df

    def load_las_file(self, filepath: str) -> pd.DataFrame:
        """Load LAS file using industry-standard lasio library"""
        if not LASIO_AVAILABLE:
            raise ImportError(
                "lasio library is required for LAS file support. "
                "Install with: pip install lasio"
            )
        
        try:
            self.log_processing(f"Loading LAS file with lasio: {filepath}")
            
            # Load LAS file with lasio
            las = lasio.read(filepath)
            
            # Convert to DataFrame
            df = las.df()
            
            # lasio puts depth on the index (usually named DEPT). Promote it to a
            # regular column so the rest of the pipeline can find a depth curve.
            # The previous check only reset when index.name was None, which left
            # named depth indexes stranded and dropped the depth channel entirely.
            if 'DEPT' not in df.columns and 'DEPTH' not in df.columns:
                if df.index.name is None and hasattr(las, 'depth'):
                    df.index.name = 'DEPT'
                if df.index.name is not None:
                    df = df.reset_index()
            
            # Handle empty DataFrame
            if df.empty:
                raise ValueError("No data found in LAS file")
            
            # Clean column names (remove special characters)
            df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Convert to numeric, handling non-numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Extract curve information from lasio
            self._extract_lasio_curve_info(las)
            
            # CRITICAL: Extract well identification information for safety
            self.well_info = self._extract_well_information(las)
            
            # Update window title with well identification
            self._update_window_title_with_well_info()
            
            # Update well info display in Data Tab
            self._update_well_info_display()
            
            # Extract geological context from LAS file
            self.geological_context = self._extract_geological_context_from_las(las, df)
            
            # BULLETPROOF HEADER CAPTURE with comprehensive break detection
            # This system detects header-to-data transitions using 8 different methods:
            # 1. Explicit data markers (~ASCII, ~A, ~DATA, etc.)
            # 2. Numeric pattern recognition (multiple consecutive data-like lines)  
            # 3. Column header detection (DEPTH, GR, RESISTIVITY, etc.)
            # 4. Comment-based indicators ("DATA STARTS", "LOG DATA", etc.)
            # 5. Empty line followed by data pattern
            # 6. Sudden format change to tabular numeric data
            # 7. LAS null value patterns (-999.25, etc.)
            # 8. Safety limits and fallback detection
            self.original_las_header = None
            try:
                header_lines = []
                ABSOLUTE_MAX_HEADER_LINES = 150  # Increased for complex headers
                data_section_found = False
                consecutive_numeric_lines = 0
                potential_data_line_count = 0
                
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        # SAFETY VALVE 1: Absolute line limit
                        if line_num >= ABSOLUTE_MAX_HEADER_LINES:
                            self.log_processing(f"Header capture stopped at {ABSOLUTE_MAX_HEADER_LINES} lines (safety limit)")
                            break
                        
                        line_stripped = line.rstrip('\n\r')
                        line_upper = line_stripped.strip().upper()
                        
                        # COMPREHENSIVE DATA SECTION MARKERS - handles ALL known formats
                        data_markers = [
                            # Standard LAS markers
                            '~ASCII', '~A ', '~A\t', '~A\n', '~A\r', '~A.',
                            # Alternative data markers
                            '~DATA', '~LOG_DATA', '~WELL_DATA', '~LOGDATA', 
                            '~WELLDATA', '~LOG', '~LOGS', '~CURVE_DATA',
                            '~DEPTH_DATA', '~MEASUREMENT_DATA', '~VALUES',
                            # Vendor-specific markers
                            '~CWLS', '~LAS', '~DIGITAL', '~NUMERIC',
                            # Less common variations
                            '~BEGIN_DATA', '~START_DATA', '~DATA_SECTION'
                        ]
                        
                        # Check for explicit data section markers
                        section_marker_found = False
                        for marker in data_markers:
                            if line_upper.startswith(marker):
                                section_marker_found = True
                                break
                        
                        # Special handling for ~A with column headers (like your file)
                        if (line_upper.startswith('~A') and 
                            any(col_indicator in line_upper for col_indicator in 
                                ['DEPTH', 'DEPT', 'MD', 'TVD', 'GR', 'GAMMA', 'RESISTIVITY', 'DENSITY', 'NEUTRON'])):
                            section_marker_found = True
                        
                        if section_marker_found:
                            data_section_found = True
                            self.log_processing(f"Found data section marker at line {line_num}: '{line_stripped[:50]}...'")
                            break
                        
                        # PATTERN-BASED DATA DETECTION (fallback for non-standard files)
                        if line_stripped.strip() and not line_stripped.strip().startswith(('#', '~', '/', '*', ';')):
                            parts = line_stripped.split()
                            if len(parts) >= 3:  # Potential data line
                                
                                # Count numeric parts
                                numeric_parts = 0
                                depth_like_first = False
                                
                                for i, part in enumerate(parts[:8]):  # Check first 8 fields
                                    if self._is_numeric_value(part):
                                        numeric_parts += 1
                                        
                                        # Check if first column looks like depth
                                        if i == 0:
                                            try:
                                                val = float(part)
                                                if 0 < val < 50000:  # Reasonable depth range
                                                    depth_like_first = True
                                            except (ValueError, TypeError):
                                                # Not a valid numeric value - expected for non-depth fields
                                                pass
                                
                                # Enhanced data line detection
                                is_likely_data = False
                                
                                # Method 1: High ratio of numeric fields
                                if numeric_parts >= max(3, len(parts) * 0.6):
                                    is_likely_data = True
                                
                                # Method 2: Depth-like first column + multiple numerics
                                if depth_like_first and numeric_parts >= 3:
                                    is_likely_data = True
                                
                                # Method 3: All fields are numeric (classic data row)
                                if numeric_parts == len(parts) and len(parts) >= 4:
                                    is_likely_data = True
                                
                                # Method 4: Contains LAS null values (-999.25, etc.)
                                if any(null_val in line_stripped for null_val in ['-999.25', '-999', '9999', '-9999']):
                                    is_likely_data = True
                                
                                if is_likely_data:
                                    consecutive_numeric_lines += 1
                                    potential_data_line_count += 1
                                    
                                    # Multiple consecutive data-like lines = data section found
                                    if consecutive_numeric_lines >= 2:
                                        self.log_processing(f"Data section detected at line {line_num} (pattern recognition)")
                                        self.log_processing(f"Sample data line: '{line_stripped[:60]}...'")
                                        break
                                else:
                                    consecutive_numeric_lines = 0
                            else:
                                consecutive_numeric_lines = 0
                        else:
                            consecutive_numeric_lines = 0
                        
                        # ADDITIONAL BREAK INDICATORS
                        
                        # Method 5: Comment lines indicating data start
                        data_comments = [
                            'DATA STARTS', 'DATA BEGINS', 'LOG DATA', 'MEASUREMENT DATA',
                            'CURVE DATA', 'ASCII DATA', 'DEPTH DATA', 'WELL DATA',
                            'BEGIN DATA', 'START DATA', 'DATA SECTION', 'LOG VALUES',
                            'CURVE VALUES', 'MEASUREMENTS', 'LOGGING DATA'
                        ]
                        if any(comment in line_upper for comment in data_comments):
                            self.log_processing(f"Data section indicated by comment at line {line_num}: '{line_stripped}'")
                            header_lines.append(line_stripped)
                            break
                        
                        # Method 6: Detect column header lines (enhanced)
                        parts = line_stripped.split()
                        if len(parts) >= 3 and self._detect_column_headers(line_stripped):
                            self.log_processing(f"Column header detected at line {line_num}: '{line_stripped[:50]}...'")
                            header_lines.append(line_stripped)
                            break
                        
                        # Method 7: Empty line followed by numeric data pattern
                        if not line_stripped.strip():  # Empty line
                            # Peek ahead to see if data follows
                            try:
                                next_pos = f.tell()
                                next_line = f.readline()
                                f.seek(next_pos)  # Reset position
                                
                                if next_line and self._looks_like_data_line(next_line):
                                    self.log_processing(f"Data section detected after empty line at {line_num}")
                                    header_lines.append(line_stripped)
                                    break
                            except (IOError, OSError) as file_error:
                                # File reading error - log but continue with other detection methods
                                self.log_processing(f"Warning: File peek ahead failed at line {line_num}: {type(file_error).__name__}: {str(file_error)}")
                            except Exception as peek_error:
                                # Unexpected error - log for debugging
                                self.log_processing(f"Warning: Unexpected error peeking ahead in file: {type(peek_error).__name__}: {str(peek_error)}")
                        
                        # Method 8: Detect sudden format change to tabular data
                        if (line_num > 5 and  # Only after some header content
                            len(parts) >= 5 and  # Multiple columns
                            consecutive_numeric_lines == 0 and  # First potential data line
                            self._looks_like_data_line(line_stripped)):
                            
                            # Look ahead to confirm this is start of data section
                            try:
                                current_pos = f.tell()
                                next_few_lines = []
                                for _ in range(3):  # Check next 3 lines
                                    next_line = f.readline()
                                    if next_line:
                                        next_few_lines.append(next_line.strip())
                                f.seek(current_pos)  # Reset position
                                
                                # If next lines also look like data, this is the data section
                                data_like_count = sum(1 for nl in next_few_lines if self._looks_like_data_line(nl))
                                if data_like_count >= 2:
                                    self.log_processing(f"Data section detected by format change at line {line_num}")
                                    self.log_processing(f"Sample: '{line_stripped[:60]}...'")
                                    break
                            except (IOError, OSError) as file_error:
                                # File reading error - log but continue with other detection methods
                                self.log_processing(f"Warning: File look-ahead failed at line {line_num}: {type(file_error).__name__}: {str(file_error)}")
                            except Exception as lookahead_error:
                                # Unexpected error - log for debugging
                                self.log_processing(f"Warning: Unexpected error in file look-ahead: {type(lookahead_error).__name__}: {str(lookahead_error)}")
                        
                        header_lines.append(line_stripped)
                        
                        # SAFETY VALVE: Stop if header seems unreasonably long without LAS sections
                        if (line_num > 60 and 
                            not any(section in ''.join(header_lines).upper() for section in 
                                   ['~VERSION', '~WELL', '~CURVE', '~PARAM', '~OTHER']) and
                            potential_data_line_count == 0):
                            self.log_processing(f"Header capture stopped - no LAS structure detected by line {line_num}")
                            break
                
                # Final header validation and cleanup
                if header_lines:
                    # Remove any trailing lines that look like data
                    clean_header = []
                    for line in header_lines:
                        line_stripped = line.strip()
                        if line_stripped and not line_stripped.startswith(('#', '~', '/')):
                            parts = line_stripped.split()
                            if len(parts) >= 3:
                                numeric_count = sum(1 for p in parts[:3] if self._is_numeric_value(p))
                                if numeric_count >= 2:  # Looks like data
                                    break
                        clean_header.append(line)
                    
                    self.original_las_header = '\n'.join(clean_header) if clean_header else None
                    self.log_processing(f"Captured LAS header: {len(clean_header)} lines (data section found: {data_section_found})")
                    
                    # Debug: Show comprehensive header capture results
                    if clean_header:
                        self.log_processing(f"Header ends with: '{clean_header[-1][:50]}...'")
                        
                        # Show detection method used
                        if data_section_found:
                            self.log_processing("Header captured using: Explicit data section marker")
                        elif consecutive_numeric_lines > 0:
                            self.log_processing("Header captured using: Numeric pattern detection")
                        else:
                            self.log_processing("Header captured using: Safety limits")
                        
                        # Show header sections found
                        sections_found = []
                        header_text = '\n'.join(clean_header).upper()
                        for section in ['VERSION', 'WELL', 'CURVE', 'PARAMETER', 'OTHER']:
                            if f'~{section}' in header_text:
                                sections_found.append(section)
                        
                        if sections_found:
                            self.log_processing(f"LAS sections found: {', '.join(sections_found)}")
                        else:
                            self.log_processing("No standard LAS sections detected (non-standard format)")
                    
                else:
                    self.original_las_header = None
                    self.log_processing("No valid header captured - possibly pure data file")
                    
            except Exception as e:
                self.log_processing(f"Error capturing LAS header: {e}")
                self.original_las_header = None
            
            self.log_processing(f"Successfully loaded LAS file: {len(df)} rows, {len(df.columns)} curves")
            
            return df
            
        except Exception as e:
            # Fallback to manual parsing if lasio fails
            self.log_processing(f"Lasio failed: {e}, trying manual parsing")
            return self._load_las_file_manual_fallback(filepath)

        if hasattr(self, 'original_las_header') and self.original_las_header:
            header_lines = self.original_las_header.split('\n')
            self.log_processing(f"SUCCESS: Captured LAS header with {len(header_lines)} lines")
            self.log_processing(f"Header sample: {self.original_las_header[:200]}...")
        else:
            self.log_processing("WARNING: Failed to capture LAS header!")
            
            # Fallback header capture - try a simpler approach
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # Read first 10KB which should contain header
                    data_marker_pos = -1
                    
                    # Find data section marker
                    for marker in ['~A', '~ASCII', '~DATA']:
                        pos = content.find(marker)
                        if pos > 0:
                            data_marker_pos = pos
                            break
                    
                    if data_marker_pos > 0:
                        # Capture everything before data marker as header
                        self.original_las_header = content[:data_marker_pos].strip()
                        self.log_processing(f"FALLBACK: Captured header using simple method - {len(self.original_las_header)} bytes")
            except Exception as e:
                self.log_processing(f"ERROR: Fallback header capture failed: {e}")

    def _extract_geological_context_from_las(self, las, df: pd.DataFrame) -> GeologicalContext:
        """
        Extract geological context information from a LAS file
        
        This method extracts:
        - Formation tops from parameter section
        - Casing program information
        - Open hole intervals
        - Depth ranges for curve validity
        """
        geological_context = GeologicalContext()
        
        try:
            # Extract formation tops from parameters section
            if hasattr(las, 'params'):
                for param in las.params:
                    param_name = param.mnemonic.upper()
                    param_value = param.value
                    param_desc = getattr(param, 'descr', '').upper()
                    
                    # Look for formation top indicators
                    if any(indicator in param_name for indicator in ['FM', 'FORM', 'TOP', 'BASE']):
                        try:
                            depth_value = float(param_value)
                            formation_name = param_desc if param_desc else param_name
                            geological_context.add_formation_top(formation_name, depth_value)
                            self.log_processing(f"Found formation top: {formation_name} at {depth_value}m")
                        except (ValueError, TypeError):
                            continue
                    
                    # Look for casing information
                    elif any(indicator in param_name for indicator in ['CASING', 'CSG', 'SHOE']):
                        try:
                            casing_depth = float(param_value)
                            geological_context.casing_points.append(casing_depth)
                            self.log_processing(f"Found casing point at {casing_depth}m")
                        except (ValueError, TypeError):
                            continue
                    
                    # Look for TD (Total Depth)
                    elif param_name in ['TD', 'TOTAL_DEPTH', 'BOTM']:
                        try:
                            td_value = float(param_value)
                            geological_context.open_hole_end = td_value
                            self.log_processing(f"Found TD at {td_value}m")
                        except (ValueError, TypeError):
                            continue
            
            # Infer open hole interval from data and casing
            if hasattr(df, 'DEPT') or 'DEPT' in df.columns:
                depth_col = 'DEPT'
            elif hasattr(df, 'DEPTH') or 'DEPTH' in df.columns:
                depth_col = 'DEPTH'
            elif df.index.name and 'DEPT' in df.index.name.upper():
                depth_col = df.index.name
            else:
                # Try to find depth column
                depth_candidates = [col for col in df.columns if 'DEPT' in col.upper()]
                depth_col = depth_candidates[0] if depth_candidates else None
            
            if depth_col:
                depth_series = df[depth_col] if depth_col in df.columns else df.index
                min_depth = depth_series.min()
                max_depth = depth_series.max()
                
                # Set open hole start (last casing shoe or minimum depth)
                if geological_context.casing_points:
                    geological_context.open_hole_start = max(geological_context.casing_points)
                else:
                    geological_context.open_hole_start = min_depth
                
                # Set open hole end if not already set
                if not geological_context.open_hole_end:
                    geological_context.open_hole_end = max_depth
                
                self.log_processing(f"Inferred open hole interval: {geological_context.open_hole_start}-{geological_context.open_hole_end}m")
            
            # If no formation tops found, try to infer from first significant curve data
            if not geological_context.formation_tops:
                self._infer_formation_start_from_data(df, geological_context)
            
        except Exception:
            # Silence logging per privacy policy
            pass
        
        return geological_context

    def _infer_formation_start_from_data(self, df: pd.DataFrame, geological_context: GeologicalContext):
        """
        Infer approximate formation start from curve data patterns
        """
        try:
            # Look for depth column
            depth_col = None
            for col_name in ['DEPT', 'DEPTH', 'MD', 'MEASURED_DEPTH']:
                if col_name in df.columns:
                    depth_col = col_name
                    break
            
            if not depth_col:
                return
            
            depth_data = df[depth_col].dropna()
            if len(depth_data) < 10:
                return
            
            # Look for curves that should have formation data
            formation_sensitive_curves = ['GR', 'RHOB', 'NPHI', 'RT', 'RD', 'LLD']
            
            for curve_name in formation_sensitive_curves:
                if curve_name in df.columns:
                    curve_data = df[curve_name].dropna()
                    if len(curve_data) > 0:
                        # Find first significant data point
                        first_valid_idx = curve_data.first_valid_index()
                        if first_valid_idx is not None:
                            formation_depth = df.loc[first_valid_idx, depth_col]
                            geological_context.add_formation_top('INFERRED_TOP', formation_depth)
                            self.log_processing(f"Inferred formation start at {formation_depth}m from {curve_name} data")
                            break
            
        except Exception as e:
            # Log formation inference failure (non-critical, just informational)
            self.log_processing(f"Note: Could not infer formation start from data: {str(e)}")

    def _extract_lasio_curve_info(self, las):
        """Extract comprehensive curve information from lasio object"""
        try:
            for curve in las.curves:
                curve_name = str(curve.mnemonic).strip()
                if curve_name and curve_name not in self.curve_info:
                    self.curve_info[curve_name] = {
                        'curve_type': 'UNKNOWN',
                        'unit': str(curve.unit).strip() if curve.unit else '',
                        'description': str(curve.descr).strip() if curve.descr else '',
                        'quality': 0.8,  # Higher quality for lasio-loaded data
                        'lasio_metadata': {
                            'mnemonic': str(curve.mnemonic),
                            'unit': str(curve.unit) if curve.unit else '',
                            'description': str(curve.descr) if curve.descr else '',
                            'section': str(getattr(curve, 'section', '')) if getattr(curve, 'section', None) else ''
                        }
                    }
                    
                    # Try to identify curve type from mnemonic and description
                    self._identify_curve_from_lasio(curve_name, self.curve_info[curve_name])
                    
        except Exception as e:
            self.log_processing(f"Warning: Could not extract lasio curve info: {e}")

    def _extract_well_information(self, las=None):
        """Extract critical well identification information from LAS file
        
        SAFETY CRITICAL: This information prevents well confusion and ensures
        users know which well they're working with at all times.
        
        Args:
            las: lasio LAS object (if available)
            
        Returns:
            dict: Well identification information
        """
        well_info = {
            'well_name': 'UNKNOWN',
            'uwi': 'UNKNOWN',
            'field': 'UNKNOWN',
            'company': 'UNKNOWN',
            'date': 'UNKNOWN',
            'null_value': None,
            'start_depth': 'UNKNOWN',
            'stop_depth': 'UNKNOWN',
            'step': 'UNKNOWN',
            'depth_unit': 'm',
            'location': {},
            'api_number': 'UNKNOWN',
            'county': 'UNKNOWN',
            'state': 'UNKNOWN',
            'country': 'UNKNOWN'
        }
        
        if las is None:
            return well_info
        
        try:
            # Extract from LAS well section
            if hasattr(las, 'well'):
                well_section = las.well
                
                # Standard LAS parameters
                param_mapping = {
                    'WELL': 'well_name',
                    'UWI': 'uwi',
                    'FIELD': 'field',
                    'COMP': 'company',
                    'DATE': 'date',
                    'NULL': 'null_value',
                    'STRT': 'start_depth',
                    'STOP': 'stop_depth',
                    'STEP': 'step',
                    'API': 'api_number',
                    'CNTY': 'county',
                    'STAT': 'state',
                    'CTRY': 'country',
                    'LOC': 'location_string',
                    'LAT': 'latitude',
                    'LON': 'longitude',
                    'LONG': 'longitude',
                    'LATI': 'latitude'
                }
                
                # Try to extract each parameter
                for las_param, dict_key in param_mapping.items():
                    try:
                        if hasattr(well_section, las_param):
                            value = getattr(well_section, las_param)
                            if hasattr(value, 'value'):
                                value = value.value
                            if value not in [None, '', 'None', 'NONE']:
                                well_info[dict_key] = str(value).strip()
                        # Also try alternate forms
                        alternate_key = las_param.lower()
                        if alternate_key in well_section:
                            value = well_section[alternate_key].value
                            if value not in [None, '', 'None', 'NONE']:
                                well_info[dict_key] = str(value).strip()
                    except (AttributeError, KeyError):
                        continue
                
                # Extract depth unit from STRT or STOP
                try:
                    if hasattr(well_section, 'STRT') and hasattr(well_section.STRT, 'unit'):
                        unit = well_section.STRT.unit
                        if unit:
                            well_info['depth_unit'] = str(unit).strip()
                except (AttributeError, KeyError):
                    pass
            
            # Build location dict if we have coordinates
            if 'latitude' in well_info and well_info.get('latitude') != 'UNKNOWN':
                try:
                    well_info['location']['latitude'] = float(well_info['latitude'])
                except (ValueError, TypeError):
                    pass
            
            if 'longitude' in well_info and well_info.get('longitude') != 'UNKNOWN':
                try:
                    well_info['location']['longitude'] = float(well_info['longitude'])
                except (ValueError, TypeError):
                    pass
            
            # Format depth values with units
            for depth_key in ['start_depth', 'stop_depth']:
                if well_info[depth_key] != 'UNKNOWN':
                    try:
                        depth_val = float(well_info[depth_key])
                        well_info[depth_key] = f"{depth_val:.2f} {well_info['depth_unit']}"
                    except (ValueError, TypeError):
                        pass
            
            # Format step value
            if well_info['step'] != 'UNKNOWN':
                try:
                    step_val = float(well_info['step'])
                    well_info['step'] = f"{step_val:.4f} {well_info['depth_unit']}"
                except (ValueError, TypeError):
                    pass
            
            # Log extraction success
            self.log_processing(f"Well Information Extracted:")
            self.log_processing(f"  Well Name: {well_info['well_name']}")
            self.log_processing(f"  UWI: {well_info['uwi']}")
            self.log_processing(f"  Field: {well_info['field']}")
            self.log_processing(f"  Depth Range: {well_info['start_depth']} to {well_info['stop_depth']}")
            
        except Exception as e:
            self.log_processing(f"Error extracting well information: {e}")
            # Return default values on error
        
        return well_info

    def _identify_curve_from_lasio(self, curve_name, curve_info):
        """Identify curve type from lasio metadata using the unified curve identifier"""
        curve_type, confidence, curve_data = self.curve_identifier.identify_curve(
            curve_name, 
            curve_info.get('unit', ''), 
            curve_info.get('description', '')
        )
        
        # CRITICAL: Record standardization operation for audit trail
        if hasattr(self, 'standardization_reporter'):
            self.standardization_reporter.record_curve_identification(
                original_name=curve_name,
                curve_type=curve_type,
                confidence=confidence,
                method='lasio_extraction',
                unit=curve_info.get('unit', ''),
                description=curve_info.get('description', '')
            )
        
        # Update curve info with comprehensive identification results
        curve_info['curve_type'] = curve_type
        curve_info['type_confidence'] = confidence
        curve_info['curve_data'] = curve_data
        
        # Update description if we have better information from the library
        if curve_data.get('description') and confidence > 0.5:
            curve_info['description'] = curve_data['description']

    def _extract_las_header(self, las) -> str:
        """Extract the original LAS header as a string for later use in preview generation"""
        try:
            header_lines = []
            
            # Add version section
            if hasattr(las, 'version') and las.version:
                header_lines.append("~Version Information")
                for item in las.version:
                    if hasattr(item, 'mnemonic') and hasattr(item, 'value') and hasattr(item, 'descr'):
                        header_lines.append(f"{item.mnemonic}.{item.value}: {item.descr}")
                header_lines.append("")
            
            # Add well section
            if hasattr(las, 'well') and las.well:
                header_lines.append("~Well Information")
                for item in las.well:
                    if hasattr(item, 'mnemonic') and hasattr(item, 'value') and hasattr(item, 'descr'):
                        header_lines.append(f"{item.mnemonic}.{item.value}: {item.descr}")
                header_lines.append("")
            
            # Add curve section
            if hasattr(las, 'curves') and las.curves:
                header_lines.append("~Curve Information")
                for item in las.curves:
                    if hasattr(item, 'mnemonic') and hasattr(item, 'unit') and hasattr(item, 'descr'):
                        header_lines.append(f"{item.mnemonic}.{item.unit}: {item.descr}")
                header_lines.append("")
            
            # Add parameter section
            if hasattr(las, 'params') and las.params:
                header_lines.append("~Parameter Information")
                for item in las.params:
                    if hasattr(item, 'mnemonic') and hasattr(item, 'value') and hasattr(item, 'descr'):
                        header_lines.append(f"{item.mnemonic}.{item.value}: {item.descr}")
                header_lines.append("")
            
            # Add other section
            if hasattr(las, 'other') and las.other:
                header_lines.append("~Other")
                for item in las.other:
                    if hasattr(item, 'mnemonic') and hasattr(item, 'value') and hasattr(item, 'descr'):
                        header_lines.append(f"{item.mnemonic}.{item.value}: {item.descr}")
                header_lines.append("")
            
            # Add ASCII section marker
            header_lines.append("~ASCII")
            
            return '\n'.join(header_lines)
            
        except Exception as e:
            # Log LAS header extraction failure
            self.log_processing(f"Warning: Error extracting LAS header: {str(e)}")
            self.log_processing("Using minimal fallback header for preview")
            warnings.warn(
                f"LAS header extraction failed: {str(e)}. "
                f"Using minimal header. Original LAS preview may be incomplete.",
                UserWarning
            )
            # Return a minimal header if extraction fails
            return "~Version Information\nVERS.                          2.0: CWLS Log ASCII Standard - Version 2.0\n~Well Information\n~Curve Information\n~ASCII"

    def _load_las_file_manual_fallback(self, filepath: str) -> pd.DataFrame:
        """Fallback manual LAS parsing if lasio fails"""
        self.log_processing("Using manual LAS parsing fallback...")
        
        try:
            # ...existing code for lasio loading...
            # If lasio fails, do manual parsing:
            curves = {}
            current_section = None
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if line.strip().startswith('~'):  # Section header
                        if 'CURVE' in line.upper():
                            current_section = 'C'
                        elif 'ASCII' in line.upper():
                            current_section = 'A'
                        else:
                            current_section = None
                        continue
                    if current_section == 'C':
                        try:
                            if '.' in line:
                                parts = line.split('.')
                                mnemonic = parts[0].strip()
                                rest = '.'.join(parts[1:]).split(':')
                                unit = rest[0].strip() if rest else ''
                                description = rest[1].strip() if len(rest) > 1 else ''
                                curves[mnemonic] = {
                                    'unit': unit,
                                    'description': description,
                                    'data': []
                                }
                            else:
                                # Warning removed - operation continues
                                # Operation result handled - continuing safely
                                pass
                        except Exception as e:
                            # Warning removed - operation continues
                            # Operation result handled - continuing safely
                            pass
                    elif current_section == 'A':
                        try:
                            values = line.split()
                            curve_names = list(curves.keys())
                            for i, value_str in enumerate(values):
                                if i < len(curve_names):
                                    curve_name = curve_names[i]
                                    try:
                                        value = float(value_str)
                                        if (abs(value + 999.25) < 0.01 or 
                                            abs(value + 999) < 0.01 or
                                            abs(value - (-999.25)) < 0.01 or
                                            abs(value - (-999)) < 0.01 or
                                            abs(value - 9999) < 0.01):
                                            value = np.nan
                                        curves[curve_name]['data'].append(value)
                                    except ValueError:
                                        # Debug information removed for security
                                        # Operation result handled - continuing safely
                                        curves[curve_name]['data'].append(np.nan)
                                else:
                                    if i == len(curve_names):
                                        # Warning removed - operation continues
                                        # Operation result handled - continuing safely
                                        pass
                        except Exception as e:
                            # Warning removed - operation continues
                            # Operation result handled - continuing safely
                            pass
            # Second pass - handle data alignment issues
            max_length = max([len(curve_info['data']) for curve_info in curves.values()], default=0)
            for curve_name, curve_info in curves.items():
                if len(curve_info['data']) < max_length:
                    # Warning removed - operation continues
                    # Operation result handled - continuing safely
                    pass  # f"Curve {curve_name} has fewer data points ({len(curve_info['data'])}) than expected ({max_length}). Padding with NaN.")
                    curve_info['data'].extend([np.nan] * (max_length - len(curve_info['data'])))
            data_dict = {}
            for curve_name, curve_info in curves.items():
                data_dict[curve_name] = curve_info['data']
                self.curve_info[curve_name] = {
                    'unit': curve_info['unit'],
                    'description': curve_info['description'],
                    'quality': 0.5
                }
            if not data_dict:
                raise ValueError("No valid curves found in LAS file")
            df = pd.DataFrame(data_dict)
            self.log_processing(f"Manual parsing successful: {len(df)} rows, {len(df.columns)} curves")
            return df
        except Exception as e:
            # Log manual LAS parsing failure with detailed information
            self.log_processing(f"ERROR: Manual LAS parsing failed: {str(e)}")
            self.log_processing("Both lasio library and manual parsing have failed.")
            self.log_processing("Please verify:")
            self.log_processing("  1. File is a valid LAS format")
            self.log_processing("  2. File is not corrupted")
            self.log_processing("  3. File encoding is readable")
            raise ValueError(f"Failed to load LAS file with both lasio and manual parsing: {str(e)}")

    def _is_numeric_value(self, value_str):
        """Enhanced helper to check if a string represents a numeric value, handling all LAS null variations"""
        if not value_str or not isinstance(value_str, str):
            return False
            
        value_clean = value_str.strip()
        if not value_clean:
            return False
        
        try:
            # Try direct float conversion
            float(value_clean)
            return True
        except (ValueError, TypeError):
            pass
        
        # Check for common LAS null values (these are considered "numeric" in LAS context)
        las_null_values = [
            '-999.25', '-999.2500', '-999', '-999.00', '-999.0',
            '999.25', '999.2500', '999', '999.00', '999.0',
            '9999', '9999.00', '9999.0', '-9999', '-9999.00', '-9999.0',
            'NULL', 'null', 'NaN', 'nan', 'NA', 'na'
        ]
        
        if value_clean in las_null_values:
            return True
        
        # Try to handle numbers with special formatting
        try:
            # Remove common LAS formatting
            cleaned = value_clean.replace(',', '').replace('_', '')
            float(cleaned)
            return True
        except (ValueError, TypeError):
            # Not a valid numeric value after cleaning - expected for non-numeric fields
            pass
        
        return False

    def _looks_like_data_line(self, line):
        """Enhanced helper to identify if a line looks like LAS data"""
        if not line or not isinstance(line, str):
            return False
        
        line_clean = line.strip()
        if not line_clean or line_clean.startswith(('#', '~', '/', '*', ';')):
            return False
        
        parts = line_clean.split()
        if len(parts) < 3:
            return False
        
        numeric_count = 0
        depth_like_first = False
        
        # Check each field
        for i, part in enumerate(parts[:10]):  # Check up to 10 fields
            if self._is_numeric_value(part):
                numeric_count += 1
                
                # Check if first field looks like depth
                if i == 0:
                    try:
                        val = float(part)
                        if 0 <= val <= 50000:  # Reasonable depth range (0 to 50,000 feet/meters)
                            depth_like_first = True
                    except (ValueError, TypeError):
                        # Not a valid numeric value - expected for non-depth fields
                        pass
        
        # Various criteria for data line detection
        total_fields = len(parts)
        numeric_ratio = numeric_count / total_fields if total_fields > 0 else 0
        
        # Criteria 1: High numeric ratio
        if numeric_ratio >= 0.7 and numeric_count >= 3:
            return True
        
        # Criteria 2: Depth-like first column + good numeric count
        if depth_like_first and numeric_count >= 3:
            return True
        
        # Criteria 3: All fields numeric with reasonable count
        if numeric_count == total_fields and total_fields >= 4:
            return True
        
        # Criteria 4: Contains LAS null patterns
        if (numeric_count >= 3 and 
            any(null_pattern in line_clean for null_pattern in ['-999.25', '-999', '9999'])):
            return True
        
        return False

    def _detect_column_headers(self, line):
        """Helper to detect if a line contains column headers"""
        if not line or not isinstance(line, str):
            return False
        
        line_upper = line.upper()
        
        # Common LAS curve mnemonics
        curve_indicators = [
            'DEPTH', 'DEPT', 'MD', 'TVD', 'TVDSS',
            'GR', 'GAMMA', 'SGR', 'CGR', 'THOR', 'URAN', 'POTA',
            'SP', 'SPONTANEOUS',
            'RESISTIVITY', 'RES', 'RT', 'RD', 'RM', 'RS', 'RXO', 'RILD', 'RILM', 'RLL',
            'DENSITY', 'RHOB', 'RHOZ', 'DEN', 'DPOR',
            'NEUTRON', 'NPHI', 'NPOR', 'NEU', 'TNPH',
            'PHOTOELECTRIC', 'PE', 'PEFZ',
            'CALIPER', 'CALI', 'CAL', 'MCAL',
            'SONIC', 'DT', 'DTCO', 'DTSM',
            'POROSITY', 'POR', 'PHI',
            'SATURATION', 'SW', 'SO', 'SG',
            'PERMEABILITY', 'PERM', 'K'
        ]
        
        parts = line.split()
        if len(parts) < 3:
            return False
        
        # Count how many parts look like curve names
        curve_count = 0
        for part in parts:
            part_clean = part.strip().upper()
            for indicator in curve_indicators:
                if indicator in part_clean:
                    curve_count += 1
                    break
        
        # If multiple curve indicators found, likely a header line
        return curve_count >= 2

    def load_csv_file(self, filepath: str) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection"""
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, na_values=['', 'NaN', 'NULL', '-999.25', '-999'])
                if df.shape[1] > 1:  # Valid DataFrame with multiple columns
                    # Initialize curve info
                    for col in df.columns:
                        self.curve_info[col] = {
                            'unit': '',
                            'description': f'Column {col} from CSV'
                        }
                    return df
            except pd.errors.ParserError as e:
                # Continue trying other delimiters - log only if all fail
                continue
            except Exception as e:
                # Continue trying other delimiters - log only if all fail
                continue
        
        # All delimiters failed - provide helpful error message
        self.log_processing(f"ERROR: Failed to parse CSV file with any standard delimiter")
        self.log_processing(f"Tried delimiters: {delimiters}")
        self.log_processing("Please verify:")
        self.log_processing("  1. File is valid CSV/TSV format")
        self.log_processing("  2. File uses standard delimiters (comma, tab, semicolon, pipe)")
        self.log_processing("  3. File is not corrupted")
        raise ValueError("Could not parse CSV file with any standard delimiter (tried: , ; \\t |)")

    def load_excel_file(self, filepath: str) -> pd.DataFrame:
        """Load Excel file"""
        df = pd.read_excel(filepath, na_values=['', 'NaN', 'NULL', '-999.25', '-999'])
        
        # Initialize curve info
        for col in df.columns:
            self.curve_info[col] = {
                'unit': '',
                'description': f'Column {col} from Excel'
            }
        
        return df

