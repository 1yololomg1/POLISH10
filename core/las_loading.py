"""
LAS Standards Compliance Module

Full LAS 2.0/3.0 compliance validation and standards-aware parsing.

Extracted from advanced_preprocessing_system10.py for modular architecture.

ARCHITECTURE OVERVIEW:

MAIN CLASS:
- LASStandardsCompliance: Validates LAS structure and parses with compliance checks

KEY FUNCTIONS:
- validate_complete_las_compliance(): Run all section/parameter/data checks
- parse_las_with_full_compliance(): Validate then parse into DataFrame + metadata
- _extract_section() / _parse_curves_compliant() / _parse_data_compliant(): Section helpers

DATA FLOW:
LAS filepath → section validation → curve/data parse → DataFrame + curve_info + metadata
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LASStandardsCompliance:
    """Full LAS 2.0/3.0 compliance validator and processor"""
    
    def __init__(self):
        self.las_version = "2.0"
        self.required_sections = {
            '2.0': ['~V', '~W', '~C', '~A'],
            '3.0': ['~V', '~W', '~C', '~A', '~D']
        }
        self.required_well_params = ['STRT', 'STOP', 'STEP', 'NULL']
        
    def validate_complete_las_compliance(self, filepath):
        """Complete LAS file validation"""
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
        
        validation_results = {
            'version_compliance': self._validate_version_section(content),
            'section_compliance': self._validate_required_sections(content),
            'well_params_compliance': self._validate_well_parameters(content),
            'curve_compliance': self._validate_curve_section(content),
            'data_compliance': self._validate_data_section(content),
            'wrap_mode_support': self._detect_wrap_mode(content)
        }
        
        # Overall compliance
        all_valid = all(result['valid'] for result in validation_results.values() 
                       if isinstance(result, dict) and 'valid' in result)
        
        return all_valid, validation_results
    
    def _validate_version_section(self, content):
        """Validate ~V section compliance"""
        version_section = self._extract_section(content, '~V')
        
        if not version_section:
            return {'valid': False, 'error': 'Missing ~V section'}
        
        # Check required version parameters
        required_params = ['VERS', 'WRAP']
        found_params = {}
        
        for line in version_section:
            if '.' in line:
                param = line.split('.')[0].strip().upper()
                if param in required_params:
                    value = self._extract_parameter_value(line)
                    found_params[param] = value
        
        missing = set(required_params) - set(found_params.keys())
        
        if missing:
            return {'valid': False, 'error': f'Missing required parameters: {missing}'}
        
        # Validate version
        version = found_params.get('VERS', '')
        if version not in ['2.0', '3.0']:
            return {'valid': False, 'error': f'Unsupported LAS version: {version}'}
        
        self.las_version = version
        
        return {'valid': True, 'version': version, 'wrap': found_params.get('WRAP', 'NO')}
    
    def _extract_section(self, content, section_marker):
        """Extract specific LAS section"""
        in_section = False
        section_lines = []
        
        for line in content:
            line_clean = line.strip().upper()
            
            if line_clean.startswith('~'):
                if section_marker.upper() in line_clean[:3]:
                    in_section = True
                    section_lines.append(line.strip())
                else:
                    if in_section:
                        break  # End of current section
            elif in_section:
                section_lines.append(line.strip())
        
        return section_lines
    
    def _validate_data_section(self, content):
        """Validate ASCII data section for parameter alignment"""
        data_section = self._extract_section(content, '~A')
        
        if not data_section:
            return {'valid': False, 'error': 'Missing ~A section'}
        
        # Get curve count from ~C section
        curve_section = self._extract_section(content, '~C')
        curve_count = len([line for line in curve_section if '.' in line and not line.startswith('#')])
        
        # Validate data lines
        data_lines = [line for line in data_section if line and not line.startswith('#') and not line.startswith('~')]
        
        if not data_lines:
            return {'valid': False, 'error': 'No data in ~A section'}
        
        alignment_issues = []
        
        for i, line in enumerate(data_lines[:100]):  # Check first 100 lines
            values = line.split()
            if len(values) != curve_count:
                alignment_issues.append({
                    'line': i + 1,
                    'expected_columns': curve_count,
                    'actual_columns': len(values)
                })
        
        if alignment_issues:
            return {
                'valid': False,
                'error': 'Parameter alignment issues',
                'alignment_issues': alignment_issues[:10]  # Show first 10 issues
            }
        
        return {'valid': True, 'data_lines': len(data_lines), 'curves': curve_count}
    
    def parse_las_with_full_compliance(self, filepath):
        """Parse LAS file with complete standards compliance"""
        
        # First validate compliance
        is_compliant, validation_results = self.validate_complete_las_compliance(filepath)
        
        if not is_compliant:
            raise ValueError(f"LAS file not compliant: {validation_results}")
        
        # Parse with compliance assurance
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
        
        # Extract metadata preserving original format
        original_metadata = self._preserve_original_metadata(content)
        
        # Parse data with proper NULL handling
        well_section = self._extract_section(content, '~W')
        null_value = self._extract_null_value(well_section)
        
        # Parse curves with proper formatting
        curve_info = self._parse_curves_compliant(content)
        
        # Parse data with parameter alignment validation
        data_dict = self._parse_data_compliant(content, curve_info, null_value)
        
        return {
            'data': pd.DataFrame(data_dict),
            'curve_info': curve_info,
            'original_metadata': original_metadata,
            'null_value': null_value,
            'las_version': self.las_version,
            'compliance_validated': True
        }
    
    def _validate_required_sections(self, content):
        """Validate all required sections are present"""
        required = self.required_sections.get(self.las_version, [])
        found_sections = []
        
        for line in content:
            if line.strip().startswith('~'):
                section = line.strip().upper()[:3]
                if section in required and section not in found_sections:
                    found_sections.append(section)
        
        missing = set(required) - set(found_sections)
        
        if missing:
            return {'valid': False, 'error': f'Missing required sections: {missing}'}
        
        return {'valid': True, 'sections_found': found_sections}
    
    def _validate_well_parameters(self, content):
        """Validate required well parameters"""
        well_section = self._extract_section(content, '~W')
        
        if not well_section:
            return {'valid': False, 'error': 'Missing ~W section'}
        
        found_params = {}
        for line in well_section:
            if '.' in line:
                param = line.split('.')[0].strip().upper()
                if param in self.required_well_params:
                    value = self._extract_parameter_value(line)
                    found_params[param] = value
        
        missing = set(self.required_well_params) - set(found_params.keys())
        
        if missing:
            return {'valid': False, 'error': f'Missing required well parameters: {missing}'}
        
        return {'valid': True, 'parameters': found_params}
    
    def _validate_curve_section(self, content):
        """Validate curve section format"""
        curve_section = self._extract_section(content, '~C')
        
        if not curve_section:
            return {'valid': False, 'error': 'Missing ~C section'}
        
        curve_lines = [line for line in curve_section if '.' in line and not line.startswith('#')]
        
        if not curve_lines:
            return {'valid': False, 'error': 'No curves defined in ~C section'}
        
        return {'valid': True, 'curve_count': len(curve_lines)}
    
    def _detect_wrap_mode(self, content):
        """Detect and validate wrap mode"""
        version_section = self._extract_section(content, '~V')
        
        for line in version_section:
            if 'WRAP' in line.upper():
                wrap_mode = self._extract_parameter_value(line)
                if wrap_mode not in ['YES', 'NO']:
                    return {'valid': False, 'error': f'Invalid wrap mode: {wrap_mode}'}
                return {'valid': True, 'wrap_mode': wrap_mode}
        
        return {'valid': True, 'wrap_mode': 'NO'}  # Default
    
    def _extract_parameter_value(self, line):
        """Extract parameter value from LAS line"""
        if '.' in line:
            parts = line.split('.', 1)
            if len(parts) > 1:
                value_part = parts[1].split(':', 1)
                if len(value_part) > 1:
                    return value_part[1].strip()
        return ''
    
    def _preserve_original_metadata(self, content):
        """Preserve original LAS metadata format"""
        metadata = {}
        current_section = None
        
        for line in content:
            line_clean = line.strip()
            if line_clean.startswith('~'):
                current_section = line_clean[:3]
                metadata[current_section] = []
            elif current_section and line_clean:
                metadata[current_section].append(line_clean)
        
        return metadata
    
    def _extract_null_value(self, well_section):
        """Extract NULL value from well section"""
        for line in well_section:
            if 'NULL' in line.upper():
                null_str = self._extract_parameter_value(line)
                try:
                    return float(null_str)
                except ValueError:
                    return -999.25  # Default LAS NULL value
        return -999.25
    
    def _parse_curves_compliant(self, content):
        """Parse curves with LAS compliance"""
        curve_section = self._extract_section(content, '~C')
        curves = {}
        
        for line in curve_section:
            if '.' in line and not line.startswith('#'):
                parts = line.split('.', 1)
                if len(parts) > 1:
                    curve_name = parts[0].strip()
                    value_part = parts[1].split(':', 1)
                    if len(value_part) > 1:
                        unit = value_part[1].strip()
                        curves[curve_name] = {'unit': unit}
        
        return curves
    
    def _parse_data_compliant(self, content, curve_info, null_value):
        """Parse data with parameter alignment validation"""
        data_section = self._extract_section(content, '~A')
        data_lines = [line for line in data_section if line and not line.startswith('#') and not line.startswith('~')]
        
        curve_names = list(curve_info.keys())
        data_dict = {name: [] for name in curve_names}
        
        for line in data_lines:
            values = line.split()
            if len(values) == len(curve_names):
                for i, name in enumerate(curve_names):
                    try:
                        value = float(values[i])
                        if value == null_value:
                            data_dict[name].append(np.nan)
                        else:
                            data_dict[name].append(value)
                    except ValueError:
                        data_dict[name].append(np.nan)
        
        return data_dict
