"""
Petrophysical Relationship Validation Module

Validates that processed wireline data preserves known petrophysical
cross-curve relationships (Archie, density-porosity, sonic-porosity, etc.).

Extracted from advanced_preprocessing_system10.py for modular architecture.

ARCHITECTURE OVERVIEW:

MAIN CLASS:
- PetrophysicalRelationshipValidator: Checks expected correlations after processing

KEY FUNCTIONS:
- validate_relationships(): Compare observed vs expected curve pair correlations
- _calculate_relationship(): Correlation (including log-resistivity forms)
- _validate_single_relationship(): Threshold check by relationship polarity

DATA FLOW:
Processed DataFrame + curve info → pair-wise correlation checks → results + warnings
"""

from __future__ import annotations

import numpy as np


class PetrophysicalRelationshipValidator:
    """Validate that processed data maintains known petrophysical relationships"""
    
    def __init__(self):
        self.known_relationships = {
            # Archie's Law relationships
            ('RT', 'NPHI'): {'type': 'negative_log', 'strength': 'strong', 'r_threshold': -0.6},
            ('RM', 'NPHI'): {'type': 'negative_log', 'strength': 'strong', 'r_threshold': -0.6},
            
            # Density-Porosity relationships
            ('RHOB', 'NPHI'): {'type': 'negative', 'strength': 'strong', 'r_threshold': -0.7},
            
            # Sonic-Porosity relationships (Wyllie Time Average)
            ('DT', 'NPHI'): {'type': 'positive', 'strength': 'moderate', 'r_threshold': 0.5},
            
            # Gamma Ray relationships
            ('GR', 'RHOB'): {'type': 'negative', 'strength': 'weak', 'r_threshold': -0.3},
            ('GR', 'RT'): {'type': 'negative', 'strength': 'weak', 'r_threshold': -0.3},
            
            # Photoelectric-Density relationships
            ('PE', 'RHOB'): {'type': 'positive', 'strength': 'moderate', 'r_threshold': 0.4},
        }
    
    def validate_relationships(self, processed_data, curve_info):
        """Validate that processing preserved known petrophysical relationships"""
        
        validation_results = {}
        warnings = []
        
        for (curve1, curve2), expected in self.known_relationships.items():
            if curve1 in processed_data.columns and curve2 in processed_data.columns:
                
                # Calculate actual relationship
                actual_result = self._calculate_relationship(
                    processed_data[curve1], 
                    processed_data[curve2], 
                    expected['type']
                )
                
                # Validate against expected
                is_valid = self._validate_single_relationship(actual_result, expected)
                
                validation_results[f"{curve1}_{curve2}"] = {
                    'expected': expected,
                    'actual': actual_result,
                    'valid': is_valid,
                    'deviation': abs(actual_result['correlation'] - expected['r_threshold'])
                }
                
                if not is_valid:
                    warnings.append(
                        f"WARNING: {curve1}-{curve2} relationship invalid. "
                        f"Expected {expected['type']} correlation {expected['r_threshold']}, "
                        f"got {actual_result['correlation']:.3f}"
                    )
        
        return validation_results, warnings
    
    def _calculate_relationship(self, curve1_data, curve2_data, relationship_type):
        """Calculate relationship between two curves"""
        
        # Clean data
        valid_mask = ~np.isnan(curve1_data) & ~np.isnan(curve2_data)
        clean_c1 = curve1_data[valid_mask]
        clean_c2 = curve2_data[valid_mask]
        
        if len(clean_c1) < 10:
            return {'correlation': 0, 'valid_points': len(clean_c1), 'relationship_type': relationship_type}
        
        # Calculate correlation based on relationship type
        if relationship_type == 'negative_log':
            # Log transform first curve (typically resistivity)
            log_c1 = np.log10(np.maximum(clean_c1, 0.01))  # Avoid log(0)
            correlation = np.corrcoef(log_c1, clean_c2)[0, 1]
        else:
            correlation = np.corrcoef(clean_c1, clean_c2)[0, 1]
        
        return {
            'correlation': correlation,
            'valid_points': len(clean_c1),
            'relationship_type': relationship_type,
            'curve1_range': (np.min(clean_c1), np.max(clean_c1)),
            'curve2_range': (np.min(clean_c2), np.max(clean_c2))
        }
    
    def _validate_single_relationship(self, actual, expected):
        """Validate a single petrophysical relationship"""
        correlation = actual['correlation']
        threshold = expected['r_threshold']
        
        if expected['type'] in ['negative', 'negative_log']:
            return correlation <= threshold  # More negative is better
        else:  # positive
            return correlation >= threshold  # More positive is better
