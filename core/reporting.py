"""
Standardization Reporting System - Critical for Professional Operations

Tracks and reports all data standardization operations for audit trails.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


class StandardizationReporter:
    """
    Comprehensive tracking and reporting of all standardization operations.
    
    CRITICAL FEATURE: Without standardization reporting, users cannot audit
    what changes were made to their data, violating professional data
    management standards.
    
    Tracks:
    - Curve name standardization (original → standardized mnemonic)
    - Unit conversions (original → standardized)
    - Curve identification confidence and method
    - Conflicts and resolutions
    - Quality metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset reporter for a new file/well"""
        self.curve_identifications = []  # Original name, identified type, confidence, method
        self.curve_renames = {}  # Original name → Standardized name
        self.unit_conversions = []  # Curve name, original unit → standardized unit, method
        self.fractional_standardizations = []  # Upload-time % → v/v conversions
        self.conflicts = []  # Naming conflicts and resolutions
        self.total_operations = 0
    
    def record_curve_identification(self, original_name: str, curve_type: str, 
                                    confidence: float, method: str = 'auto',
                                    unit: str = '', description: str = ''):
        """Record a curve identification operation"""
        self.curve_identifications.append({
            'original_name': original_name,
            'identified_type': curve_type,
            'confidence': confidence,
            'method': method,
            'unit': unit,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
        self.total_operations += 1
    
    def record_curve_rename(self, original_name: str, standardized_name: str,
                           reason: str = 'standardization', confidence: float = 1.0):
        """Record a curve name standardization (rename)"""
        if original_name != standardized_name:
            self.curve_renames[original_name] = {
                'standardized_name': standardized_name,
                'reason': reason,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            self.total_operations += 1
    
    def record_unit_conversion(self, curve_name: str, original_unit: str,
                              standardized_unit: str, method: str = 'auto',
                              factor: Optional[float] = None, validated: bool = True):
        """Record a unit conversion"""
        self.unit_conversions.append({
            'curve_name': curve_name,
            'original_unit': original_unit,
            'standardized_unit': standardized_unit,
            'method': method,
            'conversion_factor': factor,
            'validated': validated,
            'timestamp': datetime.now().isoformat()
        })
        self.total_operations += 1
    
    def record_fractional_standardization(self, curve_name: str, original_unit: str,
                                         original_value_sample: Optional[float] = None,
                                         standardized_value_sample: Optional[float] = None):
        """Record upload-time fractional standardization (% → v/v)"""
        self.fractional_standardizations.append({
            'curve_name': curve_name,
            'original_unit': original_unit,
            'standardized_unit': 'v/v',
            'original_sample': original_value_sample,
            'standardized_sample': standardized_value_sample,
            'timestamp': datetime.now().isoformat()
        })
        self.total_operations += 1
    
    def record_conflict(self, curve_name: str, conflict_type: str, 
                       resolution: str, alternatives: Optional[List[str]] = None):
        """Record a naming/identification conflict and its resolution"""
        self.conflicts.append({
            'curve_name': curve_name,
            'conflict_type': conflict_type,  # 'naming', 'unit', 'identification'
            'resolution': resolution,
            'alternatives': alternatives or [],
            'timestamp': datetime.now().isoformat()
        })
        self.total_operations += 1
    
    def get_standardization_report(self) -> str:
        """Generate comprehensive standardization report"""
        report = []
        
        report.append("╔" + "═" * 78 + "╗")
        report.append("║" + " " * 20 + "STANDARDIZATION REPORT" + " " * 35 + "║")
        report.append("╠" + "═" * 78 + "╣")
        report.append(f"║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<64} ║")
        report.append(f"║  Total Operations Recorded: {self.total_operations:<47} ║")
        report.append("╚" + "═" * 78 + "╝")
        report.append("")
        
        # Curve Identifications
        if self.curve_identifications:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 25 + "CURVE IDENTIFICATIONS" + " " * 30 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            for idx, ident in enumerate(self.curve_identifications, 1):
                confidence_pct = ident['confidence'] * 100
                report.append(f"║  {idx:3d}. {ident['original_name']:<20} → {ident['identified_type']:<25} ║")
                report.append(f"║      Confidence: {confidence_pct:5.1f}% | Method: {ident['method']:<15} ║")
                if ident.get('unit'):
                    report.append(f"║      Unit: {ident['unit']:<20} | Description: {ident.get('description', 'N/A'):<29} ║")
                report.append("║" + " " * 76 + "║")
            
            report.append("╚" + "═" * 78 + "╝")
            report.append("")
        
        # Curve Renames
        if self.curve_renames:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 28 + "CURVE RENAMES" + " " * 37 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            for original, rename_info in self.curve_renames.items():
                report.append(f"║  {original:<35} → {rename_info['standardized_name']:<35} ║")
                report.append(f"║      Reason: {rename_info['reason']:<20} | Confidence: {rename_info['confidence']*100:.1f}% ║")
                report.append("║" + " " * 76 + "║")
            
            report.append("╚" + "═" * 78 + "╝")
            report.append("")
        
        # Unit Conversions
        if self.unit_conversions:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 28 + "UNIT CONVERSIONS" + " " * 33 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            for conv in self.unit_conversions:
                factor_str = f"×{conv['conversion_factor']:.6f}" if conv['conversion_factor'] else "function"
                validated_str = "✓ Validated" if conv['validated'] else "⚠ Unvalidated"
                report.append(f"║  {conv['curve_name']:<25} {conv['original_unit']:<10} → {conv['standardized_unit']:<10} ║")
                report.append(f"║      Method: {conv['method']:<15} | Factor: {factor_str:<15} | {validated_str:<20} ║")
                report.append("║" + " " * 76 + "║")
            
            report.append("╚" + "═" * 78 + "╝")
            report.append("")
        
        # Fractional Standardizations (Upload-time)
        if self.fractional_standardizations:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 22 + "UPLOAD-TIME FRACTIONAL STANDARDIZATION" + " " * 23 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            for frac in self.fractional_standardizations:
                if frac.get('original_sample') is not None and frac.get('standardized_sample') is not None:
                    sample_str = f"{frac['original_sample']:.3f}% → {frac['standardized_sample']:.3f} v/v"
                else:
                    sample_str = "Conversion applied"
                report.append(f"║  {frac['curve_name']:<35} {frac['original_unit']:<10} → {frac['standardized_unit']:<10} ║")
                report.append(f"║      Sample: {sample_str:<62} ║")
                report.append("║" + " " * 76 + "║")
            
            report.append("╚" + "═" * 78 + "╝")
            report.append("")
        
        # Conflicts and Resolutions
        if self.conflicts:
            report.append("╔" + "═" * 78 + "╗")
            report.append("║" + " " * 27 + "CONFLICTS & RESOLUTIONS" + " " * 28 + "║")
            report.append("╠" + "═" * 78 + "╣")
            
            for conflict in self.conflicts:
                report.append(f"║  Curve: {conflict['curve_name']:<30} | Type: {conflict['conflict_type']:<15} ║")
                report.append(f"║      Resolution: {conflict['resolution']:<60} ║")
                if conflict.get('alternatives'):
                    alt_str = ", ".join(conflict['alternatives'][:3])
                    if len(conflict['alternatives']) > 3:
                        alt_str += f" (+{len(conflict['alternatives'])-3} more)"
                    report.append(f"║      Alternatives Considered: {alt_str:<48} ║")
                report.append("║" + " " * 76 + "║")
            
            report.append("╚" + "═" * 78 + "╝")
            report.append("")
        
        # Summary Statistics
        report.append("╔" + "═" * 78 + "╗")
        report.append("║" + " " * 29 + "SUMMARY STATISTICS" + " " * 32 + "║")
        report.append("╠" + "═" * 78 + "╣")
        report.append(f"║  Total Curve Identifications: {len(self.curve_identifications):<53} ║")
        report.append(f"║  Total Curve Renames: {len(self.curve_renames):<58} ║")
        report.append(f"║  Total Unit Conversions: {len(self.unit_conversions):<55} ║")
        report.append(f"║  Total Fractional Standardizations: {len(self.fractional_standardizations):<48} ║")
        report.append(f"║  Total Conflicts Resolved: {len(self.conflicts):<57} ║")
        
        # Quality Metrics
        if self.curve_identifications:
            avg_confidence = sum(i['confidence'] for i in self.curve_identifications) / len(self.curve_identifications)
            high_confidence = sum(1 for i in self.curve_identifications if i['confidence'] >= 0.8)
            report.append(f"║  Average Identification Confidence: {avg_confidence*100:.1f}% ║")
            report.append(f"║  High-Confidence Identifications (≥80%): {high_confidence}/{len(self.curve_identifications)} ║")
        
        report.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(report)
    
    def get_summary_for_log(self) -> str:
        """Get brief summary for processing log"""
        summary = []
        if self.curve_renames:
            summary.append(f"{len(self.curve_renames)} curve(s) renamed to standard mnemonics")
        if self.unit_conversions:
            summary.append(f"{len(self.unit_conversions)} unit conversion(s) applied")
        if self.fractional_standardizations:
            summary.append(f"{len(self.fractional_standardizations)} fractional curve(s) standardized on upload")
        if self.conflicts:
            summary.append(f"{len(self.conflicts)} conflict(s) resolved")
        return "; ".join(summary) if summary else "No standardization operations recorded"

