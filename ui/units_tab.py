"""
Units tab widget construction

STRUCTURE:
- UnitsTabMixin: Mixin providing create_units_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class UnitsTabMixin:
    """Units tab widget construction."""

    def create_units_tab(self):
        """Create unit standardization tab"""
        units_frame = ttk.Frame(self.notebook)
        self.notebook.add(units_frame, text=" Units")
        
        # Add unit standardization UI to this tab
        self.unit_standardizer.add_unit_standardization_ui(units_frame)
        
        # Add additional unit-related information
        info_frame = ttk.LabelFrame(units_frame, text=" Unit Information", padding="10")
        info_frame.pack(fill='x', pady=(0, 10))
        
        info_text = """
Unit standardization automatically converts wireline data to industry-standard units:

• Depth: FT → M (×0.3048)
• Density: G/CC → KG/M3 (×1000.0)  
• Resistivity: OHM-M → OHMM (×1.0)
• Sonic: USEC/FT → US/M (×3.28084)
• Porosity: PERCENT → V/V (×0.01)

This ensures consistent data interpretation and fixes depth validation issues.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify='left', wraplength=600)
        info_label.pack(anchor='w', pady=5)
    
