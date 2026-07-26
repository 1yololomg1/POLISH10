"""
Visualization tab widget construction

STRUCTURE:
- VisualizationTabMixin: Mixin providing create_visualization_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.constants import DIALOG_DESELECT_ALL, DIALOG_SELECT_ALL


class VisualizationTabMixin:
    """Visualization tab widget construction."""

    def create_visualization_tab(self):
        """Create advanced visualization tab with multi-curve capability"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Control panel
        control_frame = ttk.Frame(viz_frame)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        # First row of controls
        row1 = ttk.Frame(control_frame)
        row1.pack(fill='x', pady=(0, 5))
        
        ttk.Label(row1, text="Visualization Type:", style='Card.TLabel').pack(side='left')
        self.viz_type_var = tk.StringVar(value="comparison")
        viz_types = ["single_curve", "single_curve_comparison", "comparison", "uncertainty", "quality_metrics", "correlation_matrix", "scatter_plot", "3d_visualization", "multi_curve", "log_display", "unprocessed_curves", "quality_overview", "histogram"]
        viz_combo = ttk.Combobox(row1, textvariable=self.viz_type_var, values=viz_types, width=22)
        viz_combo.pack(side='left', padx=10)
        
        # Bind event to viz type changes to update UI
        viz_combo.bind('<<ComboboxSelected>>', self.on_viz_type_change_enhanced)
        
        # Second row for curve selection
        row2 = ttk.Frame(control_frame)
        row2.pack(fill='x', pady=5)
        
        ttk.Label(row2, text="Primary Curve:", style='Card.TLabel').pack(side='left')
        self.viz_curve_var = tk.StringVar()
        self.viz_curve_combo = ttk.Combobox(row2, textvariable=self.viz_curve_var, width=20)
        self.viz_curve_combo.pack(side='left', padx=10)
        
        ttk.Label(row2, text="Secondary Curve:", style='Card.TLabel').pack(side='left', padx=(20, 0))
        self.viz_curve2_var = tk.StringVar()
        self.viz_curve2_combo = ttk.Combobox(row2, textvariable=self.viz_curve2_var, width=20)
        self.viz_curve2_combo.pack(side='left', padx=10)
        
        # Third row for 3D visualization third curve
        self.third_curve_frame = ttk.Frame(control_frame)
        self.third_curve_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.third_curve_frame, text="Third Curve (3D only):", style='Card.TLabel').pack(side='left')
        self.viz_curve3_var = tk.StringVar()
        self.viz_curve3_combo = ttk.Combobox(self.third_curve_frame, textvariable=self.viz_curve3_var, width=20)
        self.viz_curve3_combo.pack(side='left', padx=10)
        
        # Third row for multi-curve selection
        self.multi_curve_frame = ttk.Frame(control_frame)
        self.multi_curve_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.multi_curve_frame, text="Select Multiple Curves:", style='Card.TLabel').pack(side='left')
        
        # Create a frame for the multi-select listbox and scrollbar
        listbox_frame = ttk.Frame(self.multi_curve_frame)
        listbox_frame.pack(side='left', padx=10, fill='x', expand=True)
        
        # Create multi-select listbox
        self.curve_listbox = tk.Listbox(listbox_frame, selectmode='multiple', height=4, width=50)
        curve_scroll_v = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.curve_listbox.yview)
        curve_scroll_h = ttk.Scrollbar(listbox_frame, orient='horizontal', command=self.curve_listbox.xview)
        self.curve_listbox.configure(yscrollcommand=curve_scroll_v.set, xscrollcommand=curve_scroll_h.set)
        
        self.curve_listbox.pack(side='left', fill='both', expand=True)
        curve_scroll_v.pack(side='right', fill='y')
        curve_scroll_h.pack(side='bottom', fill='x')
        
        # Select/Deselect All buttons
        select_all_btn = ttk.Button(self.multi_curve_frame, text=DIALOG_SELECT_ALL, 
                                   command=lambda: self.curve_listbox.select_set(0, tk.END))
        select_all_btn.pack(side='left', padx=(10, 5))
        
        deselect_all_btn = ttk.Button(self.multi_curve_frame, text=DIALOG_DESELECT_ALL, 
                                     command=lambda: self.curve_listbox.selection_clear(0, tk.END))
        deselect_all_btn.pack(side='left')
        
        # Hide multi-curve frame initially
        self.multi_curve_frame.pack_forget()
        
        # Visualization display option
        viz_display_frame = ttk.Frame(control_frame)
        viz_display_frame.pack(fill='x', pady=(5, 10))
        
        # Use existing instance created during __init__
        # self.plot_in_new_window_var is already initialized; bind to UI only
        ttk.Checkbutton(viz_display_frame, 
                       text="Open plots in new window (recommended for detailed analysis and dual monitors)",
                       variable=self.plot_in_new_window_var).pack(anchor='w')
        
        # Update button
        update_btn = self.ui.create_button(control_frame, text="Update Plot", 
                                          command=self.update_visualization_enhanced, button_type='primary', width=25)
        update_btn.pack(pady=10)
        
        # Visualization area
        viz_card, self.viz_content = self.ui.create_card(
            viz_frame, "Interactive Visualization",
            help_text="Render professional depth-based plots, multi-curve tracks, and comparisons. Use Export for PNG/PDF."
        )
        viz_card.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Initialize figure and canvas as None - will be created dynamically
        self.fig = None
        self.canvas = None
    
