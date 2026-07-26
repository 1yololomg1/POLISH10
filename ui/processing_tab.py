"""
Processing tab widget construction

STRUCTURE:
- ProcessingTabMixin: Mixin providing create_processing_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.constants import EVENT_CONFIGURE, FONT_DEFAULT


class ProcessingTabMixin:
    """Processing tab widget construction."""

    def create_processing_tab(self):
        """Create processing configuration and execution tab with uniformization settings"""
        process_frame = ttk.Frame(self.notebook)
        self.notebook.add(process_frame, text="Processing")
        
        # Left panel - Configuration (no nested scrolling)
        config_frame = ttk.Frame(process_frame)
        config_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Make left configuration panel scrollable
        config_canvas = tk.Canvas(config_frame)
        config_scrollbar = ttk.Scrollbar(config_frame, orient='vertical', command=config_canvas.yview)
        config_canvas.configure(yscrollcommand=config_scrollbar.set)
        config_scrollbar.pack(side='right', fill='y')
        config_canvas.pack(side='left', fill='both', expand=True)

        # Inner frame that holds all configuration widgets
        config_inner = ttk.Frame(config_canvas)
        config_canvas.create_window((0, 0), window=config_inner, anchor='nw')

        # Update scrollable region when inner frame changes size
        def _config_on_configure(event):
            config_canvas.configure(scrollregion=config_canvas.bbox('all'))
        config_inner.bind(EVENT_CONFIGURE, _config_on_configure)
        
        # Create notebook for configuration categories
        config_notebook = ttk.Notebook(config_inner)
        config_notebook.pack(fill='both', expand=True)
        
        # Create tabs for each configuration category
        uniformization_tab = ttk.Frame(config_notebook)
        gap_filling_tab = ttk.Frame(config_notebook)
        denoising_tab = ttk.Frame(config_notebook)
        advanced_tab = ttk.Frame(config_notebook)
        cohort_tab = ttk.Frame(config_notebook)
        
        # Add tabs to notebook
        config_notebook.add(uniformization_tab, text="Uniformization")
        config_notebook.add(gap_filling_tab, text="Gap Filling")
        config_notebook.add(denoising_tab, text="Denoising")
        config_notebook.add(advanced_tab, text="Advanced")
        config_notebook.add(cohort_tab, text="Cross-Well Cohort")
        
        # Uniformization tab content
        # Standard depth spacing
        ttk.Label(uniformization_tab, text="Standard Depth Spacing:", style='Card.TLabel').pack(anchor='w', padx=10, pady=(10, 5))
        # Initialize with metric default here as well
        self.depth_spacing_var = tk.DoubleVar(value=0.1)
        spacing_frame = ttk.Frame(uniformization_tab)
        spacing_frame.pack(fill='x', pady=5, padx=10)
        
        # Quick preset buttons
        spacing_values = [0.1, 0.25, 0.5, 1.0]
        for val in spacing_values:
            ttk.Radiobutton(spacing_frame, text=f"{val} m", value=val, 
                           variable=self.depth_spacing_var).pack(side='left', padx=10)
        
        # Custom depth spacing entry
        custom_spacing_frame = ttk.Frame(uniformization_tab)
        custom_spacing_frame.pack(fill='x', pady=5, padx=10)
        
        ttk.Label(custom_spacing_frame, text="Custom Spacing:").pack(side='left', padx=(0, 5))
        custom_spacing_entry = ttk.Entry(custom_spacing_frame, textvariable=self.depth_spacing_var, width=10)
        custom_spacing_entry.pack(side='left', padx=(0, 5))
        ttk.Label(custom_spacing_frame, text="meters (affects gap thresholds, filter windows, and resampling)").pack(side='left')
        
        # Curve renaming to standard mnemonics
        self.rename_curves_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(uniformization_tab, text="Rename Curves to Standard Mnemonics", 
                       variable=self.rename_curves_var).pack(anchor='w', pady=5, padx=10)
        
        # Standard null value handling
        ttk.Label(uniformization_tab, text="Standard Null Value:", style='Card.TLabel').pack(anchor='w', pady=(10, 5), padx=10)
        self.null_value_var = tk.StringVar(value="-999.25")
        null_combo = ttk.Combobox(uniformization_tab, textvariable=self.null_value_var, 
                                 values=["-999.25", "-999", "-9999", "NaN"], 
                                 state='readonly', width=15)
        null_combo.pack(anchor='w', pady=5, padx=10)
        
        # Standardize units
        self.standardize_units_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(uniformization_tab, text="Standardize Units", 
                       variable=self.standardize_units_var).pack(anchor='w', pady=5, padx=10)

        # Move full Units controls into Uniformization section
        try:
            self.unit_standardizer.add_unit_standardization_ui(uniformization_tab)
        except Exception as e:
            self.log_processing(f"Warning: Could not add unit standardization UI: {e}")

        # Normalization controls
        norm_card, norm_content = self.ui.create_card(uniformization_tab, "Normalization")
        norm_card.pack(fill='x', pady=5, padx=10)
        ttk.Label(norm_content, text="Optionally normalize curve values (excludes depth).",
                  style='Card.TLabel').pack(anchor='w', pady=(5, 5))

        self.normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(norm_content, text="Enable Normalization", 
                        variable=self.normalize_var).pack(anchor='w', pady=5)

        ttk.Label(norm_content, text="Method:").pack(anchor='w')
        self.normalize_method_var = tk.StringVar(value="zscore")
        ttk.Combobox(norm_content, textvariable=self.normalize_method_var,
                     values=["zscore", "minmax"], state='readonly', width=12).pack(anchor='w', pady=(0, 5))

        # Percent to decimal conversion controls
        conv_card, conv_content = self.ui.create_card(uniformization_tab, "Percent to Decimal Conversion")
        conv_card.pack(fill='x', pady=5, padx=10)
        ttk.Label(conv_content, text="Convert percentage values (0-100) to decimal fraction (0-1).",
                  style='Card.TLabel').pack(anchor='w', pady=(5, 5))

        conv_btns = ttk.Frame(conv_content)
        conv_btns.pack(fill='x', pady=(0, 10))

        auto_btn = self.ui.create_button(conv_btns, text="Convert % to decimal (auto-detect)",
                                         command=self.convert_columns_percent_to_decimal,
                                         button_type='primary', width=28)
        auto_btn.pack(side='left', padx=(0, 15))

        select_btn = self.ui.create_button(conv_btns, text="Select Columns...",
                                           command=self.open_percent_conversion_dialog,
                                           button_type='secondary', width=18)
        select_btn.pack(side='left')
        
        # Output format compliance
        ttk.Label(uniformization_tab, text="Output Format Compliance:", style='Card.TLabel').pack(anchor='w', pady=(10, 5), padx=10)
        self.output_format_var = tk.StringVar(value="Company Standard")
        format_combo = ttk.Combobox(uniformization_tab, textvariable=self.output_format_var, 
                                   values=["Company Standard", "LAS 2.0", "LAS 3.0"], 
                                   state='readonly', width=20)
        format_combo.pack(anchor='w', pady=5, padx=10)
        
        # Gap filling tab content
        ttk.Label(gap_filling_tab, text="Maximum Gap Size:", style='Card.TLabel').pack(anchor='w', padx=10, pady=(10, 5))
        self.max_gap_var = tk.IntVar(value=500)  # Default increased to 500
        
        # Create a frame for gap size selector with label showing current value
        gap_size_frame = ttk.Frame(gap_filling_tab)
        gap_size_frame.pack(fill='x', pady=5, padx=10)
        
        # Increased maximum to 2000
        gap_scale = ttk.Scale(gap_size_frame, from_=10, to=2000, variable=self.max_gap_var, 
                             orient='horizontal', length=200)
        gap_scale.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        # Add value display label showing both points and meters
        self.gap_size_label = ttk.Label(gap_size_frame, text="500 pts (250 m)", width=18)
        self.gap_size_label.pack(side='right')
        
        # Update label when scale or depth spacing changes
        def update_gap_label(*args):
            pts = self.max_gap_var.get()
            spacing = self.depth_spacing_var.get()
            meters = pts * spacing
            self.gap_size_label.config(text=f"{pts} pts ({meters:.1f} m)")
        
        self.max_gap_var.trace_add("write", update_gap_label)
        self.depth_spacing_var.trace_add("write", update_gap_label)
        
        # Large gap treatment options
        ttk.Label(gap_filling_tab, text="Large Gap Treatment:", style='Card.TLabel').pack(anchor='w', pady=(10, 5), padx=10)
        self.large_gap_var = tk.StringVar(value="formation_based")
        large_gap_options = [
            ("Standard Interpolation", "standard"),
            ("Formation-Based Model (Chaveste)", "formation_based"),
            ("Skip Large Gaps (>1000 points)", "skip")
        ]
        
        for text, value in large_gap_options:
            ttk.Radiobutton(gap_filling_tab, text=text, value=value, 
                           variable=self.large_gap_var).pack(anchor='w', padx=30, pady=2)
        
        # Large gap threshold
        threshold_frame = ttk.Frame(gap_filling_tab)
        threshold_frame.pack(fill='x', pady=5, padx=10)
        ttk.Label(threshold_frame, text="Large Gap Threshold:").pack(side='left')
        # Bind to existing variable initialized in __init__
        threshold_entry = ttk.Entry(threshold_frame, width=6, textvariable=self.large_gap_threshold_var)
        threshold_entry.pack(side='left', padx=5)
        self.large_gap_physical_label = ttk.Label(threshold_frame, text="points (250 m)", foreground='#666666')
        self.large_gap_physical_label.pack(side='left', padx=(5, 0))
        
        # Update physical distance label when threshold or depth spacing changes
        def update_large_gap_physical(*args):
            pts = self.large_gap_threshold_var.get()
            spacing = self.depth_spacing_var.get()
            meters = pts * spacing
            self.large_gap_physical_label.config(text=f"points ({meters:.1f} m)")
        
        self.large_gap_threshold_var.trace_add("write", update_large_gap_physical)
        self.depth_spacing_var.trace_add("write", update_large_gap_physical)
        
        # Geological gap threshold - NEW FEATURE
        ttk.Label(gap_filling_tab, text="Geological Gap Threshold:", style='Card.TLabel').pack(anchor='w', pady=(15, 5), padx=10)
        
        # Help text
        help_text = ttk.Label(gap_filling_tab, 
                             text="Gaps larger than this threshold are considered geological/logging features\n"
                                  "(e.g., cased holes, interval logging), not data errors.",
                             foreground='#666666', font=(FONT_DEFAULT, 8))
        help_text.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Geological threshold frame with slider
        geo_threshold_frame = ttk.Frame(gap_filling_tab)
        geo_threshold_frame.pack(fill='x', pady=5, padx=10)
        
        # Use instance created during __init__
        geo_scale = ttk.Scale(geo_threshold_frame, from_=50, to=2000, 
                             variable=self.geological_gap_threshold_var, 
                             orient='horizontal', length=200)
        geo_scale.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        # Value display label showing both points and meters
        self.geo_gap_label = ttk.Label(geo_threshold_frame, text="200 pts (100 m)", width=18)
        self.geo_gap_label.pack(side='left')
        
        # Update label when scale or depth spacing changes
        def update_geo_gap_label(*args):
            pts = self.geological_gap_threshold_var.get()
            spacing = self.depth_spacing_var.get()
            meters = pts * spacing
            self.geo_gap_label.config(text=f"{pts} pts ({meters:.1f} m)")
        
        self.geological_gap_threshold_var.trace_add("write", update_geo_gap_label)
        self.depth_spacing_var.trace_add("write", update_geo_gap_label)
        
        # Method Priority for normal gaps
        ttk.Label(gap_filling_tab, text="Method Priority:", style='Card.TLabel').pack(anchor='w', pady=(10, 5), padx=10)
        self.gap_method_var = tk.StringVar(value="auto")
        methods = ['auto', 'gaussian_process', 'kriging', 'cubic_spline', 'linear']
        method_combo = ttk.Combobox(gap_filling_tab, textvariable=self.gap_method_var, values=methods, state='readonly')
        method_combo.pack(fill='x', pady=5, padx=10)
        
        self.physics_informed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gap_filling_tab, text="Physics-Informed Processing", 
                       variable=self.physics_informed_var).pack(anchor='w', pady=5, padx=10)
        
        self.multi_curve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gap_filling_tab, text="Multi-Curve Correlation", 
                       variable=self.multi_curve_var).pack(anchor='w', pady=5, padx=10)
        
        # Denoising tab content
        ttk.Label(denoising_tab, text="Denoising Method:", style='Card.TLabel').pack(anchor='w', padx=10, pady=(10, 5))
        self.denoise_method_var = tk.StringVar(value="auto")
        denoise_methods = ['auto', 'wavelet', 'bilateral', 'savgol', 'median']
        denoise_combo = ttk.Combobox(denoising_tab, textvariable=self.denoise_method_var, 
                                   values=denoise_methods, state='readonly')
        denoise_combo.pack(fill='x', pady=5, padx=10)
        
        # Advanced tab content
        # Quality control settings
        self.qc_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Enable Quality Control", 
                       variable=self.qc_enabled_var).pack(anchor='w', pady=5, padx=10)
        
        self.outlier_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Outlier Detection", 
                       variable=self.outlier_detection_var).pack(anchor='w', pady=5, padx=10)
        
        self.range_validation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Range Validation", 
                       variable=self.range_validation_var).pack(anchor='w', pady=5, padx=10)
        
        # Advanced processing options
        self.parallel_processing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_tab, text="Parallel Processing", 
                       variable=self.parallel_processing_var).pack(anchor='w', pady=5, padx=10)
        
        self.uncertainty_quantification_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Uncertainty Quantification", 
                       variable=self.uncertainty_quantification_var).pack(anchor='w', pady=5, padx=10)
        
        self.confidence_intervals_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Confidence Intervals", 
                       variable=self.confidence_intervals_var).pack(anchor='w', pady=5, padx=10)
        
        # Performance settings
        ttk.Label(advanced_tab, text="Memory Limit (MB):", style='Card.TLabel').pack(anchor='w', pady=(10, 5), padx=10)
        self.memory_limit_var = tk.IntVar(value=2048)
        memory_entry = ttk.Entry(advanced_tab, textvariable=self.memory_limit_var, width=10)
        memory_entry.pack(anchor='w', pady=5, padx=10)
        
        self.auto_cleanup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_tab, text="Auto Memory Cleanup", 
                       variable=self.auto_cleanup_var).pack(anchor='w', pady=5, padx=10)

        # Cohort/prior configuration UI (expert workflow)
        ttk.Label(cohort_tab, text="Cohort Selection & Cross-Well Priors", style='Card.TLabel').pack(anchor='w', padx=10, pady=(10, 5))
        ttk.Checkbutton(cohort_tab, text="Enable Cross-Well Priors",
                        variable=self.use_crosswell_priors_var).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(cohort_tab, text="Two-Pass Refinement (pass 1 single-well, pass 2 with priors)",
                        variable=self.two_pass_refinement_var).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(cohort_tab, text="Depth-Binned Priors (per zone/depth bins)",
                        variable=self.priors_depth_binning_var).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(cohort_tab, text="Auto-Select Cohort (analog wells by curve coverage/quality)",
                        variable=self.auto_select_cohort_var).pack(anchor='w', padx=10, pady=(2, 8))

        cohort_btns = ttk.Frame(cohort_tab)
        cohort_btns.pack(fill='x', padx=10, pady=(0, 10))
        build_btn = self.ui.create_button(cohort_btns, text="Build Cross-Well Priors",
                                          command=self.build_crosswell_priors, button_type='primary', width=25)
        build_btn.pack(side='left', padx=(0, 10))
        view_btn = self.ui.create_button(cohort_btns, text="View Priors Summary",
                                         command=self.show_cross_well_summary, button_type='secondary', width=25)
        view_btn.pack(side='left')

        # Manual cohort picker listbox (optional)
        cohort_list_frame = ttk.Frame(cohort_tab)
        cohort_list_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        ttk.Label(cohort_list_frame, text="Select Cohort Wells (exclude active)").pack(anchor='w')
        self.cohort_listbox = tk.Listbox(cohort_list_frame, selectmode='multiple', height=6)
        self.cohort_listbox.pack(fill='both', expand=True)
        # Populate from well_datasets
        try:
            for wid in self.well_datasets.keys():
                self.cohort_listbox.insert(tk.END, wid)
        except Exception:
            pass
        def _apply_cohort_selection():
            try:
                sel = self.cohort_listbox.curselection()
                self.cohort_selected_well_ids = [self.cohort_listbox.get(i) for i in sel if self.cohort_listbox.get(i) != self.active_well_id]
                self.status_label.config(text=f"Cohort set: {len(self.cohort_selected_well_ids)} well(s)")
            except Exception:
                pass
        apply_btn = self.ui.create_button(cohort_tab, text="Apply Cohort Selection",
                                          command=_apply_cohort_selection, button_type='secondary', width=25)
        apply_btn.pack(anchor='e', padx=10, pady=(0, 10))
        
        # Processing execution - placed below the notebook; now reachable via scrolling
        exec_card, exec_content = self.ui.create_card(
            config_inner, "Execute Processing",
            help_text="Run processing for the active well or all wells. Use Cross-Well Cohort to enable priors and two-pass refinement."
        )
        exec_card.pack(fill='x', pady=10)
        
        process_btn = self.ui.create_button(exec_content, text="Start Processing (Active Well)",
                                           command=self.start_processing, button_type='primary', width=30)
        process_btn.pack(fill='x', pady=(10, 5), padx=10)

        process_all_btn = self.ui.create_button(exec_content, text="Process All Wells",
                                               command=self.process_all_wells, button_type='success', width=30)
        process_all_btn.pack(fill='x', pady=(0, 5), padx=10)

        cross_summary_btn = self.ui.create_button(exec_content, text="Cross-Well Summary",
                                                 command=self.show_cross_well_summary, button_type='secondary', width=30)
        cross_summary_btn.pack(fill='x', pady=(0, 5), padx=10)

        export_all_btn = self.ui.create_button(exec_content, text="Export All Processed (LAS)",
                                              command=self.export_all_processed, button_type='secondary', width=30)
        export_all_btn.pack(fill='x', pady=(0, 10), padx=10)
        
        # Add a separator for visual clarity
        separator = ttk.Separator(exec_content, orient='horizontal')
        separator.pack(fill='x', pady=20)
        
        # Add quick visualization buttons for unprocessed curves
        viz_buttons_frame = ttk.Frame(exec_content)
        viz_buttons_frame.pack(fill='x', pady=(10, 10), padx=10)
        
        ttk.Label(viz_buttons_frame, text="Quick Visualization:", style='Card.TLabel').pack(anchor='w', pady=(0, 8))
        
        quick_viz_frame = ttk.Frame(viz_buttons_frame)
        quick_viz_frame.pack(fill='x')
        
        # Button to visualize unprocessed curves
        unprocessed_btn = self.ui.create_button(quick_viz_frame, text="View Unprocessed Curves",
                                               command=self.quick_view_unprocessed, button_type='secondary', width=20)
        unprocessed_btn.pack(side='left', padx=(0, 15))
        
        # Button to show quality overview
        quality_btn = self.ui.create_button(quick_viz_frame, text="Quality Overview",
                                           command=self.quick_quality_overview, button_type='secondary', width=18)
        quality_btn.pack(side='left', padx=(0, 15))
        
        # Button to compare all curves
        compare_btn = self.ui.create_button(quick_viz_frame, text="Compare All Curves",
                                           command=self.quick_compare_all, button_type='secondary', width=18)
        compare_btn.pack(side='left')
        
        # Right panel - Progress and results
        progress_frame = ttk.Frame(process_frame)
        progress_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Progress monitoring
        self.progress_card, self.progress_bar, self.status_label = self.ui.create_progress_card(
            progress_frame, "Processing Progress"
        )
        self.progress_card.pack(fill='x', pady=(0, 10))
        
        # Results display
        results_card, results_content = self.ui.create_card(progress_frame, "Processing Results")
        results_card.pack(fill='both', expand=True)
        
        # Create frame for text widget and scrollbars
        text_frame = ttk.Frame(results_content)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(text_frame, height=20, font=('Consolas', 10), wrap='word')
        results_v_scroll = ttk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        results_h_scroll = ttk.Scrollbar(text_frame, orient='horizontal', command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=results_v_scroll.set, xscrollcommand=results_h_scroll.set)
        
        # Pack text widget and scrollbars
        results_v_scroll.pack(side='right', fill='y')
        results_h_scroll.pack(side='bottom', fill='x')
        self.results_text.pack(side='left', fill='both', expand=True)
