"""
Data Loading tab widget construction

STRUCTURE:
- DataTabMixin: Mixin providing create_data_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.constants import FONT_DEFAULT


class DataTabMixin:
    """Data Loading tab widget construction."""

    def create_data_tab(self):
        """Create data loading and inspection tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Loading")
        
        # File loading section
        load_card, load_content = self.ui.create_card(
            data_frame, "Load Data File",
            help_text="Select and load a single LAS/CSV/Excel file. The app will analyze curves, compute stats, and prepare for processing."
        )
        load_card.pack(fill='x', pady=(0, 10))
        
        file_frame = ttk.Frame(load_content)
        file_frame.pack(fill='x', pady=10)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = self.ui.create_button(file_frame, text="Browse", 
                                          command=self.browse_file, button_type='secondary', width=20)
        browse_btn.pack(side='right')
        
        # Create a button frame for Load and Clear buttons
        button_frame = ttk.Frame(load_content)
        button_frame.pack(fill='x', pady=15)
        
        load_btn = self.ui.create_button(button_frame, text="Load & Analyze File",
                                        command=self.load_file, button_type='success', width=25)
        load_btn.pack(side='left', padx=(0, 15))
        
        clear_btn = self.ui.create_button(button_frame, text="Clear Data",
                                         command=self.clear_data, button_type='warning', width=20)
        clear_btn.pack(side='left')
        
        # New: Load multiple files (multiwell)
        multi_btn = self.ui.create_button(button_frame, text="Load Multiple Files",
                                         command=self.load_multiple_files, button_type='primary', width=25)
        multi_btn.pack(side='left', padx=(15, 0))
        
        # CRITICAL: Well Information Card for safety
        well_card, well_content = self.ui.create_card(
            data_frame, "Well Identification",
            help_text="Shows key metadata parsed from the LAS header (well name, UWI, field, depth range)."
        )
        well_card.pack(fill='x', pady=(0, 10))
        
        # Create labels for well information (will be populated on load)
        self.well_name_label = ttk.Label(well_content, text="Well: Not loaded", 
                                         font=(FONT_DEFAULT, 10, 'bold'), foreground='#CC0000')
        self.well_name_label.pack(anchor='w', pady=2)
        
        self.field_label = ttk.Label(well_content, text="Field: Not loaded", 
                                     font=(FONT_DEFAULT, 9))
        self.field_label.pack(anchor='w', pady=2)
        
        self.uwi_label = ttk.Label(well_content, text="UWI: Not loaded", 
                                   font=(FONT_DEFAULT, 9))
        self.uwi_label.pack(anchor='w', pady=2)
        
        self.company_label = ttk.Label(well_content, text="Company: Not loaded", 
                                       font=(FONT_DEFAULT, 9))
        self.company_label.pack(anchor='w', pady=2)
        
        self.depth_range_label = ttk.Label(well_content, text="Depth Range: Not loaded", 
                                           font=(FONT_DEFAULT, 9))
        self.depth_range_label.pack(anchor='w', pady=2)

        # New: Loaded Wells manager
        wells_card, wells_content = self.ui.create_card(
            data_frame, "Loaded Wells",
            help_text="Manage multiple wells in the session. Set the active well, remove entries, or process all/selected wells."
        )
        wells_card.pack(fill='x', pady=(0, 10))
        wells_toolbar = ttk.Frame(wells_content)
        wells_toolbar.pack(fill='x', pady=(5, 5))
        self.well_listbox = tk.Listbox(wells_content, height=6, selectmode='extended')
        self.well_listbox.pack(fill='x', padx=10, pady=(0, 8))
        set_active_btn = self.ui.create_button(wells_toolbar, text="Set Active Well",
                                              command=self.on_set_active_well, button_type='secondary', width=20)
        set_active_btn.pack(side='left', padx=(0, 10))
        remove_btn = self.ui.create_button(wells_toolbar, text="Remove Selected",
                                          command=self.on_remove_selected_wells, button_type='warning', width=20)
        remove_btn.pack(side='left')
        process_all_quick_btn = self.ui.create_button(wells_toolbar, text="Process All Wells",
                                                     command=self.process_all_wells, button_type='success', width=20)
        process_all_quick_btn.pack(side='left', padx=(10, 0))
        process_sel_btn = self.ui.create_button(wells_toolbar, text="Process Selected",
                                               command=self.process_selected_wells, button_type='primary', width=20)
        process_sel_btn.pack(side='left', padx=(10, 0))
        
        # Data summary section
        summary_card, summary_content = self.ui.create_card(
            data_frame, "Data Summary",
            help_text="Per-curve overview: mnemonic, type, units, range, data quality with geological gap awareness."
        )
        summary_card.pack(fill='both', expand=True)
        
        # Create treeview for curve information
        columns = ('Mnemonic', 'Type', 'Unit', 'Range', 'Quality', 'Missing %')
        self.data_tree = ttk.Treeview(summary_content, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)
        
        # Scrollbars for treeview
        tree_scroll_v = ttk.Scrollbar(summary_content, orient='vertical', command=self.data_tree.yview)
        tree_scroll_h = ttk.Scrollbar(summary_content, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        self.data_tree.pack(side='left', fill='both', expand=True)
        tree_scroll_v.pack(side='right', fill='y')
        tree_scroll_h.pack(side='bottom', fill='x')
    
