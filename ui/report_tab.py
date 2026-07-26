"""
Report tab widget construction

STRUCTURE:
- ReportTabMixin: Mixin providing create_report_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.constants import FONT_DEFAULT


class ReportTabMixin:
    """Report tab widget construction."""

    def create_report_tab(self):
        """Create professional reporting tab with LAS preview"""
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="Report")
        
        # Report controls - redesigned with logical grouping
        control_frame = ttk.Frame(report_frame)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)

        # Tab-level Help button
        def _show_report_help():
            try:
                from tkinter import Toplevel
                dialog = Toplevel(report_frame)
                dialog.title("Help - Report")
                dialog.transient(report_frame)
                dialog.grab_set()
                dialog.resizable(True, True)
                body = ttk.Frame(dialog, padding=15)
                body.pack(fill='both', expand=True)
                text = (
                    "Generate a comprehensive processing report for the active well. "
                    "Use Cross-Well Summary to view field-wide statistics across loaded wells. "
                    "Export Data exports the active well’s processed data; Export All Processed writes LAS for every well."
                )
                lbl = ttk.Label(body, text=text, wraplength=560, justify='left')
                lbl.pack(fill='x', expand=True)
                ttk.Button(body, text='Close', command=dialog.destroy).pack(anchor='e', pady=(10, 0))
                dialog.update_idletasks()
                x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
                y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
                dialog.geometry(f"+{x}+{y}")
            except Exception:
                pass
        help_btn = ttk.Button(control_frame, text='Help', command=_show_report_help)
        help_btn.pack(side='right')
        
        # Group 1: Report Actions
        report_actions_frame = ttk.Frame(control_frame)
        report_actions_frame.pack(side='top', fill='x', pady=(0, 10))
        
        ttk.Label(report_actions_frame, text="Report Actions:", 
                 font=(FONT_DEFAULT, 9, 'bold')).pack(side='left', padx=(0, 15))
        
        generate_btn = self.ui.create_button(report_actions_frame, text="Generate Report",
                                            command=self.generate_report, button_type='success', width=20)
        generate_btn.pack(side='left', padx=(0, 15))
        
        export_btn = self.ui.create_button(report_actions_frame, text="Export Data",
                                          command=self.export_data, button_type='primary', width=18)
        export_btn.pack(side='left')

        # New: Cross-well utilities in Report tab
        cross_btn = self.ui.create_button(report_actions_frame, text="Cross-Well Summary",
                                         command=self.show_cross_well_summary, button_type='secondary', width=20)
        cross_btn.pack(side='left', padx=(15, 0))
        export_all_btn2 = self.ui.create_button(report_actions_frame, text="Export All Processed",
                                               command=self.export_all_processed, button_type='secondary', width=20)
        export_all_btn2.pack(side='left', padx=(10, 0))
        build_priors_btn = self.ui.create_button(report_actions_frame, text="Build Priors",
                                                command=self.build_crosswell_priors, button_type='secondary', width=14)
        build_priors_btn.pack(side='left', padx=(10, 0))
        
        # Group 2: LAS Preview Actions
        preview_actions_frame = ttk.Frame(control_frame)
        preview_actions_frame.pack(side='top', fill='x')
        
        ttk.Label(preview_actions_frame, text="LAS Preview Actions:", 
                 font=(FONT_DEFAULT, 9, 'bold')).pack(side='left', padx=(0, 15))
        
        preview_orig_btn = self.ui.create_button(preview_actions_frame, text="Preview Original LAS",
                                                command=self.preview_original_las, button_type='secondary', width=22)
        preview_orig_btn.pack(side='left', padx=(0, 15))
        
        preview_proc_btn = self.ui.create_button(preview_actions_frame, text="Preview Processed LAS",
                                                command=self.preview_processed_las, button_type='secondary', width=22)
        preview_proc_btn.pack(side='left')
        
        # Create notebook for report tabs
        report_notebook = ttk.Notebook(report_frame)
        report_notebook.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Report display tab
        report_tab = ttk.Frame(report_notebook)
        report_notebook.add(report_tab, text="Processing Report")
        
        self.report_text = tk.Text(report_tab, font=('Consolas', 10), wrap='word')
        report_scroll_v = ttk.Scrollbar(report_tab, orient='vertical', command=self.report_text.yview)
        report_scroll_h = ttk.Scrollbar(report_tab, orient='horizontal', command=self.report_text.xview)
        self.report_text.configure(yscrollcommand=report_scroll_v.set, xscrollcommand=report_scroll_h.set)
        
        self.report_text.pack(side='left', fill='both', expand=True)
        report_scroll_v.pack(side='right', fill='y')
        report_scroll_h.pack(side='bottom', fill='x')
        
        # Original LAS preview tab
        original_las_preview_tab = ttk.Frame(report_notebook)
        report_notebook.add(original_las_preview_tab, text="Original LAS Preview")
        
        self.original_las_preview_text = tk.Text(original_las_preview_tab, font=('Consolas', 10), wrap='none', state='disabled', selectbackground='#F0F0F0', selectforeground='black')
        original_las_preview_scroll_y = ttk.Scrollbar(original_las_preview_tab, orient='vertical', command=self.original_las_preview_text.yview)
        original_las_preview_scroll_x = ttk.Scrollbar(original_las_preview_tab, orient='horizontal', command=self.original_las_preview_text.xview)
        self.original_las_preview_text.configure(yscrollcommand=original_las_preview_scroll_y.set, xscrollcommand=original_las_preview_scroll_x.set)
        
        self.original_las_preview_text.pack(side='top', fill='both', expand=True)
        original_las_preview_scroll_y.pack(side='right', fill='y')
        original_las_preview_scroll_x.pack(side='bottom', fill='x')
        
        # Disable copy functionality for original preview
        self.original_las_preview_text.bind("<Control-c>", lambda e: "break")
        self.original_las_preview_text.bind("<Control-a>", lambda e: "break")
        self.original_las_preview_text.bind("<Control-x>", lambda e: "break")
        self.original_las_preview_text.bind("<Button-3>", lambda e: "break")  # Right-click context menu
        
        # Processed LAS preview tab
        processed_las_preview_tab = ttk.Frame(report_notebook)
        report_notebook.add(processed_las_preview_tab, text="Processed LAS Preview")
        
        self.processed_las_preview_text = tk.Text(processed_las_preview_tab, font=('Consolas', 10), wrap='none', state='disabled', selectbackground='#F0F0F0', selectforeground='black')
        processed_las_preview_scroll_y = ttk.Scrollbar(processed_las_preview_tab, orient='vertical', command=self.processed_las_preview_text.yview)
        processed_las_preview_scroll_x = ttk.Scrollbar(processed_las_preview_tab, orient='horizontal', command=self.processed_las_preview_text.xview)
        self.processed_las_preview_text.configure(yscrollcommand=processed_las_preview_scroll_y.set, xscrollcommand=processed_las_preview_scroll_x.set)
        
        self.processed_las_preview_text.pack(side='top', fill='both', expand=True)
        processed_las_preview_scroll_y.pack(side='right', fill='y')
        processed_las_preview_scroll_x.pack(side='bottom', fill='x')
        
        # Disable copy functionality for processed preview
        self.processed_las_preview_text.bind("<Control-c>", lambda e: "break")
        self.processed_las_preview_text.bind("<Control-a>", lambda e: "break")
        self.processed_las_preview_text.bind("<Control-x>", lambda e: "break")
        self.processed_las_preview_text.bind("<Button-3>", lambda e: "break")  # Right-click context menu
    
