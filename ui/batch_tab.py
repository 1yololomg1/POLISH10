"""
Batch Processing tab widget construction

STRUCTURE:
- BatchTabMixin: Mixin providing create_batch_tab() for AdvancedPreprocessingApplication

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
Widget construction only — plotting/processing handlers remain on the App class.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.batch_processing import BatchProcessingManager
from ui.constants import FONT_DEFAULT


class BatchTabMixin:
    """Batch Processing tab widget construction."""

    def create_batch_tab(self):
        """Create batch processing tab for processing multiple files"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch Processing")
        
        # Initialize batch manager
        if self.batch_manager is None:
            self.batch_manager = BatchProcessingManager(self)
        
        # Directory selection section
        dir_card, dir_content = self.ui.create_card(
            batch_frame, "Select Directory",
            help_text="Choose a directory containing LAS files to process in batch."
        )
        dir_card.pack(fill='x', pady=(0, 10), padx=10)
        
        dir_frame = ttk.Frame(dir_content)
        dir_frame.pack(fill='x', pady=10)
        
        self.batch_directory_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_frame, textvariable=self.batch_directory_var, width=60)
        dir_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_dir_btn = self.ui.create_button(dir_frame, text="Browse Directory",
                                              command=self.browse_batch_directory,
                                              button_type='secondary', width=20)
        browse_dir_btn.pack(side='right')
        
        # Recursive search option
        self.batch_recursive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dir_content, text="Search subdirectories recursively",
                        variable=self.batch_recursive_var).pack(anchor='w', pady=5)
        
        # Scan directory button
        scan_btn = self.ui.create_button(dir_content, text="Scan Directory",
                                        command=self.scan_batch_directory,
                                        button_type='primary', width=25)
        scan_btn.pack(anchor='w', pady=(10, 0))
        
        # File list section
        list_card, list_content = self.ui.create_card(
            batch_frame, "Files to Process",
            help_text="List of files found in the selected directory."
        )
        list_card.pack(fill='both', expand=True, pady=(0, 10), padx=10)
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(list_content)
        list_frame.pack(fill='both', expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.batch_file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=12)
        self.batch_file_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.batch_file_listbox.yview)
        
        # Status label
        self.batch_status_label = ttk.Label(list_content, text="No directory selected",
                                           font=(FONT_DEFAULT, 9))
        self.batch_status_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Processing controls
        control_card, control_content = self.ui.create_card(
            batch_frame, "Processing Controls",
            help_text="Configure output directory and start batch processing."
        )
        control_card.pack(fill='x', pady=(0, 10), padx=10)
        
        # Output directory
        output_frame = ttk.Frame(control_content)
        output_frame.pack(fill='x', pady=10)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side='left', padx=(0, 10))
        self.batch_output_dir_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.batch_output_dir_var, width=50)
        output_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_output_btn = self.ui.create_button(output_frame, text="Browse",
                                                  command=self.browse_batch_output_directory,
                                                  button_type='secondary', width=15)
        browse_output_btn.pack(side='right')
        
        # Action buttons
        button_frame = ttk.Frame(control_content)
        button_frame.pack(fill='x', pady=15)
        
        self.batch_process_btn = self.ui.create_button(button_frame, text="Start Batch Processing",
                                                       command=self.start_batch_processing,
                                                       button_type='success', width=25)
        self.batch_process_btn.pack(side='left', padx=(0, 10))
        
        self.batch_stop_btn = self.ui.create_button(button_frame, text="Stop Processing",
                                                    command=self.stop_batch_processing,
                                                    button_type='warning', width=20)
        self.batch_stop_btn.pack(side='left')
        self.batch_stop_btn.config(state='disabled')
        
        # Progress section
        progress_card, progress_content = self.ui.create_card(
            batch_frame, "Processing Progress",
            help_text="Progress information for batch processing operations."
        )
        progress_card.pack(fill='x', padx=10)
        
        self.batch_progress_label = ttk.Label(progress_content, text="Ready",
                                             font=(FONT_DEFAULT, 9))
        self.batch_progress_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.batch_progress_bar = ttk.Progressbar(progress_content, mode='determinate')
        self.batch_progress_bar.pack(fill='x', padx=10, pady=(0, 10))
    
