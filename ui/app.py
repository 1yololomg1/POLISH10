"""
Application UI shell — setup_ui and tab composition.

STRUCTURE:
- AppUIMixin: hosts setup_ui() and inherits per-tab mixins
- Tab mixins live in ui/*_tab.py (widget construction only)

DATA FLOW:
App.__init__ → setup_ui() → create_*_tab() mixins → notebook tabs

Moved from advanced_preprocessing_system10.py (Step 4 UI extraction).
"""

from __future__ import annotations

import importlib
import tkinter as tk
from tkinter import ttk

from ui.data_tab import DataTabMixin
from ui.processing_tab import ProcessingTabMixin
from ui.visualization_tab import VisualizationTabMixin
from ui.report_tab import ReportTabMixin
from ui.batch_tab import BatchTabMixin
from ui.units_tab import UnitsTabMixin
from ui.status import SecureStatusManager


class AppUIMixin(
    DataTabMixin,
    ProcessingTabMixin,
    VisualizationTabMixin,
    ReportTabMixin,
    BatchTabMixin,
    UnitsTabMixin,
):
    """Main-window UI construction for AdvancedPreprocessingApplication."""

    def setup_ui(self):
        """Setup the main user interface"""
        # Create main scrollable frame
        self.main_canvas = tk.Canvas(self.root)
        main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        main_scrollbar_h = ttk.Scrollbar(self.root, orient="horizontal", command=self.main_canvas.xview)
        
        # Main container
        main_frame = ttk.Frame(self.main_canvas, style='Card.TFrame')
        
        # Configure canvas
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set, xscrollcommand=main_scrollbar_h.set)
        
        # Pack scrollbars and canvas
        main_scrollbar.pack(side="right", fill="y")
        main_scrollbar_h.pack(side="bottom", fill="x")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas - store reference for width binding
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configure scrolling and ensure window width matches canvas width
        def configure_scroll_region(event=None):
            # Update scroll region based on the bounding box of all items
            try:
                # Get the bounding box of the canvas window (main_frame)
                bbox = self.main_canvas.bbox("all")
                if bbox:
                    # Set scrollregion to include padding
                    self.main_canvas.configure(scrollregion=bbox)
                
                # Ensure canvas window width matches canvas width (prevents horizontal scrolling issues)
                canvas_width = self.main_canvas.winfo_width()
                if canvas_width > 1:  # Only if canvas is visible
                    self.main_canvas.itemconfig(self.main_canvas_window, width=canvas_width)
            except (tk.TclError, AttributeError) as canvas_error:
                # Canvas may not be fully initialized or destroyed - cosmetic, continue
                pass
            except Exception as canvas_error:
                # Unexpected error configuring canvas - log for debugging
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: Canvas configuration failed: {type(canvas_error).__name__}: {str(canvas_error)}")
                    except Exception:
                        pass  # Can't log logging failure
        
        # Enhanced configure handler that properly calculates scroll region
        def enhanced_configure_handler(event=None):
            try:
                # Update the canvas to get accurate measurements
                self.main_canvas.update_idletasks()
                
                # Get the actual size of the frame
                frame_width = main_frame.winfo_reqwidth()
                frame_height = main_frame.winfo_reqheight()
                
                # Calculate scroll region based on frame dimensions
                if frame_width > 0 and frame_height > 0:
                    scroll_region = f"0 0 {frame_width} {frame_height}"
                    self.main_canvas.configure(scrollregion=scroll_region)
                
                # Also try bbox method as fallback
                bbox = self.main_canvas.bbox("all")
                if bbox:
                    self.main_canvas.configure(scrollregion=bbox)
                
                # Ensure canvas window width matches canvas width
                canvas_width = self.main_canvas.winfo_width()
                if canvas_width > 1:
                    self.main_canvas.itemconfig(self.main_canvas_window, width=canvas_width)
            except (tk.TclError, AttributeError) as canvas_error:
                # Canvas may not be fully initialized or destroyed - cosmetic, continue
                pass
            except Exception as canvas_error:
                # Unexpected error configuring canvas - log for debugging
                if hasattr(self, 'log_processing'):
                    try:
                        self.log_processing(f"Warning: Canvas configuration failed: {type(canvas_error).__name__}: {str(canvas_error)}")
                    except Exception:
                        pass  # Can't log logging failure
        
        # Store configure function and frame reference for later use
        self._configure_scroll_region = enhanced_configure_handler
        self._main_frame = main_frame  # Store reference to main frame
        
        # Bind configure events to update scroll region when frame or canvas resizes
        main_frame.bind("<Configure>", lambda e: enhanced_configure_handler())
        self.main_canvas.bind("<Configure>", lambda e: enhanced_configure_handler())
        
        # Enable mouse wheel scrolling - bind to canvas and frame for better focus handling
        def on_mousewheel(event):
            # Only scroll if mouse is over canvas area
            if self.main_canvas.winfo_containing(event.x_root, event.y_root):
                self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def on_shift_mousewheel(event):
            if self.main_canvas.winfo_containing(event.x_root, event.y_root):
                self.main_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind to canvas specifically first, then bind_all as fallback
        self.main_canvas.bind("<MouseWheel>", on_mousewheel)
        self.main_canvas.bind("<Shift-MouseWheel>", on_shift_mousewheel)
        # Also bind to main_frame for when it has focus
        main_frame.bind("<MouseWheel>", on_mousewheel)
        main_frame.bind("<Shift-MouseWheel>", on_shift_mousewheel)
        # Fallback bind_all for Windows
        self.main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.main_canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)
        # Linux/X11 alternative bindings for wheel events
        def _on_button4(event):
            try:
                self.main_canvas.yview_scroll(-1, "units")
            except (tk.TclError, AttributeError) as scroll_error:
                # Canvas scrolling failed - graceful degradation
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas vertical scroll (button 4)",
                        "main_canvas",
                        graceful_degradation=True
                    )
        def _on_button5(event):
            try:
                self.main_canvas.yview_scroll(1, "units")
            except (tk.TclError, AttributeError) as scroll_error:
                # Canvas scrolling failed - graceful degradation
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas vertical scroll (button 5)",
                        "main_canvas",
                        graceful_degradation=True
                    )
        def _on_shift_button4(event):
            try:
                self.main_canvas.xview_scroll(-1, "units")
            except (tk.TclError, AttributeError) as scroll_error:
                # Canvas scrolling failed - graceful degradation
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas horizontal scroll (shift button 4)",
                        "main_canvas",
                        graceful_degradation=True
                    )
        def _on_shift_button5(event):
            try:
                self.main_canvas.xview_scroll(1, "units")
            except (tk.TclError, AttributeError) as scroll_error:
                # Canvas scrolling failed - graceful degradation
                if hasattr(self, 'handle_ui_error'):
                    self.handle_ui_error(
                        scroll_error,
                        "Canvas horizontal scroll (shift button 5)",
                        "main_canvas",
                        graceful_degradation=True
                    )
        self.main_canvas.bind_all("<Button-4>", _on_button4)
        self.main_canvas.bind_all("<Button-5>", _on_button5)
        self.main_canvas.bind_all("<Shift-Button-4>", _on_shift_button4)
        self.main_canvas.bind_all("<Shift-Button-5>", _on_shift_button5)
        
        # Enable keyboard navigation
        def on_key_press(event):
            if event.keysym == 'Up':
                self.main_canvas.yview_scroll(-1, "units")
            elif event.keysym == 'Down':
                self.main_canvas.yview_scroll(1, "units")
            elif event.keysym == 'Left':
                self.main_canvas.xview_scroll(-1, "units")
            elif event.keysym == 'Right':
                self.main_canvas.xview_scroll(1, "units")
            elif event.keysym == 'Page_Up':
                self.main_canvas.yview_scroll(-1, "pages")
            elif event.keysym == 'Page_Down':
                self.main_canvas.yview_scroll(1, "pages")
        
        self.main_canvas.bind_all("<Key>", on_key_press)
        
        # Add padding to main frame
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                               text="Advanced Wireline Data Preprocessing System",
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_data_tab()
        self.create_processing_tab()
        self.create_visualization_tab()
        self.create_report_tab()
        self.create_batch_tab()  # NEW: Batch Processing tab (Priority 1.5)
        # Units UI moved into Processing > Uniformization for better workflow
        
        # Add performance optimization
        self.root.update_idletasks()
        
        # Ensure scroll region is properly set after all widgets are created
        def finalize_scroll_setup():
            if hasattr(self, '_configure_scroll_region'):
                self._configure_scroll_region()
                # Force multiple refreshes to ensure it's set correctly
                self.root.after(10, self._configure_scroll_region)
                self.root.after(50, self._configure_scroll_region)
                self.root.after(100, self._configure_scroll_region)
        
        # Schedule scroll region update after UI is fully rendered
        self.root.after_idle(finalize_scroll_setup)
        self.root.after(100, self.check_system_resources)
        
        # Initialize status manager after all UI components are created
        try:
            if hasattr(self, 'results_text') and hasattr(self, 'status_label') and hasattr(self, 'progress_bar'):
                self.status_manager = SecureStatusManager(
                    self.results_text, 
                    self.status_label, 
                    self.progress_bar
                )
                self.log_processing("Status manager initialized successfully")
            else:
                self.log_processing("Warning: UI components not ready for status manager initialization")
        except Exception as e:
            self.log_processing(f"Error initializing status manager: {e}")
            self.status_manager = None
        
        # Set application reference for unit standardizer
        self.unit_standardizer.set_application_reference(self)
        
        # Initialize cross-well prior manager after UI creation
        # Class lives on the App host module (avoid ui ↔ monolith circular import)
        try:
            host_mod = importlib.import_module(type(self).__module__)
            prior_cls = getattr(host_mod, "CrossWellPriorManager", None)
            if prior_cls is None:
                raise ImportError("CrossWellPriorManager not found on host module")
            self.crosswell_prior_manager = prior_cls()
            self.crosswell_prior_manager.set_application_reference(self)
        except Exception:
            self.crosswell_prior_manager = None
    
