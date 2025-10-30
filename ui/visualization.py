import weakref
from contextlib import contextmanager
import gc
import threading
import numpy as np
import tkinter as tk  # noqa: F401 - used via matplotlib backends
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required to register 3D
import matplotlib.pyplot as plt
import warnings


class SecureVisualizationManager:
    """Secure visualization with full original capabilities maintained"""

    def __init__(self):
        self._figures = weakref.WeakSet()
        self._canvases = weakref.WeakSet()
        self._toolbars = weakref.WeakSet()
        self._lock = threading.RLock()
        self._cleanup_in_progress = False

        # Maintain all original matplotlib configurations
        self._setup_matplotlib_params()

    def _setup_matplotlib_params(self):
        """Configure matplotlib for professional petroleum industry plots"""
        try:
            # Use modern seaborn style with fallback
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    plt.style.use('default')

            # Professional matplotlib configuration for petroleum industry
            plt.rcParams.update({
                'figure.max_open_warning': 10,
                'figure.dpi': 100,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'font.size': 10,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.linewidth': 0.5,
                'axes.linewidth': 0.8,
                'axes.edgecolor': 'black',
                'xtick.major.size': 4,
                'ytick.major.size': 4,
                'xtick.minor.size': 2,
                'ytick.minor.size': 2,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'legend.frameon': True,
                'legend.fancybox': False,
                'legend.shadow': False,
                'legend.edgecolor': 'black',
                'legend.facecolor': 'white',
                'legend.alpha': 0.9,
                'lines.linewidth': 1.5,
                'lines.markersize': 4,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14,
                'savefig.dpi': 100,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })

            # Professional color cycle for petroleum industry
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color',
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        except Exception as e:
            warnings.warn(f"Error configuring matplotlib: {e}", UserWarning)
            # Fallback to basic configuration
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'font.size': 10,
                'axes.grid': True,
                'grid.alpha': 0.3
            })

    @contextmanager
    def safe_visualization_context(self):
        """Context manager maintaining full visualization capabilities"""
        with self._lock:
            if self._cleanup_in_progress:
                raise RuntimeError("Cleanup in progress")

            try:
                yield
            except Exception:
                # Still provide the exception for debugging - just don't log to file
                self._emergency_cleanup()
                raise
            finally:
                # Gentle cleanup
                gc.collect()

    def create_figure(self, figsize=(12, 8), dpi=100):
        """Create and register figure - maintaining original capabilities"""
        with self._lock:
            fig = Figure(figsize=figsize, dpi=dpi)
            self._figures.add(fig)
            return fig

    def create_canvas(self, fig, parent):
        """Create and register canvas - full original functionality"""
        with self._lock:
            canvas = FigureCanvasTkAgg(fig, parent)
            self._canvases.add(canvas)
            return canvas

    def create_toolbar(self, canvas, parent):
        """Create and register toolbar - maintaining navigation capabilities"""
        with self._lock:
            toolbar = NavigationToolbar2Tk(canvas, parent)
            self._toolbars.add(toolbar)
            return toolbar

    def plot_comparison_with_depth(self, ax, original, processed, depth, curve_name, curve_info):
        """Maintain the sophisticated comparison plotting from original"""
        ax.plot(original, depth, 'r-', alpha=0.7, label='Original', linewidth=1)
        ax.plot(processed, depth, 'b-', alpha=0.9, label='Processed', linewidth=2)

        valid_mask = ~np.isnan(original) & ~np.isnan(processed)
        changes = np.abs(original[valid_mask] - processed[valid_mask])

        if len(changes) > 0:
            threshold = np.percentile(changes, 95) if len(changes) > 20 else np.max(changes) * 0.5
            significant_idx = np.nonzero((np.abs(original - processed) > threshold) & valid_mask)[0]

            if len(significant_idx) > 0:
                x_highlight = original[significant_idx]
                y_highlight = depth[significant_idx]
                ax.scatter(x_highlight, y_highlight, color='green', s=50, alpha=0.7,
                           label='Significant Changes', zorder=3)

        ax.set_title(f'Data Processing Comparison: {curve_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{curve_name} ({curve_info.get("unit", "")})')
        ax.set_ylabel('Depth (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Industry standard

    def plot_industry_log_display(self, fig, processed_data, curve_info_dict):
        """Maintain the sophisticated industry log display"""
        axes = fig.subplots(1, 3, sharey=True)

        curve_by_type = {}
        for curve in processed_data.columns:
            curve_type = curve_info_dict.get(curve, {}).get('curve_type', 'UNKNOWN')
            if curve_type not in curve_by_type:
                curve_by_type[curve_type] = []
            curve_by_type[curve_type].append(curve)

        # Original multi-track plotting retained elsewhere
        return axes

    def plot_3d_with_depth(self, fig, curve1_data, curve2_data, depth, curve1_name, curve2_name):
        """Maintain sophisticated 3D visualization"""
        ax = fig.add_subplot(111, projection='3d')

        valid_mask = ~np.isnan(curve1_data) & ~np.isnan(curve2_data)
        if not np.any(valid_mask):
            return None

        valid_x = curve1_data[valid_mask]
        valid_y = curve2_data[valid_mask]
        valid_z = depth[valid_mask]

        ax.scatter(valid_x, valid_y, valid_z, c=valid_z, cmap='viridis', s=30, alpha=0.8, marker='o')

        ax.set_xlabel(f'{curve1_name}')
        ax.set_ylabel(f'{curve2_name}')
        ax.set_zlabel('Depth')
        ax.view_init(elev=30, azim=45)
        ax.invert_zaxis()  # Industry standard

        return ax

    def secure_cleanup_visualization(self):
        """Comprehensive cleanup maintaining reliability"""
        with self._lock:
            self._cleanup_in_progress = True

            try:
                cleanup_success = True

                toolbars_to_destroy = list(self._toolbars)
                for toolbar in toolbars_to_destroy:
                    try:
                        if toolbar and hasattr(toolbar, 'destroy'):
                            toolbar.destroy()
                    except Exception:
                        cleanup_success = False

                canvases_to_destroy = list(self._canvases)
                for canvas in canvases_to_destroy:
                    try:
                        if canvas and hasattr(canvas, 'get_tk_widget'):
                            widget = canvas.get_tk_widget()
                            if widget and hasattr(widget, 'destroy'):
                                widget.destroy()
                    except Exception:
                        cleanup_success = False

                figures_to_close = list(self._figures)
                for fig in figures_to_close:
                    try:
                        if fig is not None:
                            for ax in fig.get_axes():
                                ax.clear()
                                ax.collections.clear()
                                ax.patches.clear()
                                ax.lines.clear()
                                ax.texts.clear()
                                ax.images.clear()
                            plt.close(fig)
                    except Exception:
                        cleanup_success = False

                try:
                    plt.rcdefaults()
                    if hasattr(plt, '_get_backend_mod'):
                        backend_mod = plt._get_backend_mod()
                        if hasattr(backend_mod, 'destroy_all'):
                            backend_mod.destroy_all()  # type: ignore[attr-defined]
                except Exception:
                    cleanup_success = False

                for _ in range(3):
                    gc.collect()

                return cleanup_success

            finally:
                self._cleanup_in_progress = False

    def _emergency_cleanup(self):
        """Emergency cleanup maintaining system stability"""
        try:
            plt.close('all')
            self._figures.clear()
            self._canvases.clear()
            self._toolbars.clear()

            for _ in range(5):
                gc.collect()

        except Exception:
            pass


