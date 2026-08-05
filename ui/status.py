import threading
import tkinter as tk
from datetime import datetime
from typing import Optional, Dict


class SecureStatusManager:
    """Maintain excellent user feedback without security risks"""

    def __init__(self, results_text_widget, status_label_widget, progress_bar_widget):
        self.results_text = results_text_widget
        self.status_label = status_label_widget
        self.progress_bar = progress_bar_widget
        # Resolved once, here, because construction happens on the main thread.
        # winfo_toplevel() is itself a Tk call, so looking the root up lazily
        # from a worker would commit the very violation this class avoids.
        self._root = self._resolve_root()

    def _resolve_root(self):
        """Find the Tk root so work can be scheduled onto the main thread.

        The manager is constructed from widgets rather than from the root, so
        the root is reached through whichever widget is still alive. Only call
        this from the main thread.
        """
        for widget in (self.results_text, self.status_label, self.progress_bar):
            if widget is None:
                continue
            try:
                return widget.winfo_toplevel()
            except (tk.TclError, RuntimeError, AttributeError):
                continue
        return None

    def update_status(self, message: str, progress: Optional[float] = None):
        """Maintain original status update quality.

        Tk is not thread safe, and background workers call this on every log
        line. Widget writes are therefore scheduled onto the main thread rather
        than performed in place. The previous implementation held a lock across
        those writes and called update_idletasks(), which drove the Tcl
        interpreter from the calling thread; a worker inside that call while the
        main thread waited on the same lock produced a hard deadlock.
        """
        # The timestamp records when the event happened, not when the main
        # thread gets round to drawing it, so it is taken on the calling thread.
        display_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"

        if threading.current_thread() is threading.main_thread():
            self._apply_status(display_message, message, progress)
            return

        if self._root is None:
            return
        try:
            self._root.after(0, self._apply_status, display_message, message, progress)
        except (tk.TclError, RuntimeError):
            # The interpreter is shutting down or the widget tree is gone.
            # Dropping a status line beats raising inside a worker thread.
            pass

    def _apply_status(self, display_message: str, message: str,
                      progress: Optional[float] = None):
        """Perform the widget writes. Must only run on the main thread.

        No lock is taken: every caller now lands here via the event loop, which
        is what serializes access. Taking a lock here would reintroduce the
        cross-thread wait that caused the deadlock.
        """
        try:
            if self.results_text:
                self.results_text.insert(tk.END, display_message + "\n")
                self.results_text.see(tk.END)

            if self.status_label:
                self.status_label.config(text=message)

            if progress is not None and self.progress_bar:
                self.progress_bar['value'] = progress
        except tk.TclError:
            # Widget destroyed between scheduling and execution.
            pass

    def log_processing_step(self, curve_name: str, step: str, details: Optional[Dict] = None):
        """Maintain detailed processing feedback like original"""
        message = f"Processing {curve_name}: {step}"

        if details:
            if 'gaps_filled' in details:
                message += f" - Gaps filled: {details['gaps_filled']}"
            if 'quality' in details:
                message += f" - Quality: {details['quality']:.2f}"
            if 'method' in details:
                message += f" - Method: {details['method']}"

        self.update_status(message)

    def log_gap_filling_results(self, curve_name: str, gap_result: dict):
        """Maintain original gap filling feedback detail"""
        quality_metrics = gap_result.get('quality_metrics', {})

        self.update_status(f"Gap filling completed for {curve_name}")
        self.update_status(f"  - Gaps filled: {quality_metrics.get('total_gaps_filled', 0)}")
        self.update_status(f"  - Points filled: {quality_metrics.get('total_points_filled', 0)}")
        self.update_status(f"  - Methods used: {', '.join(quality_metrics.get('methods_used', []))}")
        self.update_status(f"  - Average confidence: {quality_metrics.get('average_confidence', 0):.3f}")
        self.update_status(f"  - Final completeness: {quality_metrics.get('data_completeness', 0):.1f}%")

    def log_denoising_results(self, curve_name: str, denoise_result: dict):
        """Maintain original denoising feedback detail"""
        self.update_status(f"Denoising completed for {curve_name}")
        self.update_status(f"  - Method: {denoise_result.get('method', 'unknown')}")
        self.update_status(f"  - Quality score: {denoise_result.get('quality', 0):.3f}")

        if 'noise_reduction_db' in denoise_result:
            self.update_status(f"  - Noise reduction: {denoise_result['noise_reduction_db']:.1f} dB")

        if denoise_result.get('method') == 'wavelet':
            self.update_status(f"  - Wavelet used: {denoise_result.get('wavelet_used', 'unknown')}")
            self.update_status(f"  - Decomposition levels: {denoise_result.get('levels', 0)}")

    def create_comprehensive_report(self, processing_results: dict, curve_info: dict) -> str:
        """Maintain the original comprehensive reporting quality"""
        report = []

        report.append("=" * 80)
        report.append("ADVANCED WIRELINE DATA PREPROCESSING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(report)


