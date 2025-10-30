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
        self._lock = threading.Lock()

    def update_status(self, message: str, progress: Optional[float] = None):
        """Maintain original status update quality"""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            display_message = f"[{timestamp}] {message}"

            if hasattr(self, 'results_text') and self.results_text:
                self.results_text.insert(tk.END, display_message + "\n")
                self.results_text.see(tk.END)
                self.results_text.update_idletasks()

            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.config(text=message)

            if progress is not None and hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar['value'] = progress

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


