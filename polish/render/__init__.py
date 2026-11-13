"""Headless display model. No UI toolkit is imported here. D73, D74, D78.

Structure
---------
bin_curve, PixelBins
    Per-pixel bins of one curve over a depth window.
hit_test
    The nearest actual sample to a depth.
annotate, SampleAnnotation
    What a hover readout states about one sample.
"""

from polish.render.bins import PixelBins, SampleAnnotation, annotate, bin_curve, hit_test

__all__ = ["PixelBins", "SampleAnnotation", "annotate", "bin_curve", "hit_test"]
