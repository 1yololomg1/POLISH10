"""Run one well through Stages 0-4 with its saved answers. D98.

Structure
---------
run_well
    Stages 0-4 with mappings, then pass choices, then unit choices.
WellRun
    The stage results, the question set, and which answers were used.
"""

from polish.pipeline.run import WellRun, run_well

__all__ = ["WellRun", "run_well"]
