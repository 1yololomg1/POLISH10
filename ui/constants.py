"""
Shared UI string and event constants for tab/shell modules.

STRUCTURE:
- Font, dialog, and Tk event literals used by AppUIMixin and tab mixins

Kept out of advanced_preprocessing_system10.py so ui/ can import without cycles.
"""

FONT_DEFAULT = "Arial"
FONT_SIZE_DEFAULT = 9
FONT_SIZE_HEADING = 11
FONT_SIZE_LARGE = 12

EVENT_CONFIGURE = "<Configure>"
EVENT_MOUSEWHEEL = "<MouseWheel>"
EVENT_SHIFT_MOUSEWHEEL = "<Shift-MouseWheel>"

DIALOG_SELECT_ALL = "Select All"
DIALOG_DESELECT_ALL = "Deselect All"
DIALOG_ROOT_WINDOW = "root"
