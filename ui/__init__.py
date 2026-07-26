"""
UI Module - Visualization, status, and App shell/tab construction

Note: Heavy imports (matplotlib visualization) are lazy so environments with
broken mpl DLL paths can still import ui.app / tab mixins for tests and tooling.
"""

__all__ = ['SecureVisualizationManager', 'SecureStatusManager', 'AppUIMixin']


def __getattr__(name: str):
    if name == 'SecureVisualizationManager':
        from ui.visualization import SecureVisualizationManager
        return SecureVisualizationManager
    if name == 'SecureStatusManager':
        from ui.status import SecureStatusManager
        return SecureStatusManager
    if name == 'AppUIMixin':
        from ui.app import AppUIMixin
        return AppUIMixin
    raise AttributeError(f"module 'ui' has no attribute {name!r}")

