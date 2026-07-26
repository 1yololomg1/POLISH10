"""
Centralized Error Handling System

Professional, thread-safe error handling optimized for UI/UX.
Provides consistent error messaging, categorization, and user-friendly feedback.
"""

import threading
import traceback
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    messagebox = None


class ErrorSeverity(Enum):
    """Error severity levels for appropriate UI treatment"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better user guidance"""
    MEMORY_ERROR = "memory"
    DEPENDENCY_ERROR = "dependency"
    DATA_ERROR = "data"
    FILE_ERROR = "file"
    PERMISSION_ERROR = "permission"
    TIMEOUT_ERROR = "timeout"
    NETWORK_ERROR = "network"
    PROCESSING_ERROR = "processing"
    VALIDATION_ERROR = "validation"
    GENERAL_ERROR = "general"


@dataclass
class ErrorContext:
    """Context information for error reporting"""
    operation: str
    component: Optional[str] = None
    user_action: Optional[str] = None
    remediation_hint: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class CentralizedErrorHandler:
    """
    Professional centralized error handling system.
    
    Features:
    - Thread-safe UI updates
    - Categorized error messages with user-friendly guidance
    - Automatic logging integration
    - Non-blocking error display where appropriate
    - Consistent error formatting
    - Context-aware remediation suggestions
    """
    
    def __init__(self, root=None, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize error handler.
        
        Args:
            root: Tkinter root window (for thread-safe UI updates)
            log_callback: Optional callback for logging errors
        """
        self.root = root
        self.log_callback = log_callback
        self._main_thread = threading.main_thread() if threading.main_thread() else threading.current_thread()
        
        # Error message templates for better UX
        self._error_templates = {
            ErrorCategory.MEMORY_ERROR: {
                "title": "Memory Error",
                "message_template": "The operation '{operation}' requires more memory than available.\n\n"
                                   "Suggestions:\n"
                                   "ΓÇó Close other applications\n"
                                   "ΓÇó Process smaller datasets\n"
                                   "ΓÇó Reduce processing options"
            },
            ErrorCategory.DEPENDENCY_ERROR: {
                "title": "Missing Dependency",
                "message_template": "Required library not available for '{operation}'.\n\n"
                                   "Please install: {dependency_info}"
            },
            ErrorCategory.DATA_ERROR: {
                "title": "Data Error",
                "message_template": "Invalid or corrupted data detected during '{operation}'.\n\n"
                                   "{remediation_hint}"
            },
            ErrorCategory.FILE_ERROR: {
                "title": "File Error",
                "message_template": "Unable to access file during '{operation}'.\n\n"
                                   "ΓÇó Check file path and permissions\n"
                                   "ΓÇó Ensure file is not open in another program\n"
                                   "{remediation_hint}"
            },
            ErrorCategory.PERMISSION_ERROR: {
                "title": "Permission Denied",
                "message_template": "Insufficient permissions for '{operation}'.\n\n"
                                   "ΓÇó Check file/folder permissions\n"
                                   "ΓÇó Run with appropriate privileges if needed"
            },
            ErrorCategory.PROCESSING_ERROR: {
                "title": "Processing Error",
                "message_template": "An error occurred during '{operation}'.\n\n"
                                   "{error_details}\n\n"
                                   "{remediation_hint}"
            },
            ErrorCategory.VALIDATION_ERROR: {
                "title": "Validation Error",
                "message_template": "Data validation failed for '{operation}'.\n\n"
                                   "{error_details}\n\n"
                                   "Please check your input data and try again."
            },
            ErrorCategory.GENERAL_ERROR: {
                "title": "Error",
                "message_template": "An unexpected error occurred during '{operation}'.\n\n"
                                   "{error_details}"
            }
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    show_dialog: bool = True,
                    log_error: bool = True) -> None:
        """
        Central error handling entry point.
        
        Args:
            error: The exception that occurred
            context: Error context information
            severity: Error severity level
            show_dialog: Whether to show UI dialog (default: True)
            log_error: Whether to log the error (default: True)
        """
        # Categorize error
        category = self._categorize_error(error, context.operation)
        
        # Format user-friendly message
        message = self._format_error_message(error, context, category)
        
        # Log error
        if log_error and self.log_callback:
            self._log_error(error, context, category, severity)
        
        # Show UI (thread-safe)
        if show_dialog:
            self._show_ui_message(severity, category, message, context)
    
    def handle_warning(self,
                      message: str,
                      context: ErrorContext,
                      show_dialog: bool = True) -> None:
        """
        Handle warnings with appropriate UI treatment.
        
        Args:
            message: Warning message
            context: Warning context
            show_dialog: Whether to show UI dialog
        """
        if self.log_callback:
            self.log_callback(f"[WARNING] {context.operation}: {message}")
        
        if show_dialog:
            self._show_ui_message(ErrorSeverity.WARNING, ErrorCategory.GENERAL_ERROR, 
                                message, context)
    
    def handle_info(self,
                   message: str,
                   context: ErrorContext,
                   show_dialog: bool = False) -> None:
        """
        Handle informational messages (typically non-blocking).
        
        Args:
            message: Info message
            context: Info context
            show_dialog: Whether to show UI dialog (default: False for info)
        """
        if self.log_callback:
            self.log_callback(f"[INFO] {context.operation}: {message}")
        
        if show_dialog:
            self._show_ui_message(ErrorSeverity.INFO, ErrorCategory.GENERAL_ERROR,
                                message, context)
    
    def _categorize_error(self, error: Exception, operation: str) -> ErrorCategory:
        """
        Categorize error for appropriate handling and user guidance.
        
        Args:
            error: The exception
            operation: Operation that failed
            
        Returns:
            Error category
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Memory errors
        if "memory" in error_str or "MemoryError" in error_type:
            return ErrorCategory.MEMORY_ERROR
        
        # Dependency errors
        if ("import" in error_str or 
            "ModuleNotFoundError" in error_type or 
            "ImportError" in error_type):
            return ErrorCategory.DEPENDENCY_ERROR
        
        # Data errors
        if ("ValueError" in error_type or 
            "IndexError" in error_type or 
            "KeyError" in error_type or
            "shape" in error_str or
            "dimension" in error_str):
            return ErrorCategory.DATA_ERROR
        
        # File errors
        if ("file" in error_str or 
            "FileNotFoundError" in error_type or 
            "OSError" in error_type):
            return ErrorCategory.FILE_ERROR
        
        # Permission errors
        if ("permission" in error_str or 
            "PermissionError" in error_type):
            return ErrorCategory.PERMISSION_ERROR
        
        # Timeout errors
        if ("timeout" in error_str or 
            "TimeoutError" in error_type):
            return ErrorCategory.TIMEOUT_ERROR
        
        # Network errors
        if ("network" in error_str or 
            "ConnectionError" in error_type):
            return ErrorCategory.NETWORK_ERROR
        
        # Validation errors (specific to data validation)
        if ("validation" in error_str or
            "invalid" in error_str and "data" in error_str):
            return ErrorCategory.VALIDATION_ERROR
        
        # Processing errors (catch-all for processing operations)
        if any(keyword in operation.lower() for keyword in 
               ["process", "fill", "denoise", "correct", "calculate"]):
            return ErrorCategory.PROCESSING_ERROR
        
        return ErrorCategory.GENERAL_ERROR
    
    def _format_error_message(self, 
                             error: Exception, 
                             context: ErrorContext,
                             category: ErrorCategory) -> str:
        """
        Format user-friendly error message with context and remediation.
        
        Args:
            error: The exception
            context: Error context
            category: Error category
            
        Returns:
            Formatted error message
        """
        template = self._error_templates.get(category, self._error_templates[ErrorCategory.GENERAL_ERROR])
        message_template = template["message_template"]
        
        # Extract error details (user-friendly, not technical traceback)
        error_details = self._extract_user_friendly_error(error)
        
        # Build message with context
        message = message_template.format(
            operation=context.operation,
            error_details=error_details,
            remediation_hint=context.remediation_hint or "Please check your input and try again.",
            dependency_info=context.additional_info.get("dependency", "required library") if context.additional_info else "required library"
        )
        
        # Add component info if available
        if context.component:
            message = f"[{context.component}] {message}"
        
        return message
    
    def _extract_user_friendly_error(self, error: Exception) -> str:
        """
        Extract user-friendly error message from exception.
        
        Args:
            error: The exception
            
        Returns:
            User-friendly error message
        """
        error_msg = str(error)
        
        # Clean up technical details for better UX
        if "Traceback" in error_msg:
            # Extract just the last line if it's a traceback
            lines = error_msg.split('\n')
            for line in reversed(lines):
                if line.strip() and not line.startswith('  File'):
                    return line.strip()
        
        # Limit message length for UI
        if len(error_msg) > 200:
            return error_msg[:197] + "..."
        
        return error_msg
    
    def _log_error(self, 
                  error: Exception,
                  context: ErrorContext,
                  category: ErrorCategory,
                  severity: ErrorSeverity) -> None:
        """
        Log error with full context for debugging.
        
        Args:
            error: The exception
            context: Error context
            category: Error category
            severity: Error severity
        """
        if not self.log_callback:
            return
        
        # Full error details for logging
        log_message = f"[{severity.value.upper()}] {category.value.upper()}: {context.operation}"
        
        if context.component:
            log_message += f" [{context.component}]"
        
        log_message += f"\n  Error: {type(error).__name__}: {str(error)}"
        
        if context.user_action:
            log_message += f"\n  User Action: {context.user_action}"
        
        if context.additional_info:
            log_message += f"\n  Context: {context.additional_info}"
        
        # Include traceback for critical errors
        if severity == ErrorSeverity.CRITICAL:
            log_message += f"\n  Traceback:\n{''.join(traceback.format_tb(error.__traceback__))}"
        
        self.log_callback(log_message)
    
    def _show_ui_message(self,
                        severity: ErrorSeverity,
                        category: ErrorCategory,
                        message: str,
                        context: ErrorContext) -> None:
        """
        Thread-safe UI message display.
        
        Args:
            severity: Message severity
            category: Error category
            message: Message to display
            context: Error context
        """
        if not TKINTER_AVAILABLE or not messagebox:
            # Fallback to console
            print(f"[{severity.value.upper()}] {message}")
            return
        
        def _display():
            """Display message in UI (must run on main thread)"""
            try:
                title = self._error_templates.get(category, {}).get("title", "Error")
                
                if messagebox is None:
                    # Fallback if messagebox not available
                    print(f"[{severity.value.upper()}] {title}: {message}")
                    return
                
                if severity == ErrorSeverity.CRITICAL:
                    messagebox.showerror(title, message)
                elif severity == ErrorSeverity.ERROR:
                    messagebox.showerror(title, message)
                elif severity == ErrorSeverity.WARNING:
                    messagebox.showwarning(title, message)
                elif severity == ErrorSeverity.INFO:
                    messagebox.showinfo(title, message)
            except Exception as e:
                # Last resort: print to console
                print(f"[{severity.value.upper()}] {title}: {message}")
                print(f"UI display failed: {e}")
        
        # Thread-safe execution
        if threading.current_thread() is self._main_thread:
            _display()
        else:
            # Schedule on main thread
            if self.root:
                try:
                    self.root.after(0, _display)
                except Exception:
                    # Fallback if root is destroyed
                    _display()
            else:
                # No root available, use fallback
                _display()
    
    def create_context(self,
                      operation: str,
                      component: Optional[str] = None,
                      user_action: Optional[str] = None,
                      remediation_hint: Optional[str] = None,
                      **additional_info) -> ErrorContext:
        """
        Helper to create ErrorContext with common patterns.
        
        Args:
            operation: Operation name
            component: Component/module name
            user_action: User action that triggered error
            remediation_hint: Hint for fixing the issue
            **additional_info: Additional context information
            
        Returns:
            ErrorContext instance
        """
        return ErrorContext(
            operation=operation,
            component=component,
            user_action=user_action,
            remediation_hint=remediation_hint,
            additional_info=additional_info if additional_info else None
        )


# Convenience function for quick error handling
def handle_error_quick(error: Exception,
                      operation: str,
                      handler: Optional[CentralizedErrorHandler] = None,
                      **kwargs) -> None:
    """
    Quick error handling for simple cases.
    
    Args:
        error: The exception
        operation: Operation name
        handler: Error handler instance (if None, uses fallback)
        **kwargs: Additional context
    """
    if handler:
        context = handler.create_context(operation, **kwargs)
        handler.handle_error(error, context)
    else:
        # Fallback for when handler not available
        print(f"[ERROR] {operation}: {error}")

