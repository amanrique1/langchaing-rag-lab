"""
Utility functions for RAG evaluation
"""
import time
import logging
from typing import Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class RetryHandler:
    """Handle retries for failed operations"""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    def execute(self, handler: Any, method_name: str, **kwargs) -> Any:
        """
        Executes a method on a handler object with retry logic.

        Args:
            handler (Any): The object instance on which to call the method.
            method_name (str): The name of the method to execute (e.g., "search", "save").
            **kwargs: Keyword arguments to be passed to the target method.

        Returns:
            Any: The return value of the executed method.

        Raises:
            AttributeError: If the handler does not have the specified method.
            TypeError: If the specified attribute is not a callable method.
            Exception: Re-raises the last exception if all retry attempts fail.
        """
        # --- Safety Check 1: Ensure the method exists on the handler ---
        if not hasattr(handler, method_name):
            raise AttributeError(
                f"Object of type '{type(handler).__name__}' has no method named '{method_name}'."
            )

        method_to_call: Callable = getattr(handler, method_name)

        # --- Safety Check 2: Ensure the attribute is actually a callable function ---
        if not callable(method_to_call):
            raise TypeError(
                f"Attribute '{method_name}' on object '{type(handler).__name__}' is not callable."
            )

        for attempt in range(self.max_retries):
            try:
                # Dynamically call the method with its arguments
                return method_to_call(**kwargs)
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} of calling '{method_name}' failed: {e}. Retrying in {self.delay * (attempt + 1):.1f}s..."
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts to call '{method_name}' failed. Raising exception.")
                    raise  # Re-raise the last exception
                time.sleep(self.delay * (attempt + 1))

class ProgressTracker:
    """Track progress of evaluation"""

    def __init__(self, total: int):
        self.total = total
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()

    def update(self, success: bool = True):
        """Update progress"""
        if success:
            self.successful += 1
        else:
            self.failed += 1

    def display_summary(self):
        """Display summary"""
        elapsed = time.time() - self.start_time
        print(f"\n{'─'*60}")
        print(f"Progress Summary:")
        print(f"  Total:      {self.total}")
        print(f"  Successful: {self.successful}")
        print(f"  Failed:     {self.failed}")
        print(f"  Time:       {elapsed:.2f}s")
        print(f"{'─'*60}")

def timing_decorator(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper