class SecurityViolationError(Exception):
    """Raised when the input fails guardrail validation."""
    def __init__(self, message: str, violation_source: str, violation_code: str = None):
        """
        Args:
            message: Human readable error.
            violation_source: 'REGEX' or 'LLAMA_GUARD'.
            violation_code: Specific code (e.g., 'S1', 'S2') if available.
        """
        self.violation_source = violation_source
        self.violation_code = violation_code
        super().__init__(message)