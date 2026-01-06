from pathlib import Path

class GuardrailConfig:
    """
    Central configuration for Security Grounding Layers.
    """

    # -------------------------------------------------------------------------
    # INPUT LAYER CONFIGURATION
    # -------------------------------------------------------------------------

    # Fast Rules: Regex patterns to block immediately (Zero Latency)
    # These catch standard jailbreak attempts.
    JAILBREAK_PATTERNS = [
        r"ignore (all )?instructions",
        r"system override",
        r"act as (a|an)",
        r"debug mode",
        r"developer mode",
        r"unrestricted",
        r"DAN mode"
    ]

    TEMPLATE_PATH = Path("assets/templates/query_template.txt")

    # -------------------------------------------------------------------------
    # OUTPUT LAYER CONFIGURATION
    # -------------------------------------------------------------------------

    # Redaction Rules: Regex for PII to scrub from the output stream.
    # Format: {"LABEL": r"REGEX_PATTERN"}
    SENSITIVE_PATTERNS = {
        "EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+',
        "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        # Matches Credit Cards (groups of 4 or 16 digits)
        "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        # Matches IPv4 addresses
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }