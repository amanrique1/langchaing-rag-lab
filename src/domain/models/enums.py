from enum import Enum

class StorageType(str, Enum):
    FILESYSTEM = "filesystem"
    CHROMA = "chroma"
    LANCE = "lance"

class LengthBasedChunkingMode(str, Enum):
    CHARACTER = "character"
    TOKEN = "token"


class SemanticChunkingThresholdType(str, Enum):
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    ABSOLUTE = "absolute"

class QueryExpansionStrategy(str, Enum):
    """Strategy for expanding queries to improve retrieval."""
    HYDE = "hyde"
    STEPBACK = "stepback"
    SUBQUERIES = "subqueries"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"