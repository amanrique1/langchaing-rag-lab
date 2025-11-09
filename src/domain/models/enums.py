from enum import Enum

class StorageType(str, Enum):
    LOCAL = "local"
    CHROMA = "chroma"

class LengthBasedChunkingMode(str, Enum):
    CHARACTER = "character"
    TOKEN = "token"


class SemanticChunkingThresholdType(str, Enum):
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    ABSOLUTE = "absolute"
