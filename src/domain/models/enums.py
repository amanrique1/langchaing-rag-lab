from enum import Enum


class DocumentLoaderMode(str, Enum):
    SINGLE = "single"
    ELEMENTS = "elements"


class LengthBasedChunkingMode(str, Enum):
    CHARACTER = "character"
    TOKEN = "token"


class SemanticChunkingThresholdType(str, Enum):
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    ABSOLUTE = "absolute"
