"""Expected ingestion failures suitable for an eventual API error boundary."""


class ExtractionError(Exception):
    """Base class for a user-actionable extraction failure."""


class UnsupportedFormatError(ExtractionError):
    """The supplied upload is not one of the supported input formats."""


class ExtractionQualityError(ExtractionError):
    """The upload parsed, but did not yield enough usable source content."""
