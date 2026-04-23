"""Pattern-based PII parser compatible with the OpenAI Privacy Filter schema.

The OpenAI privacy-filter ships a 1.5B-param bidirectional token classifier that
labels and masks PII. This package is the reverse: a lightweight, dependency-free
parser that *extracts* PII with regex + heuristics and emits the same span shape
used by opf (label, start, end, text). No model weights required.

Category taxonomy (v2):
    account_number, private_address, private_date, private_email,
    private_person, private_phone, private_url, secret
"""

from .api import PIIParser, PIIParseResult, DetectedSpan, parse

__all__ = ["PIIParser", "PIIParseResult", "DetectedSpan", "parse"]
__version__ = "0.1.0"
