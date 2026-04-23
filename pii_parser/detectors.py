"""Detectors for each PII category.

Each detector takes raw text and yields candidate ``DetectedSpan`` objects.
Detectors are intentionally independent; overlap resolution is a later step.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from .spans import DetectedSpan


# --- email --------------------------------------------------------------------
# Local-part per RFC 5322 practical subset; TLD >= 2 letters; no trailing dot.
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+\-]*[A-Za-z0-9])?"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*"
    r"\.[A-Za-z]{2,63}"
)


def detect_emails(text: str) -> Iterator[DetectedSpan]:
    for m in _EMAIL_RE.finditer(text):
        yield DetectedSpan("private_email", m.start(), m.end(), m.group(0))


# --- url ----------------------------------------------------------------------
# Schemed URLs only — bare domains are too noisy and the opf ground truth
# also uses schemed URLs for private_url.
_URL_RE = re.compile(
    r"\bhttps?://"
    r"[A-Za-z0-9.\-]+(?:\.[A-Za-z]{2,63})"
    r"(?::\d{2,5})?"
    r"(?:/[^\s<>\"']*)?"
)


def detect_urls(text: str) -> Iterator[DetectedSpan]:
    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        # strip trailing punctuation that regex greedily grabbed.
        stripped = raw.rstrip(".,;:!?)]}\"'")
        end = m.start() + len(stripped)
        yield DetectedSpan("private_url", m.start(), end, stripped)


# --- phone --------------------------------------------------------------------
# Matches: (415) 555-0102, +1 999 555-1234, 415-555-0102, +44 20 7946 0958 etc.
# Require >= 10 digits total, allow 7-15 digits with separators.
_PHONE_RE = re.compile(
    r"(?<![\w@/])"
    r"(?:\+?\d{1,3}[\s\-.]?)?"            # country code
    r"(?:\(\d{1,4}\)[\s\-.]?|\d{1,4}[\s\-.])"  # area code
    r"\d{1,4}[\s\-.]?\d{2,4}"
    r"(?:[\s\-.]?\d{2,4})?"
    r"(?!\w)"
)
_DIGIT_RE = re.compile(r"\d")


_ISO_DATE_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHORT_DATE_SHAPE = re.compile(r"^\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?$")


def detect_phones(text: str) -> Iterator[DetectedSpan]:
    for m in _PHONE_RE.finditer(text):
        raw = m.group(0)
        digits = _DIGIT_RE.findall(raw)
        if not 7 <= len(digits) <= 15:
            continue
        # Require at least one phone separator (space, dash, dot, paren, +).
        if not re.search(r"[\s().+]", raw):
            continue
        clean = raw.strip(" \t\n\r.,;:)")
        # Skip strings that look like ISO or slash dates.
        if _ISO_DATE_SHAPE.match(clean) or _SHORT_DATE_SHAPE.match(clean):
            continue
        ls = raw.lstrip(" \t\n\r")
        start = m.start() + (len(raw) - len(ls))
        clean = ls.rstrip(" \t\n\r.,;:)")
        end = start + len(clean)
        yield DetectedSpan("private_phone", start, end, clean)


# --- date ---------------------------------------------------------------------
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_WEEKDAYS = (
    r"Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:r(?:s(?:day)?)?)?|"
    r"Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?"
)

# ISO: 2026-05-17; US: 5/23 or 12/31/2025; prose: May 17, 2026 / 17 May 2026;
# Weekday + short date: "Friday, 5/23"
_DATE_PATTERNS = [
    # ISO 8601 date (strict: 4-digit year, 2-digit month/day)
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # M/D or M/D/YYYY or M-D-YYYY
    re.compile(r"(?<!\d)\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?(?!\d)"),
    # Month DD, YYYY
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:,\s*\d{{2,4}})?\b"),
    # DD Month YYYY
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{2,4}})?\b"),
]
_WEEKDAY_RE = re.compile(rf"\b(?:{_WEEKDAYS})\b", re.IGNORECASE)


def detect_dates(text: str) -> Iterator[DetectedSpan]:
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            # Filter M/D without year that is probably a fraction: require
            # month 1-12 and day 1-31.
            if "/" in raw or (raw.count("-") and not raw[:4].isdigit()):
                parts = re.split(r"[/\-]", raw)
                try:
                    mo, dy = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if not (1 <= mo <= 12 and 1 <= dy <= 31):
                    continue
            yield DetectedSpan("private_date", m.start(), m.end(), raw)


# --- account number -----------------------------------------------------------
# Long digit runs (>= 10 digits), not part of a phone or date.
_ACCOUNT_RE = re.compile(r"(?<![\d\-])\d{10,24}(?![\d\-])")


def detect_accounts(text: str) -> Iterator[DetectedSpan]:
    for m in _ACCOUNT_RE.finditer(text):
        yield DetectedSpan("account_number", m.start(), m.end(), m.group(0))


# --- secret -------------------------------------------------------------------
# Token-shaped strings: sk-..., AKIA..., ghp_..., plus high-entropy passphrases
# of the form Word-Word-Word with mixed case/digits.
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{4,}"),
    re.compile(r"\bpk-[A-Za-z0-9_\-]{4,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b"),
    # Word-Word-Word style passphrase with at least one digit somewhere.
    re.compile(
        r"\b(?=[A-Za-z0-9\-]*\d)(?=[A-Za-z0-9\-]*[A-Za-z])"
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,}\b"
    ),
]


def detect_secrets(text: str) -> Iterator[DetectedSpan]:
    for pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            yield DetectedSpan("secret", m.start(), m.end(), m.group(0))


# --- person name --------------------------------------------------------------
# Heuristic only — capitalised multi-token names, optionally with particles.
# Stop-words prevent picking up sentence-initial pronouns, headings, and
# country/city words that also capitalise.
_NAME_STOPWORDS = {
    "I", "A", "An", "The", "This", "That", "These", "Those",
    "Mr", "Mrs", "Ms", "Dr", "Mx", "Prof", "Sir", "Madam",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "USA", "UK", "EU", "US",
    # Common sentence-initial verbs/nouns that capitalise and are not names.
    "Contact", "Call", "Email", "Reset", "Rotate", "Support", "Reach",
    "Send", "Visit", "See", "From", "To", "Dear", "Hello", "Hi",
    "Please", "Thanks", "Thank", "Regards", "Sincerely",
    "Reply", "Login", "Sign", "Ping", "Notify", "Tell", "Ask",
    "Write", "Forward", "Follow", "Meet", "Greet", "Remind",
    "Register", "Subscribe", "Unsubscribe", "Update", "Confirm",
    "Hey", "Hola", "Re", "Fwd",
}
_NAME_TOKEN = r"(?:[A-Z][a-z]+(?:'[a-z]+)?|[A-Z]\.)"
_NAME_RE = re.compile(
    rf"(?<![A-Za-z])"
    rf"(?:Mr|Mrs|Ms|Dr|Mx|Prof)\.?\s+"
    rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})*"
    rf"|"
    rf"(?<![A-Za-z])"
    rf"{_NAME_TOKEN}(?:\s+(?:de|van|von|del|della|la|le|bin|ibn|al))?"
    rf"(?:\s+{_NAME_TOKEN}){{1,3}}"
)


def detect_persons(text: str) -> Iterator[DetectedSpan]:
    for m in _NAME_RE.finditer(text):
        raw = m.group(0)
        start = m.start()
        tokens = raw.split()
        while tokens and tokens[0].rstrip(".") in _NAME_STOPWORDS:
            dropped = tokens.pop(0)
            # advance start past the dropped token and its trailing whitespace.
            advance = len(dropped)
            while start + advance < len(text) and text[start + advance].isspace():
                advance += 1
            start += advance
        if len(tokens) < 2:
            continue
        trimmed = " ".join(tokens)
        end = start + len(trimmed)
        # Final sanity: the slice must be present verbatim.
        if text[start:end] != trimmed:
            continue
        yield DetectedSpan("private_person", start, end, trimmed)


# --- address ------------------------------------------------------------------
# US-style street address: <number> <Title-Cased-Street> [, City [State|Country]]
_STREET_SUFFIX = (
    r"Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|"
    r"Court|Ct\.?|Lane|Ln\.?|Way|Place|Pl\.?|Plaza|Square|Sq\.?|Terrace|"
    r"Parkway|Pkwy\.?|Circle|Cir\.?|Highway|Hwy\.?|Trail|Tr\.?"
)
_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}\s+"
    rf"(?:[A-Z][A-Za-z0-9'\-]*\s+){{0,6}}"
    rf"(?:{_STREET_SUFFIX})"
    rf"(?:,\s+[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*"
    rf"(?:,?\s+(?:[A-Z]{{2}}|USA|UK|[A-Z][a-z]+))?)?",
)


def detect_addresses(text: str) -> Iterator[DetectedSpan]:
    for m in _ADDRESS_RE.finditer(text):
        raw = m.group(0).rstrip(".,;")
        end = m.start() + len(raw)
        yield DetectedSpan("private_address", m.start(), end, raw)


# --- dispatch -----------------------------------------------------------------
ALL_DETECTORS: tuple = (
    detect_emails,
    detect_urls,
    detect_accounts,
    detect_phones,
    detect_dates,
    detect_secrets,
    detect_addresses,
    detect_persons,
)


def run_all(text: str) -> list[DetectedSpan]:
    out: list[DetectedSpan] = []
    for det in ALL_DETECTORS:
        out.extend(det(text))
    return out
