"""Label taxonomy (mirrors opf v2 category version)."""

from typing import Final

LABELS: Final[tuple[str, ...]] = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)

REDACTED_PLACEHOLDER: Final[str] = "<REDACTED>"

# Priority used when two spans overlap and must be reduced to one.
# Higher wins. Emails beat urls beat person names, etc.
PRIORITY: Final[dict[str, int]] = {
    "private_email": 100,
    "private_url": 90,
    "secret": 80,
    "account_number": 70,
    "private_phone": 60,
    "private_date": 50,
    "private_address": 40,
    "private_person": 30,
}
