"""MVP text normalization (PRD §19 Phase 0 subset of §10).

Lowercase + whitespace/punctuation collapse only -- NO lemmatization, NO
stemming, NO suffix standardization table (that's Phase 3, PRD §10). This
exists only so blocking/feature code has a stable string to key/compare on.
"""
from __future__ import annotations

import re

_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9 ]+")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _NON_ALNUM_SPACE.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s
