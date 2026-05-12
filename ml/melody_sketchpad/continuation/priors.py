"""BiMMuDa-derived priors for section-conditional melody continuation.

The transitions JSON contains per-section-pair delta statistics
(mean, std) derived from the BiMMuDa corpus (Hamilton & Pearce, TISMIR
2024). Only transitions with at least MIN_TRANSITION_SAMPLES songs are
considered reliable; everything below silently degrades to primer-relative
constraint checks.

Tier-A transitions wired into the constraint pipeline:
    verse -> chorus       (n=155)
    verse -> pre_chorus   (n=97)
    chorus -> verse       (n=95)
    chorus -> bridge      (n=94)
    pre_chorus -> chorus  (n=83)
    verse -> bridge       (n=40)

Anything else in the JSON is either too low-n to trust, or targets a
section (instrumental, post_chorus) that HumMuse does not expose in the UI.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Sample-size floor below which a transition's deltas are not statistically
# usable. Six transitions in the JSON meet this; the rest fall back to
# primer-relative constraints.
MIN_TRANSITION_SAMPLES = 30

_TRANSITIONS_PATH = Path(__file__).resolve().parent / "bimmuda_section_transitions.json"


@lru_cache(maxsize=1)
def _load_raw_transitions() -> dict[str, Any]:
    with _TRANSITIONS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_transition_profile(
    primer_section: str | None,
    target_section: str | None,
) -> dict[str, Any] | None:
    """Return a transition profile if both labels are set and reliable.

    Returns None when either label is missing, the transition is absent
    from the JSON, or its sample size is below MIN_TRANSITION_SAMPLES.
    A None return is the caller's signal to use primer-relative checks.
    """
    if not primer_section or not target_section:
        return None
    key = f"{primer_section}->{target_section}"
    raw = _load_raw_transitions().get("transitions", {}).get(key)
    if raw is None:
        return None
    if int(raw.get("n_songs", 0)) < MIN_TRANSITION_SAMPLES:
        return None
    return raw


def available_transitions() -> list[str]:
    """List transitions meeting the sample-size threshold, sorted by name."""
    raw = _load_raw_transitions().get("transitions", {})
    return sorted(
        key for key, val in raw.items()
        if int(val.get("n_songs", 0)) >= MIN_TRANSITION_SAMPLES
    )
