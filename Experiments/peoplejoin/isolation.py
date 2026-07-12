"""Information isolation guards for PeopleJoin-Reactive."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence, Union

FORBIDDEN_PEOPLEJOIN_KEYS: tuple[str, ...] = (
    "context_latents",
    "latent_requester_preferences",
    "latent_candidate_preferences",
    "latent_interpersonal_affinity",
    "latent_risk_tolerance",
    "latent_opportunity_bias",
    "oracle",
    "outcome",
    "reward",
    "MapScore",
    "S_cap",
    "S_need",
)


def _serialize_payload(payload: Union[str, Mapping[str, Any], Sequence[Any], None]) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, default=str)


def _collect_keys(obj: Any, out: set[str]) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.add(str(key))
            _collect_keys(value, out)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            _collect_keys(item, out)


def _json_key_mentioned(serialized: str, key: str) -> bool:
    """True if `key` appears as a JSON/object field name, not mere prose."""
    # Quoted JSON keys: "context_latents" / 'MapScore'
    if re.search(rf'["\']{re.escape(key)}["\']\s*:', serialized):
        return True
    if re.search(rf'["\']{re.escape(key)}["\']', serialized):
        # Bare quoted token often indicates a serialized field name in dumps.
        return True
    return False


def assert_peoplejoin_public_inputs(
    *payloads: Union[str, Mapping[str, Any], Sequence[Any], None],
    extra_forbidden: Iterable[str] = (),
) -> None:
    """Raise if prompts/observations contain forbidden field names.

    Instructional prose may mention forbidden concepts (e.g. "do not use oracle
    outcomes"); this guard looks for field-like leakage: mapping keys and
    quoted JSON field names.
    """
    forbidden = list(FORBIDDEN_PEOPLEJOIN_KEYS) + list(extra_forbidden)
    for payload in payloads:
        if isinstance(payload, (Mapping, list, tuple)):
            keys: set[str] = set()
            _collect_keys(payload, keys)
            for key in forbidden:
                if key in keys:
                    raise AssertionError(
                        f"Hidden/oracle key leaked into PeopleJoin public input: {key}"
                    )
            # Also inspect serialized form for nested stringified JSON.
            serialized = _serialize_payload(payload)
            for key in forbidden:
                if _json_key_mentioned(serialized, key):
                    raise AssertionError(
                        f"Hidden/oracle key leaked into PeopleJoin public input: {key}"
                    )
            continue

        serialized = _serialize_payload(payload)
        # Try parse as JSON object/array first.
        try:
            parsed = json.loads(serialized)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, (Mapping, list, tuple)):
            keys = set()
            _collect_keys(parsed, keys)
            for key in forbidden:
                if key in keys or _json_key_mentioned(serialized, key):
                    raise AssertionError(
                        f"Hidden/oracle key leaked into PeopleJoin public input: {key}"
                    )
            continue

        for key in forbidden:
            if _json_key_mentioned(serialized, key):
                raise AssertionError(
                    f"Hidden/oracle key leaked into PeopleJoin public input: {key}"
                )
