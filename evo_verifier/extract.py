"""Turn a prompt into a contract, once, offline.

This is the only part of the verifier that talks to a model, and it runs before
verification rather than during it. The contracts land on disk and freeze there,
so a score is reproducible next year, auditable by anyone who opens the JSON, and
costs nothing to recompute.

The model translates; it never scores. It is asked for requirements and quotes,
not for judgements, and every explicit requirement is checked against the prompt
text before the contract is accepted.

Gemini by default, deliberately: 587 of the 607 annotated assets were generated
by OpenAI models, so extracting their requirements with an OpenAI model risks
reproducing the same misreading that produced the asset. Correlated errors would
hide exactly the failures this is meant to find.

The key is read from the environment or a ``.env`` file and sent in a header --
never in the URL, where it would end up in logs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evo_verifier.contract import Contract, ContractError, _joint, _part

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.6-flash"
KEY_NAME = "GEMINI_API_KEY"

INSTRUCTION = """\
You read a text description of an articulated 3D object and write down what the \
description requires. You do not design, judge, or score anything.

Return JSON only, with this shape:

{
  "expected_parts": [
    {"name": "...", "count": 4, "source": "explicit", "confidence": 1.0, "quote": "..."}
  ],
  "expected_joints": [
    {"child": "...", "parent": "...", "kind": "revolute", "axis_hint": "...",
     "location_hint": "...", "count": 4, "source": "explicit", "confidence": 1.0, "quote": "..."}
  ]
}

Rules:

1. `quote` must be copied VERBATIM from the description -- an exact substring, not \
a paraphrase. A requirement whose quote is not found in the description is thrown away.
2. `source` is "explicit" when the description says it. If you add something the \
description only implies by common sense, use "prior", set confidence at most 0.7, \
and leave the quote empty. Prefer explicit; do not pad the contract with priors.
3. `child` is the part that moves. `parent` is what it moves against -- the hinge \
plate, the column, the chute, the post. Name both the way the description names \
them. If the description does not say what it attaches to, leave `parent` empty \
rather than guessing.
4. `kind` is one of:
   - "revolute": rotation with a stop -- a hinge, a lid, a door, a tilting head
   - "continuous": rotation without end -- a wheel, a rotor, a spinning blade
   - "prismatic": sliding or pushing along a line -- a drawer, a button, a carriage
   - "fixed": explicitly rigid
   Use null if the description does not say.
5. `count` only when the description gives a number. Otherwise null -- null means \
unspecified, not one.
6. `axis_hint` and `location_hint` are short phrases from the description about \
direction ("vertical", "about its axle") and placement ("at the panel edges", \
"rear"). Empty string when absent.
7. Do not invent parts. Do not describe geometry, quality, or correctness.
"""


class ExtractionError(RuntimeError):
    """The model call failed, or its answer would not become a valid contract."""


def read_api_key(env_file: Path | str | None = None, *, name: str = KEY_NAME) -> str:
    """Take the key from the environment, or from a ``.env`` file.

    Returns the value for immediate use in a request header. It is never logged,
    printed, or written to disk by anything in this package.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key.strip() == name:
                return raw.strip().strip("\"'")
    raise ExtractionError(f"{name} is not set (pass --env-file to read it from a .env)")


def build_request(prompt: str, *, model: str, api_key: str) -> urllib.request.Request:
    body = {
        "contents": [{"parts": [{"text": f"{INSTRUCTION}\n\nDescription:\n{prompt}"}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
    }
    return urllib.request.Request(
        ENDPOINT.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )


def call_model(prompt: str, *, model: str, api_key: str, timeout: float = 60.0) -> str:
    """POST one prompt and return the model's raw text answer."""
    request = build_request(prompt, model=model, api_key=api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise ExtractionError(f"HTTP {error.code} from {model}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise ExtractionError(f"could not reach {model}: {error}") from None
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ExtractionError(f"unexpected response shape: {json.dumps(payload)[:400]}") from None


def contract_from_answer(
    answer: str | Mapping[str, Any],
    *,
    record_id: str,
    prompt: str,
    extractor: str,
) -> Contract:
    """Parse a model answer into a validated contract. Raises if it will not hold."""
    if isinstance(answer, str):
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError as error:
            raise ExtractionError(f"{record_id}: answer is not JSON: {error}") from None
    else:
        payload = answer
    if not isinstance(payload, Mapping):
        raise ExtractionError(f"{record_id}: answer is not a JSON object")
    contract = Contract(
        record_id=record_id,
        prompt=prompt,
        parts=tuple(_part(item) for item in payload.get("expected_parts", ())),
        joints=tuple(_joint(item) for item in payload.get("expected_joints", ())),
        extractor=extractor,
    )
    contract.validate()
    return contract


def drop_unsupported(contract: Contract) -> Contract:
    """Demote explicit requirements whose quote is missing, instead of failing.

    A second-best path for a batch run: the requirement survives as a prior at low
    confidence, flagged in notes, so one bad quote does not lose a whole contract.
    """
    prompt = contract.prompt
    kept_parts, kept_joints, notes = [], [], list(contract.notes)
    stray = set(contract.unsupported_quotes())
    for part in contract.parts:
        label = f"{part.name}: {part.quote!r}"
        if label in stray:
            notes.append(f"unquoted, demoted to prior: {part.name}")
            kept_parts.append(
                _part({**part.to_dict(), "source": "prior", "confidence": 0.5, "quote": ""})
            )
        else:
            kept_parts.append(part)
    for joint in contract.joints:
        label = f"{joint.child}: {joint.quote!r}"
        if label in stray:
            notes.append(f"unquoted, demoted to prior: {joint.child}")
            kept_joints.append(
                _joint({**joint.to_dict(), "source": "prior", "confidence": 0.5, "quote": ""})
            )
        else:
            kept_joints.append(joint)
    return Contract(
        record_id=contract.record_id,
        prompt=prompt,
        parts=tuple(kept_parts),
        joints=tuple(kept_joints),
        extractor=contract.extractor,
        notes=notes,
    )


def extract(
    record_id: str,
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    demote_unquoted: bool = True,
) -> Contract:
    """One prompt in, one validated contract out."""
    answer = call_model(prompt, model=model, api_key=api_key, timeout=timeout)
    try:
        return contract_from_answer(answer, record_id=record_id, prompt=prompt, extractor=model)
    except ContractError:
        if not demote_unquoted:
            raise
    loose = Contract(
        record_id=record_id,
        prompt=prompt,
        parts=tuple(_part(i) for i in json.loads(answer).get("expected_parts", ())),
        joints=tuple(_joint(i) for i in json.loads(answer).get("expected_joints", ())),
        extractor=model,
    )
    repaired = drop_unsupported(loose)
    repaired.validate()
    return repaired
