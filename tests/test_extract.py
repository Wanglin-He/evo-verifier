from __future__ import annotations

import json

import pytest

from evo_verifier.contract import ContractError, ExpectedJoint, Source
from evo_verifier.extract import (
    ExtractionError,
    build_request,
    contract_from_answer,
    drop_unsupported,
    read_api_key,
)

PROMPT = (
    "A wide-feed juicer with a broad rectangular base and a large clear lid. "
    "The pusher slides prismatically along the chute guide, and the main lid "
    "rotates on a rear hinge."
)

ANSWER = {
    "expected_parts": [
        {"name": "lid", "count": 1, "source": "explicit", "quote": "a large clear lid"}
    ],
    "expected_joints": [
        {
            "child": "pusher",
            "parent": "chute guide",
            "kind": "prismatic",
            "axis_hint": "",
            "location_hint": "along the chute guide",
            "source": "explicit",
            "confidence": 1.0,
            "quote": "The pusher slides prismatically along the chute guide",
        },
        {
            "child": "lid",
            "parent": "base",
            "kind": "revolute",
            "location_hint": "rear",
            "source": "explicit",
            "confidence": 1.0,
            "quote": "the main lid rotates on a rear hinge",
        },
    ],
}


def test_parses_an_answer_into_a_contract():
    contract = contract_from_answer(
        json.dumps(ANSWER), record_id="rec_j", prompt=PROMPT, extractor="test-model"
    )
    assert [joint.child for joint in contract.joints] == ["pusher", "lid"]
    assert contract.joints[0].parent == "chute guide", "the connection object B7 reads"
    assert contract.joints[0].kind == "prismatic"
    assert contract.extractor == "test-model"


def test_an_invented_requirement_is_rejected():
    """The model may not claim the prompt says something it does not."""
    answer = {
        "expected_joints": [
            {
                "child": "drawer",
                "parent": "cabinet",
                "kind": "prismatic",
                "source": "explicit",
                "quote": "a drawer slides out of the cabinet",
            }
        ]
    }
    with pytest.raises(ContractError, match="not found in the prompt"):
        contract_from_answer(json.dumps(answer), record_id="rec_j", prompt=PROMPT, extractor="m")


def test_demotion_keeps_the_requirement_but_lowers_its_standing():
    """One bad quote in a batch run should cost that claim, not the contract."""
    contract = contract_from_answer(
        json.dumps(ANSWER), record_id="rec_j", prompt=PROMPT, extractor="m"
    )
    contract.joints = (
        *contract.joints,
        ExpectedJoint(child="spout", kind="revolute", quote="the spout swings aside"),
    )
    with pytest.raises(ContractError):
        contract.validate()

    repaired = drop_unsupported(contract)
    repaired.validate()
    assert len(repaired.joints) == 3
    assert repaired.joints[-1].source is Source.PRIOR
    assert repaired.joints[-1].confidence == 0.5
    assert repaired.joints[-1].quote == ""
    assert len(repaired.explicit_joints()) == 2
    assert any("spout" in note for note in repaired.notes)


def test_a_non_json_answer_fails_loudly():
    with pytest.raises(ExtractionError, match="not JSON"):
        contract_from_answer("I think the lid opens.", record_id="r", prompt=PROMPT, extractor="m")


def test_the_key_travels_in_a_header_never_the_url():
    """A key in a query string ends up in server logs and proxies."""
    request = build_request("hello", model="gemini-3.6-flash", api_key="secret-value")
    assert "secret-value" not in request.full_url
    assert request.get_header("X-goog-api-key") == "secret-value"
    assert request.full_url.endswith("gemini-3.6-flash:generateContent")


def test_the_request_asks_for_json_at_zero_temperature():
    request = build_request("hello", model="m", api_key="k")
    body = json.loads(request.data)
    assert body["generationConfig"] == {"responseMimeType": "application/json", "temperature": 0.0}
    assert "Description:\nhello" in body["contents"][0]["parts"][0]["text"]


def test_reads_the_key_from_a_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="abc123"\nOTHER=1\n', encoding="utf-8")
    assert read_api_key(env) == "abc123"


def test_a_missing_key_says_how_to_supply_one(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(ExtractionError, match="not set"):
        read_api_key(tmp_path / ".env")


def test_an_answer_wrapped_in_a_single_element_array_is_unwrapped():
    """Seen once in 31 real calls: the object arrives inside a list."""
    contract = contract_from_answer(
        json.dumps([ANSWER]), record_id="rec_j", prompt=PROMPT, extractor="m"
    )
    assert len(contract.joints) == 2


def test_an_answer_that_is_still_not_an_object_shows_what_arrived():
    with pytest.raises(ExtractionError, match="not a JSON object"):
        contract_from_answer(json.dumps(["a", "b"]), record_id="r", prompt=PROMPT, extractor="m")
