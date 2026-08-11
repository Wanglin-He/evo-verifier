from __future__ import annotations

import pytest

from evo_verifier.contract import (
    Contract,
    ContractError,
    ExpectedJoint,
    ExpectedPart,
    Source,
    load_contracts,
    save_contracts,
)

PROMPT = (
    "A ceiling-suspended cassette air conditioning machine with a square panel, a central "
    "intake grille, and four outlet flaps around the perimeter. The four outlet flaps rotate "
    "on their own horizontal hinges at the panel edges."
)


def cabinet() -> Contract:
    return Contract(
        record_id="rec_ac",
        prompt=PROMPT,
        parts=(
            ExpectedPart(
                name="outlet flap", count=4, quote="four outlet flaps around the perimeter"
            ),
        ),
        joints=(
            ExpectedJoint(
                child="outlet flap",
                parent="panel",
                kind="revolute",
                axis_hint="horizontal",
                location_hint="at the panel edges",
                count=4,
                quote="rotate on their own horizontal hinges at the panel edges",
            ),
        ),
        extractor="test",
    )


def test_a_valid_contract_passes():
    contract = cabinet()
    contract.validate()
    assert contract.unsupported_quotes() == []
    assert contract.movable() == contract.joints


def test_an_explicit_requirement_must_quote_the_prompt():
    """The guard against an extractor inventing requirements."""
    contract = cabinet()
    contract.joints = (
        ExpectedJoint(
            child="lid", parent="body", kind="revolute", quote="the lid opens on a hinge"
        ),
    )
    assert contract.unsupported_quotes()
    with pytest.raises(ContractError, match="not found in the prompt"):
        contract.validate()


def test_quote_matching_ignores_case_and_whitespace():
    contract = cabinet()
    contract.joints = (
        ExpectedJoint(child="outlet flap", quote="Four   Outlet Flaps\naround the perimeter"),
    )
    contract.validate()


def test_a_prior_requirement_needs_no_quote():
    """Category common sense is allowed, at reduced confidence, unquoted."""
    contract = cabinet()
    contract.joints = (
        *contract.joints,
        ExpectedJoint(child="panel", parent="ceiling", source=Source.PRIOR, confidence=0.6),
    )
    contract.validate()
    assert len(contract.explicit_joints()) == 1


def test_an_unknown_joint_kind_is_rejected():
    contract = cabinet()
    contract.joints = (ExpectedJoint(child="flap", kind="hinge", source=Source.PRIOR),)
    with pytest.raises(ContractError, match="unknown joint kind"):
        contract.validate()


def test_a_joint_needs_a_moving_part():
    contract = cabinet()
    contract.joints = (ExpectedJoint(child="", parent="panel", source=Source.PRIOR),)
    with pytest.raises(ContractError, match="no moving part"):
        contract.validate()


def test_count_none_means_unspecified_not_one():
    contract = Contract.from_dict(
        {
            "record_id": "rec_x",
            "prompt": "a thing with buttons",
            "expected_parts": [{"name": "button", "quote": "a thing with buttons"}],
        }
    )
    assert contract.parts[0].count is None


def test_round_trip(tmp_path):
    original = cabinet()
    save_contracts([original], tmp_path)
    loaded = load_contracts(tmp_path)
    assert list(loaded) == ["rec_ac"]
    assert loaded["rec_ac"].to_dict() == original.to_dict()


def test_loading_rejects_an_unsupported_contract(tmp_path):
    broken = cabinet()
    broken.joints = (ExpectedJoint(child="lid", quote="not in the prompt at all"),)
    broken.save(tmp_path / "rec_ac.json")
    with pytest.raises(ContractError):
        load_contracts(tmp_path)
