from __future__ import annotations

import pytest

from evo_verifier.items import (
    BY_COLUMN,
    BY_ID,
    BY_KEY,
    FAMILY_WEIGHTS,
    ITEMS,
    items_in_group,
)


def test_fourteen_items_with_unique_keys():
    assert len(ITEMS) == 14
    assert len(BY_ID) == len(BY_KEY) == len(BY_COLUMN) == 14


def test_family_weights_sum_to_one():
    assert sum(FAMILY_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_item_has_a_weighted_family():
    assert {item.family for item in ITEMS} <= set(FAMILY_WEIGHTS)


def test_groups_partition_the_items():
    groups = ("A1-A6", "B7-B10", "B11-B14")
    assert sum(len(items_in_group(group)) for group in groups) == 14
    assert [item.item_id for item in items_in_group("B7-B10")] == ["B7", "B8", "B9", "B10"]


def test_only_b11_to_b14_need_the_simulator():
    """B7-B10 is graph, parser fields and FK. It never routes to dynamics."""
    assert {item.item_id for item in ITEMS if item.requires_simulator} == {
        "B11",
        "B12",
        "B13",
        "B14",
    }
