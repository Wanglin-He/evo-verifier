"""The fourteen human items, frozen.

The annotation platform's CSV column names are the only definition of these items
that already exists and that human annotators already answered, so they are the
anchor here. Everything else -- English keys, families, weights -- is bookkeeping
layered on top and must never change what a column means.

Item semantics, the score formula and the family weights are frozen by protocol.
Test procedures and per-item thresholds are the parts allowed to evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["semantic", "static", "motion", "physics"]
Group = Literal["A1-A6", "B7-B10", "B11-B14"]

COLUMNS: dict[str, str] = {
    "A1": "零件拆分与活动属性是否合理",
    "A2": "零件数量和种类是否符合描述",
    "A3": "必要结构是否完整",
    "A4": "几何形状、尺寸和比例是否合理",
    "A5": "零件位置、朝向与装配关系是否合理",
    "A6": "初始状态是否不存在非预期穿插、悬浮或脱离",
    "B7": "父子关系与连接对象是否正确",
    "B8": "活动零件是否具有对应的运动学关节",
    "B9": "运动学关节类型是否正确",
    "B10": "运动学关节的位置与轴向是否正确",
    "B11": "运动学关节的运动范围是否合理",
    "B12": "活动零件是否具有可信的实体连接或支撑结构",
    "B13": "运动过程中是否不存在非预期穿模或几何干涉",
    "B14": "多部件机构的连接与运动关系是否合理",
}
"""Exact CSV headers on the annotation platform. Do not edit; re-export instead."""


@dataclass(frozen=True)
class Item:
    """One human item: what a person was asked, and where it lands downstream."""

    item_id: str
    key: str
    column: str
    family: Family
    """Primary diagnostic family. Provisional -- see FAMILY_ASSIGNMENT_NOTE."""
    group: Group
    """Which of the three people owns the detector."""
    requires_simulator: bool
    """Full coverage needs time-stepped dynamics, not just geometry and FK."""
    blank_in_old_rows: bool
    """Legacy-converted rows leave this column empty. Empty is not a pass."""


# id     key                       family      group       simulator  blank in old rows
_TABLE: tuple[tuple[str, str, Family, Group, bool, bool], ...] = (
    ("A1", "part_activity_split", "semantic", "A1-A6", False, False),
    ("A2", "part_count_type", "semantic", "A1-A6", False, False),
    ("A3", "structure_complete", "semantic", "A1-A6", False, False),
    ("A4", "geometry_realism", "semantic", "A1-A6", False, True),
    ("A5", "assembly_relation", "static", "A1-A6", False, True),
    ("A6", "initial_integrity", "static", "A1-A6", False, True),
    ("B7", "parent_child_relation", "semantic", "B7-B10", False, False),
    ("B8", "joint_present", "semantic", "B7-B10", False, False),
    ("B9", "joint_type", "semantic", "B7-B10", False, True),
    ("B10", "joint_origin_axis", "static", "B7-B10", False, False),
    ("B11", "joint_range", "motion", "B11-B14", True, False),
    ("B12", "physical_mount_support", "physics", "B11-B14", True, False),
    ("B13", "motion_interference", "motion", "B11-B14", True, True),
    ("B14", "multi_part_mechanism", "motion", "B11-B14", True, True),
)

ITEMS: tuple[Item, ...] = tuple(
    Item(item_id, key, COLUMNS[item_id], family, group, simulator, blank)
    for item_id, key, family, group, simulator, blank in _TABLE
)

FAMILY_WEIGHTS: dict[Family, float] = {
    "semantic": 0.20,
    "static": 0.25,
    "motion": 0.30,
    "physics": 0.25,
}
"""Frozen by protocol. Never tune these -- tuning them is how a score drifts."""

FAMILY_ASSIGNMENT_NOTE = """
The proposal maps items to families many-to-many: B10 has both a semantic and a
geometric slice, B12 appears under static and physics, B13 under static and motion.
A4 is not assigned to any family at all in the source table.

Score_full needs one family per item, so each item gets a primary family here.
That choice is ours, not the protocol's, and it is the first thing to revisit if
family scores look wrong. Per-item outputs never depend on it.
"""

BY_ID: dict[str, Item] = {item.item_id: item for item in ITEMS}
BY_KEY: dict[str, Item] = {item.key: item for item in ITEMS}
BY_COLUMN: dict[str, Item] = {item.column: item for item in ITEMS}

ITEM_IDS: tuple[str, ...] = tuple(item.item_id for item in ITEMS)


def items_in_group(group: Group) -> tuple[Item, ...]:
    """The items one person owns."""
    return tuple(item for item in ITEMS if item.group == group)


def items_in_family(family: Family) -> tuple[Item, ...]:
    return tuple(item for item in ITEMS if item.family == family)
