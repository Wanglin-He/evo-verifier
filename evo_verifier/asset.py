"""Read the declared articulation graph out of an Articraft ``model.py``.

The dataset ships no URDF. ``cache/record_materialization/`` was never committed,
so every one of the 10,788 records is source only, and getting geometry means
running generated CAD code. But B7-B10 do not need geometry to read the graph:
parts, parent/child, joint type, origin and axis are all declared in the file.

So this reads the file instead of running it. It walks the AST of
``build_object_model``, folding module and local constants, unrolling loops whose
iterable is itself constant. It handles the shapes this dataset actually
contains; it is not a Python interpreter.

**Nothing is guessed.** A value that will not fold is recorded in ``notes`` and
left as ``None``. That distinction carries all the way to the report: an
unresolved origin is partial coverage, never a B10 failure. Reporting a missing
measurement as an asset defect is the one error this whole project is built to
avoid.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Vec3 = tuple[float, float, float]

MODEL_FACTORIES = frozenset({"part", "get_part"})
"""``model.part("door")`` and ``model.get_part("door")`` both name a part."""

PRIMITIVES = frozenset({"Box", "Cylinder", "Sphere"})
"""Analytic shapes. Everything else is a mesh we cannot see without running CAD."""

_MATH_CONSTANTS = {"pi": math.pi, "tau": math.tau, "e": math.e}

_MATH_CALLS: dict[str, Any] = {
    name: getattr(math, name)
    for name in (
        "radians",
        "degrees",
        "sin",
        "cos",
        "tan",
        "atan2",
        "sqrt",
        "hypot",
        "floor",
        "ceil",
    )
}

_PURE_CALLS: dict[str, Any] = {
    "range": lambda *args: tuple(float(value) for value in range(*(int(a) for a in args))),
    "enumerate": lambda items: tuple((float(i), item) for i, item in enumerate(items)),
    "len": lambda items: float(len(items)),
    "float": float,
    "int": lambda value: float(int(value)),
    "abs": abs,
    "round": lambda value, digits=0: round(value, int(digits)),
    "min": min,
    "max": max,
    "sum": lambda items: float(sum(items)),
    "tuple": tuple,
    "list": tuple,
    "str": str,
    "sorted": lambda items: tuple(sorted(items)),
    "reversed": lambda items: tuple(reversed(items)),
    "zip": lambda *iterables: tuple(zip(*iterables, strict=False)),
}
"""Builtins with no side effects, folded so loop bounds and sizes resolve."""

_COMPARISONS: dict[type, Any] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}


class _Unresolvable(Exception):
    """A value this reader will not invent."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PartRef:
    """A local variable bound to ``model.part(...)``."""

    name: str


@dataclass(frozen=True)
class Origin:
    """A pose in the parent frame. ``None`` means the file did not resolve."""

    xyz: Vec3 | None = (0.0, 0.0, 0.0)
    rpy: Vec3 | None = (0.0, 0.0, 0.0)

    @property
    def is_rotated(self) -> bool:
        return self.rpy is not None and any(abs(angle) > 1e-9 for angle in self.rpy)


@dataclass(frozen=True)
class Shape:
    """One visual or collision element of a part, in the part frame."""

    kind: str
    """``box``, ``cylinder``, ``sphere`` or ``mesh``."""
    origin: Origin
    name: str = ""
    size: Vec3 | None = None
    radius: float | None = None
    height: float | None = None

    def half_extents(self) -> Vec3 | None:
        """Axis-aligned half sizes, or ``None`` when the shape is not analytic."""
        if self.kind == "box" and self.size is not None:
            return (self.size[0] / 2, self.size[1] / 2, self.size[2] / 2)
        if self.kind == "cylinder" and self.radius is not None and self.height is not None:
            return (self.radius, self.radius, self.height / 2)
        if self.kind == "sphere" and self.radius is not None:
            return (self.radius, self.radius, self.radius)
        return None

    def aabb(self) -> tuple[Vec3, Vec3] | None:
        """Bounds in the part frame. ``None`` if the shape or its pose is unknown.

        A rotated shape returns ``None`` rather than a wrong box: rotating an
        axis-aligned extent needs the real geometry, and B10 divides by these.
        """
        half = self.half_extents()
        if half is None or self.origin.xyz is None or self.origin.is_rotated:
            return None
        centre = self.origin.xyz
        return (
            (centre[0] - half[0], centre[1] - half[1], centre[2] - half[2]),
            (centre[0] + half[0], centre[1] + half[1], centre[2] + half[2]),
        )


@dataclass(frozen=True)
class Part:
    """A link: a name and the shapes attached to it."""

    name: str
    shapes: tuple[Shape, ...] = ()

    @property
    def geometry_complete(self) -> bool:
        """Every shape is analytic and axis-aligned, so the bounds are exact."""
        return bool(self.shapes) and all(shape.aabb() is not None for shape in self.shapes)

    def aabb(self) -> tuple[Vec3, Vec3] | None:
        """Union of the shapes that resolved, or ``None`` when none did.

        Partial bounds are still returned when some shapes resolve; read
        ``geometry_complete`` before using this as a normaliser.
        """
        boxes = [box for shape in self.shapes if (box := shape.aabb()) is not None]
        if not boxes:
            return None
        low = tuple(min(box[0][axis] for box in boxes) for axis in range(3))
        high = tuple(max(box[1][axis] for box in boxes) for axis in range(3))
        return ((low[0], low[1], low[2]), (high[0], high[1], high[2]))


@dataclass(frozen=True)
class Limits:
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None

    @property
    def travel(self) -> float | None:
        """``|q_max - q_min|``, the normaliser several B-items divide by."""
        if self.lower is None or self.upper is None:
            return None
        return abs(self.upper - self.lower)


@dataclass(frozen=True)
class Articulation:
    """One declared joint. ``None`` fields did not resolve statically."""

    name: str
    kind: str | None
    """``revolute``, ``continuous``, ``prismatic`` or ``fixed``."""
    parent: str | None
    child: str | None
    origin: Origin
    axis: Vec3 | None
    limits: Limits | None = None
    mimic: str | None = None
    """Raw source of a ``mimic=`` argument. The coupling contract B14 needs."""

    @property
    def is_movable(self) -> bool:
        return self.kind is not None and self.kind != "fixed"


@dataclass
class Asset:
    """One record's declared model, as far as reading the source can tell."""

    record_id: str
    robot_name: str = ""
    source: Path | None = None
    parts: dict[str, Part] = field(default_factory=dict)
    articulations: list[Articulation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """What would not resolve, one line each. Empty means a complete read."""

    @property
    def complete(self) -> bool:
        return not self.notes

    def movable(self) -> list[Articulation]:
        return [joint for joint in self.articulations if joint.is_movable]

    def children_of(self, part: str) -> list[str]:
        return [j.child for j in self.articulations if j.parent == part and j.child]

    def joint_for_child(self, part: str) -> Articulation | None:
        """The joint that moves this part. A tree gives each part at most one."""
        return next((j for j in self.articulations if j.child == part), None)

    def descendants(self, part: str) -> set[str]:
        """Everything that moves when this part moves, itself excluded.

        The subtree a joint carries. B7 reads it: a drawer that hangs off the lid
        travels with the lid, which is the wrong thing to be attached to even
        though the drawer's own joint looks fine.
        """
        children: dict[str, list[str]] = {}
        for joint in self.articulations:
            if joint.parent and joint.child:
                children.setdefault(joint.parent, []).append(joint.child)
        seen: set[str] = set()
        stack = list(children.get(part, ()))
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            stack.extend(children.get(name, ()))
        return seen

    def roots(self) -> list[str]:
        """Parts that are nobody's child. A well-formed tree has exactly one."""
        children = {j.child for j in self.articulations}
        return [name for name in self.parts if name not in children]

    def diagonal(self) -> float | None:
        """``D``: the whole-object bounding box diagonal at the declared pose.

        The normaliser B10 divides anchor distances by. ``None`` unless every
        part resolved -- a diagonal computed from half the parts is worse than
        no diagonal, because it silently rescales every threshold.
        """
        if not self.parts or not all(part.geometry_complete for part in self.parts.values()):
            return None
        boxes = [box for part in self.parts.values() if (box := part.aabb()) is not None]
        if not boxes:
            return None
        low = [min(box[0][axis] for box in boxes) for axis in range(3)]
        high = [max(box[1][axis] for box in boxes) for axis in range(3)]
        return math.dist(low, high)


def find_model_py(data_dir: Path | str, record_id: str) -> Path:
    """Locate a record's active ``model.py`` through its ``record.json``."""
    record_dir = Path(data_dir) / "records" / record_id
    manifest = record_dir / "record.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no record.json for {record_id}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relative = (payload.get("artifacts") or {}).get("model_py")
    if not relative:
        revision = payload.get("active_revision_id", "rev_000001")
        relative = f"revisions/{revision}/model.py"
    path = record_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"no model.py for {record_id}: {path}")
    return path


def read_prompt(data_dir: Path | str, record_id: str) -> str:
    """The full prompt, not the 150-character preview the annotation export carries."""
    record_dir = Path(data_dir) / "records" / record_id
    payload = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    relative = (payload.get("artifacts") or {}).get("prompt_txt")
    if not relative:
        revision = payload.get("active_revision_id", "rev_000001")
        relative = f"revisions/{revision}/prompt.txt"
    return (record_dir / relative).read_text(encoding="utf-8").strip()


def load_asset(source: Path | str, record_id: str = "") -> Asset:
    """Read one ``model.py``. Never executes it."""
    path = Path(source)
    return read_model_source(path.read_text(encoding="utf-8"), record_id, source=path)


def read_model_source(code: str, record_id: str = "", *, source: Path | None = None) -> Asset:
    asset = Asset(record_id=record_id, source=source)
    tree = ast.parse(code)
    reader = _Reader(asset)
    reader.run(tree)
    return asset


class _Reader:
    """Folds constants and records declarations. Deliberately not an interpreter."""

    def __init__(self, asset: Asset) -> None:
        self.asset = asset
        self.env: dict[str, Any] = {}
        self.shapes: dict[str, list[Shape]] = {}

    def run(self, tree: ast.Module) -> None:
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                self._assign(statement, quiet=True)
        builder = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "build_object_model"
            ),
            None,
        )
        if builder is None:
            self.asset.notes.append("no build_object_model function")
            return
        self._block(builder.body)
        self.asset.parts = {
            name: Part(name=name, shapes=tuple(shapes)) for name, shapes in self.shapes.items()
        }

    # -- statements ----------------------------------------------------------

    def _block(self, body: list[ast.stmt]) -> None:
        for statement in body:
            self._statement(statement)

    def _statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            self._assign(statement)
        elif isinstance(statement, ast.Expr):
            self._call(statement.value)
        elif isinstance(statement, ast.For):
            self._loop(statement)
        elif isinstance(statement, ast.AugAssign):
            self._aug_assign(statement)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self._assign(ast.Assign(targets=[statement.target], value=statement.value))
        elif isinstance(statement, ast.If):
            self._branch(statement)
        elif isinstance(
            statement,
            (ast.Return, ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Pass),
        ):
            return
        else:
            self.asset.notes.append(f"skipped {type(statement).__name__} statement")

    def _assign(self, statement: ast.Assign, *, quiet: bool = False) -> None:
        value = self._maybe(statement.value)
        for target in statement.targets:
            if value is None:
                self._unbind(target)
            else:
                self._bind(target, value)
        if not quiet:
            self._call(statement.value)

    def _bind(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            for element, item in zip(target.elts, value, strict=False):
                self._bind(element, item)

    def _unbind(self, target: ast.expr) -> None:
        """A name reassigned to something we cannot fold must stop resolving."""
        if isinstance(target, ast.Name):
            self.env.pop(target.id, None)
        elif isinstance(target, ast.Tuple):
            for element in target.elts:
                self._unbind(element)

    def _aug_assign(self, statement: ast.AugAssign) -> None:
        value = self._maybe(
            ast.BinOp(left=statement.target, op=statement.op, right=statement.value)
        )
        if value is None:
            self._unbind(statement.target)
        else:
            self._bind(statement.target, value)

    def _branch(self, statement: ast.If) -> None:
        """Take the branch the condition selects, or neither if it will not fold."""
        test = self._maybe(statement.test)
        if test is None:
            self.asset.notes.append(f"branch not taken: if {ast.unparse(statement.test)}")
            return
        self._block(statement.body if test else statement.orelse)

    def _loop(self, statement: ast.For) -> None:
        """Unroll when the iterable is constant; otherwise say so and move on."""
        try:
            iterable = self._fold(statement.iter)
        except _Unresolvable as unresolved:
            self.asset.notes.append(f"loop not unrolled: {unresolved.reason}")
            return
        if not isinstance(iterable, (tuple, list)):
            self.asset.notes.append("loop over a non-constant iterable")
            return
        for item in iterable:
            self._bind(statement.target, item)
            self._block(statement.body)

    # -- calls ---------------------------------------------------------------

    def _call(self, node: ast.expr) -> None:
        if not isinstance(node, ast.Call):
            return
        if isinstance(node.func, ast.Name):
            if node.func.id == "ArticulatedObject":
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                name = self._maybe_str(node.args[0] if node.args else keywords.get("name"))
                self.asset.robot_name = name or ""
            return
        if not isinstance(node.func, ast.Attribute):
            return
        attribute = node.func.attr
        if attribute == "articulation":
            self._articulation(node)
        elif attribute in ("visual", "collision"):
            self._shape(node)
        elif attribute in MODEL_FACTORIES and node.args:
            try:
                self.shapes.setdefault(str(self._fold(node.args[0])), [])
            except _Unresolvable as unresolved:
                self.asset.notes.append(f"part name unresolved: {unresolved.reason}")

    def _articulation(self, node: ast.Call) -> None:
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        name = self._maybe_str(node.args[0]) if node.args else None
        kind = self._kind(node.args[1]) if len(node.args) > 1 else self._kind(keywords.get("kind"))
        joint = Articulation(
            name=name or "<unnamed>",
            kind=kind,
            parent=self._part_name(keywords.get("parent")),
            child=self._part_name(keywords.get("child")),
            origin=self._origin(keywords.get("origin")),
            axis=self._maybe_vec3(keywords.get("axis")),
            limits=self._limits(keywords.get("motion_limits")),
            mimic=ast.unparse(keywords["mimic"]) if "mimic" in keywords else None,
        )
        for field_name, value in (
            ("name", name),
            ("kind", joint.kind),
            ("parent", joint.parent),
            ("child", joint.child),
        ):
            if value is None:
                self.asset.notes.append(f"joint {joint.name}: {field_name} unresolved")
        if joint.origin.xyz is None:
            self.asset.notes.append(f"joint {joint.name}: origin unresolved")
        if joint.axis is None and joint.kind != "fixed":
            self.asset.notes.append(f"joint {joint.name}: axis unresolved")
        self.asset.articulations.append(joint)

    def _shape(self, node: ast.Call) -> None:
        owner = node.func
        if not isinstance(owner, ast.Attribute) or not isinstance(owner.value, ast.Name):
            return
        part = self.env.get(owner.value.id)
        if not isinstance(part, PartRef):
            return
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        shape = self._geometry(node.args[0] if node.args else None)
        self.shapes.setdefault(part.name, []).append(
            Shape(
                kind=shape[0],
                size=shape[1],
                radius=shape[2],
                height=shape[3],
                origin=self._origin(keywords.get("origin")),
                name=self._maybe_str(keywords.get("name")) or "",
            )
        )

    def _geometry(
        self, node: ast.expr | None
    ) -> tuple[str, Vec3 | None, float | None, float | None]:
        """``Box``/``Cylinder``/``Sphere`` resolve; anything else is an opaque mesh."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return ("mesh", None, None, None)
        name = node.func.id
        if name not in PRIMITIVES:
            return ("mesh", None, None, None)
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if name == "Box":
            return (
                "box",
                self._maybe_vec3(node.args[0] if node.args else keywords.get("size")),
                None,
                None,
            )
        radius = self._maybe_float(node.args[0] if node.args else keywords.get("radius"))
        if name == "Sphere":
            return ("sphere", None, radius, None)
        height = self._maybe_float(node.args[1] if len(node.args) > 1 else keywords.get("height"))
        return ("cylinder", None, radius, height)

    def _limits(self, node: ast.expr | None) -> Limits | None:
        if not isinstance(node, ast.Call):
            return None
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        return Limits(
            lower=self._maybe_float(keywords.get("lower")),
            upper=self._maybe_float(keywords.get("upper")),
            effort=self._maybe_float(keywords.get("effort")),
            velocity=self._maybe_float(keywords.get("velocity")),
        )

    def _origin(self, node: ast.expr | None) -> Origin:
        if node is None:
            return Origin()
        if isinstance(node, ast.Call):
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            xyz = self._maybe_vec3(keywords["xyz"]) if "xyz" in keywords else (0.0, 0.0, 0.0)
            rpy = self._maybe_vec3(keywords["rpy"]) if "rpy" in keywords else (0.0, 0.0, 0.0)
            return Origin(xyz=xyz, rpy=rpy)
        value = self._maybe_vec3(node)
        return Origin(xyz=value)

    def _part_name(self, node: ast.expr | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            value = self.env.get(node.id)
            return value.name if isinstance(value, PartRef) else None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MODEL_FACTORIES
            and node.args
        ):
            return self._maybe_str(node.args[0])
        return self._maybe_str(node)

    def _kind(self, node: ast.expr | None) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ArticulationType"
        ):
            return node.attr.lower()
        return self._maybe_str(node)

    # -- folding -------------------------------------------------------------

    def _maybe_str(self, node: ast.expr | None) -> str | None:
        value = self._maybe(node)
        return value if isinstance(value, str) else None

    def _maybe_float(self, node: ast.expr | None) -> float | None:
        value = self._maybe(node)
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    def _maybe_vec3(self, node: ast.expr | None) -> Vec3 | None:
        value = self._maybe(node)
        if (
            isinstance(value, (tuple, list))
            and len(value) == 3
            and all(isinstance(item, (int, float)) for item in value)
        ):
            return (float(value[0]), float(value[1]), float(value[2]))
        return None

    def _maybe(self, node: ast.expr | None) -> Any:
        if node is None:
            return None
        try:
            return self._fold(node)
        except _Unresolvable:
            return None

    def _fold(self, node: ast.expr) -> Any:
        """Constant expressions only. Raises rather than approximating."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            raise _Unresolvable(f"unknown name {node.id}")
        if isinstance(node, ast.UnaryOp):
            value = self._fold(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            raise _Unresolvable("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._fold(element) for element in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._fold(key): self._fold(value)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.BoolOp):
            values = [self._fold(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        if isinstance(node, ast.Slice):
            return slice(
                None if node.lower is None else int(self._fold(node.lower)),
                None if node.upper is None else int(self._fold(node.upper)),
                None if node.step is None else int(self._fold(node.step)),
            )
        if isinstance(node, ast.JoinedStr):
            return "".join(
                str(self._format(piece)) if isinstance(piece, ast.FormattedValue) else piece.value
                for piece in node.values
            )
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr in _MATH_CONSTANTS
            ):
                return _MATH_CONSTANTS[node.attr]
            raise _Unresolvable(f"attribute {ast.unparse(node)}")
        if isinstance(node, ast.Call):
            return self._fold_call(node)
        raise _Unresolvable(f"unsupported expression {type(node).__name__}")

    def _format(self, piece: ast.FormattedValue) -> Any:
        value = self._fold(piece.value)
        return int(value) if isinstance(value, float) and value.is_integer() else value

    def _subscript(self, node: ast.Subscript) -> Any:
        container = self._fold(node.value)
        index = self._fold(node.slice)
        try:
            if isinstance(container, dict) or isinstance(index, (str, slice)):
                return container[index]
            return container[int(index)]
        except (TypeError, KeyError, IndexError, ValueError) as error:
            raise _Unresolvable(f"bad index: {error}") from None

    def _compare(self, node: ast.Compare) -> bool:
        left = self._fold(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self._fold(comparator)
            try:
                outcome = _COMPARISONS[type(operator)](left, right)
            except KeyError:
                raise _Unresolvable(f"comparison {type(operator).__name__}") from None
            except TypeError as error:
                raise _Unresolvable(f"comparison: {error}") from None
            if not outcome:
                return False
            left = right
        return True

    def _binary(self, node: ast.BinOp) -> Any:
        left, right = self._fold(node.left), self._fold(node.right)
        operator = type(node.op)
        try:
            if operator is ast.Add:
                return left + right
            if operator is ast.Sub:
                return left - right
            if operator is ast.Mult:
                return left * right
            if operator is ast.Div:
                return left / right
            if operator is ast.FloorDiv:
                return left // right
            if operator is ast.Mod:
                return left % right
            if operator is ast.Pow:
                return left**right
        except (TypeError, ZeroDivisionError, ValueError) as error:
            raise _Unresolvable(f"bad arithmetic: {error}") from None
        raise _Unresolvable(f"unsupported operator {operator.__name__}")

    def _fold_call(self, node: ast.Call) -> Any:
        """Only the calls that produce constants or name a part.

        Everything else -- a CAD helper, a mesh builder, a local function -- is
        left unresolved. Running it is the thing this reader exists to avoid.
        """
        if isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in MODEL_FACTORIES and node.args:
                return PartRef(str(self._fold(node.args[0])))
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                if attribute not in _MATH_CALLS:
                    raise _Unresolvable(f"math.{attribute}()")
                return _MATH_CALLS[attribute](*[self._fold(a) for a in node.args])
            if attribute in ("items", "keys", "values"):
                container = self._fold(node.func.value)
                if isinstance(container, dict):
                    return tuple(getattr(container, attribute)())
            raise _Unresolvable(f"call {attribute}()")
        if not isinstance(node.func, ast.Name):
            raise _Unresolvable("call on a computed callable")
        name = node.func.id
        if name not in _PURE_CALLS:
            raise _Unresolvable(f"call {name}()")
        arguments = [self._fold(argument) for argument in node.args]
        try:
            return _PURE_CALLS[name](*arguments)
        except (TypeError, ValueError, IndexError, ZeroDivisionError) as error:
            raise _Unresolvable(f"{name}(): {error}") from None
