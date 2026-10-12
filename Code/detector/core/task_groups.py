"""Immutable task/class mappings, independent of tensor and data backends."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


# Public ModuleDict/Module attributes cannot be used as module registration keys.
# Keep this contract pure; the head also checks against its installed torch API.
_RESERVED_NAMES = frozenset((
    "training", "forward", "items", "keys", "values", "update", "clear", "pop",
    "get", "modules", "children", "parameters", "buffers", "state_dict",
    "load_state_dict", "add_module", "register_module", "register_parameter",
    "register_buffer", "get_submodule", "set_submodule", "get_parameter",
    "get_buffer", "named_modules", "named_children", "named_parameters",
    "named_buffers", "train", "eval", "requires_grad_", "zero_grad", "apply",
    "to", "to_empty", "cpu", "cuda", "xpu", "ipu", "mtia", "float", "double",
    "half", "bfloat16", "type", "compile", "share_memory", "extra_repr",
    "get_extra_state", "set_extra_state", "call_super_init", "dump_patches",
    "register_forward_hook", "register_forward_pre_hook", "register_backward_hook",
    "register_full_backward_hook", "register_full_backward_pre_hook",
    "register_load_state_dict_pre_hook", "register_load_state_dict_post_hook",
    "register_state_dict_pre_hook", "register_state_dict_post_hook",
))


def validate_objects(objects):
    """Validate dense zero-based foreground IDs, regardless of mapping order."""
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError("data.kitti.objects must be a nonempty class-to-ID mapping")
    if any(not isinstance(name, str) or not name for name in objects):
        raise ValueError("Class names must be nonempty strings")
    ids = tuple(objects.values())
    if (any(type(i) is not int for i in ids) or
            sorted(ids) != list(range(len(ids)))):
        raise ValueError("Global class IDs must be distinct integers 0..num_classes-1")


@dataclass(frozen=True)
class TaskGroup:
    name: str
    classes: tuple[str, ...]
    global_ids: tuple[int, ...]

    @property
    def num_classes(self):
        return len(self.global_ids)

    def local_to_global(self, local_id):
        """Gaussian foreground channel index to global foreground class ID."""
        if type(local_id) is not int or not 0 <= local_id < self.num_classes:
            raise ValueError(f"Invalid local class ID for group {self.name}: {local_id!r}")
        return self.global_ids[local_id]

    def global_to_local(self, global_id):
        if type(global_id) is not int or global_id not in self.global_ids:
            raise ValueError(f"Global class ID {global_id!r} is not in group {self.name}")
        return self.global_ids.index(global_id)

    def binary_to_global(self, label):
        """Binary label 0 is background (None); foreground labels start at 1."""
        if type(label) is not int or not 0 <= label <= self.num_classes:
            raise ValueError(f"Invalid binary label for group {self.name}: {label!r}")
        return None if label == 0 else self.local_to_global(label - 1)

    def global_to_binary(self, global_id):
        return 0 if global_id is None else self.global_to_local(global_id) + 1

    def to_dict(self):
        return {"name": self.name, "classes": list(self.classes),
                "global_ids": list(self.global_ids)}


def resolve_task_groups(head_groups, objects):
    """Resolve an exact partition without sorting group/class order or mutation.

    Ordering is semantic: it controls local channels and checkpoint identity.
    Global IDs always come from objects, never from list position in a group.
    """
    validate_objects(objects)
    if not isinstance(head_groups, (list, tuple)) or not head_groups:
        raise ValueError("Grouped mode requires nonempty data.head_groups")
    names, assigned, result = set(), set(), []
    for group in head_groups:
        if not isinstance(group, Mapping) or set(group) != {"name", "classes"}:
            raise ValueError("Each head group must contain exactly name and classes")
        name, classes = group["name"], group["classes"]
        if (not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
                or name in _RESERVED_NAMES):
            raise ValueError(f"Invalid or reserved task group name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate task group name: {name}")
        if not isinstance(classes, (list, tuple)) or not classes:
            raise ValueError(f"Group {name} requires a nonempty class list")
        for cls in classes:
            if not isinstance(cls, str) or cls not in objects:
                raise ValueError(f"Unknown class in group {name}: {cls!r}")
            if cls in assigned:
                raise ValueError(f"Class {cls} occurs more than once in head_groups")
            assigned.add(cls)
        names.add(name)
        result.append(TaskGroup(name, tuple(classes), tuple(objects[c] for c in classes)))
    if assigned != set(objects):
        raise ValueError(f"head_groups omit active classes: {sorted(set(objects) - assigned)}")
    return tuple(result)


def validate_resolved_groups(groups):
    """Revalidate metadata at direct constructor boundaries, including both imports.

    Legacy code can import core.* or detector.core.*; use the shared contract
    rather than relying on dataclass identity across those package aliases.
    """
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("Grouped mode requires nonempty resolved task groups")
    definitions, objects = [], {}
    for group in groups:
        try:
            name, classes, ids = group.name, group.classes, group.global_ids
        except AttributeError as exc:
            raise ValueError("task_groups must contain resolved task group metadata") from exc
        if (not isinstance(classes, (list, tuple)) or not isinstance(ids, (list, tuple))
                or len(classes) != len(ids)):
            raise ValueError("Resolved group classes and global_ids must have matching lengths")
        for cls, global_id in zip(classes, ids):
            if not isinstance(cls, str) or cls in objects:
                raise ValueError("Resolved groups contain invalid or duplicate classes")
            objects[cls] = global_id
        definitions.append({"name": name, "classes": classes})
    return resolve_task_groups(definitions, objects)
