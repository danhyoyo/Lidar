"""Pure, stable class-space contracts shared by targets, heads and decode."""

import copy
import importlib
import subprocess
import sys

import pytest


OBJECTS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
GROUPS = [{"name": "car", "classes": ["Car"]},
          {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]


def resolve(groups=GROUPS, objects=OBJECTS):
    return importlib.import_module("detector.core.task_groups").resolve_task_groups(groups, objects)


def test_mapping_roundtrip_and_binary_background():
    car, ped = resolve()
    assert (car.name, car.classes, car.global_ids) == ("car", ("Car",), (0,))
    assert ped.local_to_global(0) == 1
    assert ped.local_to_global(1) == 2
    assert ped.global_to_local(2) == 1
    assert ped.binary_to_global(0) is None
    assert ped.global_to_binary(None) == 0
    assert ped.binary_to_global(1) == 1
    assert ped.global_to_binary(2) == 2
    assert ped.to_dict() == {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"],
                             "global_ids": [1, 2]}
    for value in [-1, 2, True, 0.0]:
        with pytest.raises(ValueError):
            ped.local_to_global(value)
    for value in [-1, 3, True, 1.0]:
        with pytest.raises(ValueError):
            ped.binary_to_global(value)
    with pytest.raises(ValueError):
        ped.global_to_local(0)


def test_reordered_groups_classes_and_nondefault_objects_preserve_order():
    objects = {"Cyclist": 2, "Car": 0, "Pedestrian": 1}
    groups = [{"name": "vru", "classes": ["Cyclist", "Pedestrian"]}, GROUPS[0]]
    saved = copy.deepcopy((groups, objects))
    vru, car = resolve(groups, objects)
    assert vru.global_ids == (2, 1)
    assert vru.global_to_local(2) == 0
    assert car.global_ids == (0,)
    assert (groups, objects) == saved
    nondefault = resolve([{"name": "bus", "classes": ["Bus"]}], {"Bus": 0})
    assert nondefault[0].global_ids == (0,)


@pytest.mark.parametrize("objects", [{}, {"Car": 1}, {"Car": True}, {"Car": 0.0},
                                    {"Car": -1}, {"Car": 0, "Bus": 0},
                                    {"": 0}, {1: 0}, []])
def test_invalid_global_class_maps(objects):
    with pytest.raises(ValueError):
        resolve(GROUPS, objects)


@pytest.mark.parametrize("groups", [None, [], {},
    [{"name": "car", "classes": []}],
    [{"name": "car", "classes": ["Car"]}],
    [{"name": "car", "classes": ["Van", "Pedestrian", "Cyclist"]}],
    [{"name": "all", "classes": ["Car", "Car", "Pedestrian", "Cyclist"]}],
    [GROUPS[0], {"name": "car", "classes": ["Pedestrian", "Cyclist"]}],
    [GROUPS[0], {"name": "others", "classes": ["Car", "Pedestrian", "Cyclist"]}],
    [{"name": "all", "classes": "Car"}],
    [{"name": "all", "classes": ["Car", "Pedestrian", "Cyclist"], "use_iou": True}],
])
def test_invalid_partitions_and_per_group_options(groups):
    with pytest.raises(ValueError):
        resolve(groups)


@pytest.mark.parametrize("name", ["Car", "1car", "car.x", "car-x", "", "_car", "items", "training"])
def test_invalid_or_reserved_names(name):
    with pytest.raises(ValueError):
        resolve([{**GROUPS[0], "name": name}, GROUPS[1]])


def test_contract_import_does_not_load_tensor_or_encoding_backends():
    script = "import sys; import detector.core.task_groups; assert not {'torch', 'numpy', 'numba'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
