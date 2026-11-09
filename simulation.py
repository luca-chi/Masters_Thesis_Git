"""
simulation.py

Version 1 merged generic simulation engine for patchy-particle systems.

Merged engine functionality:
    - Bio shared-grid placement for non-overlapping mixed initial states
    - Bio velocity diagnostics and debug-state checkpoints
    - Bio resize-stage rethermalization behavior
    - Lang selectable ConstantVolume/Bussi or Langevin integration method
    - Lang interaction generation behavior: no extra patch-patch WCA branch

Normal use:
    1. Edit prepare.py.
    2. Run python simulation.py.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import time
from typing import Any
from collections.abc import Callable

import gsd.hoomd
import hoomd
import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# Config + polymer specifications
# =============================================================================


def load_config(filename: str | None = None) -> dict[str, Any]:
    """Load simulation settings.

    Default behavior is to import make_config() from prepare.py,
    so day-to-day use is simply:
        1. Edit prepare.py.
        2. Run python simulation.py.

    A JSON filename is still accepted for backward compatibility.
    """

    if filename is not None:
        with open(filename, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        import importlib.util
        from pathlib import Path

        prepare_path = Path(__file__).with_name("prepare.py")
        spec = importlib.util.spec_from_file_location(
            "prepare",
            prepare_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import config from {prepare_path}")

        prepare_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_module)
        config = prepare_module.make_config()

    normalize_config(config)
    return config


def expand_repeating_sequence(
    sequence_unit: list[dict[str, Any]],
    chain_length: int,
) -> list[dict[str, Any]]:
    """Expand a short monomer sequence into an exact-length polymer chain.

    Example: [Crowder, 3PPA] with chain_length=30 produces 30 monomers:
    Crowder, 3PPA, Crowder, 3PPA, ... truncated exactly at 30.
    """

    if chain_length < 0:
        raise ValueError("chain_length must be non-negative.")
    if chain_length > 0 and len(sequence_unit) == 0:
        raise ValueError("sequence_unit/repeat_sequence cannot be empty when chain_length > 0.")

    expanded: list[dict[str, Any]] = []
    for i in range(chain_length):
        expanded.append(dict(sequence_unit[i % len(sequence_unit)]))
    return expanded


def normalize_polymer_sequences(config: dict[str, Any]) -> None:
    """Support compact repeated polymer definitions in config.json.

    Existing explicit `sequence` definitions still work. New configs may provide
    `sequence_unit` or `repeat_sequence` plus `chain_length`. The expanded
    sequence is written back into each polymer dictionary before registry/force
    setup so the rest of the engine remains unchanged.
    """

    for poly in config.get("polymers", []):
        repeat_sequence = poly.get("sequence_unit", poly.get("repeat_sequence"))
        chain_length = poly.get("chain_length")

        if repeat_sequence is not None or chain_length is not None:
            if repeat_sequence is None:
                repeat_sequence = poly.get("sequence", [])
            if chain_length is None:
                chain_length = len(repeat_sequence)
            poly["sequence"] = expand_repeating_sequence(
                sequence_unit=repeat_sequence,
                chain_length=int(chain_length),
            )


def find_polymer_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    for poly in config.get("polymers", []):
        if poly["name"] == name:
            return poly
    raise KeyError(f"No polymer named {name!r} in config['polymers'].")


def compute_polymer_anchor_positions(
    sim,
    poly: dict[str, Any],
    catalog: Catalog,
    margin: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (start, end) positions near opposite faces of the *current*
    (already expanded) box, along `direction`.
    The default is to place the polymer in the center of the box in the other two dimensions.
    Can choose between central placement or side placement (70% of the box size in the other two dimensions).
    """

    box = sim.state.box
    direction = poly.get("direction", "z")
    box_sizes = {"x": box.Lx, "y": box.Ly, "z": box.Lz}
    L = box_sizes[direction]

    sequence = poly["sequence"]
    if margin is None:
        first_size = catalog.template_size(sequence[0]["template"])
        last_size = catalog.template_size(sequence[-1]["template"])
        margin = 0.5 * max(first_size, last_size)
    half = 0.5 * L - margin

    placement = poly.get("placement", "center_line")
    if placement == "center_line":
        d = direction_vector(direction)
        return -half * d, half * d
    elif placement == "side_box":
        # Implies the positiong of the anchors at 70% of the maximum side values (L/2 for a cubic box) in the plane perpendicular to the direction vector.
        start_anchor = -half * direction_vector(direction)
        end_anchor = half * direction_vector(direction)
        for d in box_sizes.keys():
            if d != direction:
                start_anchor += 0.5 * box_sizes[d] * 0.7 * direction_vector(d)
                end_anchor += 0.5 * box_sizes[d] * 0.7 * direction_vector(d)
        return start_anchor, end_anchor
    elif placement == "custom_x_y":
        # Implies the direction of the polymer being "z", and custom x and y positioning
        custom_x_y = poly.get("placement_custom", [0.0, 0.0])
        start_anchor = np.array([custom_x_y[0], custom_x_y[1], -half])
        end_anchor = np.array([custom_x_y[0], custom_x_y[1], half])
        return start_anchor, end_anchor
    else:
        raise ValueError(f"Unsupported polymer placement: {placement}")


def build_polymer_bead_positions(
    poly: dict[str, Any],
    start_position: np.ndarray,
    end_position: np.ndarray,
) -> list[np.ndarray]:
    """Evenly space every monomer between start_position and end_position,
    so the chain always exactly spans the two anchors regardless of length.
    """

    n = len(poly["sequence"])
    if n == 0:
        return []
    if n == 1:
        return [0.5 * (start_position + end_position)]

    return [
        start_position + (end_position - start_position) * (i / (n - 1))
        for i in range(n)
    ]


def normalize_dynamics_config(config: dict[str, Any]) -> None:
    """Fill in defaults for the selectable MD integration method."""

    dynamics = config.setdefault("dynamics", {})
    method = dynamics.get("method", "constant_volume")

    aliases = {
        "constant": "constant_volume",
        "constantvolume": "constant_volume",
        "constant_volume_method": "constant_volume",
        "bussi": "constant_volume",
        "langevin_dynamics": "langevin",
    }
    method = aliases.get(str(method).strip().lower(), str(method).strip().lower())
    dynamics["method"] = method

    allowed = {"constant_volume", "langevin"}
    if method not in allowed:
        raise ValueError(
            "DYNAMICS['method'] must be 'constant_volume' or 'langevin'. "
            f"Got {dynamics.get('method')!r}."
        )

    dynamics.setdefault("langevin_default_gamma", 1.0)
    dynamics.setdefault("langevin_default_gamma_r", (1.0, 1.0, 1.0))
    dynamics.setdefault("langevin_tally_reservoir_energy", False)
    dynamics.setdefault("langevin_gamma_by_type", {})
    dynamics.setdefault("langevin_gamma_r_by_type", {})


def normalize_config(config: dict[str, Any]) -> None:
    normalize_polymer_sequences(config)
    normalize_dynamics_config(config)


# =============================================================================
# Catalog
# =============================================================================


class Catalog:
    """
    Translation layer between prepare.py and simulation.py.

    It knows:
        - primitive bead types
        - molecule templates
        - aliases used in populations and polymers
        - metadata for every actual HOOMD particle type
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.bead_types = config["bead_types"]
        self.templates = config["templates"]

    def bead(self, bead_name: str) -> dict[str, Any]:
        if bead_name not in self.bead_types:
            raise KeyError(f"Unknown bead type: {bead_name}")
        return self.bead_types[bead_name]

    def template(self, template_name: str) -> dict[str, Any]:
        if template_name not in self.templates:
            raise KeyError(f"Unknown template: {template_name}")
        return self.templates[template_name]

    def collect_alias_records(self) -> list[dict[str, Any]]:
        """
        Collect every template-alias pair used by populations and polymers.
        """

        records: list[dict[str, Any]] = []

        for pop in self.config["populations"]:
            records.append(
                {
                    "template": pop["template"],
                    "alias": pop["alias"],
                    "type_names": pop.get("type_names", {}),
                }
            )

        for poly in self.config["polymers"]:
            for mon in poly["sequence"]:
                records.append(
                    {
                        "template": mon["template"],
                        "alias": mon["alias"],
                        "type_names": mon.get("type_names", {}),
                    }
                )

        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for rec in records:
            key = (
                rec["template"],
                rec["alias"],
                json.dumps(rec.get("type_names", {}), sort_keys=True),
            )
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        return unique

    def actual_type_for_bead(
        self,
        template_name: str,
        alias: str,
        bead_name: str,
        type_names: dict[str, str] | None = None,
        is_core: bool = False,
        is_single_bead: bool = False,
    ) -> str:
        """
        Return the actual HOOMD type name for a bead in a template instance.

        type_names allows prepare.py to preserve old names like:
            core  -> 3PP
            PatchA -> 3PP_patch
        """

        type_names = type_names or {}

        if is_single_bead:
            return type_names.get("bead", alias)

        if is_core:
            return type_names.get("core", f"{alias}_core")

        return type_names.get(bead_name, f"{alias}_{bead_name}")

    def alias_particle_types(
        self,
        template_name: str,
        alias: str,
        type_names: dict[str, str] | None = None,
    ) -> list[str]:
        template = self.template(template_name)
        type_names = type_names or {}

        if template["kind"] == "single_bead":
            return [
                self.actual_type_for_bead(
                    template_name=template_name,
                    alias=alias,
                    bead_name=template["bead"],
                    type_names=type_names,
                    is_single_bead=True,
                )
            ]

        if template["kind"] == "patchy":
            types = [
                self.actual_type_for_bead(
                    template_name=template_name,
                    alias=alias,
                    bead_name=template["core"],
                    type_names=type_names,
                    is_core=True,
                )
            ]

            for patch_name in template["patches"]:
                patch_type = self.actual_type_for_bead(
                    template_name=template_name,
                    alias=alias,
                    bead_name=patch_name,
                    type_names=type_names,
                )
                if patch_type not in types:
                    types.append(patch_type)

            return types

        raise ValueError(f"Unknown template kind: {template['kind']}")

    def anchor_type(
        self,
        template_name: str,
        alias: str,
        type_names: dict[str, str] | None = None,
    ) -> str:
        template = self.template(template_name)
        type_names = type_names or {}

        if template["kind"] == "single_bead":
            return self.actual_type_for_bead(
                template_name=template_name,
                alias=alias,
                bead_name=template["bead"],
                type_names=type_names,
                is_single_bead=True,
            )

        if template["kind"] == "patchy":
            return self.actual_type_for_bead(
                template_name=template_name,
                alias=alias,
                bead_name=template["core"],
                type_names=type_names,
                is_core=True,
            )

        raise ValueError(f"Unknown template kind: {template['kind']}")

    def type_metadata_for_alias(
        self,
        template_name: str,
        alias: str,
        type_names: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        template = self.template(template_name)
        type_names = type_names or {}
        metadata: dict[str, dict[str, Any]] = {}

        if template["kind"] == "single_bead":
            bead_name = template["bead"]
            bead = self.bead(bead_name)
            type_name = self.actual_type_for_bead(
                template_name=template_name,
                alias=alias,
                bead_name=bead_name,
                type_names=type_names,
                is_single_bead=True,
            )
            metadata[type_name] = self._metadata_from_bead(bead_name, bead)
            return metadata

        if template["kind"] == "patchy":
            core_name = template["core"]
            core = self.bead(core_name)
            core_type = self.actual_type_for_bead(
                template_name=template_name,
                alias=alias,
                bead_name=core_name,
                type_names=type_names,
                is_core=True,
            )
            metadata[core_type] = self._metadata_from_bead(core_name, core)

            for patch_name in template["patches"]:
                patch = self.bead(patch_name)
                patch_type = self.actual_type_for_bead(
                    template_name=template_name,
                    alias=alias,
                    bead_name=patch_name,
                    type_names=type_names,
                )
                metadata[patch_type] = self._metadata_from_bead(patch_name, patch)

            return metadata

        raise ValueError(f"Unknown template kind: {template['kind']}")

    def _metadata_from_bead(
        self,
        bead_name: str,
        bead: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "bead_type": bead_name,
            "role": bead["role"],
            "sigma": float(bead["sigma"]),
            "mass": float(bead["mass"]),
            "patch_group": bead.get("patch_group"),
        }

    def build_type_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}

        for record in self.collect_alias_records():
            metadata.update(
                self.type_metadata_for_alias(
                    template_name=record["template"],
                    alias=record["alias"],
                    type_names=record.get("type_names", {}),
                )
            )

        return metadata

    def template_size(self, template_name: str) -> float:
        template = self.template(template_name)

        if template["kind"] == "single_bead":
            return float(self.bead(template["bead"])["sigma"])

        if template["kind"] == "patchy":
            return float(self.bead(template["core"])["sigma"])

        raise ValueError(f"Unknown template kind: {template['kind']}")


# =============================================================================
# Registry
# =============================================================================


class TypeRegistry:
    def __init__(self):
        self.particle_types: list[str] = []
        self.bond_types: list[str] = []
        self.angle_types: list[str] = []

    def register_particle_type(self, name: str) -> int:
        if name not in self.particle_types:
            self.particle_types.append(name)
        return self.particle_types.index(name)

    def register_bond_type(self, name: str) -> int:
        if name not in self.bond_types:
            self.bond_types.append(name)
        return self.bond_types.index(name)

    def register_angle_type(self, name: str) -> int:
        if name not in self.angle_types:
            self.angle_types.append(name)
        return self.angle_types.index(name)

    def particle_typeid(self, name: str) -> int:
        if name not in self.particle_types:
            raise KeyError(f"Unknown particle type: {name}")
        return self.particle_types.index(name)

    def bond_typeid(self, name: str) -> int:
        if name not in self.bond_types:
            raise KeyError(f"Unknown bond type: {name}")
        return self.bond_types.index(name)

    def angle_typeid(self, name: str) -> int:
        if name not in self.angle_types:
            raise KeyError(f"Unknown angle type: {name}")
        return self.angle_types.index(name)


def patch_bond_type(alias: str) -> str:
    return f"{alias}_core_patch_bond"


def patch_angle_type(alias: str) -> str:
    return f"{alias}_patch_core_patch_angle"


def build_registry(config: dict[str, Any], catalog: Catalog) -> TypeRegistry:
    registry = TypeRegistry()

    for record in catalog.collect_alias_records():
        for type_name in catalog.alias_particle_types(
            template_name=record["template"],
            alias=record["alias"],
            type_names=record.get("type_names", {}),
        ):
            registry.register_particle_type(type_name)

    for record in catalog.collect_alias_records():
        template = catalog.template(record["template"])
        alias = record["alias"]

        if template["kind"] == "patchy":
            registry.register_bond_type(patch_bond_type(alias))

            if len(template["patches"]) >= 2:
                registry.register_angle_type(patch_angle_type(alias))

    for poly in config["polymers"]:
        if not poly["backbone_fixed"]:
            continue
        
        if len(poly["sequence"]) >= 2:
            registry.register_bond_type(poly["backbone_bond_type"])

        if len(poly["sequence"]) >= 3:
            registry.register_angle_type(poly["backbone_angle_type"])

    return registry


# =============================================================================
# Geometry
# =============================================================================


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / norm


def default_geometry_for_patch_count(patch_count: int) -> str:
    defaults = {
        2: "linear",
        3: "trigonal_planar",
        4: "tetrahedral",
        5: "trigonal_bipyramidal",
        6: "octahedral",
        7: "pentagonal_bipyramidal",
        8: "cube",
    }
    if patch_count not in defaults:
        raise ValueError("Supported patch counts are 2 through 8.")
    return defaults[patch_count]


def regular_polygon_xy(count: int, z: float = 0.0, phase: float = 0.0) -> list[list[float]]:
    return [
        [float(np.cos(phase + 2.0 * np.pi * i / count)),
         float(np.sin(phase + 2.0 * np.pi * i / count)),
         float(z)]
        for i in range(count)
    ]


def normalized_positions(vectors: list[list[float]], distance: float) -> np.ndarray:
    return distance * np.array([normalize(np.asarray(v, dtype=float)) for v in vectors])


def supported_patch_geometries() -> dict[int, list[str]]:
    return {
        2: ["linear", "linear_hor", "bent_90", "bent_120"],
        3: ["trigonal_planar", "trigonal_pyramid", "t_shape"],
        4: ["tetrahedral", "square_planar"],
        5: ["trigonal_bipyramidal", "square_pyramidal", "pentagonal_planar"],
        6: ["octahedral", "hexagonal_planar", "trigonal_prism"],
        7: ["pentagonal_bipyramidal", "capped_octahedral", "capped_trigonal_prism"],
        8: ["cube", "square_antiprism", "bicapped_octahedral"],
    }


def patch_positions(
    patch_count: int,
    distance: float,
    geometry: str = "auto",
    custom_patch_positions: list[list[float]] | None = None,
) -> np.ndarray:
    """Return relative patch positions for supported patch counts/geometries."""

    if geometry == "auto":
        geometry = default_geometry_for_patch_count(patch_count)

    if geometry == "custom":
        if custom_patch_positions is None:
            raise ValueError("custom_patch_positions must be provided for custom geometry.")
        arr = np.asarray(custom_patch_positions, dtype=float)
        if arr.shape != (patch_count, 3):
            raise ValueError(f"custom_patch_positions must have shape ({patch_count}, 3).")
        return normalized_positions(arr.tolist(), distance)

    allowed = supported_patch_geometries().get(patch_count)
    if allowed is None:
        raise ValueError("Supported patch counts are 2 through 8.")
    if geometry not in allowed:
        raise ValueError(
            f"Unsupported {patch_count}-patch geometry: {geometry}. "
            f"Available options: {allowed}, plus 'custom'."
        )

    if patch_count == 2:
        if geometry == "linear":
            vectors = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        elif geometry == "linear_hor":
            vectors = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        elif geometry == "bent_90":
            vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        else:  # bent_120
            vectors = [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0, 0.0]]
        return normalized_positions(vectors, distance)

    if patch_count == 3:
        if geometry == "trigonal_planar":
            vectors = regular_polygon_xy(3)
        elif geometry == "trigonal_pyramid":
            vectors = [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ]
        else:  # t_shape
            vectors = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        return normalized_positions(vectors, distance)

    if patch_count == 4:
        if geometry == "tetrahedral":
            vectors = [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        else:  # square_planar
            vectors = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        return normalized_positions(vectors, distance)

    if patch_count == 5:
        if geometry == "trigonal_bipyramidal":
            vectors = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] + regular_polygon_xy(3)
        elif geometry == "square_pyramidal":
            vectors = [[0.0, 0.0, 1.0]] + regular_polygon_xy(4, z=-0.35, phase=np.pi / 4.0)
        else:  # pentagonal_planar
            vectors = regular_polygon_xy(5)
        return normalized_positions(vectors, distance)

    if patch_count == 6:
        if geometry == "octahedral":
            vectors = [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ]
        elif geometry == "hexagonal_planar":
            vectors = regular_polygon_xy(6)
        else:  # trigonal_prism
            vectors = regular_polygon_xy(3, z=0.7) + regular_polygon_xy(3, z=-0.7, phase=np.pi / 3.0)
        return normalized_positions(vectors, distance)

    if patch_count == 7:
        if geometry == "pentagonal_bipyramidal":
            vectors = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] + regular_polygon_xy(5)
        elif geometry == "capped_octahedral":
            vectors = [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        else:  # capped_trigonal_prism
            vectors = regular_polygon_xy(3, z=0.7) + regular_polygon_xy(3, z=-0.7, phase=np.pi / 3.0) + [[0.0, 0.0, 1.6]]
        return normalized_positions(vectors, distance)

    if patch_count == 8:
        if geometry == "cube":
            vectors = [
                [1.0, 1.0, 1.0], [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0], [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
            ]
        elif geometry == "square_antiprism":
            vectors = regular_polygon_xy(4, z=0.7, phase=np.pi / 4.0) + regular_polygon_xy(4, z=-0.7)
        else:  # bicapped_octahedral
            vectors = [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0],
            ]
        return normalized_positions(vectors, distance)

    raise ValueError("Supported patch counts are 2 through 8.")


def auto_patch_angle(patch_count: int) -> float:
    if patch_count == 2:
        return np.pi
    if patch_count == 3:
        return 2.0944                       # 2*np.pi/3
    if patch_count == 4:
        return float(np.arccos(-1.0 / 3.0))
    if patch_count in (5, 6, 7, 8):
        return np.pi / 2.0
    return np.pi


# =============================================================================
# Placement
# =============================================================================


def box_lengths(config: dict[str, Any]) -> tuple[float, float, float]:
    box = config["box"]
    return (
        float(box["initial_box_x"]),
        float(box["initial_box_y"]),
        float(box["initial_box_z"]),
    )


def generate_positions(
    count: int,
    placement: str,
    spacing: float,
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Return the first `count` positions from a fresh placement grid.

    This function is kept for backward compatibility and for old scripts that
    call it directly. SnapshotBuilder now uses PlacementManager below, which
    consumes positions from a shared grid so later populations do not restart
    at the same coordinates as earlier populations.
    """

    if count <= 0 or placement == "none":
        return []

    positions = generate_position_candidates(placement, spacing, config)

    if len(positions) < count:
        raise ValueError(
            f"Not enough positions for {placement}. Requested {count}, "
            f"capacity {len(positions)}, spacing {spacing}."
        )

    return positions[:count]


def generate_position_candidates(
    placement: str,
    spacing: float,
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Return every grid candidate for a placement region.

    The returned list is deliberately not truncated. PlacementManager consumes
    this list progressively: after one species receives positions, those slots
    are removed from the available pool before the next species is placed.
    """

    if placement == "none":
        return []

    if placement == "full_box":
        return full_box_candidate_grid(spacing, config)

    if placement == "central_slab":
        return central_slab_candidate_grid(spacing, config)

    if placement == "outside_slab":
        return outside_slab_candidate_grid(spacing, config)

    if placement == "center_line":
        return center_line_candidate_positions(spacing, config)

    raise ValueError(f"Unknown placement: {placement}")


def full_box_candidate_grid(spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    Lx, Ly, Lz = box_lengths(config)
    return grid_in_bounds_no_check(
        spacing=spacing,
        x_min=-Lx / 2,
        x_max=Lx / 2,
        y_min=-Ly / 2,
        y_max=Ly / 2,
        z_min=-Lz / 2,
        z_max=Lz / 2,
    )


def full_box_grid(count: int, spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    positions = full_box_candidate_grid(spacing, config)

    if len(positions) < count:
        raise ValueError(
            f"Not enough grid positions for full_box. Requested {count}, capacity {len(positions)}, "
            f"spacing {spacing}."
        )

    return positions[:count]


def central_slab_candidate_grid(spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    Lx, Ly, Lz = box_lengths(config)
    xratio = float(config["box"]["xratio"])
    Lx_slab = Lx * xratio

    return grid_in_bounds_no_check(
        spacing=spacing,
        x_min=-Lx_slab / 2,
        x_max=Lx_slab / 2,
        y_min=-Ly / 2,
        y_max=Ly / 2,
        z_min=-Lz / 2,
        z_max=Lz / 2,
    )


def central_slab_grid(count: int, spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    positions = central_slab_candidate_grid(spacing, config)

    if len(positions) < count:
        raise ValueError(
            f"Not enough grid positions for central_slab. Requested {count}, capacity {len(positions)}, "
            f"spacing {spacing}."
        )

    return positions[:count]


def outside_slab_side_grids(
    spacing: float,
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    Lx, Ly, Lz = box_lengths(config)
    xratio = float(config["box"]["xratio"])
    Lx_slab = Lx * xratio

    left = grid_in_bounds_no_check(
        spacing=spacing,
        x_min=-Lx / 2,
        x_max=-Lx_slab / 2,
        y_min=-Ly / 2,
        y_max=Ly / 2,
        z_min=-Lz / 2,
        z_max=Lz / 2,
    )

    right = grid_in_bounds_no_check(
        spacing=spacing,
        x_min=Lx_slab / 2,
        x_max=Lx / 2,
        y_min=-Ly / 2,
        y_max=Ly / 2,
        z_min=-Lz / 2,
        z_max=Lz / 2,
    )

    return left, right


def outside_slab_candidate_grid(spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    left, right = outside_slab_side_grids(spacing, config)

    # Interleave the two outside regions so that sequential shared-grid
    # consumption does not fill the entire left side before touching the right.
    positions: list[np.ndarray] = []
    for i in range(max(len(left), len(right))):
        if i < len(left):
            positions.append(left[i])
        if i < len(right):
            positions.append(right[i])

    return positions


def outside_slab_grid(count: int, spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    # Preserve the original fresh-grid behavior for direct calls to this helper:
    # roughly half the requested objects are placed on the left side and half on
    # the right side. SnapshotBuilder uses PlacementManager instead.
    left, right = outside_slab_side_grids(spacing, config)

    positions = []
    left_target = count // 2
    positions.extend(left[:left_target])
    positions.extend(right[: count - len(positions)])

    if len(positions) < count:
        remaining_pool = left[left_target:] + right[count - left_target :]
        positions.extend(remaining_pool[: count - len(positions)])

    if len(positions) < count:
        raise ValueError(
            f"Not enough outside_slab positions. Requested {count}, got {len(positions)}."
        )

    return positions


def center_line_candidate_positions(spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    Lx, Ly, Lz = box_lengths(config)
    ny = max(1, int(Ly // spacing))
    nz = max(1, int(Lz // spacing))

    ys = np.linspace(-Ly / 2 + spacing / 2, Ly / 2 - spacing / 2, ny)
    zs = np.linspace(-Lz / 2 + spacing / 2, Lz / 2 - spacing / 2, nz)

    positions = []
    for y in ys:
        for z in zs:
            positions.append(np.array([-0.5 * spacing, y, z], dtype=float))

    return positions


def center_line_positions(count: int, spacing: float, config: dict[str, Any]) -> list[np.ndarray]:
    if count == 1:
        return [np.array([-0.5 * spacing, 0.0, 0.0], dtype=float)]

    positions = center_line_candidate_positions(spacing, config)

    if len(positions) < count:
        raise ValueError(
            f"Not enough center_line positions. Requested {count}, capacity {len(positions)}."
        )

    return positions[:count]


def grid_in_bounds(
    count: int,
    spacing: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    label: str,
) -> list[np.ndarray]:
    positions = grid_in_bounds_no_check(
        spacing=spacing,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    if len(positions) < count:
        raise ValueError(
            f"Not enough grid positions for {label}. Requested {count}, capacity {len(positions)}, "
            f"spacing {spacing}."
        )

    return positions[:count]


def grid_in_bounds_no_check(
    spacing: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> list[np.ndarray]:
    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    nx = max(0, int(Lx // spacing))
    ny = max(0, int(Ly // spacing))
    nz = max(0, int(Lz // spacing))

    if nx == 0 or ny == 0 or nz == 0:
        return []

    xs = np.linspace(x_min + spacing / 2, x_max - spacing / 2, nx)
    ys = np.linspace(y_min + spacing / 2, y_max - spacing / 2, ny)
    zs = np.linspace(z_min + spacing / 2, z_max - spacing / 2, nz)

    positions = []
    for x in xs:
        for y in ys:
            for z in zs:
                positions.append(np.array([x, y, z], dtype=float))

    return positions


def direction_vector(direction: str) -> np.ndarray:
    if direction == "x":
        return np.array([1.0, 0.0, 0.0])
    if direction == "y":
        return np.array([0.0, 1.0, 0.0])
    if direction == "z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unknown direction: {direction}")


def resolve_placement_spacing(
    record: dict[str, Any],
    automatic_spacing: float,
    default_multiplier: float,
) -> float:
    """Resolve user-controlled placement spacing for populations/polymers.

    Supported keys in prepare.py dictionaries:
        placement_spacing: absolute spacing, highest priority.
        spacing: legacy/short alias for placement_spacing.
        placement_spacing_multiplier: multiplier for automatic_spacing.
        spacing_multiplier: legacy/short alias for placement_spacing_multiplier.
    """

    absolute_spacing = record.get("placement_spacing", record.get("spacing"))
    if absolute_spacing is not None:
        spacing = float(absolute_spacing)
    else:
        multiplier = float(
            record.get(
                "placement_spacing_multiplier",
                record.get("spacing_multiplier", default_multiplier),
            )
        )
        spacing = float(automatic_spacing) * multiplier

    if spacing <= 0.0:
        raise ValueError(f"Placement spacing must be positive. Got {spacing}.")

    return spacing


# =============================================================================
# Resolving placement overlapping after simulation start
# =============================================================================


def _union_find_components(
    snapshot,
    n_particles: int,
) -> tuple[dict[int, list[int]], Callable[[int], int]]:
    """Group particle tags into connected components using the bond
    topology (core-patch bonds, backbone bonds, etc). Each component is one
    physically bonded molecule/instance."""

    parent = list(range(n_particles))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if snapshot.bonds.N > 0:
        bond_group = np.asarray(snapshot.bonds.group, dtype=int)
        for a, b in bond_group:
            union(int(a), int(b))

    components: dict[int, list[int]] = {}
    for tag in range(n_particles):
        root = find(tag)
        components.setdefault(root, []).append(tag)

    return components, find


def resolve_polymer_overlaps_by_relocating_existing(
    sim,
    catalog: Catalog,
    new_tags: list[int],
    output_log: logging.Logger | None = None,
    irrelevant_roles: set[str] = {"patch"},
    overlap_threshold_factor: float = 0.5,
    grid_spacing: float = 1.0
) -> dict[str, Any]:
    """Detect existing particles overlapping the newly-added particles
    (new_tags) below a critical distance, then relocate the ENTIRE bonded
    molecule each flagged particle belongs to (e.g. core + all its patches),
    rigidly translating it -- preserving internal bond/angle geometry
    exactly -- onto a free slot of a fixed grid_spacing-unit grid spanning
    the current box. Grid slots are scanned starting from the
    (-Lx/2, -Ly/2, -Lz/2) corner, in x-major order, taking the first slot
    where the WHOLE relocated molecule has no critical overlap with any
    other currently-placed particle (bulk, unmoved existing, previously
    relocated molecules, or the new polymer itself).

    Velocities of relocated particles are resampled at kT rather than kept,
    since a molecule teleported across the box lands in a different local
    environment -- its old velocity vector has no correlated meaning at the
    new location, so a fresh Maxwell-Boltzmann draw is the physically
    sensible choice. Set resample_velocities=False to keep original
    velocities instead.
    """

    snap = sim.state.get_snapshot()
    is_root = not hasattr(snap, "communicator") or snap.communicator.rank == 0
    if not is_root:
        sim.state.set_snapshot(snap)
        return {"relocated_tags": []}

    positions = np.asarray(snap.particles.position, dtype=float).copy()
    diameters = np.asarray(snap.particles.diameter, dtype=float).copy()
    N = int(snap.particles.N)

    box = sim.state.box
    L = np.array([box.Lx, box.Ly, box.Lz], dtype=float)

    new_tag_set = set(new_tags)
    existing_tags = [t for t in range(N) if t not in new_tag_set]

    # --- Step 1: existing particles overlapping the new polymer except for the irrelevant_roles.
    type_metadata = catalog.build_type_metadata()
    type_names_list = list(snap.particles.types)

    def _particle_role(tag: int) -> str:
        type_name = type_names_list[int(snap.particles.typeid[tag])]
        return type_metadata.get(type_name, {}).get("role", "unknown")
    
    new_tags_filtered = [t for t in new_tags if _particle_role(t) not in irrelevant_roles]
    existing_tags_filtered = [t for t in existing_tags if _particle_role(t) not in irrelevant_roles]
    if not existing_tags_filtered or not new_tags_filtered:
        return {"relocated_tags": []}

    tree = cKDTree(np.mod(positions[existing_tags_filtered], L), boxsize=L)          # Have to shift to a domain of [0, L]!
    max_search_r = overlap_threshold_factor * float(np.max(diameters))

    violation_existing_tags: set[int] = set()
    for new_tag in new_tags_filtered:
        pos = np.mod(positions[new_tag], L)
        nearby_local = tree.query_ball_point(pos, r=max_search_r)
        for local_idx in nearby_local:
            existing_tag = existing_tags_filtered[local_idx]
            rij = positions[new_tag] - positions[existing_tag]
            rij -= L * np.round(rij / L)
            r = float(np.linalg.norm(rij))
            threshold = overlap_threshold_factor * 0.5 * (
                diameters[new_tag] + diameters[existing_tag]
            )
            if r < threshold:
                violation_existing_tags.add(existing_tag)

    if not violation_existing_tags:
        if output_log is not None:
            output_log.debug("[OVERLAP] no critical overlaps found; nothing to relocate.")
        return {"relocated_tags": []}

    # --- Step 2: expand each flagged tag to its whole bonded molecule.
    components, find = _union_find_components(snap, N)
    molecules_to_move: dict[int, list[int]] = {}
    for tag in violation_existing_tags:
        root = find(tag)
        molecules_to_move.setdefault(root, components[root])

    if output_log is not None:
        output_log.debug(
            "[OVERLAP] %d flagged particle(s) -> %d whole molecule(s) to relocate",
            len(violation_existing_tags), len(molecules_to_move),
        )

    # --- Step 3: candidate grid, scanned from one box corner.
    grid_candidates = grid_in_bounds_no_check(
        spacing=grid_spacing,
        x_min=-L[0] / 2, x_max=L[0] / 2,
        y_min=-L[1] / 2, y_max=L[1] / 2,
        z_min=-L[2] / 2, z_max=L[2] / 2,
    )
    if not grid_candidates:
        raise ValueError(
            f"grid_spacing={grid_spacing} produces zero candidate slots for "
            f"box lengths {L.tolist()}. Use a smaller grid_spacing."
        )

    working_positions = positions  # mutated in place as molecules get placed
    all_relocated_tags: list[int] = []
    used_grid_indices: set[int] = set()

    for root, member_tags in molecules_to_move.items():
        anchor_tag = next((t for t in member_tags if _particle_role(t) in {"core", "reg_poly_core"}), member_tags[0])
        old_anchor_pos = working_positions[anchor_tag].copy()
        member_set = set(member_tags)

        # Only check core/crowder members of this molecule against other
        # core/crowder particles -- patches ride along via the rigid
        # translation but never gate whether a candidate slot is accepted.
        member_tags_checked = [t for t in member_tags if _particle_role(t) not in irrelevant_roles]

        other_tags_arr = np.array([t for t in range(N) if t not in member_set and _particle_role(t) not in irrelevant_roles])
        other_positions = working_positions[other_tags_arr]
        other_diameters = diameters[other_tags_arr]
        tree_other = cKDTree(np.mod(other_positions, L), boxsize=L)

        placed = False
        for grid_idx, candidate_anchor in enumerate(grid_candidates):
            if grid_idx in used_grid_indices:
                continue

            translation = candidate_anchor - old_anchor_pos
            ok = True

            for t in member_tags_checked:
                cand_pos = working_positions[t] + translation
                nearby_local = tree_other.query_ball_point(np.mod(cand_pos, L), r=max_search_r)
                for local_idx in nearby_local:
                    rij = cand_pos - other_positions[local_idx]
                    rij -= L * np.round(rij / L)
                    r = float(np.linalg.norm(rij))
                    threshold = overlap_threshold_factor * 0.5 * (
                        diameters[t] + other_diameters[local_idx]
                    )
                    if r < threshold:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                for t in member_tags:
                    working_positions[t] = working_positions[t] + translation
                used_grid_indices.add(grid_idx)
                all_relocated_tags.extend(member_tags)
                placed = True
                if output_log is not None:
                    output_log.debug(
                        "[OVERLAP] relocated molecule (anchor tag %d, %d particle(s)) "
                        "to grid slot %d at %s",
                        anchor_tag, len(member_tags), grid_idx, candidate_anchor,
                    )
                break

        if not placed:
            raise RuntimeError(
                f"Could not find a non-overlapping grid slot (spacing={grid_spacing}) "
                f"for molecule anchored at tag {anchor_tag} ({len(member_tags)} particles). "
                f"Try a finer grid_spacing, a lower overlap_threshold_factor, or check "
                f"whether the box is simply too dense to fit this many relocations."
            )

    # --- Step 4: commit the moved positions.
    for t in all_relocated_tags:
        wrapped_pos = working_positions[t] - L * np.round(working_positions[t] / L)
        snap.particles.position[t] = wrapped_pos
        snap.particles.image[t] = np.zeros(3, dtype=np.int32)

    sim.state.set_snapshot(snap)
    sim.run(0)

    # if resample_velocities and all_relocated_tags:
    #     kT = float(config["temperature"]["kT"])
    #     sim.state.thermalize_particle_momenta(
    #         filter=hoomd.filter.Tags(all_relocated_tags), kT=kT,
    #     )
    #     sim.run(0)

    if output_log is not None:
        output_log.debug(
            "[OVERLAP] relocation complete: %d molecule(s), %d particle(s) moved total",
            len(molecules_to_move), len(all_relocated_tags),
        )

    return {
        "relocated_tags": all_relocated_tags,
        "num_molecules_moved": len(molecules_to_move),
    }


# =============================================================================
# Snapshot builder
# =============================================================================


class PlacementManager:
    """Consume initial-placement grid slots across populations/polymers.

    Without this manager, every species calls generate_positions(...) from a
    fresh grid and receives the same first N coordinates. With shared_grid=True,
    each placement/spacing grid is created once and then depleted in order. This
    means species 1 gets slots 0..N-1, species 2 gets slots N..2N-1, etc.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        placement_config = config.get("placement", {})
        self.shared_grid = bool(placement_config.get("shared_grid", True))
        self._remaining_by_key: dict[tuple[str, float], list[np.ndarray]] = {}
        self._reserved_position_keys: set[tuple[float, float, float]] = set()
        self.cursor_increment: int = 0                     # For now useful only for inserting polymers with the full_box placement mode.

    def reserve(
        self,
        count: int,
        placement: str,
        spacing: float,
        label: str,
        poly: dict[str, Any] = None
    ) -> list[np.ndarray]:
        if count <= 0 or placement == "none":
            return []
        local_count = count

        if not self.shared_grid:
            return generate_positions(
                count=count,
                placement=placement,
                spacing=spacing,
                config=self.config,
            )

        key = self._grid_key(placement, spacing)
        if key not in self._remaining_by_key:
            self._remaining_by_key[key] = generate_position_candidates(
                placement=placement,
                spacing=spacing,
                config=self.config,
            )

        remaining = self._remaining_by_key[key]
        if len(remaining) < local_count:
            used = len(generate_position_candidates(placement, spacing, self.config)) - len(remaining)
            raise ValueError(
                f"Not enough shared placement slots for {label} using placement={placement!r}. "
                f"Requested {count}, remaining {len(remaining)}, already used {used}, "
                f"spacing {spacing}. Increase the box size, reduce counts, or reduce placement spacing."
            )

        if placement == "full_box" and poly is not None:
            # Case for reserving more positions based on the length of the polymer.
            # Positions are saved in form of (x, y, z).
            # Not best packing, but works for now. A better packing algorithm should be implemented.
            Lx, Ly, Lz = box_lengths(self.config)
            nx = max(0, int(Lx // spacing))
            ny = max(0, int(Ly // spacing))
            nz = max(0, int(Lz // spacing))

            pol_length = len(poly["sequence"])
            pol_direction = poly["direction"]
            positions = []
            extra_pos = []

            cursor = 0
            for _ in range(count):
                if pol_direction == "x":
                    if cursor in extra_pos:
                        cursor = extra_pos[-1] + 1
                    extra_pos.extend([cursor + i*ny*nz for i in range(pol_length)])
                elif pol_direction == "y":
                    if cursor in extra_pos:
                        cursor = extra_pos[-1] + 1
                    pos = remaining[cursor]
                    if pos[1] + (pol_length - 1) * spacing > Ly / 2 - spacing / 2:
                        cursor += self.cursor_increment
                        cursor = (cursor // (ny * nz) + 1) * ny * nz
                        cursor -= self.cursor_increment
                    extra_pos.extend([cursor + i*nz for i in range(pol_length)])
                else:
                    if cursor in extra_pos:
                        cursor = extra_pos[-1] + 1
                    pos = remaining[cursor]
                    if pos[2] + (pol_length - 1) * spacing > Lz / 2 - spacing / 2:
                        cursor += self.cursor_increment
                        cursor = (cursor // nz + 1) * nz
                        cursor -= self.cursor_increment
                    extra_pos.extend([cursor + i for i in range(pol_length)])

                if cursor >= len(remaining):
                    raise ValueError(
                        f"Not enough shared placement slots for {label} using placement={placement!r}. "
                        f"Requested {count}, remaining {len(remaining) - len(extra_pos)}, already used {len(extra_pos)}, "
                        f"spacing {spacing}. Increase the box size, reduce counts, or reduce placement spacing."
                    )
                pos = remaining[cursor]
                if self._position_key(pos) not in self._reserved_position_keys:
                    positions.append(pos)
                cursor += 1
            local_count = extra_pos[-1] + 1      # Converting the last index to count (number) of positions to reserve.
        else:
            # Also drop positions reserved by any other placement grid. This matters
            # when two placement modes or spacings happen to share exact coordinates.
            remaining[:] = [
                pos for pos in remaining
                if self._position_key(pos) not in self._reserved_position_keys
            ]
            positions = remaining[:local_count]

        for pos in remaining[:local_count]:
            self._reserved_position_keys.add(self._position_key(pos))
        del remaining[:local_count]
        self.cursor_increment += local_count
        return positions

    @staticmethod
    def _position_key(position: np.ndarray) -> tuple[float, float, float]:
        arr = np.asarray(position, dtype=float)
        return tuple(np.round(arr, 12).tolist())

    @staticmethod
    def _grid_key(placement: str, spacing: float) -> tuple[str, float]:
        # Rounded spacing avoids accidental key mismatches from harmless floating
        # point representation differences, while keeping genuinely different
        # grid spacings separate.
        return placement, round(float(spacing), 12)


class SnapshotBuilder:
    def __init__(self, config: dict[str, Any], catalog: Catalog, registry: TypeRegistry):
        self.config = config
        self.catalog = catalog
        self.registry = registry
        self.type_metadata = catalog.build_type_metadata()

        self.placement_manager = PlacementManager(config)

        self.snapshot = gsd.hoomd.Frame()
        self.initialize_snapshot()

    def initialize_snapshot(self) -> None:
        Lx, Ly, Lz = box_lengths(self.config)

        self.snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

        self.snapshot.particles.N = 0
        self.snapshot.particles.types = list(self.registry.particle_types)
        self.snapshot.particles.position = []
        self.snapshot.particles.typeid = []
        self.snapshot.particles.diameter = []
        self.snapshot.particles.mass = []

        self.snapshot.bonds.N = 0
        self.snapshot.bonds.types = list(self.registry.bond_types)
        self.snapshot.bonds.typeid = []
        self.snapshot.bonds.group = []

        self.snapshot.angles.N = 0
        self.snapshot.angles.types = list(self.registry.angle_types)
        self.snapshot.angles.typeid = []
        self.snapshot.angles.group = []

    def add_particle(self, type_name: str, position: np.ndarray) -> int:
        meta = self.type_metadata[type_name]

        index = self.snapshot.particles.N
        self.snapshot.particles.N += 1
        self.snapshot.particles.position.append(np.asarray(position, dtype=float))
        self.snapshot.particles.typeid.append(self.registry.particle_typeid(type_name))
        self.snapshot.particles.diameter.append(float(meta["sigma"]))
        self.snapshot.particles.mass.append(float(meta["mass"]))

        return index

    def add_bond(self, bond_type: str, i: int, j: int) -> None:
        self.snapshot.bonds.N += 1
        self.snapshot.bonds.typeid.append(self.registry.bond_typeid(bond_type))
        self.snapshot.bonds.group.append([int(i), int(j)])

    def add_angle(self, angle_type: str, i: int, j: int, k: int) -> None:
        self.snapshot.angles.N += 1
        self.snapshot.angles.typeid.append(self.registry.angle_typeid(angle_type))
        self.snapshot.angles.group.append([int(i), int(j), int(k)])

    def build(self) -> gsd.hoomd.Frame:
        self.build_populations()
        self.build_polymers()
        return self.snapshot

    def build_populations(self) -> None:
        for pop in self.config["populations"]:
            self.build_population(pop)

    def build_population(self, pop: dict[str, Any]) -> None:
        template_name = pop["template"]
        count = int(pop["count"])
        spacing = resolve_placement_spacing(
            record=pop,
            automatic_spacing=self.catalog.template_size(template_name),
            default_multiplier=1.05,
        )

        positions = self.placement_manager.reserve(
            count=count,
            placement=pop["placement"],
            spacing=spacing,
            label=f"population {pop['alias']}",
        )

        for pos in positions:
            self.build_template_instance(
                template_name=template_name,
                alias=pop["alias"],
                type_names=pop.get("type_names", {}),
                anchor_position=pos,
            )

    def build_polymers(self) -> None:
        for poly in self.config["polymers"]:
            self.build_polymer_species(poly)

    def build_polymer_species(self, poly: dict[str, Any]) -> None:
        spacing = resolve_placement_spacing(
            record=poly,
            automatic_spacing=self.estimate_polymer_size(poly),
            default_multiplier=1.0,
        )
        positions = self.placement_manager.reserve(
            count=int(poly["count"]),
            placement=poly["placement"],
            spacing=spacing,
            label=f"polymer {poly['name']}",
            poly=poly,
        )

        for start_pos in positions:
            self.build_one_polymer(poly, start_pos)

    def build_one_polymer(self, poly: dict[str, Any], start_position: np.ndarray) -> None:
        direction = direction_vector(poly["direction"])
        cursor = np.asarray(start_position, dtype=float)
        anchor_indices: list[int] = []

        for mon in poly["sequence"]:
            template_name = mon["template"]
            alias = mon["alias"]
            type_names = mon.get("type_names", {})

            anchor = self.build_template_instance(
                template_name=template_name,
                alias=alias,
                type_names=type_names,
                anchor_position=cursor,
            )
            anchor_indices.append(anchor)
            cursor = cursor + self.catalog.template_size(template_name) * direction

        self.add_polymer_backbone(poly, anchor_indices)

    def build_template_instance(
        self,
        template_name: str,
        alias: str,
        type_names: dict[str, str],
        anchor_position: np.ndarray,
    ) -> int:
        template = self.catalog.template(template_name)

        if template["kind"] == "single_bead":
            type_name = self.catalog.anchor_type(template_name, alias, type_names)
            return self.add_particle(type_name, anchor_position)

        if template["kind"] == "patchy":
            return self.build_patchy_instance(template_name, alias, type_names, anchor_position)

        raise ValueError(f"Unknown template kind: {template['kind']}")

    def build_patchy_instance(
        self,
        template_name: str,
        alias: str,
        type_names: dict[str, str],
        core_position: np.ndarray,
    ) -> int:
        template = self.catalog.template(template_name)

        core_type = self.catalog.anchor_type(template_name, alias, type_names)
        core_index = self.add_particle(core_type, core_position)

        patches = template["patches"]
        rel_positions = patch_positions(
            patch_count=len(patches),
            distance=float(template["patch_distance"]),
            geometry=template["geometry"],
            custom_patch_positions=template.get("custom_patch_positions"),
        )

        patch_indices: list[int] = []

        for patch_name, rel_pos in zip(patches, rel_positions):
            patch_type = self.catalog.actual_type_for_bead(
                template_name=template_name,
                alias=alias,
                bead_name=patch_name,
                type_names=type_names,
            )

            patch_index = self.add_particle(patch_type, core_position + rel_pos)
            patch_indices.append(patch_index)

            self.add_bond(patch_bond_type(alias), core_index, patch_index)

        if len(patch_indices) >= 2:
            for i, k in itertools.combinations(patch_indices, 2):
                self.add_angle(patch_angle_type(alias), i, core_index, k)

        return core_index

    def add_polymer_backbone(self, poly: dict[str, Any], anchor_indices: list[int]) -> None:
        if not poly["backbone_fixed"]:
            return None
        if len(anchor_indices) >= 2:
            for i in range(1, len(anchor_indices)):
                self.add_bond(poly["backbone_bond_type"], anchor_indices[i - 1], anchor_indices[i])

        if len(anchor_indices) >= 3:
            for i in range(2, len(anchor_indices)):
                self.add_angle(
                    poly["backbone_angle_type"],
                    anchor_indices[i - 2],
                    anchor_indices[i - 1],
                    anchor_indices[i],
                )

    def estimate_polymer_size(self, poly: dict[str, Any]) -> float:
        total = 0.0
        for mon in poly["sequence"]:
            total += self.catalog.template_size(mon["template"])
        return max(total, 1.0)


def write_initial_gsd(
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    filename: str,
) -> None:
    builder = SnapshotBuilder(config, catalog, registry)
    snapshot = builder.build()

    if os.path.isfile(filename):
        os.remove(filename)

    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(snapshot)

    print(f"Wrote {filename}")
    print(f"Particle types: {snapshot.particles.types}")
    print(f"Bond types:     {snapshot.bonds.types}")
    print(f"Angle types:    {snapshot.angles.types}")
    print(f"Total particles: {snapshot.particles.N}")
    print(f"Total bonds:     {snapshot.bonds.N}")
    print(f"Total angles:    {snapshot.angles.N}")


# =============================================================================
# Interaction generation
# =============================================================================


def pair_once(type_a: str, type_b: str) -> bool:
    return type_a <= type_b


def lorentz_sigma(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> float:
    return 0.5 * (float(meta_a["sigma"]) + float(meta_b["sigma"]))


def wca_cutoff(sigma: float) -> float:
    return (2.0 ** (1.0 / 6.0)) * sigma


def expanded_mie_delta(n: float, m: float, sigma_sum: float) -> float:
    return ((n / m) * (sigma_sum ** (n - m))) ** (1.0 / (n - m))


def patch_pair_key(group_a: str, group_b: str) -> str:
    if group_a <= group_b:
        return f"{group_a}|{group_b}"
    return f"{group_b}|{group_a}"


def generate_interactions(
    config: dict[str, Any],
    catalog: Catalog,
) -> dict[str, list[dict[str, Any]]]:
    metadata = catalog.build_type_metadata()
    energy = config["energy"]
    defaults = config["interactions"]["defaults"]

    e0 = float(energy["e0"])
    e2 = float(energy["e2"])
    e1 = e0 - e2
    e_poly = float(energy["e_poly"])

    interactions: dict[str, list[dict[str, Any]]] = {
        "lj_repulsive": [],
        "lj_attractive": [],
        "expanded_mie": [],
    }

    for type_a, meta_a in metadata.items():
        for type_b, meta_b in metadata.items():
            if not pair_once(type_a, type_b):
                continue

            role_a = meta_a["role"]
            role_b = meta_b["role"]
            roles = {role_a, role_b}

            sigma = lorentz_sigma(meta_a, meta_b)

            if roles == {"core"} and defaults["enable_core_core_repulsion"]:
                interactions["lj_repulsive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e1,
                        "sigma": sigma,
                        "r_cut": wca_cutoff(sigma),
                    }
                )

            if roles == {"core"} and defaults["enable_core_core_attraction"]:
                interactions["lj_attractive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e2,
                        "sigma": sigma,
                        "r_cut": 2.5 * sigma,
                    }
                )

            if roles == {"core", "crowder"} and defaults["enable_core_crowder_repulsion"]:
                interactions["lj_repulsive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e1,
                        "sigma": sigma,
                        "r_cut": wca_cutoff(sigma),
                    }
                )

            if roles == {"crowder"} and defaults["enable_crowder_crowder_repulsion"]:
                interactions["lj_repulsive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e1,
                        "sigma": sigma,
                        "r_cut": wca_cutoff(sigma),
                    }
                )

            if roles == {"core", "reg_poly_core"} and defaults["enable_core_polycore_repulsion"]:
                interactions["lj_repulsive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e1,
                        "sigma": sigma,
                        "r_cut": wca_cutoff(sigma),
                    }
                )

            if roles == {"core", "reg_poly_core"} and defaults["enable_core_polycore_attraction"]:
                interactions["lj_attractive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e_poly,
                        "sigma": sigma,
                        "r_cut": 2.5 * sigma,
                    }
                )

            if roles == {"reg_poly_core"} and defaults["enable_core_core_repulsion"]:
                interactions["lj_repulsive"].append(
                    {
                        "type_a": type_a,
                        "type_b": type_b,
                        "epsilon": e1,
                        "sigma": sigma,
                        "r_cut": wca_cutoff(sigma),
                    }
                )

            if roles == {"patch"} and defaults["enable_patch_patch_mie"]:
                group_a = meta_a.get("patch_group")
                group_b = meta_b.get("patch_group")
                matrix_key = patch_pair_key(group_a, group_b)
                matrix = config["interactions"].get("patch_pair_matrix", {})

                if matrix_key in matrix:
                    eps = float(matrix[matrix_key].get("epsilon", config["energy"]["eps_weak"]))
                else:
                    eps = float(config["energy"]["eps_weak"])

                sigma_sum = float(meta_a["sigma"]) + float(meta_b["sigma"])

                if eps > 0.0:
                    interactions["expanded_mie"].append(
                        {
                            "type_a": type_a,
                            "type_b": type_b,
                            "epsilon": eps,
                            "sigma": sigma_sum,
                            "n": 2.0,
                            "m": 1.0,
                            "delta": -expanded_mie_delta(2.0, 1.0, sigma_sum),
                            "r_cut": sigma_sum,
                        }
                    )

    apply_overrides(interactions, config)
    return interactions


def apply_overrides(
    interactions: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    for override in config["interactions"].get("overrides", []):
        kind = override["kind"]

        if kind == "lj_repulsive":
            interactions["lj_repulsive"].append(
                {
                    "type_a": override["type_a"],
                    "type_b": override["type_b"],
                    "epsilon": float(override["epsilon"]),
                    "sigma": float(override["sigma"]),
                    "r_cut": float(override["r_cut"]),
                }
            )

        elif kind == "lj_attractive":
            interactions["lj_attractive"].append(
                {
                    "type_a": override["type_a"],
                    "type_b": override["type_b"],
                    "epsilon": float(override["epsilon"]),
                    "sigma": float(override["sigma"]),
                    "r_cut": float(override["r_cut"]),
                }
            )

        elif kind == "expanded_mie":
            interactions["expanded_mie"].append(
                {
                    "type_a": override["type_a"],
                    "type_b": override["type_b"],
                    "epsilon": float(override["epsilon"]),
                    "sigma": float(override["sigma"]),
                    "n": float(override.get("n", 2.0)),
                    "m": float(override.get("m", 1.0)),
                    "delta": float(override["delta"]),
                    "r_cut": float(override["r_cut"]),
                }
            )

        else:
            raise ValueError(f"Unknown override kind: {kind}")


# =============================================================================
# Force creation
# =============================================================================


def create_forces(
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
) -> list:
    if config["interactions"].get("use_default_interactions", True):
        interactions = generate_interactions(config, catalog)
    else:
        interactions = config["interactions"]["manual"]

    forces = []
    harmonic_angle = create_harmonic_angle_force(config, catalog)
    harmonic_bond = create_harmonic_bond_force(config, catalog)
    lj_repulsive = create_lj_force(interactions["lj_repulsive"], registry)
    lj_attractive = create_lj_force(interactions["lj_attractive"], registry)
    mie = create_expanded_mie_force(interactions["expanded_mie"], registry)

    for force in [harmonic_angle, harmonic_bond, lj_repulsive, lj_attractive, mie]:
        if force is not None:
            forces.append(force)

    return forces


def create_harmonic_bond_force(config: dict[str, Any], catalog: Catalog):
    harmonic = hoomd.md.bond.Harmonic()
    found = False

    for record in catalog.collect_alias_records():
        template = catalog.template(record["template"])
        alias = record["alias"]

        if template["kind"] == "patchy":
            harmonic.params[patch_bond_type(alias)] = dict(
                k=float(template["bond_k"]),
                r0=float(template["patch_distance"]),
            )
            found = True

    for poly in config["polymers"]:
        if not poly["backbone_fixed"]:
            continue
        if len(poly["sequence"]) >= 2:
            r0 = poly.get("backbone_bond_r0", "auto")
            if r0 == "auto":
                r0 = estimate_polymer_backbone_r0(poly, catalog)

            harmonic.params[poly["backbone_bond_type"]] = dict(
                k=float(poly["backbone_bond_k"]),
                r0=float(r0),
            )
            found = True

    return harmonic if found else None


def create_harmonic_angle_force(config: dict[str, Any], catalog: Catalog):
    harmonic = hoomd.md.angle.Harmonic()
    found = False

    for record in catalog.collect_alias_records():
        template = catalog.template(record["template"])
        alias = record["alias"]

        if template["kind"] == "patchy" and len(template["patches"]) >= 2:
            t0 = template.get("angle_t0", "auto")
            if t0 == "auto":
                t0 = auto_patch_angle(len(template["patches"]))

            harmonic.params[patch_angle_type(alias)] = dict(
                k=float(template["angle_k"]),
                t0=float(t0),
            )
            found = True

    for poly in config["polymers"]:
        if not poly["backbone_fixed"]:
            continue
        if len(poly["sequence"]) >= 3:
            harmonic.params[poly["backbone_angle_type"]] = dict(
                k=float(poly["backbone_angle_k"]),
                t0=float(poly["backbone_angle_t0"]),
            )
            found = True

    return harmonic if found else None


def estimate_polymer_backbone_r0(poly: dict[str, Any], catalog: Catalog) -> float:
    sequence = poly["sequence"]
    if len(sequence) < 2:
        return 1.0

    pair_distances = []
    for i in range(1, len(sequence)):
        size_a = catalog.template_size(sequence[i - 1]["template"])
        size_b = catalog.template_size(sequence[i]["template"])
        pair_distances.append(0.5 * (size_a + size_b))

    return float(np.mean(pair_distances))


def create_lj_force(interactions: list[dict[str, Any]], registry: TypeRegistry):
    if len(interactions) == 0:
        return None

    tree = hoomd.md.nlist.Tree(
        check_dist=True,
        buffer=0.12 / 4.0,
        exclusions=["body", "bond"],
    )

    lj = hoomd.md.pair.LJ(nlist=tree, default_r_cut=0)
    types = list(registry.particle_types)

    lj.params[(types, types)] = dict(epsilon=0.0, sigma=1.0)
    lj.r_cut[(types, types)] = 0.0

    for interaction in interactions:
        pair = (interaction["type_a"], interaction["type_b"])
        lj.params[pair] = dict(
            epsilon=float(interaction["epsilon"]),
            sigma=float(interaction["sigma"]),
        )
        lj.r_cut[pair] = float(interaction["r_cut"])

    return lj


def create_expanded_mie_force(interactions: list[dict[str, Any]], registry: TypeRegistry):
    if len(interactions) == 0:
        return None

    tree = hoomd.md.nlist.Tree(
        check_dist=True,
        buffer=0.12 / 4.0,
        exclusions=["body", "bond", "angle"],
    )

    mie = hoomd.md.pair.ExpandedMie(
        nlist=tree,
        default_r_cut=0,
        mode="shift",
    )

    types = list(registry.particle_types)
    mie.params[(types, types)] = dict(
        epsilon=0.0,
        sigma=1.0,
        n=1.0,
        m=1.0,
        delta=1.0,
    )
    mie.r_cut[(types, types)] = 0.0

    for interaction in interactions:
        pair = (interaction["type_a"], interaction["type_b"])
        mie.params[pair] = dict(
            epsilon=float(interaction["epsilon"]),
            sigma=float(interaction["sigma"]),
            n=float(interaction["n"]),
            m=float(interaction["m"]),
            delta=float(interaction["delta"]),
        )
        mie.r_cut[pair] = float(interaction["r_cut"])

    return mie


# =============================================================================
# Simulation setup
# =============================================================================


def make_device(config: dict[str, Any]):
    return hoomd.device.GPU(gpu_id=int(config["run"]["gpu_id"]))


def create_simulation(
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
):
    mode = config["run"]["mode"]

    if mode in ["create_and_expand", "create_expand_and_add_polymer", "create_no_expand"]:
        filename = config["run"]["fname"] + "_init.gsd"
        write_initial_gsd(config, catalog, registry, filename)
    elif mode in ["load_and_expand", "load_no_expand", "load_no_expand_with_polymer",
                  "phase_diagram", "phase_diagram_with_polymer"]:
        filename = config["run"]["loaded_file"]
    else:
        raise ValueError(f"Unknown run mode: {mode}")

    sim = hoomd.Simulation(
        device=make_device(config),
        seed=int(config["run"]["seed"]),
    )
    sim.create_state_from_gsd(filename=filename)
    return sim


def attach_integrator(
    sim,
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
) -> None:
    integrator = hoomd.md.Integrator(dt=float(config["time"]["dt"]))

    for force in create_forces(config, catalog, registry):
        integrator.forces.append(force)

    sim.operations.integrator = integrator



def _as_gamma_r_tuple(value) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        g = float(value)
        return (g, g, g)

    if len(value) != 3:
        raise ValueError(
            "Langevin rotational gamma values must be a scalar or a length-3 sequence. "
            f"Got {value!r}."
        )

    return tuple(float(x) for x in value)


def create_md_method(config: dict[str, Any], kT: float, particle_filter=None):
    """Create the configured HOOMD MD integration method.

    DYNAMICS["method"] in lang_prepare.py selects between:
        - constant_volume: ConstantVolume + Bussi thermostat, matching the
          original engine behavior.
        - langevin: hoomd.md.methods.Langevin with default gamma values and
          optional per-type overrides.
    """

    if particle_filter is None:
        particle_filter = hoomd.filter.All()

    dynamics = config.get("dynamics", {})
    method_name = str(dynamics.get("method", "constant_volume")).strip().lower()

    if method_name == "constant_volume":
        return hoomd.md.methods.ConstantVolume(
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=float(kT)),
            filter=particle_filter,
        )

    if method_name == "langevin":
        method = hoomd.md.methods.Langevin(
            filter=particle_filter,
            kT=float(kT),
            default_gamma=float(dynamics.get("langevin_default_gamma", 1.0)),
            default_gamma_r=_as_gamma_r_tuple(
                dynamics.get("langevin_default_gamma_r", (1.0, 1.0, 1.0))
            ),
            tally_reservoir_energy=bool(
                dynamics.get("langevin_tally_reservoir_energy", False)
            ),
        )

        for type_name, gamma in dynamics.get("langevin_gamma_by_type", {}).items():
            method.gamma[str(type_name)] = float(gamma)

        for type_name, gamma_r in dynamics.get("langevin_gamma_r_by_type", {}).items():
            method.gamma_r[str(type_name)] = _as_gamma_r_tuple(gamma_r)

        return method

    raise ValueError(
        "DYNAMICS['method'] must be 'constant_volume' or 'langevin'. "
        f"Got {method_name!r}."
    )



def _snapshot_array(array_like, dtype=None):
    arr = np.asarray(array_like)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _type_property_array(
    type_ids: np.ndarray,
    type_names: list[str],
    type_metadata: dict[str, dict[str, Any]],
    property_name: str,
    default_value: float,
) -> np.ndarray:
    values = np.empty(len(type_ids), dtype=float)
    for i, type_id in enumerate(type_ids):
        type_name = type_names[int(type_id)]
        values[i] = float(type_metadata.get(type_name, {}).get(property_name, default_value))
    return values


def _box_volume_from_state(sim) -> float:
    """Return the current simulation-box volume.

    This duplicates ThermodynamicQuantities.volume as a robust fallback and
    makes the volume available even if the thermodynamic compute has not yet
    populated its property cache.
    """

    box = sim.state.box
    return float(box.Lx * box.Ly * box.Lz)


def _safe_loggable_scalar(obj, quantity: str) -> float:
    """Read a scalar loggable quantity, returning NaN if unavailable.

    HOOMD may raise on some quantities before the first MD step, after certain
    initialization stages, or when pressure/virial data is not available. The
    GSD log still receives a stable key in every frame, which makes downstream
    analysis simpler.
    """

    try:
        return float(getattr(obj, quantity))
    except Exception:
        return float("nan")


def _write_observable_logs(frame: gsd.hoomd.Frame, sim, thermo=None) -> None:
    """Write scalar observables into frame.log for analysis.py.

    These keys are intentionally short and stable. They are normal GSD log
    chunks and can be read from each frame with:

        frame.log["observables/kinetic_temperature"]
        frame.log["observables/pressure"]
        frame.log["observables/volume"]
        frame.log["observables/potential_energy"]
        frame.log["observables/kinetic_energy"]

    Extra ThermodynamicQuantities fields are also written because they are often
    useful for sanity checks and decomposing the total kinetic energy.
    """

    # Always provide timestep and walltime-like bookkeeping when possible.
    frame.log["observables/timestep"] = np.array([int(sim.timestep)], dtype=np.int64)

    if thermo is None:
        frame.log["observables/volume"] = np.array(
            [_box_volume_from_state(sim)],
            dtype=np.float64,
        )
        return

    observable_names = [
        "kinetic_temperature",
        "pressure",
        "volume",
        "potential_energy",
        "kinetic_energy",
        "translational_kinetic_energy",
        "rotational_kinetic_energy",
        "degrees_of_freedom",
        "translational_degrees_of_freedom",
        "rotational_degrees_of_freedom",
        "num_particles",
    ]

    for quantity in observable_names:
        value = _safe_loggable_scalar(thermo, quantity)
        if quantity == "volume" and not np.isfinite(value):
            value = _box_volume_from_state(sim)

        # Short analysis-friendly key.
        frame.log[f"observables/{quantity}"] = np.array([value], dtype=np.float64)

        # HOOMD-style namespace key for users who prefer explicit provenance.
        frame.log[
            f"md/compute/ThermodynamicQuantities/{quantity}"
        ] = np.array([value], dtype=np.float64)

    # Simulation properties that are handy when plotting.
    frame.log["simulation/timestep"] = np.array([int(sim.timestep)], dtype=np.int64)
    try:
        frame.log["simulation/walltime"] = np.array([float(sim.walltime)], dtype=np.float64)
    except Exception:
        frame.log["simulation/walltime"] = np.array([float("nan")], dtype=np.float64)


def complete_frame_from_state(
    sim,
    type_metadata: dict[str, dict[str, Any]],
    thermo=None,
) -> gsd.hoomd.Frame | None:
    """Build a complete GSD frame from the current HOOMD state.

    HOOMD's normal trajectory writer may omit attributes that it considers
    static. OVITO can then show later trajectory frames without particle size
    information. This function writes diameter and mass arrays explicitly in
    every saved frame. It also explicitly writes HOOMD particle image vectors
    so periodic-boundary crossings are retained for correct trajectory
    unwrapping and MSD calculations. Scalar thermodynamic observables are
    written into frame.log so analysis.py can plot kinetic_temperature,
    pressure, volume, potential_energy, and kinetic_energy directly from the
    trajectory GSD.
    """

    snap = sim.state.get_snapshot()
    if hasattr(snap, "communicator") and snap.communicator.rank != 0:
        return None

    frame = gsd.hoomd.Frame()
    frame.configuration.step = int(sim.timestep)
    frame.configuration.box = list(snap.configuration.box)

    N = int(snap.particles.N)
    type_names = list(snap.particles.types)
    type_ids = _snapshot_array(snap.particles.typeid, dtype=np.uint32)

    frame.particles.N = N
    frame.particles.types = type_names
    frame.particles.typeid = type_ids
    frame.particles.position = _snapshot_array(snap.particles.position, dtype=float)

    image = _snapshot_array(snap.particles.image, dtype=np.int32)
    if image.shape != (N, 3):
        raise ValueError(
            f"Invalid HOOMD particle image array shape: {image.shape}; "
            f"expected ({N}, 3)."
        )
    frame.particles.image = image

    diameter = _snapshot_array(snap.particles.diameter, dtype=float)
    if diameter.shape[0] != N or np.allclose(diameter, 0.0):
        diameter = _type_property_array(type_ids, type_names, type_metadata, "sigma", 1.0)
    frame.particles.diameter = diameter

    mass = _snapshot_array(snap.particles.mass, dtype=float)
    if mass.shape[0] != N or np.allclose(mass, 0.0):
        mass = _type_property_array(type_ids, type_names, type_metadata, "mass", 1.0)
    frame.particles.mass = mass

    if hasattr(snap.particles, "velocity"):
        velocity = _snapshot_array(snap.particles.velocity, dtype=float)
        if velocity.shape[0] == N:
            frame.particles.velocity = velocity

    if hasattr(snap.particles, "orientation"):
        orientation = _snapshot_array(snap.particles.orientation, dtype=float)
        if orientation.shape[0] == N:
            frame.particles.orientation = orientation

    frame.bonds.N = int(snap.bonds.N)
    frame.bonds.types = list(snap.bonds.types)
    if frame.bonds.N > 0:
        frame.bonds.typeid = _snapshot_array(snap.bonds.typeid, dtype=np.uint32)
        frame.bonds.group = _snapshot_array(snap.bonds.group, dtype=np.uint32)

    frame.angles.N = int(snap.angles.N)
    frame.angles.types = list(snap.angles.types)
    if frame.angles.N > 0:
        frame.angles.typeid = _snapshot_array(snap.angles.typeid, dtype=np.uint32)
        frame.angles.group = _snapshot_array(snap.angles.group, dtype=np.uint32)

    _write_observable_logs(frame, sim=sim, thermo=thermo)

    return frame


def particle_volume_from_state(sim, type_metadata: dict[str, dict[str, Any]] | None = None) -> float:
    """Return total spherical particle volume from the current simulation state.

    All particles are included: cores, patches, crowders, and polymer beads.
    The volume convention is the HOOMD/OVITO sphere diameter convention:
        volume_i = (pi / 6) * diameter_i**3
    """

    snap = sim.state.get_snapshot()
    if hasattr(snap, "communicator") and snap.communicator.rank != 0:
        return 0.0

    N = int(snap.particles.N)
    if N == 0:
        return 0.0

    diameters = _snapshot_array(snap.particles.diameter, dtype=float)

    if diameters.shape[0] != N or np.allclose(diameters, 0.0):
        if type_metadata is None:
            raise ValueError(
                "Cannot compute volume fraction because particle diameters are missing "
                "and no type metadata was provided."
            )
        type_names = list(snap.particles.types)
        type_ids = _snapshot_array(snap.particles.typeid, dtype=np.uint32)
        diameters = _type_property_array(type_ids, type_names, type_metadata, "sigma", 1.0)

    return float(np.sum((math.pi / 6.0) * diameters**3))


def current_volume_fraction_from_state(
    sim,
    type_metadata: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Return the current all-particle volume fraction."""

    particle_volume = particle_volume_from_state(sim, type_metadata=type_metadata)
    box = sim.state.box
    box_volume = float(box.Lx * box.Ly * box.Lz)
    if box_volume <= 0.0:
        raise ValueError(f"Box volume must be positive. Got {box_volume}.")
    return particle_volume / box_volume


def normalize_gsd_open_mode(mode: str) -> str:
    """Convert HOOMD-style binary modes to modes accepted by gsd.hoomd.open.

    hoomd.write.GSD accepts modes like "wb" and "ab", but
    gsd.hoomd.open accepts "w" and "a". This helper lets the rest of
    the code use either convention safely.
    """

    mode_map = {
        "wb": "w",
        "w": "w",
        "xb": "x",
        "x": "x",
        "ab": "a",
        "a": "a",
        "rb": "r",
        "r": "r",
    }
    if mode not in mode_map:
        raise ValueError(
            f"Unsupported GSD mode {mode!r}. Use one of {sorted(mode_map)}."
        )
    return mode_map[mode]


def write_complete_state_gsd(
    sim,
    filename: str,
    type_metadata: dict[str, dict[str, Any]],
    mode: str = "w",
    thermo=None,
) -> None:
    frame = complete_frame_from_state(sim, type_metadata, thermo=thermo)
    if frame is None:
        return

    open_mode = normalize_gsd_open_mode(mode)
    with gsd.hoomd.open(name=filename, mode=open_mode) as gsd_file:
        gsd_file.append(frame)


class CompleteGSDTrajectoryAction(hoomd.custom.Action):
    """Custom writer that saves complete trajectory frames plus observables."""

    def __init__(
        self,
        sim,
        filename: str,
        type_metadata: dict[str, dict[str, Any]],
        thermo=None,
    ):
        super().__init__()
        self.sim = sim
        self.filename = filename
        self.type_metadata = type_metadata
        self.thermo = thermo

    def act(self, timestep):
        write_complete_state_gsd(
            sim=self.sim,
            filename=self.filename,
            type_metadata=self.type_metadata,
            mode="a",
            thermo=self.thermo,
        )


def add_output_writer(sim, config: dict[str, Any], elongation: bool = False):
    # Keep this logger as a returned object for compatibility with the rest of
    # the script. The custom complete-frame GSD writer below writes frame.log
    # directly, because this script intentionally does not use hoomd.write.GSD
    # for the main trajectory.
    logger = hoomd.logging.Logger()

    if elongation:
        file_prefix = config["run"]["fname"] + "elongation"
    else:
        file_prefix = f"elongation_kT_{config['temperature']['kT']}"

    # Pressure/virial terms are needed for thermo.pressure. HOOMD computes them
    # on GSD log-writing timesteps automatically when using hoomd.write.GSD, but
    # this script writes custom frames, so request pressure availability here.
    try:
        sim.always_compute_pressure = True
    except Exception:
        pass

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    thermo_quantities = [
        "kinetic_temperature",
        "pressure",
        "volume",
        "potential_energy",
        "kinetic_energy",
        "translational_kinetic_energy",
        "rotational_kinetic_energy",
        "degrees_of_freedom",
        "translational_degrees_of_freedom",
        "rotational_degrees_of_freedom",
        "num_particles",
        "pressure_tensor",
    ]
    logger.add(thermo, quantities=thermo_quantities)

    for force in sim.operations.integrator.forces:
        logger.add(force, quantities=["energies", "forces"])

    logger.add(sim, quantities=["timestep", "walltime"])

    filename = file_prefix + "_traj.gsd"
    type_metadata = Catalog(config).build_type_metadata()

    if os.path.isfile(filename):
        os.remove(filename)

    # Write frame 0 immediately, with particle diameters/masses explicitly saved.
    write_complete_state_gsd(
        sim=sim,
        filename=filename,
        type_metadata=type_metadata,
        mode="w",
        thermo=thermo,
    )

    # Then append complete frames periodically. This replaces hoomd.write.GSD for
    # the main trajectory because the built-in writer can omit static attributes
    # such as diameter from later frames.
    complete_gsd_writer = hoomd.write.CustomWriter(
        action=CompleteGSDTrajectoryAction(
            sim=sim,
            filename=filename,
            type_metadata=type_metadata,
            thermo=thermo,
        ),
        trigger=hoomd.trigger.Periodic(int(config["time"]["writer_steps"])),
    )
    sim.operations.writers.append(complete_gsd_writer)

    return logger


def save_sim(label: str, sim, config: dict[str, Any]) -> None:
    filename = config["run"]["fname"] + "_after_" + label + ".gsd"
    type_metadata = Catalog(config).build_type_metadata()
    write_complete_state_gsd(sim=sim, filename=filename, type_metadata=type_metadata, mode="w")


def filter_excluding_tags(tags: list[int]) -> "hoomd.filter.ParticleFilter":
    """Filter matching every particle except the given tags (e.g. the two
    frozen polymer ends)."""
    if not tags:
        return hoomd.filter.All()
    return hoomd.filter.SetDifference(hoomd.filter.All(), hoomd.filter.Tags(tags))


def append_polymer_to_state(
    sim,
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    poly: dict[str, Any],
    bead_positions: list[np.ndarray],
    output_log: logging.Logger | None = None,
) -> dict[str, list[int]]:
    """Append one polymer instance's beads/bonds/angles to the live sim
    state. Requires every particle/bond/angle type used by `poly` to already
    be registered in `registry`. Supports both single_bead and patchy monomers
    in the same sequence.
    bead_positions gives the CORE/anchor position for each monomer (same
    convention as build_polymer_bead_positions) -- for patchy monomers, patch
    positions are derived from the core position using that template's own
    geometry/patch_distance, exactly like SnapshotBuilder.build_patchy_instance
    does for the initial GSD build.
    The backbone bond/angle chain is built across monomer ANCHORS (core
    index for patchy monomers, the single bead's index otherwise) -- matching
    build_one_polymer's convention in SnapshotBuilder.
    Note: like the rest of this script, this assumes a single-rank/local GPU
    run. It does not attempt an MPI-safe broadcast of the new tags.
    """

    snap = sim.state.get_snapshot()
    is_root = not hasattr(snap, "communicator") or snap.communicator.rank == 0

    sequence = poly["sequence"]
    n_new = len(sequence)
    if n_new != len(bead_positions):
        raise ValueError(
            f"Expected {n_new} bead positions for polymer {poly['name']!r}, "
            f"got {len(bead_positions)}."
        )

    new_tags: list[int] = []       # every particle added (core + patches)
    anchor_tags: list[int] = []    # one per monomer, in sequence order
    all_bond_specs: list[tuple[str, int, int]] = []
    all_angle_specs: list[tuple[str, int, int, int]] = []

    if is_root:
        type_metadata = catalog.build_type_metadata()
        old_N = int(snap.particles.N)

        # --- Pass 1: figure out how many particle slots we need, and the
        # per-monomer layout, before touching the snapshot arrays.
        monomer_plans = []
        n_add = 0
        for mon in sequence:
            template_name = mon["template"]
            alias = mon["alias"]
            type_names = mon.get("type_names", {})
            template = catalog.template(template_name)

            if template["kind"] == "single_bead":
                type_name = catalog.anchor_type(template_name, alias, type_names)
                monomer_plans.append({
                    "kind": "single_bead",
                    "core_type": type_name,
                    "patch_types": [],
                })
                n_add += 1

            elif template["kind"] == "patchy":
                core_type = catalog.anchor_type(template_name, alias, type_names)
                patch_names = template["patches"]
                patch_types = [
                    catalog.actual_type_for_bead(
                        template_name=template_name, alias=alias,
                        bead_name=p, type_names=type_names,
                    )
                    for p in patch_names
                ]
                monomer_plans.append({
                    "kind": "patchy",
                    "template_name": template_name,
                    "alias": alias,
                    "core_type": core_type,
                    "patch_types": patch_types,
                    "geometry": template["geometry"],
                    "patch_distance": float(template["patch_distance"]),
                    "custom_patch_positions": template.get("custom_patch_positions")
                })
                n_add += 1 + len(patch_types)

            else:
                raise ValueError(f"Unknown template kind: {template['kind']}")

        snap.particles.N = old_N + n_add
        cursor = old_N

        # --- Pass 2: fill particle slots and collect bond/angle specs.
        def _init_slot(idx: int, type_name: str, position: np.ndarray) -> None:
            meta = type_metadata[type_name]
            snap.particles.typeid[idx] = registry.particle_typeid(type_name)
            snap.particles.position[idx] = position
            snap.particles.diameter[idx] = float(meta["sigma"])
            snap.particles.mass[idx] = float(meta["mass"])
            snap.particles.velocity[idx] = np.zeros(3)
            # snap.particles.image[idx] = np.zeros(3, dtype=np.int32)
            # snap.particles.charge[idx] = 0.0
            # snap.particles.body[idx] = -1
            # snap.particles.orientation[idx] = np.array([1.0, 0.0, 0.0, 0.0])
            # snap.particles.moment_inertia[idx] = np.zeros(3)
            # snap.particles.angmom[idx] = np.zeros(4)

        for mon, plan, core_pos in zip(sequence, monomer_plans, bead_positions):
            core_pos = np.asarray(core_pos, dtype=float)
            core_idx = cursor
            _init_slot(core_idx, plan["core_type"], core_pos)
            cursor += 1
            new_tags.append(core_idx)
            anchor_tags.append(core_idx)  # backbone connects to the core

            if plan["kind"] == "patchy":
                rel_positions = patch_positions(
                    patch_count=len(plan["patch_types"]),
                    distance=plan["patch_distance"],
                    geometry=plan["geometry"],
                    custom_patch_positions=plan["custom_patch_positions"],
                )

                patch_indices = []
                for patch_type, rel_pos in zip(plan["patch_types"], rel_positions):
                    patch_idx = cursor
                    _init_slot(patch_idx, patch_type, core_pos + rel_pos)
                    cursor += 1
                    new_tags.append(patch_idx)
                    patch_indices.append(patch_idx)

                    all_bond_specs.append(
                        (patch_bond_type(plan["alias"]), core_idx, patch_idx)
                    )

                if len(patch_indices) >= 2:
                    for i, k in itertools.combinations(patch_indices, 2):
                        all_angle_specs.append(
                            (patch_angle_type(plan["alias"]), i, core_idx, k)
                        )

        # --- Backbone bonds/angles across monomer anchors (cores).
        if poly["backbone_fixed"] and len(anchor_tags) >= 2:
            for i in range(1, len(anchor_tags)):
                all_bond_specs.append(
                    (poly["backbone_bond_type"], anchor_tags[i - 1], anchor_tags[i])
                )

        if poly["backbone_fixed"] and len(anchor_tags) >= 3:
            for i in range(2, len(anchor_tags)):
                all_angle_specs.append(
                    (poly["backbone_angle_type"], anchor_tags[i - 2], anchor_tags[i - 1], anchor_tags[i])
                )

        # --- Write bonds/angles in one resize each.
        old_bonds_N = int(snap.bonds.N)
        snap.bonds.N = old_bonds_N + len(all_bond_specs)
        for i, (bond_type, a, b) in enumerate(all_bond_specs):
            snap.bonds.typeid[old_bonds_N + i] = registry.bond_typeid(bond_type)
            snap.bonds.group[old_bonds_N + i] = [a, b]

        old_angles_N = int(snap.angles.N)
        snap.angles.N = old_angles_N + len(all_angle_specs)
        for i, (angle_type, a, b, c) in enumerate(all_angle_specs):
            snap.angles.typeid[old_angles_N + i] = registry.angle_typeid(angle_type)
            snap.angles.group[old_angles_N + i] = [a, b, c]

        # --- Defensive bounds check, same rationale as before.
        box = sim.state.box
        half = np.array([box.Lx, box.Ly, box.Lz]) / 2.0
        all_new_positions = np.asarray(
            [snap.particles.position[t] for t in new_tags], dtype=float
        )
        bad = np.where(np.any(np.abs(all_new_positions) > half + 1e-8, axis=1))[0]
        if len(bad) > 0:
            raise ValueError(
                f"Polymer {poly['name']!r}: particle(s) at local index {list(bad)} "
                f"(tags {[new_tags[i] for i in bad]}) fall outside the current box "
                f"half-lengths {half}. Check compute_polymer_anchor_positions()'s "
                f"`direction`/`margin`, and patch_distance for the monomer geometry."
            )

        if output_log is not None:
            output_log.debug(
                "Appended polymer %s: %d monomer(s), %d total particle(s) "
                "(tags %d..%d), anchors=%s, %d bond(s), %d angle(s)",
                poly["name"], len(anchor_tags), len(new_tags), new_tags[0], new_tags[-1],
                anchor_tags, len(all_bond_specs), len(all_angle_specs),
            )

    sim.state.set_snapshot(snap)
    return {"monomer_tags": anchor_tags, "all_tags": new_tags}


# =============================================================================
# Debugging
# =============================================================================


def debug_pos_F_printer(sim, output_log: logging.Logger, config: dict[str, Any], label: str = ""):
    if not config["output"].get("write_debug_forces", False):
        return

    N = int(config["output"].get("debug_particle_count", 10))
    snap = sim.state.get_snapshot()

    if snap.particles.N == 0:
        output_log.debug(f"[{label}] Snapshot has no particles.")
        return

    type_names = snap.particles.types
    type_ids = snap.particles.typeid
    diameters = snap.particles.diameter

    output_log.debug(f"[{label}] Particle info first {N}:")
    for i in range(min(N, snap.particles.N)):
        pos = snap.particles.position[i]
        type_name = type_names[type_ids[i]]
        size = diameters[i]
        output_log.debug(f"Particle {i} [{type_name}], size={size:.3f}, pos={pos}")

    for force in sim.operations.integrator.forces:
        output_log.debug(f"[{label}] Forces from {type(force).__name__}:")
        try:
            with force.gpu_local_force_arrays as arrays:
                for i in range(min(N, len(arrays.force))):
                    type_name = type_names[type_ids[i]]
                    size = diameters[i]
                    output_log.debug(
                        f"Particle {i} [{type_name}], size={size:.3f}, force={arrays.force[i]}"
                    )
        except Exception as err:
            output_log.debug(f"Could not access local force arrays: {err}")



def _safe_stat(values: np.ndarray, reducer, default: float = float("nan")) -> float:
    if values.size == 0:
        return default
    return float(reducer(values))


def _root_snapshot(sim):
    """Return a HOOMD snapshot on rank 0, otherwise None."""
    snap = sim.state.get_snapshot()
    if hasattr(snap, "communicator") and snap.communicator.rank != 0:
        return None
    return snap


def log_velocity_diagnostics(
    sim,
    output_log: logging.Logger,
    config: dict[str, Any],
    label: str,
) -> None:
    """Log direct velocity/kinetic-energy diagnostics from the HOOMD state.

    This is meant to distinguish three cases:
      1. HOOMD really has near-zero bead velocities.
      2. Bead velocities are nonzero, but core/COM velocities are tiny.
      3. The GSD/OVITO visualization is misleading.
    """

    if not config.get("output", {}).get("write_velocity_diagnostics", True):
        return

    snap = _root_snapshot(sim)
    if snap is None:
        return

    N = int(snap.particles.N)
    if N == 0:
        output_log.debug("[VEL %s] no particles", label)
        return

    type_metadata = Catalog(config).build_type_metadata()

    position = np.asarray(snap.particles.position, dtype=float)
    velocity = np.asarray(snap.particles.velocity, dtype=float)
    mass = np.asarray(snap.particles.mass, dtype=float)
    typeid = np.asarray(snap.particles.typeid, dtype=int)
    type_names = list(snap.particles.types)

    if velocity.shape != (N, 3):
        output_log.debug(
            "[VEL %s] velocity array has unexpected shape %s for N=%d",
            label,
            velocity.shape,
            N,
        )
        return

    if mass.shape[0] != N or np.allclose(mass, 0.0):
        mass = _type_property_array(
            type_ids=typeid,
            type_names=type_names,
            type_metadata=type_metadata,
            property_name="mass",
            default_value=1.0,
        )

    speed = np.linalg.norm(velocity, axis=1)
    kinetic_energy = 0.5 * np.sum(mass * np.sum(velocity * velocity, axis=1))
    dof_3n = max(1, 3 * N)
    dof_3n_minus_3 = max(1, 3 * N - 3)
    kT_from_3n = 2.0 * kinetic_energy / dof_3n
    kT_from_3n_minus_3 = 2.0 * kinetic_energy / dof_3n_minus_3

    total_mass = float(np.sum(mass))
    total_momentum = np.sum(mass[:, None] * velocity, axis=0)
    com_velocity = total_momentum / total_mass if total_mass > 0.0 else np.array([np.nan, np.nan, np.nan])

    box = sim.state.box
    half_lengths = np.array([box.Lx, box.Ly, box.Lz], dtype=float) / 2.0
    outside = np.any(np.abs(position) > (half_lengths + 1e-8), axis=1)

    output_log.debug(
        "[VEL %s] timestep=%d box=(%.6g, %.6g, %.6g) N=%d",
        label,
        int(sim.timestep),
        float(box.Lx),
        float(box.Ly),
        float(box.Lz),
        N,
    )
    output_log.debug(
        "[VEL %s] speed min/mean/median/rms/max = %.6g %.6g %.6g %.6g %.6g",
        label,
        _safe_stat(speed, np.min),
        _safe_stat(speed, np.mean),
        _safe_stat(speed, np.median),
        float(np.sqrt(np.mean(speed * speed))) if speed.size else float("nan"),
        _safe_stat(speed, np.max),
    )
    output_log.debug(
        "[VEL %s] zero-ish speed counts: <1e-12=%d, <1e-8=%d, <1e-6=%d out of %d",
        label,
        int(np.sum(speed < 1e-12)),
        int(np.sum(speed < 1e-8)),
        int(np.sum(speed < 1e-6)),
        N,
    )
    output_log.debug(
        "[VEL %s] KE=%.8g estimated_kT_3N=%.8g estimated_kT_3Nminus3=%.8g target_kT=%.8g",
        label,
        float(kinetic_energy),
        float(kT_from_3n),
        float(kT_from_3n_minus_3),
        float(config["temperature"]["kT"]),
    )
    output_log.debug(
        "[VEL %s] total_momentum=(%.6g, %.6g, %.6g) COM_velocity=(%.6g, %.6g, %.6g)",
        label,
        float(total_momentum[0]),
        float(total_momentum[1]),
        float(total_momentum[2]),
        float(com_velocity[0]),
        float(com_velocity[1]),
        float(com_velocity[2]),
    )
    output_log.debug(
        "[VEL %s] particles outside current primary box by coordinates = %d / %d",
        label,
        int(np.sum(outside)),
        N,
    )

    # Role-level statistics: core vs patch vs crowder.
    roles = []
    for tid in typeid:
        type_name = type_names[int(tid)]
        roles.append(type_metadata.get(type_name, {}).get("role", "unknown"))
    roles = np.asarray(roles, dtype=object)

    for role in sorted(set(roles.tolist())):
        mask = roles == role
        role_speed = speed[mask]
        role_mass = mass[mask]
        role_velocity = velocity[mask]
        role_ke = 0.5 * np.sum(role_mass * np.sum(role_velocity * role_velocity, axis=1))
        role_dof = max(1, 3 * int(np.sum(mask)))
        output_log.debug(
            "[VEL %s] role=%s count=%d speed mean/median/rms/max=%.6g %.6g %.6g %.6g KE=%.6g kT_3N=%.6g",
            label,
            role,
            int(np.sum(mask)),
            _safe_stat(role_speed, np.mean),
            _safe_stat(role_speed, np.median),
            float(np.sqrt(np.mean(role_speed * role_speed))) if role_speed.size else float("nan"),
            _safe_stat(role_speed, np.max),
            float(role_ke),
            float(2.0 * role_ke / role_dof),
        )

    # Type-level statistics help catch a single frozen type or core-only issue.
    for tid, type_name in enumerate(type_names):
        mask = typeid == tid
        if not np.any(mask):
            continue
        type_speed = speed[mask]
        output_log.debug(
            "[VEL %s] type=%s count=%d speed mean/median/max=%.6g %.6g %.6g zero<1e-8=%d",
            label,
            type_name,
            int(np.sum(mask)),
            _safe_stat(type_speed, np.mean),
            _safe_stat(type_speed, np.median),
            _safe_stat(type_speed, np.max),
            int(np.sum(type_speed < 1e-8)),
        )

    sample_count = int(config.get("output", {}).get("debug_particle_count", 10))
    for i in range(min(sample_count, N)):
        type_name = type_names[int(typeid[i])]
        output_log.debug(
            "[VEL %s] sample i=%d type=%s pos=(%.6g, %.6g, %.6g) vel=(%.6g, %.6g, %.6g) speed=%.6g mass=%.6g",
            label,
            i,
            type_name,
            float(position[i, 0]),
            float(position[i, 1]),
            float(position[i, 2]),
            float(velocity[i, 0]),
            float(velocity[i, 1]),
            float(velocity[i, 2]),
            float(speed[i]),
            float(mass[i]),
        )

    write_velocity_diagnostics_csv(
        config=config,
        label=label,
        timestep=int(sim.timestep),
        box_lengths=(float(box.Lx), float(box.Ly), float(box.Lz)),
        N=N,
        speed=speed,
        kinetic_energy=float(kinetic_energy),
        kT_from_3n=float(kT_from_3n),
        kT_from_3n_minus_3=float(kT_from_3n_minus_3),
        outside_count=int(np.sum(outside)),
    )


def write_velocity_diagnostics_csv(
    config: dict[str, Any],
    label: str,
    timestep: int,
    box_lengths: tuple[float, float, float],
    N: int,
    speed: np.ndarray,
    kinetic_energy: float,
    kT_from_3n: float,
    kT_from_3n_minus_3: float,
    outside_count: int,
) -> None:
    """Append compact velocity diagnostics to a CSV file."""

    csv_name = config.get("output", {}).get("velocity_debug_csv", "velocity_debug.csv")
    if not csv_name:
        return

    file_exists = os.path.isfile(csv_name)
    with open(csv_name, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "label,timestep,Lx,Ly,Lz,N,"
                "speed_min,speed_mean,speed_median,speed_rms,speed_max,"
                "zero_lt_1e_minus_12,zero_lt_1e_minus_8,zero_lt_1e_minus_6,"
                "kinetic_energy,kT_from_3N,kT_from_3Nminus3,outside_count\n"
            )
        f.write(
            f"{label},{timestep},{box_lengths[0]:.12g},{box_lengths[1]:.12g},{box_lengths[2]:.12g},{N},"
            f"{_safe_stat(speed, np.min):.12g},{_safe_stat(speed, np.mean):.12g},"
            f"{_safe_stat(speed, np.median):.12g},"
            f"{(np.sqrt(np.mean(speed * speed)) if speed.size else float('nan')):.12g},"
            f"{_safe_stat(speed, np.max):.12g},"
            f"{int(np.sum(speed < 1e-12))},{int(np.sum(speed < 1e-8))},{int(np.sum(speed < 1e-6))},"
            f"{kinetic_energy:.12g},{kT_from_3n:.12g},{kT_from_3n_minus_3:.12g},{outside_count}\n"
        )


def save_debug_state(sim, config: dict[str, Any], label: str, output_log: logging.Logger) -> None:
    """Write a full-state checkpoint GSD for debugging selected moments."""

    if not config.get("output", {}).get("write_debug_state_gsds", True):
        return

    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    filename = f"{config['run']['fname']}_debug_{safe_label}.gsd"
    type_metadata = Catalog(config).build_type_metadata()
    write_complete_state_gsd(
        sim=sim,
        filename=filename,
        type_metadata=type_metadata,
        mode="w",
    )
    output_log.debug("[DEBUG_STATE %s] wrote %s", label, filename)


# =============================================================================
# Run protocols
# =============================================================================


def run_nvt_equilibration(sim, config: dict[str, Any], output_log: logging.Logger):
    kT = float(config["temperature"]["kT"])
    nvt = create_md_method(config, kT=kT, particle_filter=hoomd.filter.All())
    sim.operations.integrator.methods.append(nvt)
    
    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "nvt_before_initial_thermalize")
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "nvt_after_initial_thermalize")
    save_debug_state(sim, config, "nvt_after_initial_thermalize", output_log)

    output_log.debug("Starting NVT")

    sim_steps = int(config["time"]["sim_steps_NVT_EQ"])
    loop_steps = int(config["time"]["loop_steps_NVT"])
    steps_per_loop = int(sim_steps / loop_steps)

    for i in range(loop_steps):
        begin = time.time()
        sim.run(steps_per_loop)
        end = time.time()

        debug_pos_F_printer(sim, output_log, config, label="NVT")
        output_log.debug(f"{i} {sim.state.box.Lx} {sim.state.box.Ly} {sim.state.box.Lz}")
        output_log.debug(end - begin)

    log_velocity_diagnostics(sim, output_log, config, "nvt_end_before_save")
    save_debug_state(sim, config, "nvt_end_before_save", output_log)
    save_sim("nvt", sim, config)
    sim.operations.integrator.methods.remove(nvt)


def normalize_expansion_geometry_name(geometry: str) -> str:
    """Return the canonical expansion-geometry name.

    Accepted aliases:
        - constant_VF          -> constant_VF_droplet
        - constnat_VF_droplet -> constant_VF_droplet  # typo-safe alias
    """

    aliases = {
        "constant_VF": "constant_VF_droplet",
        "constnat_VF_droplet": "constant_VF_droplet",
    }
    return aliases.get(geometry, geometry)


def target_volume_from_constant_vf(sim, config: dict[str, Any]) -> float:
    target_vf = float(config["box"].get("target_volume_fraction", 0.0))
    if not (0.0 < target_vf < 1.0):
        raise ValueError(
            "BOX['target_volume_fraction'] must be between 0 and 1 for "
            f"constant-VF expansion modes. Got {target_vf}."
        )

    type_metadata = Catalog(config).build_type_metadata()
    particle_volume = particle_volume_from_state(sim, type_metadata=type_metadata)
    if particle_volume <= 0.0:
        raise ValueError(
            "Cannot use a constant-VF expansion mode because the total particle "
            "volume is zero. Check particle diameters."
        )

    return particle_volume / target_vf


def box_volume(box) -> float:
    return float(box.Lx * box.Ly * box.Lz)


def cubic_box_from_volume(volume: float):
    if volume <= 0.0:
        raise ValueError(f"Box volume must be positive. Got {volume}.")
    side = volume ** (1.0 / 3.0)
    return hoomd.box.Box(Lx=side, Ly=side, Lz=side)


def asym_box_from_volume(volume: float, axis: str, fin_value: float):
    if volume <= 0.0:
        raise ValueError(f"Box volume must be positive. Got {volume}.")
    side = volume ** (1.0 / 3.0)
    if axis == "x":
        return hoomd.box.Box(Lx=fin_value, Ly=side, Lz=side)
    elif axis == "y":
        return hoomd.box.Box(Lx=side, Ly=fin_value, Lz=side)
    elif axis == "z":
        return hoomd.box.Box(Lx=side, Ly=side, Lz=fin_value)
    else:
        raise ValueError(f"Unknown elongation_axis: {axis}")


def slab_box_with_target_volume(initial_box, axis: str, target_volume: float):
    """Return a slab box with the target volume by changing only one axis."""

    if target_volume <= 0.0:
        raise ValueError(f"Target box volume must be positive. Got {target_volume}.")

    Lx = float(initial_box.Lx)
    Ly = float(initial_box.Ly)
    Lz = float(initial_box.Lz)

    if axis == "x":
        Lx = target_volume / (Ly * Lz)
    elif axis == "y":
        Ly = target_volume / (Lx * Lz)
    elif axis == "z":
        Lz = target_volume / (Lx * Ly)
    else:
        raise ValueError(f"Unknown elongation_axis: {axis}")

    return hoomd.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)


def expansion_boxes_from_mode(sim, config: dict[str, Any]) -> tuple[str, list[tuple[str, Any]]]:
    """Return the expansion path as named box stages.

    Single-stage modes return one final stage.
    Droplet modes return two stages:
        1. slab stage: change only elongation_axis to form a slab
        2. droplet stage: morph the slab into a cubic box at fixed final volume
    """

    initial_box = sim.state.box
    factor = float(config["box"]["elongation_factor"])
    geometry = normalize_expansion_geometry_name(
        config["box"].get("simulation_geometry", "slab_simulation")
    )
    axis = config["box"].get("elongation_axis", "x")
    final_value_asym = config["box"].get("asymmetric", None)

    current_volume = box_volume(initial_box)

    if geometry == "slab_simulation":
        final_volume = current_volume * factor
        return geometry, [("slab", slab_box_with_target_volume(initial_box, axis, final_volume))]

    if geometry == "droplet_simulation":
        final_volume = current_volume * factor
        slab_box = slab_box_with_target_volume(initial_box, axis, final_volume)
        droplet_box = cubic_box_from_volume(final_volume)
        return geometry, [("slab", slab_box), ("droplet", droplet_box)]

    if geometry == "droplet_simulation_asym":
        final_volume = current_volume * factor
        slab_box = slab_box_with_target_volume(initial_box, axis, final_volume)
        droplet_cuboid = asym_box_from_volume(final_volume, axis, final_value_asym)  #TODO: check if this sort of elongation works
        return geometry, [("slab", slab_box), ("droplet", droplet_cuboid)]

    if geometry == "constant_VF_slab":
        final_volume = target_volume_from_constant_vf(sim, config)
        return geometry, [("constant_VF_slab", slab_box_with_target_volume(initial_box, axis, final_volume))]

    if geometry == "constant_VF_droplet":
        final_volume = target_volume_from_constant_vf(sim, config)
        slab_box = slab_box_with_target_volume(initial_box, axis, final_volume)
        droplet_box = cubic_box_from_volume(final_volume)
        return geometry, [("constant_VF_slab_stage", slab_box), ("constant_VF_droplet", droplet_box)]

    raise ValueError(
        "BOX['simulation_geometry'] must be 'slab_simulation', "
        "'droplet_simulation', 'constant_VF_slab', or "
        "'constant_VF_droplet'. "
        f"Got {geometry!r}."
    )


def final_box_from_expansion_mode(sim, config: dict[str, Any]):
    """Return only the final box for compatibility with older helper code."""

    _, stages = expansion_boxes_from_mode(sim, config)
    return stages[-1][1]


def add_box_resize(sim, initial_box, final_box, config: dict[str, Any], duration_steps: int):
    duration_steps = max(1, int(duration_steps))
    extendbox = hoomd.variant.box.Interpolate(
        initial_box,
        final_box,
        variant=hoomd.variant.Ramp(
            A=0,
            B=1,
            t_start=sim.timestep,
            t_ramp=duration_steps,
        ),
    )

    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(int(config["time"]["elongation_trigger"])),
        filter=hoomd.filter.Null(),
        box=extendbox,
    )
    sim.operations.updaters.append(box_resize)
    return box_resize


def run_steps_with_logging(
    sim,
    output_log: logging.Logger,
    total_steps: int,
    loop_steps: int,
    label: str,
) -> None:
    total_steps = int(total_steps)
    loop_steps = max(1, int(loop_steps))

    if total_steps <= 0:
        return

    base_steps = total_steps // loop_steps
    remainder = total_steps % loop_steps

    for i in range(loop_steps):
        steps_this_loop = base_steps + (1 if i < remainder else 0)
        if steps_this_loop <= 0:
            continue

        begin = time.time()
        sim.run(steps_this_loop)
        end = time.time()
        output_log.debug(
            f"{label} {i} {sim.state.box.Lx} {sim.state.box.Ly} {sim.state.box.Lz}"
        )
        output_log.debug(end - begin)


def run_resize_stage_with_optional_polymer(
    sim,
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    output_log: logging.Logger,
    stage_name: str,
    stage_box,
    stage_steps: int,
    stage_loop_steps: int,
    stage_duration: int,
    kT: float,
    active_method_box: list,   # 1-element mutable list holding the current nvt method
    anchor_tags_box: list,     # 1-element mutable list holding the current anchor tags ([] until inserted)
    poly_names: list[str] | None = None
) -> None:
    """Same behavior as run_resize_stage_with_rethermalization, except that
    if poly_name is given, the polymer is inserted right after the resize
    finishes (and before rethermalization), so the stage's own
    rethermalization + remaining relaxation steps double as the polymer's
    introduction + relaxation. The two polymer end beads are excluded from
    all integration and thermalization from that point on.
    """

    stage_steps = int(stage_steps)
    stage_loop_steps = max(1, int(stage_loop_steps))
    stage_duration = max(1, int(stage_duration))

    if stage_steps <= 0:
        return

    resize_steps = min(stage_duration, stage_steps)
    remaining_steps = stage_steps - resize_steps

    output_log.debug(
        "Starting resize stage %s: resize_steps=%d, remaining_steps=%d",
        stage_name, resize_steps, remaining_steps,
    )

    box_resize = add_box_resize(
        sim=sim, initial_box=sim.state.box, final_box=stage_box,
        config=config, duration_steps=stage_duration,
    )

    run_steps_with_logging(
        sim=sim, output_log=output_log, total_steps=resize_steps,
        loop_steps=stage_loop_steps, label=f"{stage_name}_resize",
    )

    log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_before_remove")
    save_debug_state(sim, config, f"{stage_name}_after_resize_before_remove", output_log)

    sim.operations.updaters.remove(box_resize)
    sim.run(0)

    log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_after_remove_before_rethermalize")
    save_debug_state(sim, config, f"{stage_name}_after_resize_after_remove_before_rethermalize", output_log)

    if poly_names is not None:
        anchor_tags_box[0] = []
        for poly_name in poly_names:
            poly = find_polymer_spec(config, poly_name)
            start, end = compute_polymer_anchor_positions(sim, poly, catalog)
            bead_positions = build_polymer_bead_positions(poly, start, end)
            output_log.debug("The bead positions of the polymer %s are: %s", poly_name, bead_positions)

            result = append_polymer_to_state(
                sim, config, catalog, registry, poly, bead_positions, output_log=output_log,
            )
            anchor_tags_box[0].extend([result["monomer_tags"][0], result["monomer_tags"][-1]])       # The non-movement tags are specifically the first and last monomer beads.

            relocation = resolve_polymer_overlaps_by_relocating_existing(
                sim, catalog, new_tags=result["all_tags"], output_log=output_log,
            )
            sim.run(0)
            log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_polymer_insertion")
            save_debug_state(sim, config, f"{stage_name}_after_polymer_insertion", output_log)
        # Swap the active integration method so the anchors are excluded
        # from all integration for the remainder of the run.
        sim.operations.integrator.methods.remove(active_method_box[0])
        new_method = create_md_method(
            config, kT=kT, particle_filter=filter_excluding_tags(anchor_tags_box[0]),
        )
        sim.operations.integrator.methods.append(new_method)
        active_method_box[0] = new_method

    output_log.debug("Re-thermalizing particle momenta after resize stage %s", stage_name)
    # thermalize everyone currently in the sim except the anchors (empty
    # anchor list == ordinary hoomd.filter.All(), identical to before the
    # polymer exists). This covers the bulk and the freshly-inserted
    # non-anchor polymer beads (which start at zero velocity) in one call.
    sim.state.thermalize_particle_momenta(
        filter=filter_excluding_tags(anchor_tags_box[0]), kT=kT,
    )
    sim.run(0)

    log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_after_rethermalize")
    save_debug_state(sim, config, f"{stage_name}_after_resize_after_rethermalize", output_log)

    if remaining_steps > 0:
        run_steps_with_logging(
            sim=sim, output_log=output_log, total_steps=remaining_steps,
            loop_steps=stage_loop_steps, label=f"{stage_name}_post_rethermalized",
        )
        log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_end_post_rethermalized")
        save_debug_state(sim, config, f"{stage_name}_end_post_rethermalized", output_log)


def run_elongation(sim, config: dict[str, Any], output_log: logging.Logger):
    kT = float(config["temperature"]["kT"])

    nvt = create_md_method(config, kT=kT, particle_filter=hoomd.filter.All())
    sim.operations.integrator.methods.append(nvt)

    sim.run(0)

    log_velocity_diagnostics(sim, output_log, config, "elongation_before_initial_thermalize")
    save_debug_state(sim, config, "elongation_before_initial_thermalize", output_log)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "elongation_after_initial_thermalize")
    save_debug_state(sim, config, "elongation_after_initial_thermalize", output_log)

    geometry, stages = expansion_boxes_from_mode(sim, config)
    final_box = stages[-1][1]

    if geometry in {"constant_VF_slab", "constant_VF_droplet"}:
        type_metadata = Catalog(config).build_type_metadata()
        initial_vf = current_volume_fraction_from_state(sim, type_metadata=type_metadata)
        final_volume = box_volume(final_box)
        particle_volume = particle_volume_from_state(sim, type_metadata=type_metadata)
        final_vf = particle_volume / final_volume
        output_log.debug(
            "Expansion mode %s: initial VF %.6g -> target/final VF %.6g / %.6g",
            geometry,
            initial_vf,
            float(config["box"].get("target_volume_fraction", 0.0)),
            final_vf,
        )

    output_log.debug("Expansion mode %s", geometry)
    for stage_name, stage_box in stages:
        output_log.debug(
            "  stage %s -> box %.6g %.6g %.6g volume %.6g",
            stage_name,
            stage_box.Lx,
            stage_box.Ly,
            stage_box.Lz,
            box_volume(stage_box),
        )

    add_output_writer(sim, config, elongation=True)
    output_log.debug("Starting Elongation")

    sim_steps = int(config["time"]["sim_steps_Elongation"])
    loop_steps = int(config["time"]["loop_steps_elongation"])
    elongation_duration = int(config["time"]["elongation_duration"])

    def run_resize_stage_with_rethermalization(
        stage_name: str,
        stage_box,
        stage_steps: int,
        stage_loop_steps: int,
        stage_duration: int,
    ):
        """Run one box-resize stage, then re-thermalize after the resize ends.

        The total number of MD steps for the stage remains stage_steps.
        The resize updater is active only for resize_steps. After that, particle
        momenta are re-sampled at kT, and the remaining steps continue at fixed
        box size.
        """

        stage_steps = int(stage_steps)
        stage_loop_steps = max(1, int(stage_loop_steps))
        stage_duration = max(1, int(stage_duration))

        if stage_steps <= 0:
            return

        resize_steps = min(stage_duration, stage_steps)
        remaining_steps = stage_steps - resize_steps

        output_log.debug(
            "Starting resize stage %s: resize_steps=%d, remaining_steps=%d",
            stage_name,
            resize_steps,
            remaining_steps,
        )

        box_resize = add_box_resize(
            sim=sim,
            initial_box=sim.state.box,
            final_box=stage_box,
            config=config,
            duration_steps=stage_duration,
        )

        run_steps_with_logging(
            sim=sim,
            output_log=output_log,
            total_steps=resize_steps,
            loop_steps=stage_loop_steps,
            label=f"{stage_name}_resize",
        )

        log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_before_remove")
        save_debug_state(sim, config, f"{stage_name}_after_resize_before_remove", output_log)

        sim.operations.updaters.remove(box_resize)
        sim.run(0)

        log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_after_remove_before_rethermalize")
        save_debug_state(sim, config, f"{stage_name}_after_resize_after_remove_before_rethermalize", output_log)

        output_log.debug("Re-thermalizing particle momenta after resize stage %s", stage_name)
        sim.state.thermalize_particle_momenta(
            filter=hoomd.filter.All(),
            kT=kT,
        )
        sim.run(0)

        log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_after_resize_after_rethermalize")
        save_debug_state(sim, config, f"{stage_name}_after_resize_after_rethermalize", output_log)

        if remaining_steps > 0:
            run_steps_with_logging(
                sim=sim,
                output_log=output_log,
                total_steps=remaining_steps,
                loop_steps=stage_loop_steps,
                label=f"{stage_name}_post_rethermalized",
            )
            log_velocity_diagnostics(sim, output_log, config, f"{stage_name}_end_post_rethermalized")
            save_debug_state(sim, config, f"{stage_name}_end_post_rethermalized", output_log)

    if len(stages) == 1:
        stage_name, stage_box = stages[0]

        run_resize_stage_with_rethermalization(
            stage_name=stage_name,
            stage_box=stage_box,
            stage_steps=sim_steps,
            stage_loop_steps=loop_steps,
            stage_duration=elongation_duration,
        )

    else:
        slab_fraction = float(config["box"].get("droplet_slab_stage_fraction", 0.5))
        if not (0.0 < slab_fraction < 1.0):
            raise ValueError(
                "BOX['droplet_slab_stage_fraction'] must be between 0 and 1. "
                f"Got {slab_fraction}."
            )

        stage1_steps = max(1, int(round(sim_steps * slab_fraction)))
        stage2_steps = max(1, sim_steps - stage1_steps)
        stage1_loops = max(1, int(round(loop_steps * slab_fraction)))
        stage2_loops = max(1, loop_steps - stage1_loops)
        stage1_duration = max(1, int(round(elongation_duration * slab_fraction)))
        stage2_duration = max(1, elongation_duration - stage1_duration)

        # Stage 1: form the slab by changing only elongation_axis.
        stage1_name, stage1_box = stages[0]
        run_resize_stage_with_rethermalization(
            stage_name=stage1_name,
            stage_box=stage1_box,
            stage_steps=stage1_steps,
            stage_loop_steps=stage1_loops,
            stage_duration=stage1_duration,
        )

        # Stage 2: shrink the slab axis and expand the other two axes into the
        # final cubic/uniform droplet box at fixed final volume.
        stage2_name, stage2_box = stages[1]
        run_resize_stage_with_rethermalization(
            stage_name=stage2_name,
            stage_box=stage2_box,
            stage_steps=stage2_steps,
            stage_loop_steps=stage2_loops,
            stage_duration=stage2_duration,
        )

    log_velocity_diagnostics(sim, output_log, config, "elongation_end_before_save")
    save_debug_state(sim, config, "elongation_end_before_save", output_log)
    save_sim("elongated", sim, config)
    save_sim(f"elongated{config['temperature']['kT']}", sim, config)
    sim.operations.integrator.methods.remove(nvt)


def run_elongation_with_polymer(
    sim,
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    poly_names: list[str],
    output_log: logging.Logger,
) -> list[int]:
    """Same structure as run_elongation, except the polymer named
    `poly_name` is inserted right after the box's *second* resize stage
    finishes (the slab -> droplet morph in two-stage geometries). That
    stage's own post-resize rethermalization and remaining relaxation
    steps/loop_steps are reused as the polymer's introduction+relaxation,
    so no separate relax_steps/loop_steps are needed.

    Geometries with only a single resize stage (slab_simulation,
    constant_VF_slab) have no "second stage" -- the polymer is inserted
    right after that one stage's resize instead, with a note in the log.

    Returns the anchor tags of the two frozen polymer end beads, so later
    protocol stages can keep excluding them via filter_excluding_tags(...).
    """

    kT = float(config["temperature"]["kT"])

    nvt = create_md_method(config, kT=kT, particle_filter=hoomd.filter.All())
    sim.operations.integrator.methods.append(nvt)

    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "elongation_before_initial_thermalize")
    save_debug_state(sim, config, "elongation_before_initial_thermalize", output_log)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "elongation_after_initial_thermalize")
    save_debug_state(sim, config, "elongation_after_initial_thermalize", output_log)

    geometry, stages = expansion_boxes_from_mode(sim, config)
    final_box = stages[-1][1]

    if geometry in {"constant_VF_slab", "constant_VF_droplet"}:
        type_metadata = Catalog(config).build_type_metadata()
        initial_vf = current_volume_fraction_from_state(sim, type_metadata=type_metadata)
        final_volume = box_volume(final_box)
        particle_volume = particle_volume_from_state(sim, type_metadata=type_metadata)
        final_vf = particle_volume / final_volume
        output_log.debug(
            "Expansion mode %s: initial VF %.6g -> target/final VF %.6g / %.6g",
            geometry, initial_vf,
            float(config["box"].get("target_volume_fraction", 0.0)), final_vf,
        )

    output_log.debug("Expansion mode %s", geometry)
    for stage_name, stage_box in stages:
        output_log.debug(
            "  stage %s -> box %.6g %.6g %.6g volume %.6g",
            stage_name, stage_box.Lx, stage_box.Ly, stage_box.Lz, box_volume(stage_box),
        )

    add_output_writer(sim, config, elongation=True)
    output_log.debug("Starting Elongation (with polymer insertion at second stage)")

    sim_steps = int(config["time"]["sim_steps_Elongation"])
    loop_steps = int(config["time"]["loop_steps_elongation"])
    elongation_duration = int(config["time"]["elongation_duration"])

    active_method_box = [nvt]
    anchor_tags_box: list = [[]]

    if len(stages) == 1:
        output_log.debug(
            "Geometry %s has only one resize stage; inserting polymer "
            "after its resize (no separate 'second stage' exists).",
            geometry,
        )
        stage_name, stage_box = stages[0]
        run_resize_stage_with_optional_polymer(
            sim, config, catalog, registry, output_log,
            stage_name=stage_name, stage_box=stage_box,
            stage_steps=sim_steps, stage_loop_steps=loop_steps,
            stage_duration=elongation_duration,
            kT=kT, active_method_box=active_method_box, anchor_tags_box=anchor_tags_box,
            poly_names=poly_names,
        )

    else:
        slab_fraction = float(config["box"].get("droplet_slab_stage_fraction", 0.5))
        if not (0.0 < slab_fraction < 1.0):
            raise ValueError(
                "BOX['droplet_slab_stage_fraction'] must be between 0 and 1. "
                f"Got {slab_fraction}."
            )

        stage1_steps = max(1, int(round(sim_steps * slab_fraction)))
        stage2_steps = max(1, sim_steps - stage1_steps)
        stage1_loops = max(1, int(round(loop_steps * slab_fraction)))
        stage2_loops = max(1, loop_steps - stage1_loops)
        stage1_duration = max(1, int(round(elongation_duration * slab_fraction)))
        stage2_duration = max(1, elongation_duration - stage1_duration)

        # Stage 1: form the slab. No polymer yet.
        stage1_name, stage1_box = stages[0]
        run_resize_stage_with_optional_polymer(
            sim, config, catalog, registry, output_log,
            stage_name=stage1_name, stage_box=stage1_box,
            stage_steps=stage1_steps, stage_loop_steps=stage1_loops,
            stage_duration=stage1_duration,
            kT=kT, active_method_box=active_method_box, anchor_tags_box=anchor_tags_box,
            poly_names=None,
        )

        # Stage 2: morph into the final cubic/uniform box. Polymer(s) go(es) in
        # right after this stage's resize; its post-resize rethermalization
        # + remaining relaxation steps are this stage's own.
        stage2_name, stage2_box = stages[1]
        run_resize_stage_with_optional_polymer(
            sim, config, catalog, registry, output_log,
            stage_name=stage2_name, stage_box=stage2_box,
            stage_steps=stage2_steps, stage_loop_steps=stage2_loops,
            stage_duration=stage2_duration,
            kT=kT, active_method_box=active_method_box, anchor_tags_box=anchor_tags_box,
            poly_names=poly_names,
        )

    log_velocity_diagnostics(sim, output_log, config, "elongation_end_before_save")
    save_debug_state(sim, config, "elongation_end_before_save", output_log)
    save_sim("elongated_with_polymer", sim, config)
    save_sim(f"elongated_with_polymer{config['temperature']['kT']}", sim, config)

    sim.operations.integrator.methods.remove(active_method_box[0])
    return anchor_tags_box[0]


def run_no_elongation(sim, config: dict[str, Any], output_log: logging.Logger):
    kT = float(config["temperature"]["kT"])
    nvt = create_md_method(config, kT=kT, particle_filter=hoomd.filter.All())
    sim.operations.integrator.methods.append(nvt)

    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "no_elongation_before_initial_thermalize")
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "no_elongation_after_initial_thermalize")
    save_debug_state(sim, config, "no_elongation_after_initial_thermalize", output_log)

    add_output_writer(sim, config, elongation=True)

    output_log.debug("Starting no-elongation continuation")

    sim_steps = int(config["time"]["sim_steps_Elongation"])
    loop_steps = int(config["time"]["loop_steps_elongation"])
    steps_per_loop = int(sim_steps / loop_steps)

    for i in range(loop_steps):
        begin = time.time()
        sim.run(steps_per_loop)
        end = time.time()
        output_log.debug(f"{i} {sim.state.box.Lx} {sim.state.box.Ly} {sim.state.box.Lz}")
        output_log.debug(end - begin)

    save_sim("elongated", sim, config)
    save_sim(f"elongated{config['temperature']['kT']}", sim, config)
    sim.operations.integrator.methods.remove(nvt)


def run_no_elongation_with_polymer(sim, config: dict[str, Any], catalog: Catalog, poly_names: list[str], output_log: logging.Logger, anchor_tags: list|None = None):
    '''
    Run a simulation without elongation and with a polymer. In this case, to keep the continuity between the
    equilibration from the previous trajectory, the method does not rethermalize the system. The two end beads of the polymer(s) are excluded.
    '''
    kT = float(config["temperature"]["kT"])
    snap = sim.state.get_snapshot()
    current_N = int(snap.particles.N)

    if anchor_tags is None:
        anchor_tags: list = []
        index_end = 0
        for poly_name in reversed(poly_names):
            # The ORDER of poly_names matters here (has to be the same as in the prepare.py of the loaded file!!), because the anchor tags are computed in order of the polymer names.
            # The first polymer's first bead is the first anchor tag, and the last polymer's last bead is the last anchor tag.
            # Because the for-loop is reversed, the first index of the added polymer is always 1, such that tah_start is current_N - n_added.
            poly = find_polymer_spec(config, poly_name)
            sequence = poly["sequence"]
            n_added = 0
            for i, mon in enumerate(sequence):
                if i == len(sequence) - 1:
                    index_end = n_added + 1        # This takes into account if the last monomer is a patchy particle too.

                template_name = mon["template"]
                alias = mon["alias"]
                type_names = mon.get("type_names", {})
                template = catalog.template(template_name)

                if template["kind"] == "single_bead":
                    if mon["alias"] not in snap.particles.types:
                        raise ValueError(f"Unknown particle type: {mon['alias']}. Check that the prepare.py file matches the corresponding trajectory file.")
                    n_added += 1

                elif template["kind"] == "patchy":
                    if f"{mon['alias']}_core" not in snap.particles.types:
                        raise ValueError(f"Unknown particle type: {mon['alias']}. Check that the prepare.py file matches the corresponding trajectory file.")
                    patch_names = template["patches"]
                    patch_types = [
                        catalog.actual_type_for_bead(
                            template_name=template_name, alias=alias,
                            bead_name=p, type_names=type_names,
                        )
                        for p in patch_names
                    ]
                    n_added += 1 + len(patch_types)

                else:
                    raise ValueError(f"Unknown template kind: {template['kind']}")

            tag_start = current_N - n_added
            tag_end = current_N - n_added + index_end - 1
            anchor_tags.extend([tag_start, tag_end])
            current_N -= n_added

    output_log.debug("[ANCHOR TAGS CHECK] The start and end anchor tags are: %s", anchor_tags)
    # Now use the anchor tags to exclude the end beads from the integration method, and run the rest of the simulation
    nvt = create_md_method(config, kT=kT, particle_filter=filter_excluding_tags(anchor_tags))
    sim.operations.integrator.methods.append(nvt)

    sim.run(0)
    log_velocity_diagnostics(sim, output_log, config, "no_elongation_initial_state")
    save_debug_state(sim, config, "no_elongation_initial_state", output_log)

    add_output_writer(sim, config, elongation=True)

    output_log.debug("Starting no-elongation continuation")

    sim_steps = int(config["time"]["sim_steps_Elongation"])
    loop_steps = int(config["time"]["loop_steps_elongation"])
    steps_per_loop = int(sim_steps / loop_steps)

    for i in range(loop_steps):
        begin = time.time()
        sim.run(steps_per_loop)
        end = time.time()
        output_log.debug(f"{i} {sim.state.box.Lx} {sim.state.box.Ly} {sim.state.box.Lz}")
        output_log.debug(end - begin)

    save_sim("elongated", sim, config)
    save_sim(f"elongated{config['temperature']['kT']}", sim, config)
    sim.operations.integrator.methods.remove(nvt)


def run_protocol(
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    output_log: logging.Logger,
) -> None:
    mode = config["run"]["mode"]

    if "phase_diagram" in mode:
        if "_with_polymer" in mode:
            poly_names = config["run"]["added_polymer_names"]
            run_phase_diagram(config, catalog, registry, output_log, poly_names=poly_names)
        else:
            run_phase_diagram(config, catalog, registry, output_log)
        return

    sim = create_simulation(config, catalog, registry)
    attach_integrator(sim, config, catalog, registry)
    log_configured_interactions(config, catalog, output_log)
    add_output_writer(sim, config, elongation=False)

    output_log.debug(f"Particle types: {registry.particle_types}")
    output_log.debug(f"Bond types: {registry.bond_types}")
    output_log.debug(f"Angle types: {registry.angle_types}")
    # output_log.debug(f"The metadata dictionary contains: {catalog.build_type_metadata()}")

    if mode == "create_and_expand":
        run_nvt_equilibration(sim, config, output_log)
        run_elongation(sim, config, output_log)

    elif mode == "create_expand_and_add_polymer":
        run_nvt_equilibration(sim, config, output_log)
        anchor_tags = run_elongation_with_polymer(sim, config, catalog, registry, poly_names=config["run"]["added_polymer_names"], output_log=output_log)
        output_log.debug(f"Final fixed anchor tags (no forces to act here) to copy for future use of this trajectory: {anchor_tags}")

    elif mode == "create_no_expand":
        run_nvt_equilibration(sim, config, output_log)
        run_no_elongation(sim, config, output_log)

    elif mode == "load_and_expand":
        run_elongation(sim, config, output_log)

    elif mode == "load_no_expand":
        run_no_elongation(sim, config, output_log)

    elif mode == "load_no_expand_with_polymer":
        run_no_elongation_with_polymer(sim, config, catalog, poly_names=config["run"]["added_polymer_names"], output_log=output_log, anchor_tags=config["run"].get("anchor_tags", None))

    else:
        raise ValueError(f"Unknown run mode: {mode}")


def run_phase_diagram(
    config: dict[str, Any],
    catalog: Catalog,
    registry: TypeRegistry,
    output_log: logging.Logger,
    poly_names: list[str] | None = None
) -> None:
    initial_file = config["run"]["loaded_file"]

    for T in config["temperature"]["phase_diagram_temperatures"]:
        config["temperature"]["kT"] = float(T)
        config["run"]["loaded_file"] = initial_file

        output_log.debug(f"Starting phase diagram temperature {T}")
        output_log.debug(f"Loading {initial_file}")

        sim = create_simulation(config, catalog, registry)
        attach_integrator(sim, config, catalog, registry)
        add_output_writer(sim, config, elongation=False)

        if poly_names is not None:
            run_no_elongation_with_polymer(sim, config, catalog, poly_names=poly_names, output_log=output_log, anchor_tags=config["run"].get("anchor_tags", None))
        else:
            run_no_elongation(sim, config, output_log)

        initial_file = f"elongation_kT_{T}_traj.gsd"


# =============================================================================
# Logging and main
# =============================================================================


def setup_logger(config: dict[str, Any]) -> logging.Logger:
    output_log = logging.getLogger("main")
    output_log.setLevel(logging.DEBUG)

    if output_log.handlers:
        return output_log

    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

    file_handler = logging.FileHandler(config["output"]["log_file"])
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    output_log.addHandler(file_handler)
    output_log.addHandler(stream_handler)

    return output_log


def log_configured_interactions(
    config: dict[str, Any],
    catalog: Catalog,
    output_log: logging.Logger,
) -> None:
    """Log every (type_a, type_b) pair with a nonzero configured interaction."""

    if config["interactions"].get("use_default_interactions", True):
        interactions = generate_interactions(config, catalog)
    else:
        interactions = config["interactions"]["manual"]

    for kind, entries in interactions.items():
        output_log.debug(f"[FORCES] {kind}: {len(entries)} active pair(s)")
        for entry in entries:
            output_log.debug(
                f"[FORCES]   {kind} {entry['type_a']}<->{entry['type_b']} "
                f"eps={entry['epsilon']:.6g} sigma={entry.get('sigma', float('nan')):.6g} "
                f"r_cut={entry['r_cut']:.6g}"
            )


def main() -> None:
    config = load_config()

    catalog = Catalog(config)
    registry = build_registry(config, catalog)
    output_log = setup_logger(config)

    run_protocol(config, catalog, registry, output_log)


if __name__ == "__main__":
    main()