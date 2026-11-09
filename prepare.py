"""
prepare.py

Version 1 merged experiment sheet for generic patchy-particle simulations.

Merged feature choices:
    - reduced/dimensionless Lang physical units and default run parameters
    - selectable ConstantVolume/Bussi or Langevin dynamics from Lang
    - Bio's 2-through-8 patch particle templates for easy model editing
    - Bio's shared placement-grid option for mixtures
    - Bio's like-only A-G patch-group affinity matrix
    - Lang-style interaction engine: no additional explicit patch-patch WCA branch
    - Bio diagnostic output settings used by the merged simulation engine

Day-to-day model edits belong here. Run with:
    python simulation.py
"""

from __future__ import annotations

import json
from copy import deepcopy
import numpy as np


# =============================================================================
# Helper constructors
# =============================================================================


def core_type(sigma=1.0, mass=1.0):
    return {
        "role": "core",
        "sigma": sigma,
        "mass": mass,
    }


def patch_type(sigma=0.12, mass=None, patch_group="A"):
    if mass is None:
        mass = sigma**3 / 0.5**3

    return {
        "role": "patch",
        "sigma": sigma,
        "mass": mass,
        "patch_group": patch_group,
    }


def crowder_type(sigma=1.0, mass=1.0):
    return {
        "role": "crowder",
        "sigma": sigma,
        "mass": mass,
    }


def reg_poly_core_type(sigma=1.0, mass=1.0):
    return {
        "role": "reg_poly_core",              # By "regular" it implies a simple bead-spring model (with a soft LJ repulsion so the beads don't overlap, but NO LJ ATTRACTION).
        "sigma": sigma,
        "mass": mass,
    }


def bead_template(bead_type):
    return {
        "kind": "single_bead",
        "bead": bead_type,
    }


def patchy_template(
    core,
    patches,
    geometry="auto",
    patch_distance=0.465,
    bond_k=1000.0,
    angle_k=10.0,
    angle_t0="auto",
    custom_patch_positions=None,
):
    return {
        "kind": "patchy",
        "core": core,
        "patches": patches,
        "geometry": geometry,
        "patch_distance": patch_distance,
        "bond_k": bond_k,
        "angle_k": angle_k,
        "angle_t0": angle_t0,
        "custom_patch_positions": custom_patch_positions,
    }


def population(
    template,
    alias,
    count,
    placement,
    type_names=None,
    placement_spacing=None,
    placement_spacing_multiplier=None,
):
    """Create one free-particle population.

    placement_spacing:
        Absolute grid spacing used for placement. If None, simulation.py uses
        template_size * placement_spacing_multiplier. Smaller values fit more
        particles in a smaller box, but can create overlaps if too small.

    placement_spacing_multiplier:
        Multiplier for the automatic spacing. The default is 1.05, so a bead
        with sigma=1 is placed on a grid with spacing 1.05.
    """

    result = {
        "template": template,
        "alias": alias,
        "count": count,
        "placement": placement,
        "type_names": type_names or {},
    }

    if placement_spacing is not None:
        result["placement_spacing"] = placement_spacing
    if placement_spacing_multiplier is not None:
        result["placement_spacing_multiplier"] = placement_spacing_multiplier

    return result


def monomer(template, alias, type_names=None):
    return {
        "template": template,
        "alias": alias,
        "type_names": type_names or {},
    }


def repeated_sequence(sequence_unit, chain_length):
    """Expand a short monomer pattern into an exact-length chain.

    Example: repeated_sequence([Crowder, 3PPB], 30) gives 30 monomers:
    Crowder, 3PPB, Crowder, 3PPB, ... truncated exactly at 30.
    """

    if chain_length < 0:
        raise ValueError("chain_length must be non-negative.")
    if chain_length > 0 and len(sequence_unit) == 0:
        raise ValueError("sequence_unit cannot be empty when chain_length > 0.")

    return [deepcopy(sequence_unit[i % len(sequence_unit)]) for i in range(chain_length)]


def polymer(
    name,
    count,
    sequence=None,
    placement="center_line",
    placement_custom=None,
    direction="x",
    backbone_bond_k=1000.0,
    backbone_bond_r0="auto",
    backbone_angle_k=1.0,
    backbone_angle_t0=np.pi,
    backbone_fixed=True,
    sequence_unit=None,
    chain_length=None,
    placement_spacing=None,
    placement_spacing_multiplier=None
):
    if sequence_unit is not None or chain_length is not None:
        if sequence_unit is None:
            sequence_unit = sequence or []
        if chain_length is None:
            chain_length = len(sequence_unit)
        sequence = repeated_sequence(sequence_unit, int(chain_length))

    if sequence is None:
        sequence = []

    result = {
        "name": name,
        "count": count,
        "sequence": sequence,
        "placement": placement,
        "direction": direction,
        "backbone_bond_type": f"{name}_backbone_bond",
        "backbone_bond_k": backbone_bond_k,
        "backbone_bond_r0": backbone_bond_r0,
        "backbone_angle_type": f"{name}_backbone_angle",
        "backbone_angle_k": backbone_angle_k,
        "backbone_angle_t0": backbone_angle_t0,
        "backbone_fixed": backbone_fixed
    }

    if placement_spacing is not None:
        result["placement_spacing"] = placement_spacing
    if placement_spacing_multiplier is not None:
        result["placement_spacing_multiplier"] = placement_spacing_multiplier

    if placement_custom is not None:
        result["placement_custom"] = placement_custom

    return result


# =============================================================================
# Run settings
# =============================================================================


# Available RUN["mode"] values:
#
#   "create_and_expand"
#       Build a fresh initial configuration from the settings below, write
#       fname + "_init.gsd", equilibrate, then expand the box.
#
#   "create_expand_and_add_polymer"
#       Build a fresh initial configuration from the settings below, write
#       fname + "_init.gsd", equilibrate, then expand the box (preferentially using a "droplet" mode) and add a polymer with fixed end points.
#
#   "create_no_expand"
#       Build a fresh initial configuration, equilibrate, then continue at
#       fixed box size without expansion.
#
#   "load_and_expand"
#       Load RUN["loaded_file"] and expand the box. Use this when you already
#       have a saved starting structure.
#
#   "load_no_expand"
#       Load RUN["loaded_file"] and continue at fixed box size.
#
#   "load_no_expand_with_polymer"
#       Load RUN["loaded_file"] and continue at fixed box size, where the loaded file contains a polymer with fixed end points.
#
#   "phase_diagram" ("phase_diagram_with_polymer")
#       Loop over TEMPERATURE["phase_diagram_temperatures"]. Each temperature
#       loads the previous trajectory endpoint in sequence. If the loaded file
#       contains a polymer, the "phase_diagram_with_polymer" mode needs to be used.
#
# Other RUN fields:
#   added_polymer_names: name(s) of the polymer to introduce after elongation. Only used by the "create_expand_and_add_polymer" mode.
#   fname             : prefix for output files.
#   loaded_file       : input GSD used by load_* and phase_diagram modes.
#   gpu_id            : GPU index passed to hoomd.device.GPU.
#   seed              : random seed for HOOMD.

RUN = {
    "mode": "load_and_expand",
    "added_polymer_names": ["FixedChain"],
    "anchor_tags": None,                                        # OPTIONAL: add list[int] of tags when the "mode" is "load_no_expand_with_polymer" OR "phase_diagram_with_polymer".
                                                                # Need to look in the final line of the output.txt of the respective loaded_file: should be a list of indices.
    "fname": "slab_reg_pol5_",
    "loaded_file": "reg_pol5__after_nvt.gsd",
    "gpu_id": 0,
    "seed": 1,
}

write_observables: True      # or True
observables_steps: 100000     # only used when write_observables=True

TEMPERATURE = {
    "kT": 0.063,
    "phase_diagram_temperatures": [0.064, 0.066, 0.0665, 0.067],
}


DYNAMICS = {
    # Integration method selector.
    #
    # Available DYNAMICS["method"] values:
    #
    #   "constant_volume"
    #       Current behavior: ConstantVolume integration with a Bussi
    #       thermostat at TEMPERATURE["kT"].
    #
    #   "langevin"
    #       Langevin dynamics at TEMPERATURE["kT"] using the default drag
    #       coefficients below unless type-specific overrides are provided.
    "method": "constant_volume",

    # Langevin defaults. These are passed to hoomd.md.methods.Langevin.
    "langevin_default_gamma": 1.0,
    "langevin_default_gamma_r": (1.0, 1.0, 1.0),
    "langevin_tally_reservoir_energy": False,

    # Optional per-particle-type overrides. Leave empty to use the defaults
    # above for every HOOMD particle type.
    #
    # Example:
    # "langevin_gamma_by_type": {
    #     "3PP": 1.0,
    #     "3PP_patch": 1.0,
    #     "Crowder": 2.0,
    # },
    # "langevin_gamma_r_by_type": {
    #     "3PP": (1.0, 1.0, 1.0),
    #     "3PP_patch": (1.0, 1.0, 1.0),
    #     "Crowder": (2.0, 2.0, 2.0),
    # },
    "langevin_gamma_by_type": {},
    "langevin_gamma_r_by_type": {},
}


BOX = {
    "xratio": 0.6,
    "initial_box_x": 30.75,
    "initial_box_y": 30.75,
    "initial_box_z": 30.75,
    "elongation_factor": 3.0,

    # simulation_geometry options:
    #   "slab_simulation"
    #       Single-stage slab expansion. Expand only elongation_axis by
    #       elongation_factor. Example for elongation_factor = 3 along x:
    #       Lx * Ly * Lz -> (3 Lx) * Ly * Lz.
    #
    #   "droplet_simulation"
    #       Two-stage droplet expansion. First make a slab by expanding only
    #       elongation_axis to reach the requested final volume, then morph that
    #       slab into a cubic/uniform box with the same final volume.
    #       Example for elongation_factor = 3:
    #       Lx * Ly * Lz -> (3 Lx) * Ly * Lz -> S * S * S,
    #       where S = (3 * Lx * Ly * Lz)^(1/3).
    #
    #   "constant_VF_slab"
    #       Choose the final box volume from target_volume_fraction, then change
    #       only elongation_axis so the final VF matches the target.
    #       If the target volume is 3 times the initial volume and elongation_axis="x":
    #       Lx * Ly * Lz -> (3 Lx) * Ly * Lz.
    #
    #   "constant_VF_droplet"
    #       Two-stage constant-VF droplet expansion. First make a slab whose
    #       volume gives the requested target_volume_fraction, then morph that
    #       slab into a cubic/uniform box with the same target volume.
    #       Example if target_volume is 3 times the initial volume:
    #       Lx * Ly * Lz -> (3 Lx) * Ly * Lz -> S * S * S,
    #       where S = (target_volume)^(1/3).
    #
    #   Backward-compatible aliases accepted by simulation.py:
    #       "constant_VF"         -> "constant_VF_droplet"
    #       "constnat_VF_droplet" -> "constant_VF_droplet"  # typo-safe alias
    #
    # For both constant-VF modes, the volume fraction uses all particles in the
    # state, including cores, patches, crowders, and polymer beads:
    #     VF = sum_i [(pi/6) * diameter_i^3] / box_volume
    #
    # elongation_axis options: "x", "y", or "z".
    # target_volume_fraction is only used by the constant-VF modes.
    # droplet_slab_stage_fraction is only used by the two droplet modes and
    # controls what fraction of the elongation run is spent forming the slab
    # before morphing into the cubic final box.
    "simulation_geometry": "slab_simulation",
    "elongation_axis": "x",
    "asymmetric": 75.00,
    "target_volume_fraction": 0.05,
    "droplet_slab_stage_fraction": 0.05,
}

PLACEMENT = {
    # Reserve candidate positions across all populations and polymers so mixed
    # systems do not begin with species stacked at identical coordinates.
    "shared_grid": True,
}


TIME_PROFILES = {
    # Very short smoke test. Use this to check that the setup runs.
    "test": {
        "dt": 0.001,
        "sim_steps_NVT_EQ": 1000,
        "loop_steps_NVT": 50,
        "sim_steps_NPT_EQ": 2000,
        "loop_steps_NPT": 50,
        "sim_steps_Elongation": 70000,
        "loop_steps_elongation": 50,
        "tau_thermo": 2500 * 0.001,
        "elongation_duration": 5000,
        "elongation_trigger": 10,
        "writer_steps": 500,
    },

    # Your standard/default production settings.
    "normal": {
        "dt": 0.001,
        "sim_steps_NVT_EQ": 10000000,
        "loop_steps_NVT": 500,
        "sim_steps_NPT_EQ": 20000000,
        "loop_steps_NPT": 500,
        "sim_steps_Elongation": 70000000,
        "loop_steps_elongation": 500,
        "tau_thermo": 2500 * 0.001,
        "elongation_duration": 50000,
        "elongation_trigger": 100,
        "writer_steps": 500000,
    },

    # 10x shorter than normal. Sampling is scaled 10x shorter too, so the
    # number of saved frames is approximately comparable to normal, but the
    # saved data resolves shorter time scales.
    "short_time": {
        "dt": 0.001,
        "sim_steps_NVT_EQ": 1000000,
        "loop_steps_NVT": 500,
        "sim_steps_NPT_EQ": 2000000,
        "loop_steps_NPT": 500,
        "sim_steps_Elongation": 7000000,
        "loop_steps_elongation": 500,
        "tau_thermo": 2500 * 0.001,
        "elongation_duration": 5000,
        "elongation_trigger": 10,
        "writer_steps": 50000,
    },

    # 10x longer than normal. Sampling is scaled 10x longer too, so the
    # number of saved frames is approximately comparable to normal, but the
    # trajectory covers longer time scales.
    "long_time": {
        "dt": 0.001,
        "sim_steps_NVT_EQ": 100000000,
        "loop_steps_NVT": 500,
        "sim_steps_NPT_EQ": 200000000,
        "loop_steps_NPT": 500,
        "sim_steps_Elongation": 700000000,
        "loop_steps_elongation": 500,
        "tau_thermo": 2500 * 0.001,
        "elongation_duration": 500000,
        "elongation_trigger": 1000,
        "writer_steps": 5000000,
    },

    # Custom time: feel free to change this however you need to
    "custom": {
        "dt": 0.001,
        "sim_steps_NVT_EQ": 10000000,
        "loop_steps_NVT": 500,
        "sim_steps_NPT_EQ": 20000000,
        "loop_steps_NPT": 5,
        "sim_steps_Elongation": 700000,
        "loop_steps_elongation": 50,
        "tau_thermo": 2500 * 0.001,
        "elongation_duration": 50000,
        "elongation_trigger": 1,
        "writer_steps": 25000,
    }
}

# Choose one: "test", "normal", "short_time", "long_time" or "custom".
TIME_PROFILE = "custom"
TIME = deepcopy(TIME_PROFILES[TIME_PROFILE])

ENERGY = {
    "e0": 0.06,
    "e2": 0.01,             # Regular strength of attractive LJ.
    "e_poly": 0.04,         # This is the strength (epsilon) of the polymer beads with respect to the other particles, of an ATTRACTIVE LJ potential (includes the positive slope past the minimum).
                            
    "eps_weak": 9.0,
}


OUTPUT = {
    "log_file": "sample.log",
    "write_debug_forces": False,
    "write_velocity_diagnostics": True,
    "velocity_debug_csv": "velocity_debug.csv",
    "write_debug_state_gsds": False,
    "debug_particle_count": 10,
}


# =============================================================================
# Primitive bead types
# =============================================================================

BEAD_TYPES = {
    "Core": core_type(sigma=1.0, mass=1.0),

    "CrowderSmall": crowder_type(sigma=0.5, mass=1.0),
    "CrowderMedium": crowder_type(sigma=1.0, mass=1.0),
    "CrowderLarge": crowder_type(sigma=2.0, mass=1.0),

    "RegPolyCore": reg_poly_core_type(sigma=1.0, mass=1.0),
    "RegPolyCoreLarge": reg_poly_core_type(sigma=2.0, mass=1.0),
}

for _patch_group in ["A", "B", "C", "D", "E", "F", "G"]:
    BEAD_TYPES[f"Patch{_patch_group}"] = patch_type(
        sigma=0.12,
        mass=0.12**3 / 0.5**3,
        patch_group=_patch_group,
    )


# =============================================================================
# Molecule templates
# =============================================================================


# Template geometry options for patchy_template(..., geometry=...):
#
#   geometry="auto"
#       Pick the default geometry for the number of patches.
#
#   2 patches:
#       "linear", "linear_hor", "bent_90", "bent_120"
#
#   3 patches:
#       "trigonal_planar", "trigonal_pyramid", "t_shape"
#
#   4 patches:
#       "tetrahedral", "square_planar"
#
#   5 patches:
#       "trigonal_bipyramidal", "square_pyramidal", "pentagonal_planar"
#
#   6 patches:
#       "octahedral", "hexagonal_planar", "trigonal_prism"
#
#   7 patches:
#       "pentagonal_bipyramidal", "capped_octahedral", "capped_trigonal_prism"
#
#   8 patches:
#       "cube", "square_antiprism", "bicapped_octahedral"
#
#   geometry="custom"
#       Also provide custom_patch_positions as a list of N direction vectors,
#       one vector for each patch. The engine normalizes these vectors and
#       multiplies by patch_distance.
#
# The number of entries in patches=[...] determines the patch count.
# Supported patch counts: 2, 3, 4, 5, 6, 7, 8.

TEMPLATES = {
    "CoreOnly": bead_template(
        bead_type="Core",
    ),

    "RegPolyCoreBead": bead_template(
        bead_type="RegPolyCore",
    ),

    "CrowderSmallBead": bead_template(
        bead_type="CrowderSmall",
    ),

    "CrowderMediumBead": bead_template(
        bead_type="CrowderMedium",
    ),

    "CrowderLargeBead": bead_template(
        bead_type="CrowderLarge",
    ),

    "2PPA": patchy_template(
        core="Core",
        patches=["PatchA", "PatchA"],
        geometry="linear_hor",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "3PPB": patchy_template(
        core="Core",
        patches=["PatchB", "PatchB", "PatchB"],
        geometry="trigonal_planar",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "4PPC": patchy_template(
        core="Core",
        patches=["PatchC", "PatchC", "PatchC", "PatchC"],
        geometry="tetrahedral",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "5PPD": patchy_template(
        core="Core",
        patches=["PatchD", "PatchD", "PatchD", "PatchD", "PatchD"],
        geometry="trigonal_bipyramidal",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "6PPE": patchy_template(
        core="Core",
        patches=["PatchE", "PatchE", "PatchE", "PatchE", "PatchE", "PatchE"],
        geometry="octahedral",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "7PPF": patchy_template(
        core="Core",
        patches=[
            "PatchF", "PatchF", "PatchF", "PatchF",
            "PatchF", "PatchF", "PatchF",
        ],
        geometry="pentagonal_bipyramidal",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "8PPG": patchy_template(
        core="Core",
        patches=[
            "PatchG", "PatchG", "PatchG", "PatchG",
            "PatchG", "PatchG", "PatchG", "PatchG",
        ],
        geometry="cube",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "2PPA_poly": patchy_template(
        core="RegPolyCore",
        patches=["PatchA", "PatchA"],
        geometry="linear_hor",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),
}


'''
Optional templates. Move any of these into active use by referencing them in
POPULATIONS or POLYMERS.

TEMPLATES.update({
    "3PPB": patchy_template(
        core="Core",
        patches=["PatchB", "PatchB", "PatchB"],
        geometry="trigonal_planar",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0=2.0944,
    ),

    "4PP_AABB": patchy_template(
        core="Core",
        patches=["PatchA", "PatchA", "PatchB", "PatchB"],
        geometry="tetrahedral",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),

    "8PP_AAAAAAAA": patchy_template(
        core="Core",
        patches=[
            "PatchA", "PatchA", "PatchA", "PatchA",
            "PatchA", "PatchA", "PatchA", "PatchA",
        ],
        geometry="cube",
        patch_distance=0.465,
        bond_k=1000.0,
        angle_k=10.0,
        angle_t0="auto",
    ),
})
'''


# =============================================================================
# Free populations
# =============================================================================


# Population placement options:
#   placement="full_box"
#       Fill the whole initial box with a regular grid. With PLACEMENT["shared_grid"]
#       enabled, later populations continue from the remaining unused slots. Implies
#       using the SAME placement_spacing for all populations and polymers with "full_box"
#
#   placement="central_slab"
#       Fill a slab centered along x. The slab width is xratio * initial_box_x.
#
#   placement="outside_slab"
#       Fill the two outside x-regions, leaving the central slab empty.
#
#   placement="center_line"
#       Place objects on a y-z grid centered around x ~= 0. Mostly useful for
#       a small number of polymers or test particles.
#
#   placement="none"
#       Do not place this population.
#
# Spacing controls for population(...):
#   placement_spacing=None
#       Use the automatic grid spacing: template_size * placement_spacing_multiplier.
#
#   placement_spacing=<number>
#       Use this exact grid spacing. Example: placement_spacing=0.75.
#
#   placement_spacing_multiplier=<number>
#       Change automatic spacing. Default is 1.05. If you have 10x fewer
#       particles in a 10x smaller box and run out of grid positions, try
#       something like placement_spacing_multiplier=0.9 or placement_spacing=0.9.
#       Be careful: too-small spacing can create strong overlaps.
#
# Important for mixed species:
#   The order of POPULATIONS matters when PLACEMENT["shared_grid"] is True.
#   Each species consumes its positions, those slots are removed from the shared
#   grid, and the next species is placed in the remaining slots.

fixed_polymer_core_sigma = float(BEAD_TYPES[TEMPLATES["RegPolyCoreBead"]["bead"]]["sigma"])
pol_length = 5                                              # round((BOX["initial_box_x"] * BOX["initial_box_y"] * BOX["initial_box_z"] * BOX["elongation_factor"])**(1.0 / 3.0) / fixed_polymer_core_sigma)
pol_total_vol = 0.01 * float(BOX["initial_box_x"] * BOX["initial_box_y"] * BOX["initial_box_z"] * BOX["elongation_factor"])
pol_monomer_volume = (np.pi / 6) * core_type()["sigma"]**3  # HAS TO MATCH THE VOLUME OF THE MONOMER USED IN THE POLYMER SEQUENCE
count_loss_pps = round(pol_total_vol / pol_monomer_volume)  # Decide on what to control; for now we keep the volume of patchy particles constant! (no use for this param so far)

POPULATIONS = [
    population(
        template="2PPA",
        alias="2PPA",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="3PPB",
        alias="3PPB",
        count=10000,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="4PPC",
        alias="4PPC",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="5PPD",
        alias="5PPD",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="6PPE",
        alias="6PPE",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="7PPF",
        alias="7PPF",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
    population(
        template="8PPG",
        alias="8PPG",
        count=0,
        placement="full_box",
        placement_spacing=1.0,
    ),
]


'''
Optional populations. Add these dictionaries to POPULATIONS when needed.

POPULATIONS.extend([
    population(
        template="CrowderMediumBead",
        alias="Crowder",
        count=1000,
        placement="outside_slab",
        type_names={"bead": "Crowder"},
    ),

    population(
        template="CrowderSmallBead",
        alias="CrowderSmall",
        count=1000,
        placement="outside_slab",
    ),

    population(
        template="CrowderLargeBead",
        alias="CrowderLarge",
        count=1000,
        placement="outside_slab",
    ),

    population(
        template="4PP_AABB",
        alias="Free4PP_AABB",
        count=10000,
        placement="central_slab",
    ),
])
'''


# =============================================================================
# Polymers
# =============================================================================


# Polymer placement uses the same placement names as populations.
#
# Spacing controls for polymer(...):
#   placement_spacing=None
#       Use the automatic polymer-object spacing, estimated from the full chain
#       length along its build direction.
#
#   placement_spacing=<number>
#       Use this exact spacing between different polymer copies.
#
#   placement_spacing_multiplier=<number>
#       Multiply the automatic polymer-object spacing. Default is 1.0.
#
# These spacing values control where different polymer copies are initially
# placed. They do not change the spacing between monomers inside one chain; that
# still comes from each monomer template size and the backbone bond settings.

POLYMERS = [
    polymer(
        name="CoreOnlyChain",
        count=round(pol_total_vol / (pol_monomer_volume * pol_length)),
        sequence_unit=[
            monomer("CoreOnly", alias="PolyCore")
        ],
        chain_length=pol_length,
        placement="full_box",
        placement_spacing=1.0,
        direction="x",
    ),

    polymer(
        name="2PPA_mixed_Chain",
        count=0,                                            # round(pol_total_vol / (pol_monomer_volume * pol_length)),
        sequence_unit=[
            monomer("CoreOnly", alias="PolyCore"),
            monomer("CoreOnly", alias="PolyCore"),
            monomer("2PPA", alias="Poly2PPA"),
        ],
        # backbone_bond_k=0.0,
        backbone_angle_k=1.0,           # Might make bendier if necessary. BUT keep in mind the existing LJ of 0.01 (needed for consistency) btw consecutive beads.
        chain_length=pol_length,
        placement="full_box",
        placement_spacing=1.0,          # Increase spacing so that the initial config at least has a fixed size for the polymers without interacting with one another.
        direction="y",
        backbone_fixed=True,
    ),

    polymer(
        name=RUN["added_polymer_names"][0],   # Have to name the same as one added polymer from the list in the RUN dictionary!
        count=0,                             # Have to set count to 0, otherwise the polymer will be placed in the initial configuration and will not be fixed in space!
        sequence_unit=[
            monomer("RegPolyCoreBead", alias="FixedPolyBead"),
        ],
        chain_length=pol_length,
        backbone_angle_k=3.0,            # Make more rigid than the default 
        placement="custom_x_y",          # AVAILABLE OPTIONS: "center_line", "side_box" (70% of box length in plane orthogonal to direction), "custom_x_y" (requires direction="z")
        placement_custom=[0.0, 0.0],
        direction="z",
    ),
]
    

# Example repeated polymer sequence. Uncomment to make a length-30 chain with
# CrowderMediumBead, 3PPB, CrowderMediumBead, 3PPB, ... repeats.
#
# POLYMERS.append(
#     polymer(
#         name="Crowder_3PPB_RepeatingChain",
#         count=1,
#         sequence_unit=[
#             monomer(template="CrowderMediumBead", alias="PolCrowder"),
#             monomer(template="3PPB", alias="Poly3PPB"),
#         ],
#         chain_length=30,
#         placement="center_line",
#         direction="x",
#     ),
#
#     polymer(
#         name="FixedChain",           # Have to name the same as in the RUN dictionary!
#         count=0,
#         sequence_unit=[
#             monomer("2PPA_poly", alias="FixedPolyBead")
#         ],
#         chain_length=pol_length,
#         backbone_angle_k=3.0,        # Make more rigid than the default 
#         placement="center_line",     # Useless param for the FixedChain so far
#         direction="z",
#     ),
#
#     polymer(
#         name="FixedChain",           # Have to name the same as in the RUN dictionary!
#         count=0,                     # Have to set count to 0, otherwise the polymer will be placed in the initial configuration and will not be fixed in space!
#         sequence_unit=[
#             monomer("RegPolyCoreBead", alias="FixedPolyBead"),
#             monomer("RegPolyCoreBead", alias="FixedPolyBead"),
#             monomer("RegPolyCoreBead", alias="FixedPolyBead"),
#             monomer("2PPA_poly", alias="FixedPolyBead_2")
#         ],
#         chain_length=pol_length,
#         backbone_angle_k=3.0,        # Make more rigid than the default 
#         placement="center_line",     # Useless param for the FixedChain so far
#         direction="z",
#     ),
#
    # polymer(
    #     name=RUN["added_polymer_names"][1],   # Have to name the same as one added polymer from the list in the RUN dictionary!
    #     count=0,                             # Have to set count to 0, otherwise the polymer will be placed in the initial configuration and will not be fixed in space!
    #     sequence_unit=[
    #         monomer("RegPolyCoreBead", alias="FixedPolyBead_2"),    # Need different aliases between the beads from different polymers!
    #     ],
    #     chain_length=pol_length,
    #     backbone_angle_k=3.0,            # Make more rigid than the default 
    #     placement="custom_x_y",          # AVAILABLE OPTIONS: "center_line", "side_box" (70% of box length in plane orthogonal to direction), "custom_x_y" (requires direction="z")
    #     placement_custom=[7.0, 0.0],
    #     direction="z",
    # ),
# )


'''
Optional polymers. Add these dictionaries to POLYMERS when needed.

POLYMERS.extend([
    polymer(
        name="PolCrowder_Poly3PPB_PolCrowder",
        count=1,
        sequence=[
            monomer(
                template="CrowderMediumBead",
                alias="PolCrowder",
            ),
            monomer(
                template="3PPB",
                alias="Poly3PPB",
            ),
            monomer(
                template="CrowderMediumBead",
                alias="PolCrowder",
            ),
        ],
        placement="center_line",
        direction="x",
    ),

    polymer(
        name="CoreOnlyChain",
        count=5,
        sequence=[
            monomer("CoreOnly", alias="PolyCore"),
            monomer("CoreOnly", alias="PolyCore"),
            monomer("CoreOnly", alias="PolyCore"),
            monomer("CoreOnly", alias="PolyCore"),
        ],
        placement="center_line",
        direction="x",
    ),

    polymer(
        name="Poly4PP_PolyCore_PolCrowder",
        count=1,
        sequence=[
            monomer("4PP_AABB", alias="Poly4PP_AABB"),
            monomer("CoreOnly", alias="PolyCore"),
            monomer("CrowderMediumBead", alias="PolCrowder"),
        ],
        placement="center_line",
        direction="x",
    ),
])
'''


# =============================================================================
# Interaction settings
# =============================================================================
def expanded_mie_delta(n: float, m: float, sigma_sum: float) -> float:
    return ((n / m) * (sigma_sum ** (n - m))) ** (1.0 / (n - m))


def lorentz_sigma(bead_a_diam: float, bead_b_diam: float) -> float:
    return 0.5 * (bead_a_diam + bead_b_diam)


PATCH_GROUPS = ["A", "B", "C", "D", "E", "F", "G"]


def like_only_patch_pair_matrix(like_epsilon=9.0):
    """Build the A-G patch matrix.

    Like pairs attract with like_epsilon. Unlike pairs have epsilon 0.0,
    so they receive no ExpandedMie attraction. The merged v1 engine intentionally
    retains Lang behavior and does not add an extra patch-patch WCA branch.
    """

    matrix = {}
    for i, a in enumerate(PATCH_GROUPS):
        for b in PATCH_GROUPS[i:]:
            matrix[(a, b)] = {"epsilon": like_epsilon if a == b else 0.0}
    return matrix


INTERACTIONS = {
    "use_default_interactions": True,

    "defaults": {
        "enable_core_core_repulsion": True,
        "enable_core_core_attraction": True,
        "enable_core_crowder_repulsion": True,
        "enable_crowder_crowder_repulsion": True,
        "enable_patch_patch_mie": True,
        "enable_patch_core_nonbonded": False,
        "enable_patch_crowder_nonbonded": False,
        "enable_core_polycore_repulsion": True,
        "enable_core_polycore_attraction": False,
    },

    "patch_pair_matrix": like_only_patch_pair_matrix(like_epsilon=9.0),

    "overrides": [
        # {
        #     "kind": "expanded_mie",
        #     "type_a": "FixedPolyBead_PatchA",
        #     "type_b": "FixedPolyBead_PatchA",
        #     "epsilon": 0.0,
        #     "sigma": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"]),
        #     "delta": -expanded_mie_delta(2.0, 1.0, float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"])),
        #     "r_cut": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"])
        # },

        # {
        #     "kind": "expanded_mie",
        #     "type_a": "FixedPolyBead_PatchA",
        #     "type_b": "3PPB_PatchB",
        #     "epsilon": 9.0,
        #     "sigma": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchB"]["sigma"]),
        #     "delta": -expanded_mie_delta(2.0, 1.0, float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchB"]["sigma"])),
        #     "r_cut": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchB"]["sigma"])
        # }
    ]
}


'''
Optional interaction overrides. Possible types: "lj_repulsive", "lj_attractive", "expanded_mie". Add dictionaries to INTERACTIONS["overrides"].

INTERACTIONS["overrides"].extend([
    {
        "kind": "patch_pair",
        "patch_group_a": "A",
        "patch_group_b": "B",
        "epsilon": 0.0,
    },

    {
        "kind": "lj_attractive",
        "type_a": "Free4PP_AABB_core",
        "type_b": "3PP",
        "epsilon": 0.02,
        "sigma": 1.0,
        "r_cut": 2.5,
    },

    {
        "kind": "expanded_mie",
        "type_a": "Poly2PPA_PatchA",
        "type_b": "Poly2PPA_PatchA",
        "epsilon": 18.0,
        "sigma": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"]),
        "delta": -expanded_mie_delta(2.0, 1.0, float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"])),
        "r_cut": float(BEAD_TYPES["PatchA"]["sigma"]) + float(BEAD_TYPES["PatchA"]["sigma"])
    },

    {
        "kind": "lj_attractive",
        "type_a": "3PPB_core",
        "type_b": "FixedPolyBead",
        "epsilon": ENERGY["e_poly"],
        "sigma": lorentz_sigma(float(BEAD_TYPES["Core"]["sigma"]), float(BEAD_TYPES["Core"]["sigma"])),
        "r_cut": 2.5 * lorentz_sigma(float(BEAD_TYPES["Core"]["sigma"]), float(BEAD_TYPES["Core"]["sigma"]))
    },
])
'''


# =============================================================================
# JSON assembly
# =============================================================================


def make_json_safe_interactions(interactions):
    result = deepcopy(interactions)
    matrix = {}

    for key, value in interactions["patch_pair_matrix"].items():
        a, b = key
        matrix[f"{a}|{b}"] = value

    result["patch_pair_matrix"] = matrix
    return result


def make_config():
    return {
        "run": RUN,
        "temperature": TEMPERATURE,
        "dynamics": DYNAMICS,
        "box": BOX,
        "placement": PLACEMENT,
        "time_profile": TIME_PROFILE,
        "time": TIME,
        "energy": ENERGY,
        "output": OUTPUT,
        "bead_types": BEAD_TYPES,
        "templates": TEMPLATES,
        "populations": POPULATIONS,
        "polymers": POLYMERS,
        "interactions": make_json_safe_interactions(INTERACTIONS),
    }


def main():
    config = make_config()

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Wrote config.json")
    print(f"Time profile: {TIME_PROFILE}")
    print(f"Expansion geometry: {BOX['simulation_geometry']}")
    print(f"Dynamics method: {DYNAMICS['method']}")
    print(f"Shared placement grid: {PLACEMENT['shared_grid']}")
    if BOX["simulation_geometry"] in {"constant_VF_slab", "constant_VF_droplet", "constant_VF"}:
        print(f"Target volume fraction: {BOX['target_volume_fraction']}")
    print("Active populations:")
    for pop in POPULATIONS:
        print(f"  {pop['count']} x {pop['alias']} from template {pop['template']}")
    print("Active polymers:")
    for poly in POLYMERS:
        if poly['name'] in RUN["added_polymer_names"]:
            if "polymer" in RUN["mode"]:
                print(f"  1 x {poly['name']}")
                print("   The fixed polymer sigma is: ", fixed_polymer_core_sigma)
                print("   The fixed polymer length is: ", pol_length)
            else:
                print(f"  {poly['count']} x {poly['name']}")
        else:
            print(f"  {poly['count']} x {poly['name']}")

    print("Run with: python simulation.py")
    print("Note: simulation.py imports prepare.py directly; config.json is optional.")


if __name__ == "__main__":
    main()
