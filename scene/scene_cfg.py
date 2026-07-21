# HUROP - Unitree G1 Color Cube Picking Project
# Scene configuration: 3x3 grid of colored cubes + G1 (inspire hand) + fixed camera
#
# Adapted from Unitree Robotics' unitree_sim_isaaclab reference scene
# (tasks/common_scene/base_scene_pickplace_redblock.py), used under Apache License 2.0.
# https://github.com/unitreerobotics/unitree_sim_isaaclab
#
# Key difference from the reference: this is a plain InteractiveSceneCfg (no RL
# reward/termination/event manager wrapper), since our pipeline is scripted
# (CV -> IK -> grasp), not RL-trained. We also spawn 9 cubes in a fixed 3x3 grid
# with randomized colors instead of a single red block.

import os
import sys
import random

UNITREE_REPO_PATH = os.environ.get(
    "UNITREE_SIM_ISAACLAB_PATH",
    os.path.expanduser("~/unitree_sim_isaaclab"),
)
if UNITREE_REPO_PATH not in sys.path:
    sys.path.insert(0, UNITREE_REPO_PATH)
    sys.path.insert(0, os.path.expanduser("~/stubs"))

os.environ.setdefault("PROJECT_ROOT", UNITREE_REPO_PATH)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets, CameraPresets

project_root = os.environ.get("PROJECT_ROOT", UNITREE_REPO_PATH)

CUBE_SIZE = 0.06
# Moved further from the robot's resting hand position (robot init_pos y=-3.7) so
# cubes don't spawn already touching/overlapping the hand geometry, and widened
# spacing so cubes are more clearly separated in both the render and for grasp planning.
TABLE_CENTER_X = -4.25
TABLE_CENTER_Y = -4.15
TABLE_SURFACE_Z = 0.84
GRID_SPACING = 0.12

COLOR_PALETTE = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def _grid_positions():
    positions = []
    offsets = [-GRID_SPACING, 0.0, GRID_SPACING]
    for row_offset in offsets:
        for col_offset in offsets:
            x = TABLE_CENTER_X + row_offset
            y = TABLE_CENTER_Y + col_offset
            positions.append((x, y, TABLE_SURFACE_Z))
    return positions


def _random_color_assignment(seed=None):
    if seed is not None:
        random.seed(seed)
    colors = list(COLOR_PALETTE.keys())
    return [random.choice(colors) for _ in range(9)]


_GRID_POSITIONS = _grid_positions()
_GRID_COLORS = _random_color_assignment()


def make_cube_cfg(cell_index, position, color_name):
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cube_{cell_index}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=COLOR_PALETTE[color_name], metallic=0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )


@configclass
class ColorCubeGridSceneCfg(InteractiveSceneCfg):
    """Scene: G1 (inspire hand, fixed base) + table + 3x3 grid of colored cubes + front camera."""

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    robot = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
    )

    front_camera = CameraPresets.g1_front_camera()

    # Light source - the reference scene had this commented out, but our CV
    # detection step (HSV color thresholding) needs consistent, adequate lighting
    # to work reliably, so we add it back in.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    cube_0 = make_cube_cfg(0, _GRID_POSITIONS[0], _GRID_COLORS[0])
    cube_1 = make_cube_cfg(1, _GRID_POSITIONS[1], _GRID_COLORS[1])
    cube_2 = make_cube_cfg(2, _GRID_POSITIONS[2], _GRID_COLORS[2])
    cube_3 = make_cube_cfg(3, _GRID_POSITIONS[3], _GRID_COLORS[3])
    cube_4 = make_cube_cfg(4, _GRID_POSITIONS[4], _GRID_COLORS[4])
    cube_5 = make_cube_cfg(5, _GRID_POSITIONS[5], _GRID_COLORS[5])
    cube_6 = make_cube_cfg(6, _GRID_POSITIONS[6], _GRID_COLORS[6])
    cube_7 = make_cube_cfg(7, _GRID_POSITIONS[7], _GRID_COLORS[7])
    cube_8 = make_cube_cfg(8, _GRID_POSITIONS[8], _GRID_COLORS[8])


def get_ground_truth_layout():
    """Returns {cell_index: {"color": str, "position": (x,y,z)}} for validating
    the CV detection pipeline against known ground truth."""
    return {
        i: {"color": _GRID_COLORS[i], "position": _GRID_POSITIONS[i]}
        for i in range(9)
    }