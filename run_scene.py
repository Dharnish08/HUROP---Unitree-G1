# HUROP - Unitree G1 Color Cube Picking Project
# Runner: launches Isaac Sim, instantiates ColorCubeGridSceneCfg, and steps
# the simulation so you can view the scene (G1 + table + 3x3 colored cube grid).
#
# Usage:
#   ./isaaclab.sh -p run_scene.py
#   ./isaaclab.sh -p run_scene.py --num_envs 1
#
# Follows the same launch pattern as Isaac Lab's own tutorials
# (scripts/tutorials/02_scene/create_scene.py).

import argparse

from isaaclab.app import AppLauncher

# --- CLI args ---
parser = argparse.ArgumentParser(description="View the color cube grid scene with G1.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- Launch Isaac Sim (must happen before importing anything that touches omni/isaacsim) ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the launched simulator."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from scene.scene_cfg import ColorCubeGridSceneCfg, get_ground_truth_layout


def main():
    # --- Simulation context ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    sim = SimulationContext(sim_cfg)

    # Default viewer angle - looking at the table area
    sim.set_camera_view(eye=[-3.0, -4.5, 2.0], target=[-4.25, -4.03, 0.84])

    # --- Build the scene ---
    scene_cfg = ColorCubeGridSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Print ground truth so you can visually confirm cube colors/positions match
    print("\n=== Ground truth cube layout (cell_index: color, position) ===")
    for cell_index, info in get_ground_truth_layout().items():
        print(f"  cell {cell_index}: {info['color']:>6s}  at  {info['position']}")
    print("=" * 60 + "\n")

    # --- Reset and play ---
    sim.reset()
    print("[INFO] Scene setup complete. Simulation running - press Ctrl+C to stop.")

    # --- Simulation loop ---
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()