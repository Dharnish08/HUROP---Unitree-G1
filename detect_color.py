# HUROP - Unitree G1 Color Cube Picking Project
# CLI test wrapper around perception.detect_cubes() - for manual testing/debugging only.
# The actual shared function Person B's code should import is perception.detect_cubes().
#
# Usage:
#   ./isaaclab.sh -p detect_color.py --color blue --enable_cameras

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Detect cubes of a given color from the camera feed.")
parser.add_argument("--color", type=str, default="blue", choices=["red", "green", "blue", "yellow"])
parser.add_argument("--settle_steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the launched simulator."""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from scene.scene_cfg import ColorCubeGridSceneCfg, get_ground_truth_layout
from perception import detect_cubes


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-3.0, -4.5, 2.0], target=[-4.25, -4.05, 0.84])

    scene_cfg = ColorCubeGridSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print(f"[INFO] Letting scene settle for {args_cli.settle_steps} steps...")
    for _ in range(args_cli.settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    camera = scene["front_camera"]
    camera.update(dt=sim.get_physics_dt())

    print(f"\n[INFO] Detecting color: {args_cli.color}")
    detections = detect_cubes(scene, args_cli.color)
    print(f"[INFO] Found {len(detections)} cube(s) matching '{args_cli.color}'")

    print("\n=== Detected positions ===")
    for d in detections:
        x, y, z = d["position"]
        print(f"  {d['color']:>6s}  ->  world=({x:.3f}, {y:.3f}, {z:.3f})")

    print("\n=== Ground truth (for comparison) ===")
    gt = get_ground_truth_layout()
    for cell_index, info in gt.items():
        if info["color"] == args_cli.color:
            print(f"  cell {cell_index}: {info['color']:>6s}  at  {info['position']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()