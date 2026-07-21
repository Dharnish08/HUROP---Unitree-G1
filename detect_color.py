# HUROP - Unitree G1 Color Cube Picking Project
# CV Detection - manual single-frame test
#
# Loads the scene, steps the sim a few times to let physics settle, captures one
# camera frame, runs HSV color thresholding filtered by a commanded color, computes
# pixel centroids, back-projects to 3D world coordinates using depth + camera
# intrinsics/pose, and compares the result against the known ground-truth layout.
#
# Usage:
#   ./isaaclab.sh -p detect_color.py --color blue --enable_cameras

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Detect cubes of a given color from the camera feed.")
parser.add_argument("--color", type=str, default="blue", choices=["red", "green", "blue", "yellow"],
                     help="Target color to detect.")
parser.add_argument("--settle_steps", type=int, default=60,
                     help="Number of physics steps to let cubes settle before capturing a frame.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the launched simulator."""

import numpy as np
import cv2
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import unproject_depth, transform_points
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from scene.scene_cfg import ColorCubeGridSceneCfg, get_ground_truth_layout


# HSV ranges per color. These are starting points - real values will likely need
# tuning once you see actual detection results against the rendered scene's
# lighting/material appearance. Each color may need two ranges (e.g. red wraps
# around the HSV hue circle at 0/180).
HSV_RANGES = {
    "red": [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    "green": [((40, 60, 60), (80, 255, 255))],
    "blue": [((100, 100, 60), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}

MIN_CONTOUR_AREA = 50  # pixels - filters out noise; tune once you see real detections


def detect_color_centroids(rgb_image: np.ndarray, color_name: str):
    """Run HSV thresholding for the given color, return list of (u, v) pixel centroids."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (low, high) in HSV_RANGES[color_name]:
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        centroids.append((u, v, area))

    return centroids, mask

import omni.usd

def get_camera_world_pose(camera):
    stage = omni.usd.get_context().get_stage()
    prim_path = camera.cfg.prim_path.replace("env_.*", "env_0")
    prim = stage.GetPrimAtPath(prim_path)
    xform_matrix = omni.usd.get_world_transform_matrix(prim)
    translation = xform_matrix.ExtractTranslation()
    quat_gf = xform_matrix.ExtractRotationQuat()

    device = camera.data.output["rgb"].device
    pos = torch.tensor([translation[0], translation[1], translation[2]], dtype=torch.float32, device=device)
    quat_opengl = torch.tensor([[quat_gf.GetReal(), *quat_gf.GetImaginary()]], dtype=torch.float32, device=device)

    # USD/prim transforms are authored in OpenGL convention; convert to ROS
    # convention to match our manual pinhole unprojection (+Z forward, +Y down).
    quat_ros = convert_camera_frame_orientation_convention(quat_opengl, origin="opengl", target="ros")
    return pos, quat_ros[0]


def backproject_to_world(camera, pixel_centroids, depth_image: torch.Tensor):
    """Back-project pixel centroids to 3D world coordinates using depth + camera
    intrinsics/pose. Returns a list of (x, y, z) world positions."""
    intrinsics = camera.data.intrinsic_matrices[0]  # (3,3)
    cam_pos, cam_quat = get_camera_world_pose(camera)

    print("[DEBUG] intrinsics:", intrinsics)
    print("[DEBUG] cam_pos:", cam_pos)
    print("[DEBUG] cam_quat:", cam_quat)

    world_positions = []
    for (u, v, _area) in pixel_centroids:
        u_int, v_int = int(round(u)), int(round(v))
        depth_value = depth_image[v_int, u_int].item()
        print(f"[DEBUG] pixel=({u_int},{v_int}) depth_value={depth_value}")

        # unproject_depth expects a depth image + intrinsics; here we do a single-pixel
        # version by building a 1x1 "image" for that pixel, since Isaac Lab's helper
        # operates on full depth images. Simpler: manual pinhole unprojection.
        fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
        cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

        x_cam = (u - cx) * depth_value / fx
        y_cam = (v - cy) * depth_value / fy
        z_cam = depth_value

        point_cam = torch.tensor([[x_cam, y_cam, z_cam]], device=depth_image.device, dtype=torch.float32)
        point_world = transform_points(point_cam, cam_pos.unsqueeze(0), cam_quat.unsqueeze(0))
        world_positions.append(tuple(point_world[0].tolist()))

    return world_positions


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
    rgb_image = camera.data.output["rgb"][0].cpu().numpy()
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[:, :, :3]
    rgb_image = rgb_image.astype(np.uint8)

    depth_image = camera.data.output["distance_to_camera"][0].squeeze()  # (H, W)

    print(f"\n[INFO] Detecting color: {args_cli.color}")
    centroids, mask = detect_color_centroids(rgb_image, args_cli.color)
    print(f"[INFO] Found {len(centroids)} contour(s) matching '{args_cli.color}'")

    world_positions = backproject_to_world(camera, centroids, depth_image)

    print("\n=== Detected positions (pixel -> world) ===")
    for (u, v, area), (x, y, z) in zip(centroids, world_positions):
        print(f"  pixel=({u:.1f}, {v:.1f})  area={area:.0f}px  ->  world=({x:.3f}, {y:.3f}, {z:.3f})")

    print("\n=== Ground truth (for comparison) ===")
    gt = get_ground_truth_layout()
    for cell_index, info in gt.items():
        if info["color"] == args_cli.color:
            print(f"  cell {cell_index}: {info['color']:>6s}  at  {info['position']}")
    print("=" * 60)

    # Save the RGB frame and the color mask so you can visually inspect detection
    # quality outside the simulator.
    cv2.imwrite("/tmp/detect_rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("/tmp/detect_mask.png", mask)
    print("\n[INFO] Saved /tmp/detect_rgb.png and /tmp/detect_mask.png for visual inspection.")


if __name__ == "__main__":
    main()
    simulation_app.close()