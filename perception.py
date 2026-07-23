# HUROP - Unitree G1 Color Cube Picking Project
# Perception module: reusable cube detection function.
#
# This is the shared interface Person A's perception code exposes to Person B's
# manipulation code. Given a running scene (with the front_camera already in it)
# and a target color, detect_cubes() returns the 3D world positions of every
# matching cube currently visible.
#
# Shared data format (agreed with Person B):
#   {"color": str, "position": (x, y, z)}

import numpy as np
import cv2
import torch
import omni.usd

from isaaclab.utils.math import convert_camera_frame_orientation_convention


# HSV ranges per color - tuned against the actual rendered scene.
# NOTE: update these inline if further tuning is needed; keep this as the single
# source of truth (detect_color.py imports from here, don't duplicate ranges there).
HSV_RANGES = {
    "red": [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    "green": [((40, 60, 60), (80, 255, 255))],
    "blue": [((100, 100, 60), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}

MIN_CONTOUR_AREA = 50  # pixels - filters out noise


def detect_color_centroids(rgb_image: np.ndarray, color_name: str):
    """Run HSV thresholding for the given color, return list of (u, v, area) pixel centroids."""
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


def get_camera_world_pose(camera):
    """Workaround for an open Isaac Lab bug where camera.data.pos_w/quat_w_ros
    return zero/NaN (IsaacLab issues #3004, #6422). Reads the camera's world
    transform directly from the USD stage instead, then converts from the
    USD/OpenGL convention to the ROS convention our pinhole math assumes."""
    stage = omni.usd.get_context().get_stage()
    prim_path = camera.cfg.prim_path.replace("env_.*", "env_0")
    prim = stage.GetPrimAtPath(prim_path)
    xform_matrix = omni.usd.get_world_transform_matrix(prim)
    translation = xform_matrix.ExtractTranslation()
    quat_gf = xform_matrix.ExtractRotationQuat()

    device = camera.data.output["rgb"].device
    pos = torch.tensor([translation[0], translation[1], translation[2]], dtype=torch.float32, device=device)
    quat_opengl = torch.tensor([[quat_gf.GetReal(), *quat_gf.GetImaginary()]], dtype=torch.float32, device=device)

    quat_ros = convert_camera_frame_orientation_convention(quat_opengl, origin="opengl", target="ros")
    return pos, quat_ros[0]


def backproject_to_world(camera, pixel_centroids, depth_image: torch.Tensor):
    """Back-project pixel centroids to 3D world coordinates using depth + camera
    intrinsics/pose. Returns a list of (x, y, z) world positions."""
    intrinsics = camera.data.intrinsic_matrices[0]
    cam_pos, cam_quat = get_camera_world_pose(camera)

    from isaaclab.utils.math import transform_points

    world_positions = []
    for (u, v, _area) in pixel_centroids:
        u_int, v_int = int(round(u)), int(round(v))
        depth_value = depth_image[v_int, u_int].item()

        fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
        cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

        x_cam = (u - cx) * depth_value / fx
        y_cam = (v - cy) * depth_value / fy
        z_cam = depth_value

        point_cam = torch.tensor([[x_cam, y_cam, z_cam]], device=depth_image.device, dtype=torch.float32)
        point_world = transform_points(point_cam, cam_pos.unsqueeze(0), cam_quat.unsqueeze(0))
        world_positions.append(tuple(point_world[0].tolist()))

    return world_positions


def detect_cubes(scene, target_color: str, camera_name: str = "front_camera") -> list[dict]:
    """Main shared-interface function for Person B's grasp code to call.

    Args:
        scene: an already-running InteractiveScene (sim must already be stepped/settled).
        target_color: one of "red", "green", "blue", "yellow".
        camera_name: name of the camera in the scene config (default: "front_camera").

    Returns:
        list of {"color": str, "position": (x, y, z)} for every detected cube of
        that color currently visible to the camera.
    """
    camera = scene[camera_name]

    rgb_image = camera.data.output["rgb"][0].cpu().numpy()
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[:, :, :3]
    rgb_image = rgb_image.astype(np.uint8)

    depth_image = camera.data.output["distance_to_camera"][0].squeeze()

    centroids, _mask = detect_color_centroids(rgb_image, target_color)
    world_positions = backproject_to_world(camera, centroids, depth_image)

    return [{"color": target_color, "position": pos} for pos in world_positions]