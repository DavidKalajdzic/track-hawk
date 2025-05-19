import setup_path
import airsim
import cv2
import numpy as np
import os
import csv
import random
import time
import math
from datetime import datetime

# ---------- CONFIG ----------
NUM_FRAMES = 500
TARGET_MESH = "Car*"
OBSTACLE_MESHES = ["Tree*", "Cylinder*", "Car*"]
SPIRAL_THRESHOLD = 8.0   # meters to target for follow mode
MIN_DISTANCE = 2.0       # collision avoidance minimum distance in meters
DATA_DIR = "rl_data"
CAMERA_NAME = "0"
SCENE_TYPE = airsim.ImageType.Scene
DEPTH_TYPE = airsim.ImageType.DepthPerspective
STEP_VEL = 2

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- INIT ----------
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# --- Set random weather for each run (optional) ---
#client.simEnableWeather(True)
#client.simSetWeatherParameter(airsim.WeatherParameter.Rain, random.uniform(0, 1))
#client.simSetWeatherParameter(airsim.WeatherParameter.Fog, random.uniform(0, 1))
#client.simSetWeatherParameter(airsim.WeatherParameter.Snow, random.uniform(0, 1))
#client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, random.uniform(0, 1))

# --- Setup Detection Filters for Target and Obstacles ---
client.simSetDetectionFilterRadius(CAMERA_NAME, SCENE_TYPE, 2000)  # 20 meters
client.simAddDetectionFilterMeshName(CAMERA_NAME, SCENE_TYPE, TARGET_MESH)
for mesh in OBSTACLE_MESHES:
    client.simAddDetectionFilterMeshName(CAMERA_NAME, SCENE_TYPE, mesh)

# --- CSV Logging Setup ---
csv_path = os.path.join(DATA_DIR, "data_log.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame", "timestamp", "image_file", "depth_file", "action",
    "drone_pos_x", "drone_pos_y", "drone_pos_z",
    "target_pos_x", "target_pos_y", "target_pos_z"
])

# ---------- HELPERS ----------
def get_detected_objects(client, camera_name, image_type, name_filter=None):
    detections = client.simGetDetections(camera_name, image_type)
    positions = []
    if detections:
        for obj in detections:
            if name_filter is None or (name_filter.rstrip("*") in obj.name):
                positions.append((obj.x, obj.y, obj.z))
    return positions

def avoid_collisions(drone_pos, next_pos, object_positions, min_distance=MIN_DISTANCE):
    safe_pos = list(next_pos)
    for obj_pos in object_positions:
        dist = math.sqrt((safe_pos[0] - obj_pos[0]) ** 2 +
                         (safe_pos[1] - obj_pos[1]) ** 2 +
                         (safe_pos[2] - obj_pos[2]) ** 2)
        if dist < min_distance:
            # Repel from object
            dx = safe_pos[0] - obj_pos[0]
            dy = safe_pos[1] - obj_pos[1]
            dz = safe_pos[2] - obj_pos[2]
            norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) or 1
            dx, dy, dz = dx / norm, dy / norm, dz / norm
            push = min_distance - dist + 0.1
            safe_pos[0] += dx * push
            safe_pos[1] += dy * push
            safe_pos[2] += dz * push
    return tuple(safe_pos)

def get_action_from_move(start, end):
    delta = np.array(end) - np.array(start)
    action = []
    if delta[2] > 0.05:
        action.append("Z_UP")
    elif delta[2] < -0.05:
        action.append("Z_DOWN")
    if delta[0] > 0.05:
        action.append("X_UP")
    elif delta[0] < -0.05:
        action.append("X_DOWN")
    if delta[1] > 0.05:
        action.append("Y_UP")
    elif delta[1] < -0.05:
        action.append("Y_DOWN")
    return "+".join(action) if action else "NONE"

def save_image(img_res, out_path):
    if img_res.width == 0:
        return
    array = np.frombuffer(img_res.image_data_uint8, dtype=np.uint8).reshape(img_res.height, img_res.width, 3)
    cv2.imwrite(out_path, array)

def save_depth(img_res, out_path):
    if img_res.width == 0:
        return
    arr = airsim.list_to_2d_float_array(img_res.image_data_float, img_res.width, img_res.height)
    np.save(out_path, arr)

# ---------- MAIN LOOP ----------
framecounter = 1
spiral_phase = True
spiral_angle = 0
spiral_radius = 1
target_pos = None

while framecounter <= NUM_FRAMES:
    timestamp = datetime.now().isoformat()
    drone_state = client.getMultirotorState()
    drone_pos = drone_state.kinematics_estimated.position
    drone_tuple = (drone_pos.x_val, drone_pos.y_val, drone_pos.z_val)

    # 1. Get target position (from detections)
    target_positions = get_detected_objects(client, CAMERA_NAME, SCENE_TYPE, name_filter=TARGET_MESH)
    if target_positions:
        target_pos = target_positions[0]  # Take first detected
    # 2. Get all object positions for avoidance
    object_positions = get_detected_objects(client, CAMERA_NAME, SCENE_TYPE)

    # 3. Decide Next Move
    if spiral_phase:
        if target_pos:
            dist = np.linalg.norm(np.array(drone_tuple) - np.array(target_pos))
            if dist < SPIRAL_THRESHOLD:
                spiral_phase = False
        # Spiral motion
        spiral_angle += 0.22
        spiral_radius += 0.015
        x = spiral_radius * np.cos(spiral_angle)
        y = spiral_radius * np.sin(spiral_angle)
        z = drone_tuple[2] + random.uniform(-0.2, 0.2)
        next_pos = (x, y, z)
        phase_action = "SPIRAL"
    else:
        # Follow logic (if target seen)
        if target_pos:
            dx, dy, dz = (target_pos[0] - drone_tuple[0],
                          target_pos[1] - drone_tuple[1],
                          target_pos[2] - drone_tuple[2])
            step = np.clip([dx, dy, dz], -1.0, 1.0)  # Prevent huge jumps
            x = drone_tuple[0] + step[0]
            y = drone_tuple[1] + step[1]
            z = drone_tuple[2] + step[2]
            next_pos = (x, y, z)
            phase_action = "FOLLOW"
        else:
            # Wait if target lost
            next_pos = drone_tuple
            phase_action = "WAIT"

    # 4. Collision Avoidance
    safe_next_pos = avoid_collisions(drone_tuple, next_pos, object_positions, MIN_DISTANCE)
    action = get_action_from_move(drone_tuple, safe_next_pos)
    if phase_action != "WAIT":
        action = phase_action + "+" + action

    # 5. Move Drone
    client.moveToPositionAsync(*safe_next_pos, STEP_VEL).join()

    # 6. Get Images
    image_responses = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, SCENE_TYPE, False, False),
        airsim.ImageRequest(CAMERA_NAME, DEPTH_TYPE, True, False)
    ])
    image_file = os.path.join(DATA_DIR, f"rgb_{framecounter:05d}.png")
    depth_file = os.path.join(DATA_DIR, f"depth_{framecounter:05d}.npy")
    save_image(image_responses[0], image_file)
    save_depth(image_responses[1], depth_file)

    # 7. Log to CSV
    csv_writer.writerow([
        framecounter, timestamp, image_file, depth_file, action,
        drone_tuple[0], drone_tuple[1], drone_tuple[2],
        target_pos[0] if target_pos else '', target_pos[1] if target_pos else '', target_pos[2] if target_pos else ''
    ])

    framecounter += 1
    time.sleep(0.01)  # Fast loop, but not infinite loop

# Cleanup
csv_file.close()
client.enableApiControl(False)
client.armDisarm(False)
print("Data collection complete.")
