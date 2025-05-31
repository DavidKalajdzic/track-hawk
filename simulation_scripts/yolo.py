"""
Follower script – resource-safe version
Runs indefinitely until you press Ctrl-C.
Tested on macOS 14 (Apple M4) with AirSim v1.9.1-Linux/Mac.
"""

import airsim, numpy as np, time, pickle, os, cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# ── PARAMETERS ──────────────────────────────────────────────────────────────
TARGET        = "BP_SplineObject_C_3"
FOLLOW_DIST   = 1.0      # m
MAX_SPEED     = 2.2      # m/s
DT            = 0.1      # s   (main control period)
YAW_P_GAIN    = 120.0    # ° s-¹ per rad error (≈ 2 rad/s × 57.3)
IMG_EVERY_N   = 20        # capture each N-th step (set >1 to lighten the load)
DATA_DIR      = "camera_data_test_train"+str(time.time())

# ── INIT ─────────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)

client  = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()


log  = []                                   # (tracking + action history)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_target_pose():
    p = client.simGetObjectPose(TARGET)
    return None if (not p or np.isnan(p.position.x_val)) else p

def follow_point(pose):
    yaw = airsim.to_eularian_angles(pose.orientation)[2]
    back = np.array([-np.cos(yaw), -np.sin(yaw), 0.]) * FOLLOW_DIST
    return airsim.Vector3r(pose.position.x_val + back[0],
                           pose.position.y_val + back[1],
                           pose.position.z_val)
def save_images(index: int):
    """Capture & write images; every call opens its own socket and closes it."""

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene,            pixels_as_float=False, compress=True),  # PNG
        airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, pixels_as_float=True,  compress=False), # DEPTH
        airsim.ImageRequest("0", airsim.ImageType.Segmentation,     pixels_as_float=False, compress=False), # Segmentation
    ])

    for cam_id, r in enumerate(responses):
        if r.height == 0 or r.width == 0:
            continue
        fname = f"{DATA_DIR}/cam{cam_id}_dt{time.time()}_frame{index:06d}"
        if r.pixels_as_float:
            np.save(fname + ".npy",
                    np.array(r.image_data_float, dtype=np.float32)
                      .reshape(r.height, r.width))
        else:

            if r.compress:
                img = Image.open(BytesIO(r.image_data_uint8))
                img.save(fname + ".png")  # Scene as PNG
            else:
                img = Image.frombytes("RGB", (r.width, r.height), bytes(r.image_data_uint8))
                img.save(fname + ".bmp")  # Segmentation as BMP
            img.close()
    print("Done", step)


# ── MAIN LOOP ───────────────────────────────────────────────────────────────
print("Following…  (Ctrl-C to stop)")
step = 0
try:
    while step <= 18120:
        step += 1
        pose = get_target_pose()
        if pose is None:                     # no target yet
            time.sleep(DT)
            continue

        tgt = follow_point(pose)
        state = client.getMultirotorState()
        p_drone = state.kinematics_estimated.position

        rel = np.array([tgt.x_val - p_drone.x_val,
                        tgt.y_val - p_drone.y_val,
                        pose.position.z_val - p_drone.z_val])

        dist = np.linalg.norm(rel)
        direction = rel / dist if dist > 1e-3 else np.zeros(3)
        speed = np.clip((dist / FOLLOW_DIST) * MAX_SPEED, 0.2, MAX_SPEED)
        v = direction * speed

        yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        yaw_des = np.arctan2(rel[1], rel[0])
        yaw_err = np.arctan2(np.sin(yaw_des - yaw_now), np.cos(yaw_des - yaw_now))
        yaw_rate = YAW_P_GAIN * yaw_err

        action   = np.array([v[0], v[1], v[2], yaw_rate], np.float32)

        # non-blocking image save (only every IMG_EVERY_N steps)
        if step % IMG_EVERY_N == 0:
            #pass
            client.simPause(True)
            save_images(step)
            client.simPause(False)
            #save_images(step)
        

        log.append({
            "timestamp": time.time(),
            "drone_position": np.array([p_drone.x_val, p_drone.y_val, p_drone.z_val], dtype=np.float32),
            "drone_velocity": np.array([state.kinematics_estimated.linear_velocity.x_val,
                                        state.kinematics_estimated.linear_velocity.y_val,
                                        state.kinematics_estimated.linear_velocity.z_val], dtype=np.float32),
            "drone_orientation": np.array(airsim.to_eularian_angles(state.kinematics_estimated.orientation), dtype=np.float32),
            "drone_angular_velocity": np.array([state.kinematics_estimated.angular_velocity.x_val,
                                                state.kinematics_estimated.angular_velocity.y_val,
                                                state.kinematics_estimated.angular_velocity.z_val], dtype=np.float32),
            "target_position": np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=np.float32),
            "relative_position": rel.astype(np.float32),
            "distance": np.float32(dist),
            "direction": direction.astype(np.float32),
            "action": action
        })

        fut = client.moveByVelocityAsync(v[0], v[1], v[2], DT,
                                         airsim.DrivetrainType.MaxDegreeOfFreedom,
                                         airsim.YawMode(True, yaw_rate))
        
        fut.join()

except KeyboardInterrupt:
    print("\nLanding…")
finally:
    pkle_name = DATA_DIR+"/follow_data.pkl"
    with open(pkle_name, "wb") as f:
        pickle.dump(log, f)
    print(f"Saved {len(log):d} steps to {pkle_name}")

    try:
        client.landAsync()
        time.sleep(3)
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        pass

