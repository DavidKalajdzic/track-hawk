import airsim, numpy as np, time, cv2, argparse
from gr00t.eval.robot import RobotInferenceClient
from PIL import Image
from io import BytesIO

# ── PARAMETERS ──────────────────────────────────────────────────────────────
host = "localhost"
port = "5555"
STEPS_UNTIL_MODEL_TAKEOVER = 4000

TARGET = "BP_SplineObject_C_3"
MAX_SPEED = 2.2
FOLLOW_DIST = 1.0
DT = 0.1
YAW_P_GAIN = 120.0
IMG_EVERY_N = 20

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


def get_obs(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),  # RGB
        airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, pixels_as_float=True, compress=False),  # Depth
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),  # Segmentation
    ])

    # RGB
    rgb = Image.open(BytesIO(responses[0].image_data_uint8)).convert("RGB")
    rgb_np = np.array(rgb)[None, ...].astype(np.uint8)  # (1, H, W, 3)

    # Depth: convert float32 -> log -> normalize -> uint8 -> grayscale to 3ch
    depth_raw = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
        responses[1].height, responses[1].width)
    depth_log = np.log1p(np.clip(depth_raw, 0, 5000))
    norm = ((depth_log - depth_log.min()) / (depth_log.max() - depth_log.min()) * 255).astype(np.uint8)
    depth_3ch = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)[None, ...]  # (1, H, W, 3)

    # Segmentation
    seg_np = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8).reshape(
        responses[2].height, responses[2].width, 3)[None, ...].astype(np.uint8)  # (1, H, W, 3)

    # Drone state
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
    drone_state = np.array([[pos.x_val, pos.y_val, pos.z_val,
                             vel.x_val, vel.y_val, vel.z_val,
                             ori[0], ori[1], ori[2]]], dtype=np.float64)

    obs = {
        "video.rgb": rgb_np,
        "video.depth": depth_3ch,
        "video.segmentation": seg_np,
        "state.drone_state": drone_state,
        "annotation.human.action.task_description": ["look at the box and follow it"],
    }

    return obs


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

inference_client = RobotInferenceClient(host=host, port=port)

print("Following…  (Ctrl-C to stop)")
step = 0
vx, vy, vz, yaw_rate = 0, 0, 0, 0
try:
    while step <= 18120:
        ##############
        step += 1
        pose = get_target_pose()
        if pose is None:  # no target yet
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

        action_prog = np.array([v[0], v[1], v[2], yaw_rate], np.float32)
        ##############

        if step > STEPS_UNTIL_MODEL_TAKEOVER and (True or step % (IMG_EVERY_N) == 0):
            client.simPause(True)
            obs = get_obs(client)
            action_dict = inference_client.get_action(obs)
            client.simPause(False)
            action_seq = action_dict["action.drone_action"]  # (16, 4)
            # for action in action_seq:
            model_action = action_seq[0]  # pick first step (the model predicted 16 in the future)

            want_to_avg = False
            if want_to_avg:
                weights = np.linspace(1.0, 0.1, num=16)
                weights /= weights.sum()
                model_action = (action_seq.T @ weights).T

            vx, vy, vz, yaw_rate = model_action[0], model_action[1], model_action[2], model_action[3]
            print("model        follow", vx, vy, vz, yaw_rate)
            print("programmatic follow", action_prog[0], action_prog[1], action_prog[2], action_prog[3])
            difff = model_action@action_prog
            print("delta model - prog", model_action - action_prog, difff)


        vx, vy, vz, yaw_rate = action_prog[0], action_prog[1], action_prog[2], action_prog[3]
        if step < STEPS_UNTIL_MODEL_TAKEOVER:
            print(step)
        if step > STEPS_UNTIL_MODEL_TAKEOVER:
            if step == STEPS_UNTIL_MODEL_TAKEOVER:
                print("START TAKING MODEL NOW")
            vx = model_action[0]
            vy = model_action[1]
            vz = model_action[2]
            yaw_rate = model_action[3]
        fut = client.moveByVelocityAsync(float(vx), float(vy), float(vz), DT,
                                         airsim.DrivetrainType.MaxDegreeOfFreedom,
                                         airsim.YawMode(True, float(yaw_rate)))
        fut.join()

except KeyboardInterrupt:
    print("\nLanding…")
finally:
    try:
        client.simPause(False)
        client.landAsync()
        time.sleep(10)
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        pass
