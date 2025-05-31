import os
import re
import glob
import numpy as np
import cv2
import pandas as pd


def transform_df(df: pd.DataFrame, episode_idx: int) -> pd.DataFrame:
    """
    Transform a filtered raw DataFrame into the desired training format.

    Columns:
      - observation.state: concatenation of drone_position and drone_orientation (6 floats)
      - action: original action (4 floats)
      - timestamp: (row_index + 1) * 2
      - annotation.human.action.task_description: always 0
      - task_index: episode_index
      - annotation.human.validity: 1
      - episode_index: episode_index
      - index: row index (0-based)
      - next.reward: 0.0
      - next.done: False
    """
    n = len(df)
    # build each column
    states = [list(vel) + list(orient) + list(ang)
              for vel, orient, ang in zip(df['drone_velocity'], df['drone_orientation'], df['drone_angular_velocity'])]
    actions = [list(a) for a in df['action']]
    timestamps = [(i + 1) * 2 for i in range(n)]
    task_desc = [0] * n
    task_idx = [episode_idx] * n
    valid = [1] * n
    ep_idx = [episode_idx] * n
    idxs = list(range(n))
    rewards = [0.0] * n
    dones = [False] * n
    dones[-1] = True  # last frame is done

    # matching the info.json's features section:
    new_df = pd.DataFrame({
        'observation.state': states,
        'action': actions,
        'timestamp': timestamps,
        'annotation.human.action.task_description': task_desc,
        'task_index': task_idx,
        'annotation.human.validity': valid,
        'episode_index': ep_idx,
        'index': idxs,
        'next.reward': rewards,
        'next.done': dones,
    })
    return new_df


def write_parquet(df: pd.DataFrame, output_data_dir: str, episode_index: int) -> str:
    """
    Write DataFrame to parquet with episode index. Returns the file path.
    """
    os.makedirs(output_data_dir, exist_ok=True)
    file_name = f"episode_{episode_index:06d}.parquet"
    out_path = os.path.join(output_data_dir, file_name)
    df.to_parquet(out_path, index=False)
    return out_path


def create_video_for_cam(src_dir: str,
                         cam_index: int,
                         out_file: str,
                         dt: float = 0.5,
                         scale_max=np.iinfo(np.uint8).max) -> list:
    """
    Create an mp4 video for a specific camera index (0=RGB,1=Depth,2=Segmentation).
    Use a fixed timestep dt seconds per frame. Returns the list of frame indices.
    """
    # collect files for this cam
    prefix = f"cam{cam_index}_"
    all_files = [f for f in os.listdir(src_dir) if f.startswith(prefix)]
    # filter by allowed extensions
    allowed = {'.bmp', '.png', '.npy'}
    files = [f for f in all_files if os.path.splitext(f)[1].lower() in allowed]
    assert files, f"No files found for cam{cam_index} in {src_dir}"
    # sort by frame index
    files = sorted(files, key=lambda x: int(re.search(r"frame(\d+)", x).group(1)))
    frame_indices = []
    frames = []

    for fn in files:
        idx = int(re.search(r"frame(\d+)", fn).group(1))
        frame_indices.append(idx)
        ext = os.path.splitext(fn)[1].lower()
        path = os.path.join(src_dir, fn)

        if ext in ('.bmp', '.png'):
            img = cv2.imread(path)
            assert img is not None, f"Failed to read image {fn}"
        elif ext == '.npy':
            arr = np.log1p(np.clip(np.load(path), a_min=0, a_max=5000))

            arr_min, arr_max = float(arr.min()), float(arr.max())
            if arr_max > arr_min:
                norm = ((arr - arr_min) / (arr_max - arr_min) * scale_max)
            else:
                norm = np.zeros_like(arr)

            # choose dtype based on scale_max, to make it efficient
            if scale_max <= 255:
                norm_img = norm.astype(np.uint8)
            else:
                norm_img = norm.astype(np.uint16)

            # img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            if norm_img.ndim == 2:
                img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            else:
                img = norm_img
        else:
            continue
        frames.append(img)

    # compute fps from fixed dt
    fps = 1.0 / dt
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

    return frame_indices


def process_episode(episode_idx: int,
                    before_processing_dir: str,
                    output_data_dir: str,
                    output_video_dir: str,
                    dt: float = 2) -> dict:
    """
    Process a single episode version:
      - Load and filter DataFrame
      - Build videos and get frame indices per cam
      - Verify frame indices match across cams
      - Filter DataFrame by those indices
      - Write filtered DataFrame to parquet
    Returns a dict with episode_index, paths.
    """
    # locate camera_data directory
    episode_dirs = [d for d in os.listdir(before_processing_dir)
                    if os.path.isdir(os.path.join(before_processing_dir, d))
                    and re.match(fr"episode_data_e{episode_idx}_", d)]
    assert len(episode_dirs) == 1, f"Expected one camera_data dir for episode {episode_idx}, found {episode_dirs}"
    episode_dir = os.path.join(before_processing_dir, episode_dirs[0])

    # load and filter DataFrame
    pickle_file = os.path.join(episode_dir, f"follow_data.pkl")
    df = pd.DataFrame(pd.read_pickle(pickle_file))

    # prepare video output subdirectories
    images_dir = os.path.join(output_video_dir, "observation.images.rgb")
    depth_dir = os.path.join(output_video_dir, "observation.images.depth")
    seg_dir = os.path.join(output_video_dir, "observation.images.segmentation")
    for d in (images_dir, depth_dir, seg_dir):
        os.makedirs(d, exist_ok=True)

    # define output basename
    out_name = f"episode_{episode_idx:06d}.mp4"

    # create videos and collect frame indices
    rgb_path = os.path.join(images_dir, out_name)
    idx0 = create_video_for_cam(episode_dir, 0, rgb_path, dt)
    depth_path = os.path.join(depth_dir, out_name)
    idx1 = create_video_for_cam(episode_dir, 1, depth_path, dt)
    seg_path = os.path.join(seg_dir, out_name)
    idx2 = create_video_for_cam(episode_dir, 2, seg_path, dt)

    # validate frame indices match across cams
    assert idx0 == idx1 == idx2, "Frame indices differ across cameras"

    # filter DataFrame to keep only those frame indices, in order
    df = df.iloc[idx0].reset_index(drop=True)
    df = transform_df(df, episode_idx=episode_idx)

    print(df.head(5))
    print(f"Filtered DataFrame shape: {df.shape}")

    # write filtered DataFrame to parquet
    os.makedirs(output_data_dir, exist_ok=True)
    parquet_path = write_parquet(df, output_data_dir, episode_idx)

    return {
        "episode_index": episode_idx,
        "parquet": parquet_path,
        "videos": {
            "rgb": rgb_path,
            "depth": depth_path,
            "segmentation": seg_path,
        }
    }
