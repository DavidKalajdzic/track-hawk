# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    state_modality_keys: list,
    action_modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in state_modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in action_modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_joints_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in action_modality_keys],
                    axis=0,
                )
                pred_action_joints_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]

    print("SHAPESSS", state_joints_across_time.shape, gt_action_joints_across_time.shape, pred_action_joints_across_time.shape)
    # SHAPESSS (150, 9) (150, 4) (150, 4)
    #assert (
    #    state_joints_across_time.shape
    #    == gt_action_joints_across_time.shape
    #    == pred_action_joints_across_time.shape
    #)

    # calc MSE across time
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        num_state_joints = state_joints_across_time.shape[1]
        num_action_joints = gt_action_joints_across_time.shape[1]
    
        total_plots = num_state_joints + num_action_joints
        fig, axes = plt.subplots(nrows=total_plots, ncols=1, figsize=(8, 4 * total_plots))
    
        fig.suptitle(
            f"Trajectory {traj_id} - State: {', '.join(state_modality_keys)} | Action: {', '.join(action_modality_keys)}",
            fontsize=16,
            color="blue",
        )
    
        # Plot state joints
        for i in range(num_state_joints):
            ax = axes[i]
            ax.plot(state_joints_across_time[:, i], label=f"state[{i}]")
            ax.set_title(f"State Dim {i}")
            ax.legend()
    
        # Plot action joints
        for i in range(num_action_joints):
            ax = axes[num_state_joints + i]
            ax.plot(gt_action_joints_across_time[:, i], label="gt action")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action")
    
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")
    
            ax.set_title(f"Action Dim {i}")
            ax.legend()
    
        plt.tight_layout()
        plt.show()

    return mse
