# Track-Hawk

Language-Guided Drone Tracking in Simulation

## Project Summary

Track-Hawk is an autonomous drone tracking framework that interprets natural language prompts to identify, follow, and
maintain a safe distance from a specified target in a simulated environment. Leveraging pretrained Vision Language
Action (VLA) models, specifically GR00T N1 paired with parameter-efficient fine-tuning (LoRA), Track-Hawk demonstrates
that high quality drone control policies can be learned with limited computational resources. To address the scarcity of
real world robotic data, we introduce **Dr00ne**, a large-scale synthetic dataset generated using AirSim (Unreal
Engine), consisting of synchronized RGB images, depth maps, and segmentation masks.

### Key Contributions

* **Dr00ne Dataset:** Approximately 9 hours of synthetic, multi-modal video (RGB, depth, segmentation), featuring
  diverse targets such as humans, animals, and basic objects following predefined trajectories.
* **TrackHawk Model:** A GR00T N1 model fine-tuned with Low Rank Adaptation (LoRA rank 32), specifically adapted for
  drone velocity control from visual, language, and drone-state inputs.
* **Simulator Tooling & Pipelines:** Automation scripts for data collection, conversion to HuggingFace/LeRobot format,
  and model training/evaluation within AirSim (that connects to Unreal Engine).

## Dataset Collection and Simulation

Data collection and drone control scripts are located in `simulation_scripts`, facilitating the automated generation of
synchronized RGB images, depth maps, and segmentation masks. Raw data and processed datasets are available in the
following structure:

```
data_track_hawk/
├── before_processing/
│   ├── episode_data_e0_box
│   ├── episode_data_e1_human
│   ├── episode_data_e2_dog
│   ├── episode_data_e3_chicken
│   ├── episode_data_e4_buffalo
│   └── episode_data_e5_turtle
└── dataset_drone_control/
    ├── data/chunk-000
    ├── meta
    └── videos/chunk-000/
        ├── observation.images.depth
        ├── observation.images.rgb
        └── observation.images.segmentation
```

Note that the folder before_processing is available on Huggingface
at [Track-Hawk Raw Input Dataset](https://huggingface.co/datasets/DavidKalajdzic/track_hawk_before_processing/tree/main).
Note that the folder dataset_drone_control is available on Huggingface
at [Track-Hawk Dr00ne Dataset](https://huggingface.co/datasets/DavidKalajdzic/dr00ne/tree/main).

To reproduce data collection and simulation (warning this is challenging to setup!):

1. Find a MacBook Pro with M4 Max 64Gb chip
2. Download Unreal Engine 5.5.4, the forked version of [AirSim](https://github.com/OpenSourceVideoGames/AirSim)
or [here](https://drive.google.com/file/d/1JDXIQsXcHFBBpZgum81xFZM1-KNpCeXJ/view?usp=share_link) if it does not work and
the following [City Park](https://drive.google.com/file/d/1Ofippa0zpMLpgj-gZ11KP5vtPA803J5-/view?usp=sharing)
environment.     
3. Create a python conda environment with version 3.10, and execute `pip install -r requirements.txt` in the **AirSim**
repository.    
4. Copy in the file `simulation_scripts/collector_drone.py` into `AirSim/PythonClient/multirotor`.     
5. Launch Unreal Engine and CityPark's project via the CityPark.uproject file at the root of the **CityPark**
folder.    
6. Click on the green in the IDE to start the simulation.     
7. Launch within the `AirSim/PythonClient/multirotor` folder the collector script through the conda environment via
python `simulation_scripts/collector_drone.py` and wait for completion of the data collection.     
8. Create the DR00NE dataset by running the `simluation_to_DR00NE_format.ipynb` file generating it into
`dataset_drone_control`.

Check the `1_dr00ne_dataset.ipynb` notebook for more details on the dataset generation and how to visualize the results.

## Model Training and Inference

The **TrackHawk Model** leverages a modified GR00T N1 adapted for drone present in `Isaac-GR00T`, integrating:

* **Vision Encoder:** Processes visual input (RGB, depth, segmentation).
* **Language Model:** Parses and understands natural language commands.
* **Action Head:** Outputs drone control commands (velocities in x, y, z, and yaw).

Model fine-tuning strategies tested:

* **LoRA fine-tuning (rank 32)** for the action head.
* **LoRA fine-tuning of the action head (rank 0)** for the action head.
* **Full fine-tuning of action head and vision encoder** for further accuracy.

Training was executed using two to four A100 GPUs, with evaluation based on offline metrics (Mean Squared Error) and
online performance tests in AirSim.

To reproduce the finetuning:
1- go in the modified `Isaac-GR00T` folder and follow the instruction.    
2- run the script `run_finetuning.sh` at the root of the project on your dedicated cluster.    
Check at `2_dr00ne_finetune.ipynb` notebook for more details of each model we trained and with what parameters.

Models are available on HuggingFace at
[Model 1](https://huggingface.co/DavidKalajdzic/dr00ne-gr00t-lora-rank32),
[Model 2](https://huggingface.co/DavidKalajdzic/dr00ne-gr00t-lora-rank0) and
[Model 3](https://huggingface.co/DavidKalajdzic/dr00ne-gr00t-lora-rank0-vision-unfreezed) ✨

## Evaluation and Results

#### Offline Evaluation:
  We used the dataset to run the model at multiple inference points, predicting 16 future actions each time. The Mean
  Squared Error (MSE) between predictions and ground truth was computed per step and averaged. This helps assess how
  well the model fits the data. Plots in `3_evaluate_finetuned_models.ipynb` show MSE curves per task and per action
  dimension, highlighting differences between predicted and actual trajectories.

#### Online Evaluation:
  1. Make everything is setup and sure the airsim package has been installed as explained at step 2 of the dataset collection and simulation.
  2. This script is very similar to the `simulation_scripts/collector_drone.py` because we want at the beginning the drone to be following the object and then once it is ready we take over the control of the drone to the model.
  
  3. In the cluster run the inference service at port 5555
  ```bash
    python scripts/inference_service.py --server     --model_path DavidKalajdzic/dr00ne-gr00t-lora-rank0-vision-unfreezed     --embodiment_tag new_embodiment     --data_config track_hawk
  ```
  4. Then port forward the port 5555 to your local machine.

  5. And finally launch the simulation_scripts/collector_drone.py
     `/Isaac-GR00T/scripts/online_evaluation_drone.py` script and look at the results in the AirSim simulator. In the notebook you can find a link to a video.