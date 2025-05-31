# Track-Hawk

Language-Guided Drone Tracking in Simulation

## Project Summary

Track-Hawk is an autonomous drone tracking framework that interprets natural language prompts to identify, follow, and maintain a safe distance from a specified target in a simulated environment. Leveraging pretrained Vision Language Action (VLA) models, specifically GR00T N1 paired with parameter-efficient fine-tuning (LoRA), Track-Hawk demonstrates that high quality drone control policies can be learned with limited computational resources. To address the scarcity of real world robotic data, we introduce **Dr00ne**, a large-scale synthetic dataset generated using AirSim (Unreal Engine), consisting of synchronized RGB images, depth maps, and segmentation masks.
### Key Contributions

* **Dr00ne Dataset:** Approximately 9 hours of synthetic, multi-modal video (RGB, depth, segmentation), featuring diverse targets such as humans, animals, and basic objects following predefined trajectories.
* **TrackHawk Model:** A GR00T N1 model fine-tuned with Low Rank Adaptation (LoRA rank 32), specifically adapted for drone velocity control from visual, language, and drone-state inputs.
* **Simulator Tooling & Pipelines:** Automation scripts for data collection, conversion to HuggingFace/LeRobot format, and model training/evaluation within AirSim.

## Dataset Collection and Simulation

Data collection and drone control scripts are located in `simulation_scripts`, facilitating the automated generation of synchronized RGB images, depth maps, and segmentation masks. Raw data and processed datasets are available in the following structure:

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
To reproduce data collection and simulation:    
1- Download Unreal Engine 5.5.4, the forked version of [AirSim](https://github.com/OpenSourceVideoGames/AirSim) and the following [City Park](https://drive.google.com/file/d/1Ofippa0zpMLpgj-gZ11KP5vtPA803J5-/view?usp=sharing) environment.     
2- Create a python conda environment with version 3.11, and execute `pip install -r requirements.txt` in the **AirSim** repository.    
3- Copy in the file simulation_scripts/collector_drone.py into AirSim/PythonClient/multirotor.     
4- Launch Unreal Engine and CityPark's project via the CityPark.uproject file at the root of the **CityPark** folder.    
6- Click on the green in the IDE to start the simulation.     
7- Launch within the `AirSim/PythonClient/multirotor` folder the collector script through the conda environment via python simulation_scripts/collector_drone.py and wait for completion of the data collection.     
8- Create he DR00NE dataset by running the `simluation_to_DR00NE_format.ipynb` file generating it into `dataset_drone_control`.    

## Model Training and Inference

The **TrackHawk Model** leverages a modified GR00T N1 adapted for drone present in `Isaac-GR00T`, integrating:

* **Vision Encoder:** Processes visual input (RGB, depth, segmentation).
* **Language Model:** Parses and understands natural language commands.
* **Action Head:** Outputs drone control commands (velocities in x, y, z, and yaw).

Model fine-tuning strategies tested:

* **LoRA fine-tuning (rank 32)** for the action head.
* **Full fine-tuning** for further accuracy.

Training was executed using two A100 GPUs, with evaluation based on offline metrics (Mean Squared Error) and online performance tests in AirSim.

To reproduce the finetuning:     
1- go in the modified `Isaac-GR00T` folder and follow the instruction.    
2- run the script `run_finetuning.sh` at the root of the project on your dedicated cluster.    

