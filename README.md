# Track-Hawk

Language-Guided Drone Tracking in Simulation

## Project Summary

Track-Hawk is an autonomous drone tracking framework that interprets natural language prompts to identify, follow, and maintain a safe distance from a specified target in a simulated environment. Leveraging pretrained Vision Language Action (VLA) models, specifically GR00T N1—paired with parameter-efficient fine-tuning (LoRA), Track-Hawk demonstrates that high quality drone control policies can be learned with limited computational resources. To address the scarcity of real world robotic data, we introduce **Dr00ne**, a large-scale synthetic dataset generated using AirSim (Unreal Engine), consisting of synchronized RGB images, depth maps, and segmentation masks.
Key contributions:

* **Dr00ne Dataset**: ≈9 hours of multi-modal drone-centric video (RGB, depth, segmentation) capturing humans, animals, and simple objects following predefined trajectories.
* **TrackHawk Model**: A fine-tuned GR00T N1 VLA model adapted via LoRA (rank 32) to predict drone control commands (desired velocities in x, y, z and yaw) from visual inputs, language prompts, and drone state.
* **Simulator Tooling & Pipelines**: Scripts to automate dataset collection in AirSim, convert data to HuggingFace/LeRobot format, and train/evaluate the VLA-based controller.

---
## Dataset Collection and Simulation

## Model training and inference

## Repository Structure
