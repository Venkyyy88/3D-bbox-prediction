# 3D Bounding Box Prediction Project - Multimodal-Bbox Regressor
This project implements an end-to-end deep learning pipeline for 3D bounding box prediction using RGB images, LiDAR point clouds, and segmentation masks.

## Features
- Data preprocessing and augmentation using Albumentations and Kornia.
- A multi-modal network combining a ResNet-18 branch for images and a simplified PointNet++ branch for LiDAR.
- Cross-attention fusion between modalities.
- Multi-task loss for predicting box center, dimensions, and orientation (using a multi-bin approach).
- Detailed model evaluation with logging & visualization.

## Documentation
- Refer the document – detailed_methodology.pdf for design choices, data flow and model architecture flow chart.

## Logs & Outputs
- Attached intermediate data preprocessing outputs, log file, saved_model, experiment_runs, model outputs.

## Setup Instructions
1. Clone the repository:git clone https://github.com/Venkyyy88/3D-bbox-prediction.git
2. pip install -r requirements.txt
3. python main.py or main.ipynb
(Note: Since it is prototype, the code is maintained under a single script for easy follow-up - preferably .ipynb)
(Some packages (like torch-geometric and torch-scatter) might require extra installation steps depending on your CUDA version. Refer to their official installation guides)
