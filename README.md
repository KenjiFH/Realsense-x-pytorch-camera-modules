


# RealSense Live Recognition 👁️✨

This repository contains Python scripts that demonstrate how to combine the **Intel RealSense** depth camera with powerful **computer vision** and **deep learning** models for live object recognition and detection.

## Features

  - **Live Camera Feed**: Both scripts stream **RGB** and/or **depth** data from a connected Intel RealSense camera.
  - **Real-Time Recognition**: `realsense_live_recognition.py` uses a pre-trained **ResNet18** model to perform live object classification on RGB frames.
  - **Object Detection**: `realsense_live_yolo_detection.py` leverages the **YOLOv5s** model from the Ultralytics library to perform live object detection, drawing bounding boxes and labels around detected objects.
  - **Visual Feedback**: The scripts display the live camera feed, with one showing side-by-side color and depth views, and the other showing the RGB stream with bounding boxes overlaid.
  - **Easy Integration**: A straightforward example of how to combine the `pyrealsense2` library with `PyTorch` and `Ultralytics` for various robotics or AI projects.

## Prerequisites

Before running the scripts, make sure you have the following installed:

  - **Intel RealSense SDK 2.0**: The software and drivers for your camera.
  - `pyrealsense2`: The Python wrapper for the SDK.
  - `torch` and `torchvision`: The core PyTorch libraries (for `realsense_live_recognition.py`).
  - `ultralytics`: The library for YOLO models (for `realsense_live_yolo_detection.py`).
  - `numpy`: For numerical operations.
  - `opencv-python`: For image processing and display.

You can install the Python libraries using pip:

```bash
pip install pyrealsense2 torch torchvision ultralytics numpy opencv-python
```

-----

## How to Run

1.  Connect your Intel RealSense camera to your computer.
2.  Run either script from your terminal:

**For live object classification:**

```bash
python realsense_live_recognition.py
```

**For live object detection:**

```bash
python realsense_live_yolo_detection.py
```

3.  A window will pop up showing the live camera feed. Press `q` to exit the application.
