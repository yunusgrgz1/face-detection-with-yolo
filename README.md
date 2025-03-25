# Face Detection with YOLOv8

![Project Image](https://github.com/yunusgrgz1/face-detection-with-yolo/blob/main/images%20and%20outputs/video_output.gif?raw=true)

## Overview
This project implements face detection using **YOLOv8** in a **PyTorch Dataset class**. It processes images and labels in the YOLO format and prepares them for model training.

## Features
- Loads image and label files in a structured manner
- Converts YOLO annotation coordinates into NumPy arrays
- Supports image preprocessing with transformations
- Compatible with PyTorch's `DataLoader` for easy dataset handling

## Installation
Make sure you have the required dependencies installed:
```bash
pip install ultralytics torch torchvision numpy pillow
```

## Dataset
You can download the dataset from the following link:
[Dataset Link](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)

## Model Training
You can integrate this dataset into a YOLOv8 training pipeline using `ultralytics`:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pre-trained model
model.train(data='config.yaml', epochs=50)
```

## Contributing
Feel free to contribute by creating issues or submitting pull requests!

## License
This project is licensed under the MIT License.

