# Real-Time Object Detection using YOLOv5

This project implements a real-time object detection system using YOLOv5, PyTorch, and OpenCV. The system processes webcam video streams to detect and count objects such as people, bottles, cups, and cell phones.

## Features

* Real-time object detection using YOLOv5
* Webcam video processing with OpenCV
* Detection of person, bottle, cup, and cell phone
* Bounding box visualization with confidence scores
* FPS monitoring for performance tracking

## Technologies Used

* Python
* PyTorch
* YOLOv5
* OpenCV
* TorchVision

## How to Run

### 1. Install Dependencies

```
pip install torch torchvision opencv-python
```

### 2. Run the Program

```
python object_detection_webcam.py
```

## Output

The system detects objects from the webcam and displays bounding boxes with labels and confidence scores in real time.
