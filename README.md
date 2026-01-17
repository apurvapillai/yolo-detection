# YOLO26 Object Detection Demo

A simple Streamlit web app that demonstrates object detection using the pre-trained **YOLO26** model from Ultralytics.  
The app loads 3 random images from the COCO128 dataset and displays them with object detection results (before & after).

!(demo.png)


## Features
- Uses pre-trained YOLO26s model (no training required)
- Automatically displays 3 random images + detections on page load
- Sidebar controls for Confidence Threshold, IoU Threshold, and Max Detections
- Clean before/after comparison view

## Requirements
- Python 3.8–3.12
- A local copy of the COCO128 dataset (small, 128 images)

## Setup Instructions

### 1. Download the COCO128 Dataset
The app uses the small **COCO128** dataset (first 128 images from COCO train2017).

1. Download the zip file:  
   👉 https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip

2. Unzip it to a folder of your choice

3. Inside the unzipped folder, the images are usually located in:  
   `coco128/images/train2017/`  
   → This is the folder you'll point to in the code.

### 2. Install Dependencies
Open a terminal in your project folder and run:

```bash
# (Optional but recommended) Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# or source venv/bin/activate  # macOS/Linux

# Install all required packages
pip install -r requirements.txt
