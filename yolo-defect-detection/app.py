import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import glob
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# COCO128 image folder inside the repo
DATASET_IMAGES_FOLDER = os.path.join(BASE_DIR, "coco128", "images", "train2017")

# Load pre-trained model
@st.cache_resource
def load_model():
    return YOLO(os.path.join(BASE_DIR, "yolo26s.pt"))

model = load_model()

# Sidebar controls
st.sidebar.title("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
max_detections = st.sidebar.slider("Max Detections per Image", 10, 300, 100, 10)

# Main App
st.title("YOLO26 Demo: 3 Random Images with Detections")

# Load images
all_images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_images.extend(glob.glob(os.path.join(DATASET_IMAGES_FOLDER, ext)))

if len(all_images) == 0:
    st.error(f"No images found in folder: {DATASET_IMAGES_FOLDER}")
    st.write("Current app folder:", BASE_DIR)
    st.write("Expected image folder:", DATASET_IMAGES_FOLDER)
else:
    selected = random.sample(all_images, min(3, len(all_images)))
    cols = st.columns(3)

    for i, img_path in enumerate(selected):
        with cols[i]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)

                results = model(
                    img_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_detections
                )

                st.image(img, caption=f"Before Detection {i + 1}", use_container_width=True)

                annotated = results[0].plot(line_width=2, font_size=12)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(
                    annotated_rgb,
                    caption=f"After Detection {i + 1}",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error processing image {i + 1}: {e}")

st.markdown("---")
st.caption("Pre-trained YOLO26 • COCO128 sample images • Adjust settings in sidebar")
