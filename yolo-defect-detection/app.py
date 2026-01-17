import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import glob
import random

# ── CHANGE THIS PATH to where you unzipped COCO128 images ──────────────
# (Usually inside the unzipped folder: coco128/images/train2017/*.jpg)
DATASET_IMAGES_FOLDER = r"C:\Users\Apurva\OneDrive\Desktop\dataset\coco128\images\train2017"

# If images are directly in the coco128 folder, use:
# DATASET_IMAGES_FOLDER = r"C:\Users\Apurva\OneDrive\Desktop\dataset\coco128"

# ── Load pre-trained model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolo26s.pt")

model = load_model()

# ── Sidebar controls ────────────────────────────────────────────────────
st.sidebar.title("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
max_detections = st.sidebar.slider("Max Detections per Image", 10, 300, 100, 10)

# ── Main App ────────────────────────────────────────────────────────────
st.title("YOLO26 Demo – 3 Random Images with Detections")

# Load images from your local folder
all_images = glob.glob(os.path.join(DATASET_IMAGES_FOLDER, "*.jpg"))

if len(all_images) == 0:
    st.error(f"No images found in folder: {DATASET_IMAGES_FOLDER}\n"
             "Please unzip COCO128.zip and update the path above.")
else:
    # Automatically show 3 random images + detections on load
    selected = random.sample(all_images, min(3, len(all_images)))
    cols = st.columns(3)

    for i, img_path in enumerate(selected):
        with cols[i]:
            try:
                img = Image.open(img_path)
                img_array = np.array(img)

                # Run detection with sidebar settings
                results = model(
                    img_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_detections
                )

                # Before (original image)
                st.image(img, caption=f"Before Detection {i+1}", use_column_width=True)

                # After (with boxes/labels)
                annotated = results[0].plot(line_width=2, font_size=12)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption=f"After Detection {i+1}", use_column_width=True)

            except Exception as e:
                st.error(f"Error processing image {i+1}: {e}")

st.markdown("---")
st.caption("Pre-trained YOLO26 • Local COCO128 dataset • Adjust settings in sidebar")
