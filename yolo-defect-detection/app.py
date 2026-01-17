import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import glob
import random
import tempfile
import zipfile
import urllib.request
import shutil
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import av

# ── Auto-download COCO128 at runtime (no need to upload to GitHub) ──────
@st.cache_resource(show_spinner="Preparing COCO128 dataset (first time only)...")
def get_coco128_folder():
    target_dir = "coco128"
    zip_path = "coco128.zip"
    images_flat_dir = os.path.join(target_dir, "images_flat")

    if os.path.exists(images_flat_dir) and len(glob.glob(os.path.join(images_flat_dir, "*.jpg"))) >= 100:
        return images_flat_dir

    try:
        st.info("Downloading COCO128 dataset (~small, 128 images)...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # Flatten images (common structure: images/train2017/*.jpg)
        extracted_images = os.path.join(target_dir, "images", "train2017")
        if not os.path.exists(extracted_images):
            extracted_images = target_dir  # fallback

        os.makedirs(images_flat_dir, exist_ok=True)
        for file in glob.glob(os.path.join(extracted_images, "*.jpg")):
            shutil.move(file, images_flat_dir)

        # Cleanup
        os.remove(zip_path)
        shutil.rmtree(os.path.join(target_dir, "images"), ignore_errors=True)
        shutil.rmtree(os.path.join(target_dir, "labels"), ignore_errors=True)

        st.success("COCO128 ready!")
        return images_flat_dir

    except Exception as e:
        st.error(f"Dataset download failed: {e}\nPlease try refreshing or check your connection.")
        st.stop()

# ── Load model (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolo26s.pt")

model = load_model()

# ── Sidebar controls ────────────────────────────────────────────────────
st.sidebar.title("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.40, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.50, 0.05)
max_detections = st.sidebar.slider("Max Detections per Frame", 10, 300, 100, 10)

# ── Main Layout ─────────────────────────────────────────────────────────
st.title("YOLO26 Object Detection Demo – Images, Video & Webcam 🚀")

tab1, tab2, tab3 = st.tabs(["📷 Random Images", "🎥 Upload Video", "📹 Live Webcam"])

DATASET_IMAGES_FOLDER = get_coco128_folder()

# ── Tab 1: Random Images ────────────────────────────────────────────────
with tab1:
    st.subheader("3 Random Images from COCO128")

    all_images = glob.glob(os.path.join(DATASET_IMAGES_FOLDER, "*.jpg"))

    if len(all_images) == 0:
        st.error("No images found in dataset.")
    else:
        if st.button("Show 3 Random + Detections"):
            selected = random.sample(all_images, 3)
            cols = st.columns(3)

            for i, img_path in enumerate(selected):
                with cols[i]:
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    results = model(img_array, conf=conf_threshold, iou=iou_threshold, max_det=max_detections)

                    st.image(img, caption=f"Original {i+1}", use_column_width=True)

                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption=f"Detections {i+1}", use_column_width=True)

# ── Tab 2: Upload Video ─────────────────────────────────────────────────
with tab2:
    st.subheader("Upload Your Own Video")

    uploaded_video = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Process Video with Detections"):
            with st.spinner("Processing..."):
                # Save input
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                video_path = tfile.name

                # Output folder
                os.makedirs("processed_videos", exist_ok=True)
                out_filename = f"detected_{uploaded_video.name}"
                out_path = os.path.join("processed_videos", out_filename)

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w, h = int(cap.get(3)), int(cap.get(4))

                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)

                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=conf_threshold, iou=iou_threshold, max_det=max_detections)
                    annotated = results[0].plot()
                    out.write(annotated)

                    frame_idx += 1
                    progress.progress(frame_idx / total_frames)

                cap.release()
                out.release()

                st.success("Done! Download below if needed.")
                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name=out_filename, mime="video/mp4")

# ── Tab 3: Live Webcam ──────────────────────────────────────────────────
with tab3:
    st.subheader("Live Webcam Detection")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=conf_threshold, iou=iou_threshold, max_det=max_detections)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    st.info("Allow camera access. If 'network problem' appears, try Chrome & refresh.")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun.stunprotocol.org"},
                {"urls": "stun:stun.nextcloud.com:3478"},
                # Free public TURN (helps with restricted networks)
                {
                    "urls": "turn:openrelay.metered.ca:80",
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        },
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

st.markdown("---")
st.caption("YOLO26 • COCO128 auto-download • Video/Webcam support • Deploy-ready")