from ultralytics import YOLO

# Load your model
model = YOLO("yolo26s.pt")  # same as in your app

# Run validation on COCO128 (auto-downloads yaml/config if missing)
results = model.val(data="coco.yaml", imgsz=640)  # auto-downloads config
# Print key metrics
print("Validation Results:")
print(f"mAP@0.5: {results.box.map50:.3f}")  # Main accuracy score
print(f"mAP@0.5:0.95: {results.box.map:.3f}")
print(f"Precision: {results.box.p[0]:.3f}")
print(f"Recall: {results.box.r[0]:.3f}")
print(f"F1: {results.box.f1[0]:.3f}")

# Optional: Save plots/confusion matrix
results.save("val_results")  # creates images like confusion_matrix.png