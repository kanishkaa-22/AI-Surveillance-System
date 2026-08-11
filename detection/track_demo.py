from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.predict(
    source="sample_videos/test.mp4",
    classes=[0],          # class 0 = person in COCO
    save=True,
    project="sample_videos",
    name="detect_output"
)