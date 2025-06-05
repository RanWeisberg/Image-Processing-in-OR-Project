import cv2
import os
from ultralytics import YOLO
import shutil


# PARAMETERS
main_folder = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(main_folder, "Data", "surg_1.mp4")
OUTPUT_DIR = os.path.join(main_folder, "Data Visualization",'surg_1')
YOLO_WEIGHTS = os.path.join(main_folder, "models", "post_videos", "yolo11s_post_videos2", "weights", "best.pt")
# YOLO_WEIGHTS = os.path.join(main_folder, "Part 1", "aug_test", "yolo11s_augmented", "weights", "best.pt")

# Default “global” confidence threshold (used only if a class-specific threshold is not provided)
default_threshold = 0.25

# === CLASS-SPECIFIC CONFIDENCE THRESHOLDS ===
# Modify these values based on your own class indices and desired cutoffs
# For example, if your model has 3 classes (0, 1, 2), you might do:
class_conf_thresholds = {
    0: 0.90,   # e.g. “Empty”
    1: 0.30,   # e.g. “Tweezers”
    2: 0.85,   # e.g. “Needle_driver”
}

FRAME_JUMP = 3
OUTPUT_VIDEO_NAME = "bbox_video.mp4"

# === OUTPUT TOGGLES ===
OUTPUT_LABELS = True  # <--- Toggle to False if you only want to output video with drawn boxes

# === SETUP OUTPUT FOLDERS ===
images_dir = os.path.join(OUTPUT_DIR, "images")
labels_dir = os.path.join(OUTPUT_DIR, "labels")
bbox_dir   = os.path.join(OUTPUT_DIR, "bbox")

if OUTPUT_LABELS:
    # If the folders already exist, delete them and recreate
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    if os.path.exists(bbox_dir):
        shutil.rmtree(bbox_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
else:
    # If no labels, still ensure OUTPUT_DIR exists (for video)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD YOLO MODEL ===
model = YOLO(YOLO_WEIGHTS)

# === OPEN VIDEO CAPTURE ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === SETUP VIDEO WRITER FOR OUTPUT VIDEO ===
fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
video_out_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)
video_writer   = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))

# === FRAME LOOP ===
frame_idx = 0
while frame_idx < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the current frame
    results = model.predict(frame)[0]

    # Collect boxes that meet class-specific confidence thresholds
    bboxes = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())

        # Determine threshold for this class (fallback to default if not specified)
        cls_thresh = class_conf_thresholds.get(cls_id, default_threshold)

        if conf >= cls_thresh:
            # Extract xywh coordinates
            xc, yc, w, h = [float(x) for x in box.xywh[0]]
            bboxes.append([xc, yc, w, h, conf, cls_id])

    if bboxes:
        image_name = f"frame_{frame_idx:05d}.jpg"

        if OUTPUT_LABELS:
            # 1) Save the original frame as a JPEG
            image_path = os.path.join(images_dir, image_name)
            cv2.imwrite(image_path, frame)

            # 2) Write a YOLO-format .txt file (class, normalized x_center, y_center, width, height)
            label_path = os.path.join(labels_dir, f"frame_{frame_idx:05d}.txt")
            with open(label_path, "w") as f:
                for xc, yc, w, h, conf, cls_id in bboxes:
                    xc_norm = xc / frame_width
                    yc_norm = yc / frame_height
                    w_norm  = w  / frame_width
                    h_norm  = h  / frame_height
                    f.write(f"{cls_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            # 3) Draw bounding boxes + confidence labels on a copy of the frame
            vis_frame = frame.copy()
            for xc, yc, w, h, conf, cls_id in bboxes:
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_frame,
                    f"{cls_id} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # Save the visualization image and write it to the output video
            vis_path = os.path.join(bbox_dir, image_name)
            cv2.imwrite(vis_path, vis_frame)
            video_writer.write(vis_frame)

        else:
            # Only draw boxes on the frame and write to video (no label files)
            vis_frame = frame.copy()
            for xc, yc, w, h, conf, cls_id in bboxes:
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_frame,
                    f"{cls_id} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )
            video_writer.write(vis_frame)

    # Advance by FRAME_JUMP frames
    if FRAME_JUMP < 0:
        print(f"Skipping frame {frame_idx} (no detections)")
        FRAME_JUMP = 10
    frame_idx += FRAME_JUMP

# === CLEANUP ===
cap.release()
video_writer.release()
print(f"✅ Done. Output saved to '{OUTPUT_DIR}'")
