import cv2
import os
from ultralytics import YOLO

# Gets the current working directory
current_directory = os.getcwd()

# PARAMETERS
main_folder_path = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(current_directory, "id_video_data/20_2_24_1.mp4")
OUTPUT_DIR = os.path.join(current_directory, "video_output")
YOLO_WEIGHTS = os.path.join(current_directory, "models/aug_test/yolo11s_augmented/weights/best.pt")
CONF_THRESHOLD = 0.8  # This is your 'p'
FRAME_JUMP = 5                     # Process every 5th frame - 1 is every frame.
OUTPUT_VIDEO_NAME = 'bbox_video.mp4'

# === OUTPUT TOGGLES ===
OUTPUT_LABELS = True  # <--- TOGGLE THIS TO FALSE TO ONLY OUTPUT VIDEO

# === SETUP OUTPUT FOLDERS ===
images_dir = os.path.join(OUTPUT_DIR, 'images')
labels_dir = os.path.join(OUTPUT_DIR, 'labels')
bbox_dir = os.path.join(OUTPUT_DIR, 'bbox')

import shutil
if OUTPUT_LABELS:
    # if the folders already exist, delete them
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    if os.path.exists(bbox_dir):
        shutil.rmtree(bbox_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

# === LOAD MODEL ===
model = YOLO(YOLO_WEIGHTS)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === SETUP VIDEO WRITER FOR OUTPUT VIDEO ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)
video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))

# === FRAME LOOP ===
frame_idx = 0
while frame_idx < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)[0]
    bboxes = []

    for box in results.boxes:
        if box.conf.item() >= CONF_THRESHOLD:
            x_center = float(box.xywh[0][0])
            y_center = float(box.xywh[0][1])
            width = float(box.xywh[0][2])
            height = float(box.xywh[0][3])
            conf = float(box.conf.item())
            cls = int(box.cls.item())
            bboxes.append([x_center, y_center, width, height, conf, cls])

    if bboxes:
        image_name = f"frame_{frame_idx:05d}.jpg"

        if OUTPUT_LABELS:
            # Save original frame
            image_path = os.path.join(images_dir, image_name)
            cv2.imwrite(image_path, frame)

            # Save YOLO label file (normalized, without confidence)
            label_path = os.path.join(labels_dir, f"frame_{frame_idx:05d}.txt")
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    xc, yc, w, h, conf, cls = bbox
                    xc_norm = xc / frame_width
                    yc_norm = yc / frame_height
                    w_norm = w / frame_width
                    h_norm = h / frame_height
                    f.write(f"{cls} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            # Draw and save frame with bbox + confidence
            vis_frame = frame.copy()
            for bbox in bboxes:
                xc, yc, w, h, conf, cls = bbox
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f'{cls} ({conf:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            vis_path = os.path.join(bbox_dir, image_name)
            cv2.imwrite(vis_path, vis_frame)
            video_writer.write(vis_frame)
        else:
            # Only draw on frame and write to video
            vis_frame = frame.copy()
            for bbox in bboxes:
                xc, yc, w, h, conf, cls = bbox
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f'{cls} ({conf:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            video_writer.write(vis_frame)

    if FRAME_JUMP < 0:
        print(f"Skipping frame {frame_idx} (no detections)")
        FRAME_JUMP = 10
    frame_idx += FRAME_JUMP

# === CLEANUP ===
cap.release()
video_writer.release()
print(f"✅ Done. Output saved to '{OUTPUT_DIR}'")
