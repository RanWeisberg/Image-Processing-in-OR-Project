import cv2
import os
from ultralytics import YOLO

# Paths
main_folder =os.path.dirname(os.path.abspath(__file__))

weights_rel_path = os.path.join(main_folder, "Part 1", "aug_test", "yolo11s_augmented", "weights", "best.pt")
# C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\aug_test\yolo11s_augmented
video_rel_path = os.path.join(main_folder, "train_data", 'surg_1.mp4')
output_dir = "Part 1/video_results"

# Ensure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Output file names
output_video_path = os.path.join(output_dir, "output_with_boxes.mp4")
output_labels_path = os.path.join(output_dir, "yolo_bboxes_results.txt")

# Model and video
model = YOLO(weights_rel_path)
cap = cv2.VideoCapture(video_rel_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_rel_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_number = 0
with open(output_labels_path, "w") as label_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame[..., ::-1])[0]

        # Draw and write YOLO format results
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            # Save result: frame, x_center, y_center, w, h, conf, class
            label_file.write(
                f"{frame_number} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f} {cls}\n"
            )

            # Draw rectangle and label on frame
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1i, y1i-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow('YOLO Live Video', frame)
        out.write(frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

        if frame_number % 50 == 0:
            print(f"Processed frame {frame_number}")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Results saved to {output_video_path} and {output_labels_path}")
