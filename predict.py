import os
import cv2
from ultralytics import YOLO

weights_rel_path = "models/aug_test/yolo11s_augmented/weights/best.pt"
input_image_path = "labeled_image_data/images/train/a2ca750f-frame_2832.jpg"
output_dir = "image_results"
os.makedirs(output_dir, exist_ok=True)

model = YOLO(weights_rel_path)
img = cv2.imread(input_image_path)
if img is None:
    print(f"Warning: Could not open {input_image_path}")
    exit()

height, width = img.shape[:2]
results = model(img[..., ::-1])[0]

label_lines = []
for box in results.boxes:
    x1, y1, x2, y2 = map(float, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height

    label_lines.append(
        f"{x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f} {cls}"
    )

    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    label = model.names[cls]
    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1i, y1i-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

base_name = os.path.splitext(os.path.basename(input_image_path))[0]
out_img_path = os.path.join(output_dir, f"{base_name}_boxes.jpg")
out_txt_path = os.path.join(output_dir, f"{base_name}.txt")
cv2.imwrite(out_img_path, img)
with open(out_txt_path, "w") as f:
    for line in label_lines:
        f.write(line + "\n")

print(f"Processed {input_image_path}: detections saved to {out_txt_path}, visualization to {out_img_path}")
