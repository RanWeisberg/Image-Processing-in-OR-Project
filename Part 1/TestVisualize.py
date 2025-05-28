import cv2
import os

def load_yolo_labels(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def draw_boxes(image_path, label_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    boxes = load_yolo_labels(label_path, w, h)
    if not boxes:
        print("⚠️ No boxes found or label format issue.")
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"Class {int(cls)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === EXAMPLE USAGE ===
if __name__ == '__main__':
    # Provide full paths here
    image_path = '/Users/ranweisberg/PycharmProjects/Image Processing in OR Project/Image-Processing-in-OR-Project/Part 1/augmented_dataset/images/train/1c0b1584-frame_1789_rgb_dropout.jpg'
    label_path = '/Users/ranweisberg/PycharmProjects/Image Processing in OR Project/Image-Processing-in-OR-Project/Part 1/augmented_dataset/labels/train/1c0b1584-frame_1789_rgb_dropout.txt'

    draw_boxes(image_path, label_path)
