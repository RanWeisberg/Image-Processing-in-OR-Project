import os
import cv2
import yaml
import random
import numpy as np
from glob import glob
from shutil import copy2
from albumentations import (
    Rotate, RandomCrop, HueSaturationValue, RandomBrightnessContrast,
    GaussianBlur, GaussNoise, BboxParams, Compose
)

# === CONFIGURATION ===
current_directory = os.getcwd()
INPUT_DIR = ORIGINAL_DATASET_DIR = os.path.join(current_directory,'labeled_image_data')
OUTPUT_DIR = os.path.join(current_directory, 'augmented_dataset')
SELECTED_AUGMENTATIONS = ['rotate', 'crop', 'brightness', 'contrast', 'hue', 'blur', 'noise']
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

# === AUGMENTATION FUNCTIONS ===
def rotate_image(img, boxes):
    aug = Compose([Rotate(limit=10, p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def crop_image(img, boxes):
    h, w = img.shape[:2]
    crop_size = int(0.9 * min(w, h))
    aug = Compose([RandomCrop(height=crop_size, width=crop_size, p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def brightness_image(img, boxes):
    aug = Compose([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def contrast_image(img, boxes):
    aug = Compose([RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def hue_image(img, boxes):
    aug = Compose([HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def blur_image(img, boxes):
    aug = Compose([GaussianBlur(blur_limit=(3, 5), p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

def noise_image(img, boxes):
    aug = Compose([GaussNoise(var_limit=(10.0, 50.0), p=1.0)],
                  bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])

AUGMENT_FUNCS = {
    'rotate': rotate_image,
    'crop': crop_image,
    'brightness': brightness_image,
    'contrast': contrast_image,
    'hue': hue_image,
    'blur': blur_image,
    'noise': noise_image
}

def get_image_label_pairs(image_dir, label_dir):
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob(os.path.join(image_dir, f'*{ext}')))
    pairs = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f'{basename}.txt')
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    return pairs

def load_labels(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            x_min = (x - w / 2) * img_w
            y_min = (y - h / 2) * img_h
            x_max = (x + w / 2) * img_w
            y_max = (y + h / 2) * img_h
            boxes.append([x_min, y_min, x_max, y_max, int(cls)])
    return boxes

def save_labels(label_path, boxes, img_w, img_h):
    with open(label_path, 'w') as f:
        for box in boxes:
            x_min, y_min, x_max, y_max, cls = box
            x = (x_min + x_max) / 2 / img_w
            y = (y_min + y_max) / 2 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def augment_folder(split):
    image_dir = os.path.join(INPUT_DIR, 'images', split)
    label_dir = os.path.join(INPUT_DIR, 'labels', split)
    out_img_path = os.path.join(OUTPUT_DIR, 'images', split)
    out_lbl_path = os.path.join(OUTPUT_DIR, 'labels', split)
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_lbl_path, exist_ok=True)

    pairs = get_image_label_pairs(image_dir, label_dir)
    print(f"\n📁 Processing {len(pairs)} samples from {split}")

    for img_path, lbl_path in pairs:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = load_labels(lbl_path, w, h)
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Copy original
        copy2(img_path, os.path.join(out_img_path, base + '.jpg'))
        copy2(lbl_path, os.path.join(out_lbl_path, base + '.txt'))

        for aug_name in SELECTED_AUGMENTATIONS:
            aug_func = AUGMENT_FUNCS[aug_name]
            try:
                result = aug_func(img.copy(), boxes)
                aug_img = result['image']
                aug_boxes = [
                    list(b) + [cls] for b, cls in zip(result['bboxes'], result['class_labels'])
                    if b[2] > b[0] and b[3] > b[1]
                ]
                if aug_boxes:
                    aug_base = f"{base}_{aug_name}"
                    cv2.imwrite(os.path.join(out_img_path, aug_base + '.jpg'), aug_img)
                    save_labels(os.path.join(out_lbl_path, aug_base + '.txt'),
                                aug_boxes, aug_img.shape[1], aug_img.shape[0])
            except Exception as e:
                print(f"❌ Error applying {aug_name} to {base}: {e}")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print(f"✨ Starting augmentation with: {SELECTED_AUGMENTATIONS}")
    for split in ['train', 'val']:
        augment_folder(split)
    print(f"\n✅ All augmented data saved in: {OUTPUT_DIR}")
