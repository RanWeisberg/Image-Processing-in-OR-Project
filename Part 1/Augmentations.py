# --- PART 1: Imports and Config ---
import os
import cv2
import numpy as np
from glob import glob
from shutil import copy2
from albumentations import (
    RandomCrop, HueSaturationValue, RandomBrightnessContrast,
    GaussianBlur, GaussNoise, MotionBlur, GlassBlur, Downscale,
    RandomShadow, RandomSunFlare, HorizontalFlip, GridDistortion,
    BboxParams, Compose
)

# CONFIGURATION
current_directory = os.getcwd()
INPUT_DIR = os.path.join(current_directory, 'labeled_image_data')
OUTPUT_DIR = os.path.join(current_directory, 'augmented_dataset')
MODE = 'full'  # 'full' or 'single'
TARGET_AUG = 'motion_blur'  # only used in 'single' mode
SELECTED_AUGMENTATIONS = [
    'rotate', 'crop', 'brightness', 'rgb_dropout', 'motion_blur', 'glass_blur',
    'downscale', 'shadow', 'sun_flare', 'grid_elastic'
]
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

# --- PART 2: Augmentation Functions ---
def rotate_image_custom(img, boxes, angle=10):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    rotated_boxes = []
    for x_min, y_min, x_max, y_max, cls in boxes:
        corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        corners = np.hstack([corners, np.ones((4, 1))])
        transformed = M.dot(corners.T).T
        x_coords, y_coords = transformed[:, 0], transformed[:, 1]
        new_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords), cls]
        rotated_boxes.append(new_box)
    return {'image': rotated_img, 'bboxes': [b[:4] for b in rotated_boxes], 'class_labels': [b[4] for b in rotated_boxes]}

def albumentations_wrapper(augmentation):
    def wrapper(img, boxes):
        aug = Compose([augmentation], bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
        return aug(image=img, bboxes=[b[:4] for b in boxes], class_labels=[b[4] for b in boxes])
    return wrapper

def rgb_channel_dropout(img, boxes, dropout_ratio=0.5):
    out_img = img.copy()
    h, w, _ = out_img.shape
    num_pixels = int(h * w * dropout_ratio)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    channels = np.random.randint(0, 3, num_pixels)
    for y, x, ch in zip(ys, xs, channels):
        out_img[y, x, ch] = 0
    return {'image': out_img, 'bboxes': [b[:4] for b in boxes], 'class_labels': [b[4] for b in boxes]}

AUGMENT_FUNCS = {
    'rotate': lambda img, boxes: rotate_image_custom(img, boxes, angle=10),
    'crop': albumentations_wrapper(RandomCrop(height=400, width=400)),
    'brightness': albumentations_wrapper(RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0)),
    'rgb_dropout': lambda img, boxes: rgb_channel_dropout(img, boxes, dropout_ratio=0.5),
    'motion_blur': albumentations_wrapper(MotionBlur(blur_limit=(19, 21))),
    'glass_blur': albumentations_wrapper(GlassBlur(sigma=0.7, max_delta=4)),
    'downscale': albumentations_wrapper(Downscale(scale_range=(0.1, 0.3))),
    'shadow': albumentations_wrapper(RandomShadow()),
    'sun_flare': albumentations_wrapper(RandomSunFlare()),
    'h_flip': albumentations_wrapper(HorizontalFlip()),
    'grid_elastic': albumentations_wrapper(GridDistortion(num_steps=5, distort_limit=0.3)),
}

# --- PART 3: Utilities ---
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
            x_min, y_min = (x - w / 2) * img_w, (y - h / 2) * img_h
            x_max, y_max = (x + w / 2) * img_w, (y + h / 2) * img_h
            boxes.append([x_min, y_min, x_max, y_max, int(cls)])
    return boxes

def save_labels(label_path, boxes, img_w, img_h):
    with open(label_path, 'w') as f:
        for x_min, y_min, x_max, y_max, cls in boxes:
            x = (x_min + x_max) / 2 / img_w
            y = (y_min + y_max) / 2 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# --- PART 4: Augmentation Logic ---
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

        if MODE == 'full':
            copy2(img_path, os.path.join(out_img_path, base + '.jpg'))
            copy2(lbl_path, os.path.join(out_lbl_path, base + '.txt'))

        for aug_name in SELECTED_AUGMENTATIONS:
            if MODE == 'single' and aug_name != TARGET_AUG:
                continue

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

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print(f"✨ Starting in {MODE.upper()} mode...")
    if MODE == 'single':
        print(f"🎯 Targeting augmentation: {TARGET_AUG}")
    for split in ['train', 'val']:
        augment_folder(split)
    print(f"\n✅ Done. Output saved to {OUTPUT_DIR}")
