import os
import shutil
import logging
import yaml
import cv2
from ultralytics import YOLO
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import albumentations as A
import torch

# ──────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────

def setup_logger():
    """
    Sets up a basic logger with INFO level and a simple stream handler.
    Prevents duplicate handlers in case of repeated calls.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(ch)
    return logger

LOGGER = setup_logger()

# ──────────────────────────────────────────────────────────────
# Data utilities for YOLO datasets
# ──────────────────────────────────────────────────────────────

class YOLOData:
    """
    Utility class for manipulating YOLO dataset folders, visualizing samples,
    copying/merging datasets, augmenting images+labels, and generating YAML configs.
    """
    def __init__(self, control):
        # Parse control dictionary for relevant paths
        self.control   = control
        self.train_dir = control['folders']['train']
        self.val_dir   = control['folders']['val']
        LOGGER.info(f"Initialized YOLOData with train={self.train_dir}, val={self.val_dir}")

    def debug_load_images_and_labels(self, folder=None, index=None, image_path=None, label_path=None):
        """
        Debug function for visualizing YOLO images and their bounding boxes.

        (A) If folder and index are given: randomly sample 'index' images+labels from the folder.
        (B) If image_path and label_path are given: visualize that single pair.
        Draws images using matplotlib with rectangles for each bounding box.
        """
        images_list = []
        labels_list = []

        if folder is not None and index is not None:
            print(f"✅ [INFO] Using folder='{folder}' and sampling {index} pairs.")
            images_dir = os.path.join(folder, "images")
            labels_dir = os.path.join(folder, "labels")
            # Only .png images considered here (can add more extensions)
            all_img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
            all_lbl_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
            all_img_files.sort()
            all_lbl_files.sort()
            if index > len(all_img_files):
                index = len(all_img_files)
                print(f"⚠️⚠️⚠️ [WARNING] Index too large, using {index} instead.")
            random_indices = random.sample(range(len(all_img_files)), index)
            for i in random_indices:
                images_list.append(os.path.join(images_dir, all_img_files[i]))
                labels_list.append(os.path.join(labels_dir, all_lbl_files[i]))
            print(f"✅ [INFO] Collected {len(images_list)} image paths and {len(labels_list)} label paths.")
        elif image_path and label_path:
            print("✅ [INFO] Using explicit single file paths.")
            print("       Image:", image_path)
            print("       Label:", label_path)
            images_list = [image_path]
            labels_list = [label_path]
        else:
            print("❌ [DEBUG] Provide either (folder & index) or (image_path & label_path).")
            return

        # Iterate through image-label pairs and plot bboxes
        for (img_path, lbl_path) in zip(images_list, labels_list):
            print(f"✅ [INFO] Attempting to load image: {img_path}")
            if not os.path.isfile(img_path):
                print(f"❌ [DEBUG] Missing image: {img_path}, skipping.")
                continue
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"❌ [DEBUG] Cannot read image: {img_path}, skipping.")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]
            print(f"✅ [INFO] Attempting to load labels: {lbl_path}")
            if not os.path.isfile(lbl_path):
                print(f"❌ [DEBUG] Missing label: {lbl_path}, skipping.")
                continue
            with open(lbl_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            boxes = []
            for line in lines:
                # Each label line is expected to be: cls_id x_center y_center width height (all normalized)
                parts = line.split()
                if len(parts) != 5:
                    print(f"⚠️ [DEBUG] Invalid line => '{line}'")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                except ValueError:
                    print(f"❌ [DEBUG] Could not parse => '{line}'")
                    continue
                # Convert normalized bbox to absolute pixel coordinates
                x_center_abs = x_center * img_w
                y_center_abs = y_center * img_h
                w_abs = w_norm * img_w
                h_abs = h_norm * img_h
                x1 = x_center_abs - w_abs / 2
                y1 = y_center_abs - h_abs / 2
                boxes.append((cls_id, x1, y1, w_abs, h_abs))
            print(f"✅ [INFO] Found {len(boxes)} bounding boxes in {lbl_path}")
            # Plot image and bounding boxes
            fig, ax = plt.subplots(1)
            ax.imshow(image_rgb)
            for (cls_id, x, y, w_box, h_box) in boxes:
                rect = patches.Rectangle((x, y), w_box, h_box,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"{cls_id}", fontsize=6, backgroundcolor='white')
            plt.title(f"Image + BBoxes\n{os.path.basename(img_path)}")
            ax.axis('off')
            plt.show()
        print("✅ [INFO] Done displaying images + bounding boxes.")

    def copy_data(self, input_folders, output_folder, delete_existing=True):
        """
        Merges multiple input folders into a single YOLO dataset at output_folder.
        All images/labels are renamed as image_{index}.png/txt to avoid collisions.
        """
        out_img = os.path.join(output_folder, 'images')
        out_lbl = os.path.join(output_folder, 'labels')
        if delete_existing and os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        count = 0

        for folder in input_folders:
            print(f"Processing folder: {folder}")
            imgs = os.path.join(folder, 'images')
            lbls = os.path.join(folder, 'labels')
            if not os.path.isdir(imgs) or not os.path.isdir(lbls):
                LOGGER.warning(f"Skipping {folder}: missing images/ or labels/")
                continue

            for fname in sorted(os.listdir(imgs)):
                if not fname.lower().endswith(exts):
                    continue
                img_src = os.path.join(imgs, fname)
                lbl_src = os.path.join(lbls, os.path.splitext(fname)[0] + '.txt')
                if not os.path.isfile(lbl_src):
                    LOGGER.warning(f"No label for {fname}, skipping")
                    continue

                # Use a running index to guarantee unique filenames
                img_dst = os.path.join(out_img, f'image_{count}.png')
                lbl_dst = os.path.join(out_lbl, f'image_{count}.txt')
                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)
                count += 1

        LOGGER.info(f"Copied {count} samples to {output_folder}")
        return count

    def apply_augmentation(self, input_dir, output_dir, show_number=5, p=0.3, data_factor=1.0, p_noise=1):
        """
        Augments images and labels in YOLO format with Albumentations.
        - input_dir: original data folder (must contain images/ and labels/)
        - output_dir: where augmented images/labels are saved
        - show_number: number of samples to visualize after augmentation
        - p: probability for most augmentations
        - data_factor: output dataset will be this multiple of input size
        - p_noise: probability for noise augmentation (can be >1 to always apply)
        """
        out_images = os.path.join(output_dir, "images")
        out_labels = os.path.join(output_dir, "labels")
        # Remove any existing output directory
        if os.path.isdir(out_images):
            print(f"⚠️ [INFO] Deleting existing folder: {out_images}")
            shutil.rmtree(out_images)
        if os.path.isdir(out_labels):
            print(f"⚠️ [INFO] Deleting existing folder: {out_labels}")
            shutil.rmtree(out_labels)
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        # Define Albumentations transforms (customize as needed)
        T = [
            A.Affine(rotate=(-180, 180), scale=(0.5, 1), translate_percent=(-0.1, 0.1), shear=(-15, 15), p=p),
            A.GaussNoise(p=p_noise, std_range=(0.295, 0.3), per_channel=True),
            A.RandomToneCurve(scale=0.2, per_channel=False, p=p),
            A.RandomSunFlare(p=p),
            A.D4(p=1.0),  # Random flips and rotations
            A.MotionBlur(blur_limit=(19, 21), p=p),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=p),
        ]
        transform = A.Compose(
            T,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
                min_area=0.2,
                clip=True,
                filter_invalid_bboxes=True,
                check_each_transform=True)
        )
        print("✅ [INFO] Augmentation pipeline created.")
        data_pairs = YOLOData.load_data(input_dir)
        if not data_pairs:
            print(f"❌ [ERROR] No images+labels found in {input_dir}")
            return
        # Duplicate or subsample dataset according to data_factor
        dataset_length = len(data_pairs)
        if data_factor > 1.0:
            extra_count = int(dataset_length * (data_factor - 1))
            print(f"✅ [INFO] Data factor > 1.0, adding {extra_count} extra samples (×{data_factor} total).")
            extra_pairs = random.choices(data_pairs, k=extra_count)
            data_pairs = data_pairs + extra_pairs
        elif data_factor < 1.0:
            print(f"✅ [INFO] Data factor < 1.0, sampling {int(dataset_length * data_factor)} samples.")
            data_pairs = random.sample(data_pairs, int(dataset_length * data_factor))

        print(f"✅ [INFO] Found {len(data_pairs)} pairs in {input_dir}. Starting augmentation...")
        saved_pairs = []
        for i, (img_path, lbl_path) in enumerate(data_pairs):
            if i % 15 == 0:
                print(f"✅ [INFO] Processing pair {i}/{len(data_pairs)}")
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"❌ [WARN] Could not read {img_path}, skipping.")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            with open(lbl_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            bboxes_yolo = []
            class_labels = []
            for line in lines:
                # Read label in YOLO format (cls_id x_center y_center width height)
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    ww = float(parts[3])
                    hh = float(parts[4])
                except ValueError:
                    continue
                bboxes_yolo.append([x_c, y_c, ww, hh])
                class_labels.append(cls_id)
            if len(bboxes_yolo) == 0:
                print(f"⚠️ [WARN] No boxes in {lbl_path}, skipping.")
                continue
            # Apply augmentation
            transformed = transform(
                image=image_rgb,
                bboxes=bboxes_yolo,
                class_labels=class_labels
            )
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]
            if len(aug_boxes) == 0:
                print(f"⚠️ [INFO] Aug transform removed all boxes from {img_path}. Skipping.")
                continue
            # Save augmented image and label
            base_in = os.path.splitext(os.path.basename(img_path))[0]
            aug_img_name = f"{base_in}_aug_{i}.png"
            aug_lbl_name = f"{base_in}_aug_{i}.txt"
            aug_img_path = os.path.join(out_images, aug_img_name)
            aug_lbl_path = os.path.join(out_labels, aug_lbl_name)
            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            with open(aug_lbl_path, "w") as f:
                for cls_id, (xc, yc, ww, hh) in zip(aug_labels, aug_boxes):
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
            saved_pairs.append((aug_img_path, aug_lbl_path))
        print(f"✅ [INFO] Saved {len(saved_pairs)} augmented pairs into {output_dir}.")

        # Visualize random samples of the augmented dataset
        if len(saved_pairs) == 0:
            print("⚠️ [INFO] No saved pairs to display.")
            return
        show_num = min(show_number, len(saved_pairs))
        chosen = random.sample(saved_pairs, show_num)
        for (img_path, lbl_path) in chosen:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]
            with open(lbl_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            boxes = []
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c = float(parts[1]) * img_w
                    y_c = float(parts[2]) * img_h
                    w_ = float(parts[3]) * img_w
                    h_ = float(parts[4]) * img_h
                except ValueError:
                    continue
                x1 = x_c - w_ / 2
                y1 = y_c - h_ / 2
                boxes.append((cls_id, x1, y1, w_, h_))
            fig, ax = plt.subplots(1)
            ax.imshow(image_rgb)
            for (cls, x, y, bw, bh) in boxes:
                rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"Class: {cls}", fontsize=12, backgroundcolor='white')
            plt.title(f"Augmented {os.path.basename(img_path)}")
            plt.show()
        print("✅ [INFO] Done displaying random augmented images.")

    def create_yaml(self, output_path, train_path=None, val_path=None, names=None, check_split=False):
        """
        Writes a data.yaml for YOLO training. Optionally, can force a new 80/20 train/val split
        if the validation set is too small.
        """
        # Resolve provided paths, or fall back to object's train/val dirs
        train_path = train_path or self.train_dir
        val_path = val_path or self.val_dir

        train_img_dir = os.path.join(train_path, "images")
        train_lbl_dir = os.path.join(train_path, "labels")
        val_img_dir = os.path.join(val_path, "images")
        val_lbl_dir = os.path.join(val_path, "labels")

        train_imgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        val_imgs = sorted([f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

        # Optionally, re-split to 80/20 if val set is too small
        if len(val_imgs) < 0.2 * (len(train_imgs)+len(val_imgs)):
            if not check_split:
                LOGGER.warning(f"⚠️ Validation set too small: {len(val_imgs)} < 20% of {len(train_imgs)}. but check_split is False, skipping re-split.")
            else:
                LOGGER.warning("⚠️ Validation set too small. Re-splitting train/val to 80/20 ratio...")

                all_pairs = [(f, os.path.splitext(f)[0] + ".txt") for f in train_imgs]
                random.shuffle(all_pairs)

                val_count = int(0.2 * len(all_pairs))
                val_pairs = all_pairs[:val_count]
                train_pairs = all_pairs[val_count:]

                # Create new temp train/val directories
                train_temp = train_path + "_temp"
                val_temp = val_path + "_temp"
                for d in [train_temp, val_temp]:
                    os.makedirs(os.path.join(d, "images"), exist_ok=True)
                    os.makedirs(os.path.join(d, "labels"), exist_ok=True)

                def copy_pair(pair_list, src_img, src_lbl, dest_dir):
                    for img_f, lbl_f in pair_list:
                        src_img_path = os.path.join(src_img, img_f)
                        src_lbl_path = os.path.join(src_lbl, lbl_f)
                        if not (os.path.exists(src_img_path) and os.path.exists(src_lbl_path)):
                            continue
                        shutil.copy(src_img_path, os.path.join(dest_dir, "images", img_f))
                        shutil.copy(src_lbl_path, os.path.join(dest_dir, "labels", lbl_f))

                copy_pair(train_pairs, train_img_dir, train_lbl_dir, train_temp)
                copy_pair(val_pairs, train_img_dir, train_lbl_dir, val_temp)

                # Update paths to new split
                train_path = train_temp
                val_path = val_temp
                self.train_dir = train_temp
                self.val_dir = val_temp
                LOGGER.info(f"✅ New split created: {len(train_pairs)} train, {len(val_pairs)} val")

        # Generate and save YAML file
        data = {
            'train': train_path,
            'val': val_path,
            'nc': len(names) if names else None,
            'names': names
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(data, f)
        LOGGER.info(f"YAML saved to {output_path}")
        return output_path

    def load_data(folder):
        """
        Static method to load paired lists of (image_path, label_path) from YOLO-style folders.
        Returns only pairs where both image and label exist.
        """
        images_dir = os.path.join(folder, "images")
        labels_dir = os.path.join(folder, "labels")
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        label_files = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")])
        pairs = []
        for img_name in image_files:
            base, _ = os.path.splitext(img_name)
            lbl_name = base + ".txt"
            if lbl_name in label_files:
                pairs.append((os.path.join(images_dir, img_name),
                              os.path.join(labels_dir, lbl_name)))
        return pairs

# ──────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────

def train_model(control, yaml_path):
    """
    Trains a YOLO model using Ultralytics YOLO Python API and parameters in control dict.
    """
    s = control['train_settings']
    model = YOLO(s['model_type'])
    LOGGER.info(f"Training {s['model_type']} for {s['epochs']} epochs…")
    # Start training
    model.train(
        data      = yaml_path,
        epochs    = s['epochs'],
        batch     = s['batch_size'],
        device    = s['device'],
        name      = s['model_name'],
        pretrained= s['pretrained'],
        freeze    = s['freeze'],
        cos_lr=True,  # Use cosine learning rate scheduler
        dropout=    s['dropout'],  # Set to 0.0 if not using dropout
        project   = s['project'],
    )
    # Run validation at end of training and save metrics
    model.val(data=yaml_path, save_json=True, save_txt=True, conf = 0.8)
    LOGGER.info("Training completed.")

# ──────────────────────────────────────────────────────────────
# Main script logic
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Set up device for training (GPU or CPU)
    if torch.cuda.is_available():
        device = "0"
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        device = "cpu"
    print("device", device)
    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("current_folder", current_folder)
    # Define all paths and parameters in one dictionary
    CONTROL = {
        'folders': {
            'main': current_folder,
            'main_train_folder': os.path.join(current_folder, "train_data"),
            'train': os.path.join(current_folder, "train_data", "train"),
            'val': os.path.join(current_folder, "train_data", "val"),
            'all_train_data': os.path.join(current_folder, "train_data", "all_train"),
            'labeld_data': os.path.join(current_folder, "train_data", "labeld_data"),
            'video1': os.path.join(current_folder, "video_output_20_2_24_1"),
            'video2': os.path.join(current_folder, "video_output_4_2_24_B_2"),
            'augmented': os.path.join(current_folder, "train_data", "augmented_dataset"),
        },
        'train_settings': {
            'model_type': 'yolo11m',
            'epochs':      100,
            'batch_size':  12,
            'device':      device, # '0' for GPU, 'cpu' for CPU
            'model_name': 'yolo11n',
            'pretrained': True,
            'freeze':     9,
            'cos_lr': True,  # Use cosine learning rate scheduler
            'dropout': 0.1,  # Set to 0.0 if not using dropout
            'project': 'aug_test'
        }
    }

    # Build data.yaml for YOLO and start training
    yaml_path = os.path.join(CONTROL['folders']['main_train_folder'], 'data.yaml')
    ydata = YOLOData(CONTROL)

    # Merge labeled folders to one dataset
    copy_folders = [CONTROL['folders']['labeld_data']]
    training_path = CONTROL['folders']['all_train_data']
    ydata.copy_data(input_folders=copy_folders,output_folder=training_path, delete_existing=True)
    CONTROL['folders']['train'] = training_path

    # Uncomment for augmentation step
    # ydata.apply_augmentation(CONTROL['folders']['train'], CONTROL['folders']['augmented'], show_number=5, p=0.5, data_factor=6, p_noise=0.2)
    CONTROL['folders']['train'] = CONTROL['folders']['augmented']

    # Create the YAML config for YOLO training
    yaml_file = ydata.create_yaml(
        output_path= yaml_path,
        train_path = CONTROL['folders']['train'],
        val_path   = CONTROL['folders']['val'],
        names      = ['Empty', 'Tweezers', 'Needle_driver'],
        check_split = False
    )

    # Train several model variants with different sizes
    CONTROL['train_settings']['model_name'] = 'yolo11n_augmented'
    CONTROL['train_settings']['model_type'] = 'yolo11n'
    train_model(CONTROL, yaml_file)
    CONTROL['train_settings']['model_name'] = 'yolo11s_augmented'
    CONTROL['train_settings']['model_type'] = 'yolo11s'
    train_model(CONTROL, yaml_file)
    CONTROL['train_settings']['model_name'] = 'yolo11m_augmented'
    CONTROL['train_settings']['model_type'] = 'yolo11m'
    train_model(CONTROL, yaml_file)
