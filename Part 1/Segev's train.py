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

# ─── Logger setup ─────────────────────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(ch)
    return logger

LOGGER = setup_logger()

# ─── Data utilities ────────────────────────────────────────────────────────────
class YOLOData:
    def __init__(self, control):
        self.control   = control
        self.train_dir = control['folders']['train']
        self.val_dir   = control['folders']['val']
        LOGGER.info(f"Initialized YOLOData with train={self.train_dir}, val={self.val_dir}")

    def copy_data(self, input_folders, output_folder, delete_existing=True):
        """
        Copy images+labels from multiple input_folders into one YOLO-style
        dataset at output_folder/images + output_folder/labels.
        Each input folder must contain 'images/' and 'labels/'.
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

                img_dst = os.path.join(out_img, f'image_{count}.png')
                lbl_dst = os.path.join(out_lbl, f'image_{count}.txt')
                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)
                count += 1

        LOGGER.info(f"Copied {count} samples to {output_folder}")
        return count
    def apply_augmentation(self, input_dir, output_dir, show_number=5, p=0.3, data_factor=1.0, p_noise=1):
        #https://explore.albumentations.ai

        out_images = os.path.join(output_dir, "images")
        out_labels = os.path.join(output_dir, "labels")
        #delete the folders if they already exist
        if os.path.isdir(out_images):
            print(f"⚠️ [INFO] Deleting existing folder: {out_images}")
            shutil.rmtree(out_images)
        if os.path.isdir(out_labels):
            print(f"⚠️ [INFO] Deleting existing folder: {out_labels}")
            shutil.rmtree(out_labels)
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)


        #randomly images and labels from input_dir

        T = [
            A.Affine(rotate=(-180, 180), scale=(0.5, 1), translate_percent= (-0.1,0.1), shear = (-15,15),p=p),
            A.GaussNoise(p=p_noise, std_range=(0.295, 0.3), per_channel=True),
            # A.RandomCropFromBorders(crop_left=0.2, crop_right=0.2, crop_top=0.2, crop_bottom=0.2, p=p),
            # A.Perspective(scale=(0.05, 0.1), fit_output=True, keep_size=True, p=p),
            A.RandomToneCurve(scale=0.2, per_channel=False, p=p),
            A.SaltAndPepper(amount = (0.1, 0.2),  salt_vs_pepper = (0.3, 0.8),  p =p),
            A.RandomSunFlare(p=p),  # from 'sun_flare'
            # A.Downscale(scale_min=0.1, scale_max=0.3, p=p),  # from 'downscale'
            A.D4(p=1.0),
            A.MotionBlur(blur_limit=(19, 21), p=p),
            A.HueSaturationValue(hue_shift_limit = 20,sat_shift_limit = 30,val_shift_limit = 20,p =1),
            A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=p),
            # A.Downscale(p=p / 5, scale_range=(0.25, 0.75)),
            # A.ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.1, 0.1), rotate_limit=(-22, 22), p=p)
        ]
        #
        # T = [
        #     # A.Rotate(limit=10, p=p),  # from 'rotate'
        #     # A.RandomCrop(height=400, width=400, p=p),  # from 'crop'
        #     A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=p),  # from 'brightness'
        #     A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=p),  # approximate 'rgb_dropout'
        #     A.MotionBlur(blur_limit=(19, 21), p=p),  # from 'motion_blur'
        #     A.GlassBlur(sigma=0.7, max_delta=4, p=p),  # from 'glass_blur'
        #     A.Downscale(scale_min=0.1, scale_max=0.3, p=p),  # from 'downscale'
        #     A.RandomShadow(p=p),  # from 'shadow'
        #     A.RandomSunFlare(p=p),  # from 'sun_flare'
        #     A.HorizontalFlip(p=p),  # from 'h_flip'
        #     A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),  # from 'grid_elastic'
        # ]
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
        #copy randomly images and labels from input_dir to input_dir to create data factor
        dataset_length = len(data_pairs)
        if data_factor > 1.0:
            extra_count = int(dataset_length * (data_factor - 1))
            print(f"✅ [INFO] Data factor > 1.0, adding {extra_count} extra samples (×{data_factor} total).")
            # sample with replacement
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

    def create_yaml(self, output_path, train_path=None, val_path=None, names=None, check_split=True):
        """
        Writes a YOLO data.yaml listing train/val dirs and class names.
        If val set is too small compared to train, a new split is created (80/20).
        """
        # Resolve paths
        train_path = train_path or self.train_dir
        val_path = val_path or self.val_dir

        train_img_dir = os.path.join(train_path, "images")
        train_lbl_dir = os.path.join(train_path, "labels")
        val_img_dir = os.path.join(val_path, "images")
        val_lbl_dir = os.path.join(val_path, "labels")

        train_imgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        val_imgs = sorted([f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

        # Automatically re-split if val < 20% of train
        if len(val_imgs) < 0.2 * len(train_imgs) :
            LOGGER.warning("⚠️ Validation set too small. Re-splitting train/val to 80/20 ratio...")

            all_pairs = [(f, os.path.splitext(f)[0] + ".txt") for f in train_imgs]
            random.shuffle(all_pairs)

            val_count = int(0.2 * len(all_pairs))
            val_pairs = all_pairs[:val_count]
            train_pairs = all_pairs[val_count:]

            # Create temp dirs
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

            # Update paths
            train_path = train_temp
            val_path = val_temp
            self.train_dir = train_temp
            self.val_dir = val_temp
            LOGGER.info(f"✅ New split created: {len(train_pairs)} train, {len(val_pairs)} val")

        # Write the YAML file
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

# ─── Training ──────────────────────────────────────────────────────────────────
def train_model(control, yaml_path):
    s = control['train_settings']
    model = YOLO(s['model_type'])
    LOGGER.info(f"Training {s['model_type']} for {s['epochs']} epochs…")
    # model.train(data=yaml_file, epochs=epochs, batch=batch_size, device=device, cos_lr=True, dropout=drpout,
    #             workers=workers, name=model_name, pretrained=pretrained, cls=cls, box=box, dfl=dfl,
    #             augment=yolo_augment, freeze=freeze)
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
    )
    LOGGER.info("Training completed.")

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    #%% set device
    if torch.cuda.is_available():
        device = "0"
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        device = "cpu"
    print("device", device)
    CONTROL = {
        'folders': {
            'main_folder': r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\train_data",
            "simple_labels": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\train_data\train",
            'train': r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\train_data\train",
            'val':   r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\train_data\val",
            'video1': r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\video_output_20_2_24_1",
            'video2': r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\video_output_4_2_24_B_2",
            'augmented': r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\train_data\augmented_dataset"
        },
        'train_settings': {
            'model_type': 'yolo11m',
            'epochs':      100,
            'batch_size':  12,
            'device':      device, # '0' for GPU, 'cpu' for CPU
            'model_name': 'segev_yolo11n',
            'pretrained': True,
            'freeze':     9,
            'cos_lr': True,  # Use cosine learning rate scheduler
            'dropout': 0.1,  # Set to 0.0 if not using dropout
        }
    }

    ydata = YOLOData(CONTROL)
    copy_folders = [CONTROL['folders']['simple_labels'], CONTROL['folders']['video1'], CONTROL['folders']['video2']]
    training_path = CONTROL['folders']['main_folder'] + '/all_train'
    ydata.copy_data(input_folders=copy_folders,output_folder=training_path, delete_existing=True)
    CONTROL['folders']['train'] = training_path
    # ydata.apply_augmentation(CONTROL['folders']['train'], CONTROL['folders']['augmented'], show_number=5, p=0.5, data_factor=6, p_noise=0.2)
    # CONTROL['folders']['train'] = CONTROL['folders']['augmented']
    yaml_file = ydata.create_yaml(
        output_path= CONTROL['folders']['main_folder'] + '/data.yaml',
        train_path = CONTROL['folders']['train'],
        val_path   = CONTROL['folders']['val'],
        names      = ['Empty', 'Tweezers', 'Needle_driver']
    )

    # 3) Launch training:
    train_model(CONTROL, yaml_file)
