import os
import shutil
from collections import defaultdict, Counter
from glob import glob
import random
import matplotlib.pyplot as plt

# === Config ===
LABEL_DIR = 'labeled_image_data/labels/train'
IMAGE_DIR = 'labeled_image_data/images/train'
BALANCED_ROOT = 'balanced_dataset'
BALANCED_IMAGE_DIR = os.path.join(BALANCED_ROOT, 'images', 'train')
BALANCED_LABEL_DIR = os.path.join(BALANCED_ROOT, 'labels', 'train')

# === Step 1: Analyze class distribution ===
def analyze_class_distribution(label_dir):
    class_counts = Counter()
    label_map = defaultdict(list)  # class_id -> list of label file paths

    label_files = glob(os.path.join(label_dir, '*.txt'))

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
                label_map[class_id].append(label_file)

    print("\n📊 Class Distribution Report:")
    total = sum(class_counts.values())
    for cls_id, count in sorted(class_counts.items()):
        print(f"Class {cls_id}: {count} ({100 * count / total:.2f}%)")

    # Optional: plot bar chart
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()

    return class_counts, label_map

# === Step 2: Create a balanced dataset ===
def create_balanced_dataset(label_map, class_counts):
    # Create new folders
    os.makedirs(BALANCED_IMAGE_DIR, exist_ok=True)
    os.makedirs(BALANCED_LABEL_DIR, exist_ok=True)

    # Target count is the minimum across all classes
    target_count = min(class_counts.values())
    print(f"\n📦 Balancing dataset to {target_count} samples per class")

    used_files = set()
    for class_id, files in label_map.items():
        selected_files = random.sample(list(set(files)), target_count)
        for label_file in selected_files:
            if label_file in used_files:
                continue  # avoid duplicate copies if label has multiple class lines
            used_files.add(label_file)

            # Copy label
            filename = os.path.basename(label_file)
            dst_label = os.path.join(BALANCED_LABEL_DIR, filename)
            shutil.copy(label_file, dst_label)

            # Copy corresponding image
            img_name = filename.replace('.txt', '.jpg')
            src_img = os.path.join(IMAGE_DIR, img_name)
            dst_img = os.path.join(BALANCED_IMAGE_DIR, img_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"⚠️ Image not found for label: {filename}")

    print("\n✅ Balanced dataset created at:", BALANCED_ROOT)

# === Run ===
if __name__ == "__main__":
    class_counts, label_map = analyze_class_distribution(LABEL_DIR)
    create_balanced_dataset(label_map, class_counts)
