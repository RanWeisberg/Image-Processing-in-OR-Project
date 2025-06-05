import os
from collections import Counter
from glob import glob
import random
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATASET_DIR = 'labeled_image_data'
PLOT_DIR = os.path.join(DATASET_DIR, 'distribution_plots')
IMAGE_EXT = '.jpg'
SETS = ['train', 'val']

IMAGE_DIR = lambda s: os.path.join(DATASET_DIR, 'images', s)
LABEL_DIR = lambda s: os.path.join(DATASET_DIR, 'labels', s)

# === UTILITIES ===
def parse_labels(label_path):
    try:
        with open(label_path, 'r') as f:
            return [int(line.split()[0]) for line in f if line.strip()]
    except:
        return []

def collect_samples():
    dataset = {'train': [], 'val': []}
    for split in SETS:
        label_files = glob(os.path.join(LABEL_DIR(split), '*.txt'))
        for label_path in label_files:
            img_name = os.path.basename(label_path).replace('.txt', IMAGE_EXT)
            img_path = os.path.join(IMAGE_DIR(split), img_name)
            if not os.path.exists(img_path):
                continue
            class_ids = parse_labels(label_path)
            if class_ids:
                dataset[split].append({'image': img_path, 'label': label_path, 'classes': class_ids})
    return dataset

def compute_class_distribution(samples):
    class_counts = Counter()
    for item in samples:
        class_counts.update(item['classes'])
    return class_counts

def report_distribution(samples, title, save_path=None):
    class_counts = compute_class_distribution(samples)

    print(f"\n📊 {title}")
    print(f"Total samples: {len(samples)}")
    print(f"Total annotations: {sum(class_counts.values())}")

    if not class_counts:
        print("No classes found!")
        return

    sorted_classes = sorted(class_counts.keys())
    total = sum(class_counts.values())

    for cls in sorted_classes:
        count = class_counts[cls]
        pct = count / total * 100 if total > 0 else 0
        print(f"  Class {cls}: {count:6d} ({pct:5.1f}%)")

    if save_path and class_counts:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        classes = list(sorted_classes)
        counts = [class_counts[c] for c in classes]

        bars = plt.bar(classes, counts, color='lightgreen', edgecolor='black', alpha=0.75)
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{count}', ha='center', va='bottom')

        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

def compare_distributions(train_samples, val_samples):
    train_dist = compute_class_distribution(train_samples)
    val_dist = compute_class_distribution(val_samples)

    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))
    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())

    print("\n📈 Class Distribution Comparison (Train vs. Val):")
    print(f"{'Class':>6} | {'Train Count':>12} | {'Train %':>8} | {'Val Count':>10} | {'Val %':>8} | {'Diff %':>8}")
    print("-" * 70)

    for cls in all_classes:
        t_count = train_dist.get(cls, 0)
        v_count = val_dist.get(cls, 0)
        t_pct = t_count / train_total * 100 if train_total > 0 else 0
        v_pct = v_count / val_total * 100 if val_total > 0 else 0
        diff = abs(t_pct - v_pct)
        print(f"{cls:>6} | {t_count:>12} | {t_pct:>7.1f}% | {v_count:>10} | {v_pct:>7.1f}% | {diff:>7.1f}%")

# === MAIN ===
if __name__ == "__main__":
    random.seed(42)
    print("🔍 YOLO Dataset Class Distribution Analyzer")
    print("=" * 50)

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset not found: {DATASET_DIR}")
        exit(1)

    dataset = collect_samples()
    os.makedirs(PLOT_DIR, exist_ok=True)

    report_distribution(dataset['train'], "TRAIN Class Distribution", os.path.join(PLOT_DIR, "train.png"))
    report_distribution(dataset['val'], "VAL Class Distribution", os.path.join(PLOT_DIR, "val.png"))
    compare_distributions(dataset['train'], dataset['val'])

    print(f"\n✅ Distribution plots saved to: {PLOT_DIR}")
