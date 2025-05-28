import os
from glob import glob

# Gets the current working directory
current_directory = os.getcwd()

# === CONFIGURATION ===
DATASET_DIR = os.path.join(current_directory, 'Part 1/labeled_image_data')
IMAGE_EXT = '.jpg'
LABEL_SETS = ['train', 'val']  # or add 'test' if you have one

IMAGE_DIR = lambda s: os.path.join(DATASET_DIR, 'images', s)
LABEL_DIR = lambda s: os.path.join(DATASET_DIR, 'labels', s)

# === RUNNER ===
def find_images_with_class_zero():
    class_zero_images = []

    for split in LABEL_SETS:
        label_files = glob(os.path.join(LABEL_DIR(split), '*.txt'))

        for label_path in label_files:
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                classes = set(int(line.split()[0]) for line in lines)

            # Check if class '0' is present (not necessarily alone)
            if 0 in classes:
                image_name = os.path.basename(label_path).replace('.txt', IMAGE_EXT)
                image_path = os.path.join(IMAGE_DIR(split), image_name)

                if os.path.exists(image_path):
                    class_zero_images.append((split, image_path))
                else:
                    print(f"⚠️ Image not found for label file: {label_path}")

    return class_zero_images

# === EXECUTE ===
if __name__ == "__main__":
    results = find_images_with_class_zero()

    print(f"\n🔍 Found {len(results)} images containing class '0':")
    for split, path in results:
        print(f"[{split}] {path}")

    # Optional: save to file
    with open("class_zero_images.txt", 'w') as f:
        for split, path in results:
            f.write(f"[{split}] {path}\n")

    print("\n✅ Saved results to: class_zero_images.txt")
