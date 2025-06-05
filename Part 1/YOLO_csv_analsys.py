import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(file_path):
    # Determine folder and names
    folder = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    temp_name = f"temp_{base_name}"
    temp_path = os.path.join(folder, temp_name)

    # Copy the original file to a temporary file
    shutil.copyfile(file_path, temp_path)
    print(f"Copied '{file_path}' to '{temp_path}'")

    # Read the data from the temporary file
    df = pd.read_csv(temp_path)

    # Compute total train and validation losses
    df['train_total_loss'] = 7.5 * df['train/box_loss'] + 0.5 * df['train/cls_loss'] + 1.5 * df['train/dfl_loss']
    df['val_total_loss']   = 7.5 * df['val/box_loss']   + 0.5 * df['val/cls_loss']   + 1.5 * df['val/dfl_loss']

    os.remove(temp_path)
    print(f"Deleted temporary file '{temp_path}'")
    return df

if __name__ == '__main__':
    # 1) Define all CSV paths in a single dictionary
    path_files = {
        "aug_no_freeze_n": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\aug_no_freeze\yolo11n_augmented_no_freeze\results.csv",
        "aug_no_freeze_s": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\aug_no_freeze\yolo11s_augmented_no_freeze\results.csv",
        "aug_no_freeze_m": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\aug_no_freeze\yolo11m_augmented_no_freeze\results.csv",
        "aug_m": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_aug\augmented_yolo11m\results.csv",
        "aug_s": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_aug\augmented_yolo11s\results.csv",
        "aug_n": r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_aug\augmented_yolo11n\results.csv",
        "m":     r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_no_aug\yolo11m\results.csv",
        "s":     r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_no_aug\yolo11s\results.csv",
        "n":     r"C:\Users\cadsegev\Desktop\Image-Processing-in-OR-Project\Part 1\model_size_no_aug\yolo11n\results.csv",
    }

    # 2) Choose which models to load
    wanted_keys = ["aug_no_freeze_n", "aug_no_freeze_s", "aug_no_freeze_m"]

    # 3) Read each CSV into a DataFrame and store in a dict called dfs
    dfs = { key: read_csv(path_files[key]) for key in wanted_keys }

    # 4) (Optional) Print column names for each DataFrame
    for key, df in dfs.items():
        print(f"\nColumns in '{key}' DataFrame:")
        for col in df.columns:
            print(col)

    # 5) Plotting: one figure with 3 subplots (train_loss, val_loss, mAP@50)
    map50_col = 'metrics/mAP50(B)'

    fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    fig.suptitle('Comparison Across Models', fontsize=16, y=0.98)

    # Top subplot: train_total_loss
    for key, df in dfs.items():
        axs[0].plot(df['epoch'], df['train_total_loss'], label=key)
    axs[0].set_title('Train Total Loss')
    axs[0].set_ylabel('Train Total Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Middle subplot: val_total_loss
    for key, df in dfs.items():
        axs[1].plot(df['epoch'], df['val_total_loss'], label=key)
    axs[1].set_title('Validation Total Loss')
    axs[1].set_ylabel('Val Total Loss')
    axs[1].legend()
    axs[1].grid(True)

    # Bottom subplot: mAP@50
    for key, df in dfs.items():
        if map50_col in df.columns:
            axs[2].plot(df['epoch'], df[map50_col], label=key)
        else:
            print(f"Warning: '{map50_col}' not found in '{key}' DataFrame.")
    axs[2].set_title('mAP@50')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('mAP@50')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the suptitle
    plt.show()
