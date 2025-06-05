import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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
    current_folder = os.path.dirname(os.path.abspath(__file__))
    path_files = {
        "aug_no_freeze_n": os.path.join(current_folder, "models", "aug_no_freeze", "yolo11n_augmented_no_freeze", "results.csv"),
        "aug_no_freeze_s": os.path.join(current_folder, "models", "aug_no_freeze", "yolo11s_augmented_no_freeze", "results.csv"),
        "aug_no_freeze_m": os.path.join(current_folder, "models", "aug_no_freeze", "yolo11m_augmented_no_freeze", "results.csv"),
        "aug_freeze_n": os.path.join(current_folder, "models", "aug_test", "yolo11n_augmented", "results.csv"),
        "aug_freeze_s": os.path.join(current_folder, "models", "aug_test", "yolo11s_augmented", "results.csv"),
        "aug_freeze_m": os.path.join(current_folder, "models", "aug_test", "yolo11m_augmented", "results.csv"),
        "no_aug_freeze_n": os.path.join(current_folder, "models", "no_aug_test", "yolo11n_no_augmented", "results.csv"),
        "no_aug_freeze_s": os.path.join(current_folder, "models", "no_aug_test", "yolo11s_no_augmented", "results.csv"),
        "no_aug_freeze_m": os.path.join(current_folder, "models", "no_aug_test", "yolo11m_no_augmented", "results.csv"),
    }

    # 2) Choose which models to load
    wanted_keys = ["no_aug_freeze_s", "aug_freeze_s", "aug_no_freeze_s"]

    # 3) Read each CSV into a DataFrame and store in a dict called dfs
    dfs = { key: read_csv(path_files[key]) for key in wanted_keys }

    # 4) (Optional) Print column names for each DataFrame
    for key, df in dfs.items():
        print(f"\nColumns in '{key}' DataFrame:")
        for col in df.columns:
            print(col)


    # # 5) Plotting: one figure with 3 subplots (train_loss, val_loss, mAP@50)
    map50_col = 'metrics/mAP50(B)'
    toal_val_loss_col = 'val_total_loss'
    presicion_col = 'metrics/precision(B)'
    #plots as one figure
    figure(figsize=(16, 16))
    for key, df in dfs.items():
        plt.plot(df['epoch'], df[map50_col], label=f"{key} mAP@50")
    plt.title('mAP@50 Across Models', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('mAP@50', fontsize=20)
    plt.legend(fontsize=20, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mAP50_across_models.png', dpi=150)
    plt.show()
    plt.close()