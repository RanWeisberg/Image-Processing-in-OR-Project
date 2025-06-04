from ultralytics import YOLO
import torch
import os

# Gets the current working directory
current_directory = os.getcwd()

def get_best_device():
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for Metal (MPS - Apple Silicon)
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback to CPU
    else:
        return torch.device("cpu")


def train(model, dataset, epochs, results_name, freeze):
    """
    Train the YOLO model with the specified dataset and number of epochs.

    Args:
        model: The YOLO model to be trained.
        dataset: Path to the dataset configuration file.
        epochs: Number of epochs to train the model.
    """

    save_path = os.path.join(current_directory, 'results')
    _ = model.train(data=dataset, epochs=epochs, save=True, project=save_path, name=results_name, freeze = freeze)

def main():
    device = get_best_device()
    print(f"Using device: {device}")

    # Load the YOLO 11n model with pretrained weights
    model = YOLO('yolo11m. pt', verbose= True).to(device)
    dataset = os.path.join(current_directory,'augmented_dataset/data.yaml')
    epochs = 100
    results_name = 'Augmented dataset 100 epochs m'
    train(model, dataset, epochs, results_name, 9)


if __name__ == "__main__":
    main()
