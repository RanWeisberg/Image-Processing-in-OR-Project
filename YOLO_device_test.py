import torch

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

def main():
    device = get_best_device()
    print(f"Using device: {device}")

    # Example: Load a YOLOv5 model from Ultralytics hub and move it to the selected device
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True).to(device)

    # Dummy input for testing
    results = model("https://ultralytics.com/images/zidane.jpg")
    results.show()

if __name__ == "__main__":
    main()
