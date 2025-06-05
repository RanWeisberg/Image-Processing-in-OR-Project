# Surgical Tool Detection using YOLO

This project implements an object detection system for surgical tools using YOLO with limited labeled data. The system uses data augmentation techniques to improve model performance with only ~100 labeled images and demonstrates the ability to detect surgical instruments in real surgical videos.

## Example Detection Results

![Surgical Tool Detection Example](Data%20Visualization/0018fa1f-output_0063_with_boxes.jpg)
*Example of the model detecting surgical instruments and hands during a leg suturing procedure. Green bounding boxes show detected objects with their class labels.*

## Project Overview

The system detects surgical instruments (tweezers and needle drivers) and hands in surgical videos. Given only ~100 labeled images, the model uses pseudo-labeling on unlabeled surgical videos to improve performance and generalize to different camera setups.

## Dataset & Results

Due to size constraints, the dataset and video files are hosted externally:

- **Labeled Dataset (images + labels)**: [Download from Google Drive](YOUR_DRIVE_LINK_HERE)
- **Raw Videos (ID + OOD)**: [Download from Google Drive](YOUR_DRIVE_LINK_HERE)
- **Detection Results Videos (with bounding boxes)**: [Download from Google Drive](YOUR_DRIVE_LINK_HERE)

After downloading:
1. Extract the labeled dataset to the root directory of the project
2. Place raw videos in `id_video_data/` directory
3. Detection result videos can be viewed directly or placed in `video_results/`

**Note**: All trained model weights are included in this repository under the `models/` directory.

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Image-Processing-in-OR-Project.git
cd Image-Processing-in-OR-Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Image-Processing-in-OR-Project/
├── labeled_image_data/          # Labeled training data (from Drive)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── id_video_data/              # Unlabeled videos (from Drive)
├── models/                     # Trained model weights
│   └── aug_test/
│       ├── yolo11n_augmented/
│       ├── yolo11s_augmented/
│       └── yolo11m_augmented/
├── video_results/              # Output directory for video predictions
├── image_results/              # Output directory for image predictions
└── Data Visualization/         # Visualization outputs
```

## Scripts Description

### 1. **Train.py** - Main Training Pipeline
The core training script that handles the entire SSL pipeline:
- Loads and preprocesses YOLO-formatted data
- Applies data augmentation using Albumentations
- Trains YOLO models (11n, 11s, 11m variants)
- Includes utilities for dataset manipulation and visualization

**Usage:**
```bash
python Train.py
```

### 2. **Create_pseudo_labels.py** - Video Frame Extraction with Predictions
Processes videos to extract frames with high-confidence detections:
- Useful for creating additional training data from videos
- Applies confidence threshold filtering (default: 0.8)
- Saves detected frames as images with YOLO-format labels
- Creates visualization video with bounding boxes

**Usage:**
```bash
python Create_pseudo_labels.py
```

Key parameters (modify in script):
- `VIDEO_PATH`: Input video path
- `CONF_THRESHOLD`: Confidence threshold for pseudo-labels (0.8)
- `FRAME_JUMP`: Process every Nth frame (5)

### 3. **predict.py** - Single Image Inference
Runs inference on a single image and visualizes results:
- Loads trained model
- Detects objects in the image
- Saves visualization with bounding boxes
- Outputs YOLO-format predictions

**Usage:**
```bash
python predict.py
```

### 4. **video.py** - Video Inference
Processes entire videos for object detection:
- Real-time detection and visualization
- Saves output video with bounding boxes
- Exports frame-by-frame detection results in YOLO format

**Usage:**
```bash
python video.py
```

### 5. **TestVisualize.py** - Visualization Tool
Visualizes YOLO annotations on images:
- Draws bounding boxes from label files
- Useful for verifying annotations
- Saves visualized images

**Usage:**
```bash
python TestVisualize.py
```

### 6. **Class_imbalance_test.py** - Dataset Analysis
Analyzes class distribution in the dataset:
- Computes class frequencies
- Compares train/val distributions
- Generates distribution plots

**Usage:**
```bash
python Class_imbalance_test.py
```

### 7. **YOLO_csv_analysis.py** - Training Metrics Analysis
Analyzes and visualizes training metrics:
- Plots training/validation losses
- Compares mAP scores across models
- Helps identify best performing model

**Usage:**
```bash
python YOLO_csv_analysis.py
```

## Training Pipeline

### Step 1: Model Training
```bash
python Train.py
```
This trains YOLO models on the limited labeled data with extensive augmentation.

### Step 2: Generate Predictions
```bash
# For video inference
python video.py

# For single image inference
python predict.py
```

### Step 3: Visualize Results
```bash
python TestVisualize.py
```
Visualize the model's detections on test images.

## Model Performance

The project trains three YOLO11 variants:
- **YOLO11n**: Nano version (fastest, least accurate)
- **YOLO11s**: Small version (balanced)
- **YOLO11m**: Medium version (most accurate, slower)

All models are trained with:
- 100 epochs
- Batch size: 12
- Cosine learning rate scheduler
- Layer freezing (first 9 layers)
- Extensive data augmentation

## Classes

The model detects 3 classes:
- Class 0: Empty (background)
- Class 1: Tweezers
- Class 2: Needle driver

## Output Format

All detections are in YOLO format:
```
x_center y_center width height confidence class_id
```
Where all spatial values are normalized to [0, 1].

## Citation

This project was developed as part of the Computer Vision for Surgical Applications course.

## License

This project is for educational purposes only.