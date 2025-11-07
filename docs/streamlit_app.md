# Streamlit App Documentation

The Streamlit app provides an interactive web interface for the YOLO Training Template, allowing users to train models, run inference, preprocess data, and perform auto-labeling without command-line usage.

## Overview

The app features a sidebar navigation with four main pages:

1. **Training**: Train YOLO models on Kaggle datasets or uploaded ZIP files.
2. **Inference**: Perform object detection on images, videos, or webcam feed using trained models.
3. **Preprocessing**: Clean and augment datasets before training.
4. **Auto-labeling**: Automatically generate labels for images using GroundingDINO.

## Installation and Running

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser.

## Training Page

### Dataset Source
- **Kaggle Dataset**: Enter the Kaggle dataset handle (e.g., `jocelyndumlao/multi-weather-pothole-detection-mwpd`), number of classes, and class names (comma-separated).
- **Upload Dataset**: Upload a ZIP file containing the dataset with `images/` and `labels/` directories.

### Training Parameters
- **Epochs**: Number of training epochs (1-1000).
- **Image Size**: Input image size (32-2048, step 32).
- **Batch Size**: Training batch size (1-128).
- **Device**: GPU device (e.g., "0") or CPU.
- **Project Directory**: Output directory for results.
- **Experiment Name**: Name for the training run.

### Preprocessing Options
- Enable preprocessing for data cleaning and augmentation.
- Option to run augmentation only (skip training).

### Weights and Resume
- Specify pretrained weights path (optional).
- Resume from a previous checkpoint.

Click "Start Training" to begin. The app will display progress and results.

## Inference Page

### Model Upload
Upload your trained YOLO model weights (.pt file).

### Input Selection
- **Image**: Upload a single image file.
- **Video**: Upload a video file.
- **Webcam**: Use your webcam (opens in a separate window).

### Parameters
- **Confidence Threshold**: Minimum confidence for detections (0.0-1.0).

Click "Run Inference" to process the input and display results.

## Preprocessing Page

### Dataset Upload
Upload a ZIP file containing your dataset with `images/` and `labels/` directories.

### Configuration
Optionally upload a custom preprocessing config YAML file (defaults to `preprocessing_config.yaml`).

Click "Run Preprocessing" to clean and augment the data. The app will show preprocessing statistics.

## Auto-labeling Page

### Input
- **Input Images Folder**: Path to folder containing images to label.
- **Text Prompt**: Comma-separated list of classes (e.g., "car, person, dog").
- **Output Path**: Directory to save labeled dataset.

Click "Run Auto-labeling" to generate labels using GroundingDINO.

## Notes

- Uploaded files are temporarily stored and cleaned up after processing.
- For large datasets, command-line scripts may be more efficient.
- Ensure your dataset follows YOLO format (images and corresponding .txt label files).
- Preprocessing requires the `preprocessing.py` script and config file.