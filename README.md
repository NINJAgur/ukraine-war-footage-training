# YOLO Training Template

This repository provides a template for training YOLO models on any Kaggle dataset and performing inference. It includes scripts for command-line use and a notebook-style script for interactive environments.

## Files

- `scripts/main.py`: Command-line script for training YOLO on a Kaggle dataset.
- `scripts/inference.py`: Command-line script for running inference with a trained model.
- `notebooks/yolo_template.ipynb`: Notebook template to run train a YOLO model and test it.
- `docs/CONTRIBUTING.md`: Contributing guidelines.
- `example_datasets.md`: List of example Kaggle datasets for testing.
- `requirements.txt`: Dependencies for the project.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. For training: Run `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names>`
3. For inference: Run `python scripts/inference.py --model <model-path> --input <image/video/webcam>`

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on how to contribute, report issues, and run the notebook on Google Colab.

