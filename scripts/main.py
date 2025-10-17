import argparse
import logging
import os
import yaml
import kagglehub
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(dataset_handle):
    """Download the Kaggle dataset."""
    try:
        path = kagglehub.dataset_download(dataset_handle)
        logging.info(f"Downloaded dataset to: {path}")
        return path
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

def detect_dataset_structure(dataset_path):
    """Detect train/val/test images and labels paths in the dataset."""
    paths = {}
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    logging.info(f"Detected subdirs in dataset: {subdirs}")
    splits = ['train', 'valid', 'test']
    for split in splits:
        key_split = 'val' if split == 'valid' else split
        if split in subdirs:
            split_dir = os.path.join(dataset_path, split)
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            if os.path.exists(images_dir):
                paths[f'{key_split}_images'] = images_dir
            if os.path.exists(labels_dir):
                paths[f'{key_split}_labels'] = labels_dir
    if not paths:
        # Check one level deeper
        for subdir in subdirs:
            sub_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(sub_path):
                inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
                logging.info(f"Checking inner subdirs in {subdir}: {inner_subdirs}")
                for split in splits:
                    key_split = 'val' if split == 'valid' else split
                    if split in inner_subdirs:
                        split_dir = os.path.join(sub_path, split)
                        images_dir = os.path.join(split_dir, 'images')
                        labels_dir = os.path.join(split_dir, 'labels')
                        if os.path.exists(images_dir):
                            paths[f'{key_split}_images'] = images_dir
                        if os.path.exists(labels_dir):
                            paths[f'{key_split}_labels'] = labels_dir
                if paths:
                    dataset_path = sub_path
                    break
        if not paths:
            # Check two levels deeper
            for subdir in subdirs:
                sub_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(sub_path):
                    inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
                    for inner_subdir in inner_subdirs:
                        inner_path = os.path.join(sub_path, inner_subdir)
                        if os.path.isdir(inner_path):
                            deepest_subdirs = [d for d in os.listdir(inner_path) if os.path.isdir(os.path.join(inner_path, d))]
                            logging.info(f"Checking deepest subdirs in {inner_subdir}: {deepest_subdirs}")
                            for split in splits:
                                key_split = 'val' if split == 'valid' else split
                                if split in deepest_subdirs:
                                    split_dir = os.path.join(inner_path, split)
                                    images_dir = os.path.join(split_dir, 'images')
                                    labels_dir = os.path.join(split_dir, 'labels')
                                    if os.path.exists(images_dir):
                                        paths[f'{key_split}_images'] = images_dir
                                    if os.path.exists(labels_dir):
                                        paths[f'{key_split}_labels'] = labels_dir
                            if paths:
                                dataset_path = inner_path
                                break
                    if paths:
                        break
    logging.info(f"Detected paths: {paths}")
    return paths, dataset_path

def create_yaml(dataset_path, paths, nc, names):
    """Create the data.yaml file for YOLO training."""
    data_yaml = {
        "path": dataset_path,
        "train": os.path.relpath(paths.get('train_images', ''), dataset_path) if 'train_images' in paths else '',
        "val": os.path.relpath(paths.get('val_images', ''), dataset_path) if 'val_images' in paths else '',
        "test": os.path.relpath(paths.get('test_images', ''), dataset_path) if 'test_images' in paths else '',
        "nc": nc,
        "names": names,
    }
    yaml_path = f"{os.path.basename(dataset_path)}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    logging.info(f"Created YAML config at: {yaml_path}")
    return yaml_path

def train_model(yaml_path, epochs, imgsz, batch, device, project, name, weights=None, resume=False):
    """Train the YOLO model."""
    if resume:
        # Resume from last checkpoint
        logging.info("Resuming training from last checkpoint...")
        model = YOLO(f"{project}/{name}/weights/last.pt")
        results = model.train(resume=True)
    elif weights:
        # Load custom weights
        logging.info(f"Loading weights from: {weights}")
        model = YOLO(weights)
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )
    else:
        # Start fresh with pretrained YOLOv8
        logging.info("Starting training with YOLOv8m pretrained weights...")
        model = YOLO("yolov8m.pt")
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on any Kaggle dataset")
    parser.add_argument('--dataset', required=True, help='Kaggle dataset handle, e.g., jocelyndumlao/multi-weather-pothole-detection-mwpd')
    parser.add_argument('--nc', type=int, required=True, help='Number of classes')
    parser.add_argument('--names', required=True, help='Class names, comma separated, e.g., "Potholes,Cracks"')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=512, help='Image size')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='0', help='Device to use, e.g., 0 for GPU, cpu for CPU')
    parser.add_argument('--project', default='runs/train', help='Project directory for runs')
    parser.add_argument('--name', default='yolo_train', help='Experiment name')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights (e.g., runs/train/yolo_train/weights/best.pt)')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')

    args = parser.parse_args()
    names = [n.strip() for n in args.names.split(',')]

    # Check for conflicting arguments
    if args.resume and args.weights:
        logging.warning("Both --resume and --weights specified. Using --resume (ignoring --weights)")
        args.weights = None

    try:
        # If resuming, we don't need to download dataset or create yaml
        if args.resume:
            logging.info("Resume mode: skipping dataset download and yaml creation")
            results = train_model(None, args.epochs, args.imgsz, args.batch, args.device, 
                                args.project, args.name, weights=None, resume=True)
        else:
            dataset_path = download_dataset(args.dataset)
            logging.info(f"Dataset downloaded to: {dataset_path}")
            paths, dataset_path = detect_dataset_structure(dataset_path)
            if not paths:
                raise ValueError("No standard train/val/test structure found in dataset")
            yaml_path = create_yaml(dataset_path, paths, args.nc, names)
            results = train_model(yaml_path, args.epochs, args.imgsz, args.batch, args.device, 
                                args.project, args.name, weights=args.weights, resume=False)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()