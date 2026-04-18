"""
Data preprocessing pipeline for YOLO training datasets.
Includes data cleaning and augmentation capabilities.
"""

import os
import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLODataPreprocessor:
    """
    Preprocessing pipeline for YOLO datasets with cleaning and augmentation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config = self._load_config(config_path)
        self.augmentation_pipeline = self._build_augmentation_pipeline()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load preprocessing configuration from YAML file."""
        default_config = {
            'cleaning': {
                'remove_corrupted_images': True,
                'validate_annotations': True,
                'check_bbox_validity': True,
                'min_bbox_size': 1,  # minimum bbox width/height in pixels
                'max_bbox_size_ratio': 0.9  # maximum bbox size relative to image
            },
            'augmentation': {
                'enabled': True,
                'augment_factor': 2,  # how many augmented versions per image
                'transforms': {
                    'horizontal_flip': {'p': 0.5},
                    'vertical_flip': {'p': 0.1},
                    'rotate': {'limit': 15, 'p': 0.3},
                    'brightness_contrast': {'brightness_limit': 0.2,
                                           'contrast_limit': 0.2, 'p': 0.3},
                    'gaussian_blur': {'blur_limit': 3, 'p': 0.1},
                    'gaussian_noise': {'var_limit': (10, 50), 'p': 0.1},
                    'hue_saturation_value': {'hue_shift_limit': 20,
                                             'sat_shift_limit': 30,
                                             'val_shift_limit': 20, 'p': 0.2}
                }
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge user config with defaults
            self._merge_configs(default_config, user_config)

        return default_config

    def _merge_configs(self, base: Dict, update: Dict) -> None:
        """Recursively merge update config into base config."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _build_augmentation_pipeline(self) -> A.Compose:
        """Build Albumentations augmentation pipeline."""
        transforms = []

        aug_config = self.config['augmentation']['transforms']

        if aug_config.get('horizontal_flip', {}).get('p', 0) > 0:
            transforms.append(A.HorizontalFlip(p=aug_config['horizontal_flip']['p']))

        if aug_config.get('vertical_flip', {}).get('p', 0) > 0:
            transforms.append(A.VerticalFlip(p=aug_config['vertical_flip']['p']))

        if aug_config.get('rotate', {}).get('p', 0) > 0:
            transforms.append(A.Rotate(
                limit=aug_config['rotate']['limit'],
                p=aug_config['rotate']['p']
            ))

        if aug_config.get('brightness_contrast', {}).get('p', 0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_contrast']['brightness_limit'],
                contrast_limit=aug_config['brightness_contrast']['contrast_limit'],
                p=aug_config['brightness_contrast']['p']
            ))

        if aug_config.get('gaussian_blur', {}).get('p', 0) > 0:
            transforms.append(A.GaussianBlur(
                blur_limit=aug_config['gaussian_blur']['blur_limit'],
                p=aug_config['gaussian_blur']['p']
            ))

        if aug_config.get('gaussian_noise', {}).get('p', 0) > 0:
            transforms.append(A.GaussNoise(
                var_limit=aug_config['gaussian_noise']['var_limit'],
                p=aug_config['gaussian_noise']['p']
            ))

        if aug_config.get('hue_saturation_value', {}).get('p', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=aug_config['hue_saturation_value']['hue_shift_limit'],
                sat_shift_limit=aug_config['hue_saturation_value']['sat_shift_limit'],
                val_shift_limit=aug_config['hue_saturation_value']['val_shift_limit'],
                p=aug_config['hue_saturation_value']['p']
            ))

        # Add bbox parameters for YOLO format
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            )
        )

    def clean_dataset(self, images_dir: str, labels_dir: str) -> Tuple[int, int]:
        """
        Clean dataset by removing corrupted files and invalid annotations.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files

        Returns:
            Tuple of (images_removed, labels_fixed)
        """
        logging.info("Starting dataset cleaning...")

        images_removed = 0
        labels_fixed = 0

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f'*{ext}'))
            image_files.extend(images_path.glob(f'*{ext.upper()}'))

        for image_file in image_files:
            label_file = labels_path / f"{image_file.stem}.txt"

            # Read image once for all operations
            img = None
            if self.config['cleaning']['remove_corrupted_images'] or \
               (self.config['cleaning']['validate_annotations'] and self.config['cleaning']['check_bbox_validity']):
                try:
                    img = cv2.imread(str(image_file))
                    if img is None:
                        logging.warning(f"Removing corrupted image: {image_file}")
                        os.remove(image_file)
                        images_removed += 1
                        if label_file.exists():
                            os.remove(label_file)
                        continue
                except Exception as e:
                    logging.warning(f"Error reading image {image_file}: {e}")
                    os.remove(image_file)
                    images_removed += 1
                    if label_file.exists():
                        os.remove(label_file)
                    continue

            # Validate annotations
            if self.config['cleaning']['validate_annotations'] and label_file.exists():
                try:
                    valid_lines = []
                    lines_processed = 0
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f):
                            lines_processed += 1
                            parts = line.strip().split()
                            if len(parts) != 5:
                                logging.warning(f"Invalid annotation format in {label_file} line {line_num + 1}")
                                continue

                            try:
                                class_id = int(parts[0])
                                bbox = [float(x) for x in parts[1:]]
                            except ValueError:
                                logging.warning(f"Non-numeric values in {label_file} line {line_num + 1}")
                                continue

                            # Validate bbox values (YOLO format: class x_center y_center width height)
                            if not all(0 <= x <= 1 for x in bbox):
                                logging.warning(f"Invalid bbox values in {label_file} line {line_num + 1}: {bbox}")
                                continue

                            # Check bbox size constraints
                            if self.config['cleaning']['check_bbox_validity'] and img is not None:
                                width, height = bbox[2], bbox[3]
                                if (width < self.config['cleaning']['min_bbox_size'] /
                                        img.shape[1] or
                                    height < self.config['cleaning']['min_bbox_size'] /
                                        img.shape[0]):
                                    logging.warning(f"Bbox too small in {label_file} "
                                                    f"line {line_num + 1}")
                                    continue
                                if (width > self.config['cleaning']['max_bbox_size_ratio'] or
                                    height > self.config['cleaning']['max_bbox_size_ratio']):
                                    logging.warning(f"Bbox too large in {label_file} "
                                                    f"line {line_num + 1}")
                                    continue

                            valid_lines.append(line)

                    # Rewrite file with only valid lines
                    if len(valid_lines) != lines_processed:
                        with open(label_file, 'w') as f:
                            f.writelines(valid_lines)
                        labels_fixed += 1

                except Exception as e:
                    logging.error(f"Error processing label file {label_file}: {e}")

        logging.info(f"Dataset cleaning completed. Removed {images_removed} corrupted images, fixed {labels_fixed} label files.")
        return images_removed, labels_fixed

    def augment_dataset(self, images_dir: str, labels_dir: str, output_images_dir: str, output_labels_dir: str) -> int:
        """
        Augment dataset by creating additional transformed versions of images and labels.

        Args:
            images_dir: Source images directory
            labels_dir: Source labels directory
            output_images_dir: Output images directory
            output_labels_dir: Output labels directory

        Returns:
            Number of augmented images created
        """
        if not self.config['augmentation']['enabled']:
            logging.info("Augmentation disabled in config.")
            return 0

        logging.info("Starting dataset augmentation...")

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        augment_factor = self.config['augmentation']['augment_factor']
        augmented_count = 0

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f'*{ext}'))
            image_files.extend(images_path.glob(f'*{ext.upper()}'))

        for image_file in image_files:
            label_file = labels_path / f"{image_file.stem}.txt"

            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read labels
            bboxes = []
            class_labels = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            bboxes.append(bbox)
                            class_labels.append(class_id)

            # Generate augmented versions
            for i in range(augment_factor):
                # Apply augmentation
                transformed = self.augmentation_pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                augmented_image = transformed['image']
                augmented_bboxes = transformed['bboxes']
                augmented_labels = transformed['class_labels']

                # Save augmented image
                output_image_name = f"{image_file.stem}_aug_{i}{image_file.suffix}"
                output_image_path = os.path.join(output_images_dir, output_image_name)
                augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_image_path, augmented_image_bgr)

                # Save augmented labels
                output_label_name = f"{image_file.stem}_aug_{i}.txt"
                output_label_path = os.path.join(output_labels_dir, output_label_name)
                with open(output_label_path, 'w') as f:
                    for class_id, bbox in zip(augmented_labels, augmented_bboxes):
                        f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")

                augmented_count += 1

        logging.info(f"Dataset augmentation completed. Created {augmented_count} augmented images.")
        return augmented_count

    def preprocess_dataset(self, images_dir: str, labels_dir: str,
                          output_images_dir: Optional[str] = None,
                          output_labels_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Run full preprocessing pipeline: cleaning + augmentation.

        Args:
            images_dir: Source images directory
            labels_dir: Source labels directory
            output_images_dir: Output images directory (if None, augment in-place)
            output_labels_dir: Output labels directory (if None, augment in-place)

        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {}

        # Clean dataset
        images_removed, labels_fixed = self.clean_dataset(images_dir, labels_dir)
        stats['images_removed'] = images_removed
        stats['labels_fixed'] = labels_fixed

        # Augment dataset
        if output_images_dir and output_labels_dir:
            augmented_count = self.augment_dataset(
                images_dir, labels_dir, output_images_dir, output_labels_dir
            )
        else:
            # In-place augmentation (create augmented versions in same directories)
            augmented_count = self.augment_dataset(
                images_dir, labels_dir, images_dir, labels_dir
            )
        stats['augmented_images'] = augmented_count

        return stats


def create_default_config(output_path: str = "preprocessing_config.yaml") -> None:
    """
    Create a default preprocessing configuration file.

    Args:
        output_path: Path to save the config file
    """
    default_config = {
        'cleaning': {
            'remove_corrupted_images': True,
            'validate_annotations': True,
            'check_bbox_validity': True,
            'min_bbox_size': 1,
            'max_bbox_size_ratio': 0.9
        },
        'augmentation': {
            'enabled': True,
            'augment_factor': 2,
            'transforms': {
                'horizontal_flip': {'p': 0.5},
                'vertical_flip': {'p': 0.1},
                'rotate': {'limit': 15, 'p': 0.3},
                'brightness_contrast': {'brightness_limit': 0.2,
                                       'contrast_limit': 0.2, 'p': 0.3},
                'gaussian_blur': {'blur_limit': 3, 'p': 0.1},
                'gaussian_noise': {'var_limit': [10, 50], 'p': 0.1},
                'hue_saturation_value': {'hue_shift_limit': 20,
                                         'sat_shift_limit': 30,
                                         'val_shift_limit': 20, 'p': 0.2}
            }
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Default preprocessing config saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    create_default_config()