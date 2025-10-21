import cv2
import os
import yaml
import argparse
from groundingdino.util.inference import load_model, load_image, predict


def create_yolo_dataset(
    input_folder,
    text_prompt,
    output_path,
    config_path="GroundingDINO_SwinT_OGC.py",
    checkpoint_path="groundingdino_swint_ogc.pth",
    box_threshold=0.35,
    text_threshold=0.25,
):
    """
    Annotate images using GroundingDINO and create a YOLO-compatible dataset.

    Args:
        input_folder (str): Path to folder containing input images
        text_prompt (str): Text prompt for object detection (comma-separated classes)
        output_path (str): Path to store YOLO dataset
        config_path (str): Path to GroundingDINO config file
        checkpoint_path (str): Path to GroundingDINO checkpoint file
        box_threshold (float): Confidence threshold for bounding boxes
        text_threshold (float): Confidence threshold for text phrases
    """
    classes = [cls.strip() for cls in text_prompt.split(",")]
    class_dict = {cls.lower(): idx for idx, cls in enumerate(classes)}

    train_images_dir = os.path.join(output_path, "train", "images")
    train_labels_dir = os.path.join(output_path, "train", "labels")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    model = load_model(config_path, checkpoint_path)

    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, image_file)

            train_image_path = os.path.join(train_images_dir, image_file)
            cv2.imwrite(train_image_path, cv2.imread(image_path))

            image_source, image = load_image(image_path)

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            label_path = os.path.join(
                train_labels_dir, os.path.splitext(image_file)[0] + ".txt"
            )
            with open(label_path, "w") as label_file:
                for box, logit, phrase in zip(boxes, logits, phrases):
                    phrase = phrase.strip().lower()
                    if phrase in class_dict:
                        class_id = class_dict[phrase]
                        cx, cy, w, h = box.tolist()
                        label_file.write(f"{class_id} {cx} {cy} {w} {h}\n")

            print(f"Processed: {image_file}")

    data_yaml = {
        "path": os.path.abspath(output_path),
        "train": "train/images",
        "val": "train/images",  # Using train as val; update if validation split is needed
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False)

    print(f"YOLO dataset created at {output_path}. Use data.yaml for training.")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate images and create a YOLO dataset using GroundingDINO"
    )
    parser.add_argument(
        "--input_folder", required=True, help="Path to folder containing input images"
    )
    parser.add_argument(
        "--text_prompt",
        required=True,
        help="Text prompt for object detection (comma-separated classes)",
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to store YOLO dataset"
    )
    parser.add_argument(
        "--config_path",
        default="GroundingDINO_SwinT_OGC.py",
        help="Path to GroundingDINO config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="groundingdino_swint_ogc.pth",
        help="Path to GroundingDINO checkpoint file",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for bounding boxes",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for text phrases",
    )

    args = parser.parse_args()

    create_yolo_dataset(
        input_folder=args.input_folder,
        text_prompt=args.text_prompt,
        output_path=args.output_path,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )


if __name__ == "__main__":
    main()
