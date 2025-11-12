import argparse
import logging
import cv2
import subprocess
import os
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(model_path):
    """Load the YOLO model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model


def infer_image(model, image_path, conf_thresh=0.5, save_path=None, no_display=False):
    """Perform inference on a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    results = model(img, conf=conf_thresh)
    annotated = results[0].plot()
    if save_path:
        cv2.imwrite(save_path, annotated)
        logging.info(f"Annotated image saved to {save_path}")
    elif no_display:
        default_path = "inference_result.jpg"
        cv2.imwrite(default_path, annotated)
        logging.info(f"Annotated image saved to {default_path} (no display)")
    else:
        cv2.imshow("Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_video(model, video_path, conf_thresh=0.5, save_path=None, no_display=False):
    """Perform inference on a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if save_path is None and no_display:
        save_path = "inference_result.mp4"

    temp_path = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise ValueError("Failed to open video writer. Check codec support.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thresh)
        annotated = results[0].plot()
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    # Re-encode to H.264 for browser playback
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp_path,
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            "-strict",
            "experimental",
            save_path,
        ],
        check=True,
    )

    os.remove(temp_path)
    logging.info(f"Annotated video saved to {save_path}")



def infer_webcam(model, conf_thresh=0.5, no_display=False):
    """Perform real-time inference on webcam feed."""
    if no_display:
        raise ValueError("Webcam inference not supported in no-display mode")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot access webcam")
    logging.info("Starting webcam inference. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thresh)
        annotated = results[0].plot()
        cv2.imshow("Webcam Inference", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights, e.g., "
             "runs/train/weights/best.pt",
    )
    parser.add_argument(
        "--input", required=True, help='Input: image/video path or "webcam"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument("--output", help="Output path for saving results (optional)")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without displaying results",
    )
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        input_lower = args.input.lower()
        if input_lower == "webcam":
            infer_webcam(model, args.conf, args.no_display)
        elif input_lower.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            infer_image(model, args.input, args.conf, args.output, args.no_display)
        elif input_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv")):
            infer_video(model, args.input, args.conf, args.output, args.no_display)
        else:
            raise ValueError("Unsupported input type. Use image/video path or 'webcam'")
        logging.info("Inference completed successfully")
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
