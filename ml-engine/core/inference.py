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

    import tempfile
    tmp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
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



def _draw_tactical_box(frame, x1, y1, x2, y2, conf, label, color):
    """Draw tactical HUD-style box matching frontend canvas aesthetic."""
    import numpy as np

    # Translucent fill (very faint, matches canvas alpha*0.07)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.07, frame, 1.0, 0, frame)

    # Faint full box (matches canvas strokeRect alpha*0.9)
    dim = tuple(max(0, int(c * 0.55)) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), dim, 1)

    # Corner ticks — solid, thicker (matches canvas corner draws)
    tk = max(8, int(min(x2 - x1, y2 - y1) * 0.15))
    cv2.line(frame, (x1, y1 + tk), (x1, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1 + tk, y1), color, 2)
    cv2.line(frame, (x2 - tk, y1), (x2, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + tk), color, 2)
    cv2.line(frame, (x1, y2 - tk), (x1, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1 + tk, y2), color, 2)
    cv2.line(frame, (x2 - tk, y2), (x2, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - tk), color, 2)

    # Label text above top-left (no filled bar — matches canvas fillText style)
    tag = f"{label}  {int(conf * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, lthick = 0.38, 1
    (tw, th), _ = cv2.getTextSize(tag, font, scale, lthick)
    ty = max(y1 - 5, th + 2)
    cv2.putText(frame, tag, (x1 + 2, ty), font, scale, color, lthick, cv2.LINE_AA)

    # Confidence bar on LEFT edge (vertical, matches canvas left-side bar)
    box_h = y2 - y1
    bar_h = int(box_h * conf)
    cv2.rectangle(frame, (x1 - 4, y2 - bar_h), (x1 - 2, y2), dim, -1)
    cv2.rectangle(frame, (x1 - 4, y2 - bar_h), (x1 - 2, y2 - bar_h + 2), color, -1)


def infer_video_multi_model(
    models_info: list,
    video_path: str,
    conf_thresh: float = 0.4,
    save_path: str = None,
    no_display: bool = False,
) -> int:
    """
    Run multiple YOLO models on the same video, drawing all detections onto
    each frame with per-model colours. Sequential inference — models are run
    one at a time to avoid VRAM contention.

    models_info: list of (model, label_prefix: str, color_bgr: tuple)
      e.g. [(soldier_model, "SOLDIER", (0, 80, 255)), ...]

    Returns total frame count processed.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Pass 1-N: run each model, collect {frame_idx: [(box, conf, cls_name)]}
    all_detections: list[dict] = []   # one dict per model
    for model, label_prefix, _ in models_info:
        frame_dets: dict[int, list] = {}
        cap2 = cv2.VideoCapture(video_path)
        idx = 0
        while cap2.isOpened():
            ret, frame = cap2.read()
            if not ret:
                break
            results = model(frame, conf=conf_thresh, verbose=False)
            dets = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = results[0].names.get(cls_id, str(cls_id))
                dets.append((x1, y1, x2, y2, conf_val, f"{label_prefix}:{cls_name}"))
            frame_dets[idx] = dets
            idx += 1
        cap2.release()
        all_detections.append(frame_dets)

    # Final pass: render all detections onto each frame
    if save_path is None and no_display:
        save_path = "inference_multi_result.mp4"

    import tempfile
    tmp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise ValueError("Failed to open video writer.")

    cap3 = cv2.VideoCapture(video_path)
    idx = 0
    while cap3.isOpened():
        ret, frame = cap3.read()
        if not ret:
            break
        for model_idx, (_, label_prefix, color) in enumerate(models_info):
            for (x1, y1, x2, y2, conf_val, label) in all_detections[model_idx].get(idx, []):
                _draw_tactical_box(frame, x1, y1, x2, y2, conf_val, label, color)
        out.write(frame)
        idx += 1

    cap3.release()
    out.release()

    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", temp_path,
         "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental",
         save_path],
        check=True,
    )
    os.remove(temp_path)
    logging.info(f"Multi-model annotated video saved to {save_path} ({idx} frames)")
    return idx


def validate_clip(model, path, conf_thresh: float = 0.35,
                  min_rate: float = 0.15, n_samples: int = 30) -> bool:
    """Return True if ≥min_rate of sampled frames have at least one detection.
    Deletes the file and returns False when the clip fails validation."""
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        os.remove(path)
        logging.warning(f"validate_clip REJECTED (empty): {path}")
        return False

    step = max(1, total // n_samples)
    detected = sampled = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame, conf=conf_thresh, verbose=False)
        if len(results[0].boxes) > 0:
            detected += 1
        sampled += 1
        if sampled >= n_samples:
            break
    cap.release()

    rate = detected / sampled if sampled > 0 else 0.0
    logging.info(f"validate_clip {detected}/{sampled} frames ({rate:.0%}): {path}")
    if rate < min_rate:
        logging.warning(f"validate_clip REJECTED ({rate:.0%} < {min_rate:.0%}): {path}")
        os.remove(path)
        return False
    return True


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
