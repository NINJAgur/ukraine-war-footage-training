"""
ml-engine/tasks/autolabel_kaggle.py

CLI script: run GroundingDINO on any image folder, output a canonical nc=3
YOLO dataset ready for the fine-tune corpus.

Unlike auto_label.py (which handles video clips), this operates directly on
pre-existing image folders — no frame extraction needed.

Usage:
    cd ml-engine
    python tasks/autolabel_kaggle.py --path /path/to/images --output-dir /path/to/output
    python tasks/autolabel_kaggle.py --path media/kaggle_datasets/nzigulic/.../train/images
    python tasks/autolabel_kaggle.py --path media/kaggle_datasets/some_dataset --max-images 5000
    python tasks/autolabel_kaggle.py --path ... --prompt "tank . drone . soldier"
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml as _yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autolabel-kaggle")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Canonical remapping ───────────────────────────────────────────────────────
_MT_TO_CANONICAL: Dict[str, int] = {"AIRCRAFT": 0, "VEHICLE": 1, "PERSONNEL": 2}
_GDINO_TO_CANONICAL: Dict[int, int] = {
    term_idx: _MT_TO_CANONICAL[mt]
    for term_idx, mt in settings.GDINO_CLASS_TO_MODEL.items()
}
_CANONICAL_NAMES = ["aircraft", "vehicle", "personnel"]


def _build_classes(prompt: str) -> Tuple[List[str], Dict[str, int]]:
    classes = [c.strip() for c in prompt.replace(",", ".").split(".") if c.strip()]
    class_dict = {cls.lower(): idx for idx, cls in enumerate(classes)}
    return classes, class_dict


def _phrase_to_canonical(phrase: str, class_dict: Dict[str, int]) -> int:
    """
    Map a GDINO-returned phrase to a canonical class ID.
    Falls back to substring matching when an exact match fails (GDINO sometimes
    merges adjacent tokens, e.g. "armored vehicle military vehicle").
    """
    phrase = phrase.strip().lower()
    if phrase in class_dict:
        return _GDINO_TO_CANONICAL.get(class_dict[phrase], -1)
    matched: set[int] = set()
    for term, idx in class_dict.items():
        if term in phrase:
            cid = _GDINO_TO_CANONICAL.get(idx, -1)
            if cid >= 0:
                matched.add(cid)
    if len(matched) == 1:
        return matched.pop()
    if len(matched) > 1:
        for priority in (1, 0, 2):  # VEHICLE > AIRCRAFT > PERSONNEL
            if priority in matched:
                return priority
    return -1


# ── Image collection ──────────────────────────────────────────────────────────

def _collect_images(root: Path, max_images: int) -> List[Path]:
    """Recursively collect all images under root."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found under {root}")
    if max_images > 0 and len(images) > max_images:
        logger.info(f"Capping {len(images)} -> {max_images} images (--max-images)")
        images = images[:max_images]
    return images


# ── GDINO labeling ────────────────────────────────────────────────────────────

def _label_images(
    images: List[Path],
    out_labels: Path,
    out_images: Path,
    model,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    class_dict: Dict[str, int],
    log_every: int = 500,
) -> int:
    """Label a list of images with GDINO, write canonical .txt files, copy images."""
    from groundingdino.util.inference import load_image, predict

    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    labeled = 0
    for i, img_path in enumerate(images):
        if i > 0 and i % log_every == 0:
            logger.info(f"  Progress: {i}/{len(images)}  ({labeled} with detections)")

        stem = f"{img_path.parent.name}__{img_path.stem}"
        dst_img = out_images / (stem + img_path.suffix)
        shutil.copy2(img_path, dst_img)

        try:
            image_source, image = load_image(str(img_path))
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except Exception as exc:
            logger.warning(f"GDINO failed on {img_path.name}: {exc}")
            (out_labels / (stem + ".txt")).write_text("")
            continue

        lines = []
        for box, phrase in zip(boxes, phrases):
            canonical_id = _phrase_to_canonical(phrase, class_dict)
            if canonical_id < 0:
                continue
            cx, cy, w, h = box.tolist()
            lines.append(f"{canonical_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        lbl_path = out_labels / (stem + ".txt")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            labeled += 1

    return labeled


# ── Main entry point ──────────────────────────────────────────────────────────

def run_autolabel(
    input_path: Path,
    output_dir: Optional[Path] = None,
    max_images: int = 0,
    prompt: Optional[str] = None,
) -> Path:
    from groundingdino.util.inference import load_model
    import groundingdino

    images = _collect_images(input_path, max_images)
    logger.info(f"Source: {input_path}  images to label: {len(images)}")

    if output_dir is None:
        output_dir = settings.KAGGLE_CACHE_DIR / "labeled" / input_path.name
    output_dir = Path(output_dir)
    if output_dir.exists():
        logger.info(f"Removing existing: {output_dir}")
        shutil.rmtree(output_dir)

    config_path = str(
        Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_path = str(Path(__file__).parent.parent / "groundingdino_swint_ogc.pth")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"GDINO checkpoint not found: {checkpoint_path}\n"
            "Download from: https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha"
        )

    logger.info("Loading GDINO model...")
    model = load_model(config_path, checkpoint_path)

    active_prompt = prompt or settings.GDINO_TEXT_PROMPT
    classes, class_dict = _build_classes(active_prompt)
    logger.info(f"Prompt terms ({len(classes)}): {classes}")

    out_train_images = output_dir / "train" / "images"
    out_train_labels = output_dir / "train" / "labels"

    labeled = _label_images(
        images, out_train_labels, out_train_images,
        model, active_prompt,
        settings.GDINO_BOX_THRESHOLD, settings.GDINO_TEXT_THRESHOLD,
        class_dict,
    )

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        _yaml.dump(
            {
                "path": str(output_dir),
                "train": "train/images",
                "val": "train/images",
                "nc": 3,
                "names": _CANONICAL_NAMES,
            },
            f,
            default_flow_style=False,
        )

    logger.info(
        f"Done. {labeled}/{len(images)} images with detections. "
        f"Output: {output_dir}  nc=3"
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="GDINO auto-label any image folder -> nc=3 YOLO dataset"
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to folder containing images (searched recursively)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: media/kaggle_datasets/labeled/<folder-name>)",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to process (0 = all, default 0)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Override GDINO prompt (default: settings.GDINO_TEXT_PROMPT)",
    )
    args = parser.parse_args()

    out = run_autolabel(
        input_path=Path(args.path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_images=args.max_images,
        prompt=args.prompt,
    )
    print(f"\nOutput dataset: {out}")


if __name__ == "__main__":
    main()