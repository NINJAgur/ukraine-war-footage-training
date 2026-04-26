"""
One-off script: GDINO auto-label ALL piterfm images using category-aware prompts.
Category is encoded in each image's source folder name.
Outputs a clean YOLO dataset to media/kaggle_datasets/labeled/piterfm_labeled/.

Run:
    cd ml-engine
    python tasks/relabel_piterfm.py
    python tasks/relabel_piterfm.py --max-images 0   # all images (default)
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import yaml as _yaml
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("relabel-piterfm")

# category folder name → (gdino prompt, canonical class id)  (-1 = drop)
CATEGORY_MAP = {
    "Aircraft":                                  ("aircraft . jet . plane . bomber",       0),
    "Helicopters":                               ("helicopter",                              0),
    "Reconnaissance_Unmanned_Aerial_Vehicles":   ("drone . uav",                            0),
    "Unmanned_Combat_Aerial_Vehicles":           ("drone . uav",                            0),
    "Tanks":                                     ("tank",                                    1),
    "Infantry_Fighting_Vehicles":                ("armored vehicle",                         1),
    "Armoured_Fighting_Vehicles":                ("armored vehicle",                         1),
    "Armoured_Personnel_Carriers":               ("armored vehicle . apc",                   1),
    "Infantry_Mobility_Vehicles":                ("military vehicle",                         1),
    "Trucks,_Vehicles,_and_Jeeps":               ("military vehicle . truck",                1),
    "Trucks,_Vehicles,_Jeeps,_and—Trains":  ("military vehicle . truck",                1),
    "Self-Propelled_Artillery":                  ("artillery . military vehicle",             1),
    "Towed_Artillery":                           ("artillery",                                1),
    "Multiple_Rocket_Launchers":                 ("rocket launcher . military vehicle",       1),
    "Surface-To-Air_Missile_Systems":            ("missile system . military vehicle",        1),
    "Self-Propelled_Anti-Tank_Missile_Systems":  ("military vehicle",                         1),
    "Self-Propelled_Anti-Aircraft_Guns":         ("military vehicle",                         1),
    "Anti-Aircraft_Guns":                        ("military vehicle . gun",                   1),
    "Engineering_Vehicles_And_Equipment":        ("military vehicle",                         1),
    "Mine-Resistant_Ambush_Protected":           ("military vehicle",                         1),
    "Radars_And_Communications_Equipment":       ("radar . military vehicle",                 1),
    "Radars":                                    ("radar",                                    1),
    "Jammers_And_Deception_Systems":             ("military vehicle",                         1),
    "Command_Posts_And_Communications_Stations": ("military vehicle",                         1),
    "Artillery_Support_Vehicles_And_Equipment":  ("military vehicle . artillery",             1),
    "Unmanned_Ground_Vehicles":                  ("military vehicle",                         1),
    "Naval_Ships":                               (None,                                      -1),
    "Naval_Ships_and_Submarines":                (None,                                      -1),
}

# Handle any unicode/encoding variants of folder names
_EXTRA = {}
for k, v in CATEGORY_MAP.items():
    for char in ["—", "�", "–"]:
        if char not in k:
            variant = k.replace("_and_", f"_and{char}")
            _EXTRA[variant] = v
CATEGORY_MAP.update(_EXTRA)


def _collect_images(base: Path, max_images: int) -> List[Tuple[Path, str, int]]:
    """Return list of (img_path, gdino_prompt, canonical_id) for all piterfm images."""
    results = []
    unknown_cats = set()
    for snapshot in sorted(base.iterdir()):
        if not snapshot.is_dir():
            continue
        for side in sorted(snapshot.iterdir()):
            if not side.is_dir():
                continue
            for cat in sorted(side.iterdir()):
                if not cat.is_dir():
                    continue
                entry = CATEGORY_MAP.get(cat.name)
                if entry is None:
                    unknown_cats.add(cat.name)
                    continue
                prompt, canonical_id = entry
                if canonical_id == -1:
                    continue
                for img in sorted(cat.glob("*.jpg")) + sorted(cat.glob("*.jpeg")) + sorted(cat.glob("*.png")):
                    results.append((img, prompt, canonical_id))
    if unknown_cats:
        logger.warning(f"Unknown categories (skipped): {unknown_cats}")
    if max_images > 0:
        results = results[:max_images]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino

    ds_base = (
        settings.KAGGLE_CACHE_DIR / "piterfm" /
        "2022-ukraine-russia-war-equipment-losses-oryx" / "versions" / "55"
    )
    output_dir = settings.KAGGLE_CACHE_DIR / "labeled" / "piterfm_labeled"

    if output_dir.exists():
        logger.info(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    out_images = output_dir / "train" / "images"
    out_labels = output_dir / "train" / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    images = _collect_images(ds_base, args.max_images)
    logger.info(f"Images to process: {len(images)}")

    config_path = str(Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py")
    checkpoint_path = str(Path(__file__).parent.parent / "groundingdino_swint_ogc.pth")
    logger.info("Loading GDINO model...")
    model = load_model(config_path, checkpoint_path)

    labeled = empty = failed = 0
    for i, (img_path, prompt, canonical_id) in enumerate(images):
        stem = f"{img_path.parent.name}__{img_path.stem}"
        dst_img = out_images / (stem + img_path.suffix)
        shutil.copy2(img_path, dst_img)

        try:
            image_source, image = load_image(str(img_path))
            boxes, logits, phrases = predict(
                model=model, image=image, caption=prompt,
                box_threshold=settings.GDINO_BOX_THRESHOLD,
                text_threshold=settings.GDINO_TEXT_THRESHOLD,
            )
        except Exception as exc:
            logger.warning(f"GDINO failed on {img_path.name}: {exc}")
            (out_labels / (stem + ".txt")).write_text("")
            failed += 1
            continue

        lines = [f"{canonical_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes.tolist()]
        lbl_path = out_labels / (stem + ".txt")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            labeled += 1
        else:
            empty += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{len(images)}  labeled={labeled}  empty={empty}")

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        _yaml.dump({"path": str(output_dir), "train": "train/images", "val": "train/images",
                    "nc": 3, "names": ["aircraft", "vehicle", "personnel"]}, f, default_flow_style=False)

    logger.info(f"Done. labeled={labeled}  empty={empty}  failed={failed}  total={len(images)}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
