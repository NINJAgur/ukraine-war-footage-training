"""
aircraft_pipeline.py

Scrape 2 aircraft-relevant clips from Funker530 + 2 from GeoConfirmed,
validate each clip has visible aircraft (≥15% frames with detections),
then run the trained AIRCRAFT YOLO model on each → annotated MP4s.

Usage (from repo root):
    cd ml-engine && python scripts/aircraft_pipeline.py
"""
import sys
import re
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aircraft_pipeline")

REPO_ROOT        = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR    = REPO_ROOT / "ml-engine"
SCRAPER_DIR      = REPO_ROOT / "scraper-engine"
AIRCRAFT_WEIGHTS = ML_ENGINE_DIR / "runs/baseline/AIRCRAFT/baseline_AIRCRAFT_13/weights/best.pt"
OUT_DIR          = ML_ENGINE_DIR / "media/annotated"
RAW_FUNKER       = SCRAPER_DIR / "media/raw/funker530"
RAW_GEO          = SCRAPER_DIR / "media/raw/geoconfirmed"

sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(SCRAPER_DIR))

_HIGH = re.compile(
    r"\b(fpv|drone|uav|shahed|geran|lancet|orlan|bayraktar|"
    r"helicopter|ka-52|mi-8|mi-17|mi-24|mi-28|mi-35|"
    r"loitering|kamikaze drone|strike drone|recon drone)\b",
    re.IGNORECASE,
)
_MED = re.compile(
    r"\b(aircraft|jet|plane|su-24|su-25|su-27|su-30|su-34|su-35|"
    r"glide bomb|kab|fab-500|fab-1500|intercept|aerial)\b",
    re.IGNORECASE,
)


def _aircraft_score(title: str, desc: str = "") -> int:
    text = f"{title} {desc}"
    return len(_HIGH.findall(text)) * 3 + len(_MED.findall(text))


# ── 1. Cleanup ────────────────────────────────────────────────────────

def cleanup():
    for d in [RAW_FUNKER, RAW_GEO]:
        if d.exists():
            shutil.rmtree(d)
            log.info(f"Removed {d}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 2. Fetch + score + download ───────────────────────────────────────

def scrape_funker(model, n: int = 2) -> list[Path]:
    from tasks.scrape_funker530 import fetch_ukraine_posts, _download_video, get_output_path
    from core.inference import validate_clip

    log.info("Fetching Funker530 posts…")
    posts = fetch_ukraine_posts(max_count=60)
    log.info(f"  fetched {len(posts)} posts")

    scored = sorted(
        posts,
        key=lambda c: _aircraft_score(c.get("title", ""), c.get("description", "")),
        reverse=True,
    )

    paths = []
    for p in scored:
        if len(paths) >= n:
            break
        score = _aircraft_score(p.get("title", ""), p.get("description", ""))
        if score == 0:
            log.info("  remaining candidates score 0, stopping")
            break
        log.info(f"  trying (score={score}): {p.get('title', '')[:80]!r}")
        out = get_output_path(p["video_url"], "clip")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_video(p["video_url"], out)
            if validate_clip(model, out):
                paths.append(out)
        except Exception as e:
            log.error(f"  failed: {e}")

    log.info(f"  Funker530: {len(paths)}/{n} valid clips")
    return paths


def scrape_geo(model, n: int = 2) -> list[Path]:
    from tasks.scrape_geoconfirmed import extract_video_incidents, _download_video, get_output_path
    from core.inference import validate_clip

    log.info("Fetching GeoConfirmed incidents…")
    incidents = extract_video_incidents(max_incidents=60)
    log.info(f"  fetched {len(incidents)} incidents")

    scored = sorted(
        incidents,
        key=lambda c: _aircraft_score(c.get("title", ""), c.get("description", "")),
        reverse=True,
    )

    paths = []
    for inc in scored:
        if len(paths) >= n:
            break
        score = _aircraft_score(inc.get("title", ""), inc.get("description", ""))
        if score == 0:
            log.info("  remaining candidates score 0, stopping")
            break
        log.info(f"  trying (score={score}): {inc.get('title', '')[:80]!r}")
        out = get_output_path(inc["url"], "clip")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_video(inc["url"], out)
            if validate_clip(model, out):
                paths.append(out)
        except Exception as e:
            log.error(f"  failed: {e}")

    log.info(f"  GeoConfirmed: {len(paths)}/{n} valid clips")
    return paths


# ── 4. Inference ──────────────────────────────────────────────────────

def annotate(input_paths: list[Path], model) -> list[Path]:
    from core.inference import infer_video_multi_model

    output_paths = []
    for src in input_paths:
        stem = src.stem.split("_clip")[0]
        out  = OUT_DIR / f"{stem}_aircraft_annotated.mp4"
        log.info(f"Annotating {src.name} → {out.name}")
        try:
            infer_video_multi_model(
                models_info=[(model, "AIRCRAFT", (210, 130, 38))],  # BGR ≈ amber
                video_path=str(src),
                conf_thresh=0.35,
                save_path=str(out),
                no_display=True,
            )
            log.info(f"  saved: {out}")
            output_paths.append(out)
        except Exception as e:
            log.error(f"  annotation failed: {e}")

    return output_paths


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== AIRCRAFT PIPELINE START ===")

    cleanup()

    from ultralytics import YOLO
    if not AIRCRAFT_WEIGHTS.exists():
        raise FileNotFoundError(f"AIRCRAFT weights not found: {AIRCRAFT_WEIGHTS}")
    log.info(f"Loading AIRCRAFT model from {AIRCRAFT_WEIGHTS}")
    aircraft_model = YOLO(str(AIRCRAFT_WEIGHTS))

    funker_clips = scrape_funker(aircraft_model, n=2)
    geo_clips    = scrape_geo(aircraft_model, n=2)

    all_clips = funker_clips + geo_clips
    log.info(f"\nDownloaded {len(all_clips)} valid clips total")
    if len(all_clips) < 4:
        log.warning(f"Only {len(all_clips)} clips passed validation (expected 4)")
    if not all_clips:
        log.error("No valid clips — aborting")
        sys.exit(1)

    annotated = annotate(all_clips, aircraft_model)

    log.info("\n=== DONE ===")
    log.info(f"Annotated videos ({len(annotated)}):")
    for p in annotated:
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        log.info(f"  {p.name}  ({size_mb:.1f} MB)")
