"""
vehicle_pipeline.py

Scrape 2 vehicle-relevant clips from Funker530 + 2 from GeoConfirmed,
validate each clip has visible vehicles (≥15% frames with detections),
then run the VEHICLE model → annotated MP4 saved to media/annotated/vehicle/.
DB Clip entry written for each annotated clip.

Usage (from repo root):
    cd ml-engine && python scripts/vehicle_pipeline.py
"""
import sys
import re
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vehicle_pipeline")

REPO_ROOT       = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR   = REPO_ROOT / "ml-engine"
SCRAPER_DIR     = REPO_ROOT / "scraper-engine"
VEHICLE_WEIGHTS = ML_ENGINE_DIR / "runs/baseline/VEHICLE/baseline_VEHICLE_25/weights/best.pt"
OUT_DIR         = ML_ENGINE_DIR / "media/annotated/vehicle"

sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(SCRAPER_DIR))
sys.path.insert(0, str(REPO_ROOT))

_TANK_RE = re.compile(
    r"\b(tank|t-54|t-55|t-62|t-64|t-72|t-72b3|t-80|t-80bvm|t-90|t-90m|"
    r"leopard|abrams|m1a1|challenger|pt-91|amx-10)\b",
    re.IGNORECASE,
)
_APC_RE = re.compile(
    r"\b(bmp|btr|bmd|bradley|m2a2|marder|cv90|stryker|m113|mt-lb|mrap|"
    r"maxxpro|humvee|hmmwv|ifv|apc|armou?red vehicle)\b",
    re.IGNORECASE,
)
_ARTY_RE = re.compile(
    r"\b(artillery|howitzer|mlrs|himars|grad|bm-21|caesar|pzh2000|krab|"
    r"m777|paladin|msta|2s19|2s3|2s1|tos-1|solntsepyok|gepard|pantsir)\b",
    re.IGNORECASE,
)


def _vehicle_score(title: str, desc: str = "") -> int:
    text = f"{title} {desc}"
    return (
        len(_TANK_RE.findall(text)) * 3
        + len(_APC_RE.findall(text)) * 2
        + len(_ARTY_RE.findall(text))
    )


# ── 1. Scrape + download ──────────────────────────────────────────────

def scrape_funker(model, n: int = 2) -> list[Path]:
    from tasks.scrape_funker530 import fetch_ukraine_posts, _download_video, get_output_path
    from core.inference import validate_clip

    log.info("Fetching Funker530 posts…")
    posts = fetch_ukraine_posts(max_count=60)
    log.info(f"  fetched {len(posts)} posts")

    scored = sorted(
        posts,
        key=lambda c: _vehicle_score(c.get("title", ""), c.get("description", "")),
        reverse=True,
    )

    paths = []
    for p in scored:
        if len(paths) >= n:
            break
        score = _vehicle_score(p.get("title", ""), p.get("description", ""))
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
        key=lambda c: _vehicle_score(c.get("title", ""), c.get("description", "")),
        reverse=True,
    )

    paths = []
    for inc in scored:
        if len(paths) >= n:
            break
        score = _vehicle_score(inc.get("title", ""), inc.get("description", ""))
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


# ── 2. Annotate with VEHICLE model + write DB ─────────────────────────

def annotate(input_paths: list[Path], model) -> list[Path]:
    from core.inference import infer_video_multi_model
    from db.session import get_session
    from shared.db.models import Clip, ClipSource, ClipStatus
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ml_config", ML_ENGINE_DIR / "config.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    color = _mod.settings.MODEL_COLORS["VEHICLE"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for src in input_paths:
        hash_prefix = src.stem[:8]
        src_name = src.parent.name.lower()
        clip_source = ClipSource.FUNKER530 if "funker" in src_name else ClipSource.GEOCONFIRMED
        out = OUT_DIR / f"{hash_prefix}_annotated.mp4"
        log.info(f"Annotating {src.name} → {out.name}")
        try:
            infer_video_multi_model(
                models_info=[(model, "VEHICLE", color)],
                video_path=str(src),
                conf_thresh=0.4,
                save_path=str(out),
                no_display=True,
            )
        except Exception as e:
            log.error(f"  annotation failed: {e}")
            continue

        full_hash = hash_prefix.ljust(64, "0")
        with get_session() as session:
            stmt = (
                pg_insert(Clip)
                .values(
                    url=f"https://recovered/{hash_prefix}",
                    url_hash=full_hash,
                    source=clip_source,
                    title=src.stem,
                    status=ClipStatus.ANNOTATED,
                    det_class="VEHICLE",
                    mp4_path=str(out),
                    file_path=str(src),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["url_hash"],
                    set_={"mp4_path": str(out), "det_class": "VEHICLE",
                          "status": ClipStatus.ANNOTATED, "updated_at": datetime.utcnow()},
                )
            )
            session.execute(stmt)
        log.info(f"  saved + DB written  det_class=VEHICLE")
        output_paths.append(out)

    return output_paths


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== VEHICLE PIPELINE START ===")

    from ultralytics import YOLO

    if not VEHICLE_WEIGHTS.exists():
        raise FileNotFoundError(f"VEHICLE weights not found: {VEHICLE_WEIGHTS}")
    log.info(f"Loading VEHICLE model from {VEHICLE_WEIGHTS}")
    vehicle_model = YOLO(str(VEHICLE_WEIGHTS))

    funker_clips = scrape_funker(vehicle_model, n=2)
    geo_clips    = scrape_geo(vehicle_model, n=2)

    all_clips = funker_clips + geo_clips
    log.info(f"\nTotal: {len(all_clips)} valid clips")
    if len(all_clips) < 4:
        log.warning(f"Only {len(all_clips)} clips passed validation (expected 4)")
    if not all_clips:
        log.error("No valid clips — aborting")
        sys.exit(1)

    annotated = annotate(all_clips, vehicle_model)

    log.info("\n=== DONE ===")
    log.info(f"Annotated videos ({len(annotated)}):")
    for p in annotated:
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        log.info(f"  {p.name}  ({size_mb:.1f} MB)")
