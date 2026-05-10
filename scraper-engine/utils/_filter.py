"""
Shared content-quality filter for all scraper tasks.

Refined to prioritize 'Action' over 'Aftermath'. We accept FPV drones hitting
targets, but reject videos focused on aftermath/consequences with no visible
action. Filter operates on text only (title + description).

Key principle: block on EVIDENCE OF AFTERMATH (fire, smoke, ruins, wreckage),
NOT on the target type (refinery, bridge). A drone hitting a refinery is valid;
a smoke-plume-over-refinery three hours later is not.
"""
import re
from typing import Optional

# ── 1. Equipment & Personnel (Positive Match) ──────────────────────────

_AIRCRAFT = [
    "fighter", "helicopter", "ka-52", "alligator", "mi-8", "mi-17", "mi-24", "mi-28",
    "mi-35", "uh-60", "black hawk", "ah-64", "apache", "aircraft", "aviation",
    "plane", "su-24", "su-25", "su-27", "su-30", "su-34", "su-35", "mig-29",
    "mig-31", "f-16", "a-50", "il-22", "tu-22", "tu-95", "tu-160",
    "glide bomb", "kab", "fab-500", "fab-1500", "fab-3000",
    # MOVED TARGET DRONES HERE:
    "shahed", "geran", "lancet", "orlan", "bayraktar", "tb2", "switchblade", 
    "leleka", "valkyrie", "puma", "poseidon", "shark", "zala", "supercam",
    "Liutyi", "AN-196", "Molnya-2", "Molniya-2"
]

_UAS = [
    # GENERIC AND CAMERA DRONES ONLY:
    "point of view", "drone", "uav", "fpv", "mavic", "baba yaga", "quadcopter", "hexacopter"
]

_TANKS = [
    "tank", "t-54", "t-55", "t-62", "t-64", "t-72", "t-72b3", "t-80", "t-80bvm", 
    "t-90", "t-90m", "leopard", "leopard 2", "abrams", "m1a1", "challenger", 
    "challenger 2", "pt-91", "amx-10"
]

_ARMORED_VEHICLES = [
    "bmp", "bmp-1", "bmp-2", "bmp-3", "btr", "btr-60", "btr-70", "btr-80", 
    "btr-82", "btr-4", "bmd", "bmd-2", "bmd-4", "bradley", "m2a2", "marder", 
    "cv90", "stryker", "m113", "mt-lb", "mrap", "maxxpro", "humvee", "hmmwv", 
    "tigr", "typhoon", "kozak", "spartan", "kirpi", "senator", "ifv", "apc", 
    "armored vehicle", "armoured vehicle", "UGV", "armor"
]

_ARTILLERY_AIR_DEFENSE = [
    "artillery", "howitzer", "Tyulpan", "mortar", "mlrs", "m777", "pzh2000", 
    "krab", "caesar", "m109", "paladin", "dana", "zuzana", "archer", "bogdana", 
    "2s1", "gvozdika", "2s3", "akatsiya", "2s5", "giatsint", "2s7", "pion", 
    "2s19", "msta", "d-30", "msta-b", "grad", "bm-21", "uragan", "smerch", 
    "himars", "m270", "tos-1", "tos-1a", "solntsepyok", "patriot", "iris-t", 
    "nasams", "s-300", "s-400", "buk", "tor", "pantsir", "tunguska", "gepard"
]

_NAVAL_MARINE = [
    "ship", "boat", "vessel", "usv", "sea drone", "magura", "magura v5", 
    "sea baby", "landing ship", "ropucha", "tapir", "corvette", "frigate", 
    "submarine", "kilo class", "buyan", "karakurt", "slava class", "moskva", 
    "raptor", "patrol boat", "bk-16"
]

_LOGISTICS_VEHICLES = [
    "truck", "uaz", "bukhanka", "scooby-doo van", "loaf", "kamaz", "ural", 
    "desertcross", "technical", "supply truck", "logistics truck", "quad"
]

_INFANTRY_WEAPONS = [
    "ak-47", "rpg", "atgm", "javelin", "nlaw", "stugna", "stugna-p", "kornet", 
    "fagot", "konkurs", "milan", "tow", "carl gustaf", "at4", "panzerfaust",
    "anti-tank", "GoPro"
]

_PERSONNEL = [
    "soldier", "soldiers", "troops", "infantry", "infantryman", "infantrymen", "sniper",
    "fighter", "fighters", "combatant", "combatants", "marine", "marines",
    "paratrooper", "mercenary", "casualties", "evacuation team", "POW", "POWs",
    "gopro", "assault squad", "assault group", "stormtroopers", "naval infantry",

    # ── Russian units & factions ──────────────────────────────────────
    "spetsnaz", "vdv", "gru", "rosgvardia", "omon", "sobr",
    "wagner", "storm-z", "kadyrovtsy", "chechen", "chechens", "kontraktnik",

    # ── Ukrainian units & factions ────────────────────────────────────
    "sso", "gur",
    "azov", "kraken", "right sector", "svoboda", "tdf", "tro",
    "foreign fighters", "foreign legion",
    "national guard", "ngu",
]

# POV/Kamikaze (Negative Match for Aircraft Pipeline Only)
POV_KEYWORDS = [
    "fpv", "kamikaze", "drops grenade", "drone drops", "first person view", 
    "waiter drone", "loitering munition", "suicide drone", "impact"
]


# ── 2. Impact & Aftermath (Negative Match) ─────────────────────────────
# Block on VISUAL STATE (fire, smoke, ruins) not on TARGET TYPE (refinery, bridge).
# "drone hits refinery" → valid; "smoke plume over refinery" → invalid.
NEGATIVE_KEYWORDS = [
    # Civilian Personnel
    "civilian", "civilians", "child", "children", "kid", "kids", "woman", "women",
    "elderly", "non-combatant", "bystander", "pedestrian", "resident", "citizen", "paramedic",
    
    # Civilian Residential & Medical
    "apartment", "apartments", "residential", "neighborhood", "hospital", "clinic",
    "maternity", "ambulance", "medical center",

    # Civilian Education & Religion
    "school", "church", "cathedral", "monastery", "kindergarten",

    # Civilian Commercial, Public & Infrastructure
    "mall", "shopping center", "supermarket", "grocery", "market", "museum", "library",
    "park", "train", "railway", "water treatment", "plant", "power plant", "bridge", "oil depot",
    "dispatch station", "production line",
    
    # Aftermath states
    "aftermath", "ruins", "rubble", "wreckage", "debris", "remains",
    "crater", "crash site", "burning wreckage", "obliterated", "scorched", "charred",
    "smoldering", "incinerated", "crashed",
    
    # Fire/smoke states (aftermath visual evidence)
    "flames", "in flames", "engulfed", "inferno", "blaze", "burning", "smoke", 
    "smoke plume", "on fire", "explosion",
    
    # Damage assessment language (editorial framing = not raw action footage)
    "bomb damage", "battle damage", "battle damage assessment", 
    "damages", "following the strike", "following the attack", "post-strike", 
    "war damage",
]


# ── 3. Geo Markers (Soft Verification) ─────────────────────────────────
GEO_KEYWORDS = [
    "ukraine", "ukrainian", "russia", "russian", "donetsk", "luhansk", 
    "zaporizhzhia", "kherson", "kharkiv", "kyiv", "mariupol", "bakhmut", 
    "avdiivka", "dnipro", "crimea", "donbas", "donbass", "wagner", "azov"
]


# ── Filter & Scoring Functions ─────────────────────────────────────────

def _make_pattern(terms: list) -> re.Pattern:
    joined = "|".join(map(re.escape, sorted(terms, key=len, reverse=True)))
    # Explicit boundary treats hyphens as part of words, so "tank" won't match
    # inside "anti-tank" and "t-72" won't match inside "t-72b3".
    return re.compile(rf"(?<![a-zA-Z0-9-])({joined})(?![a-zA-Z0-9-])", re.IGNORECASE)

PATTERNS = {
    "aircraft": _make_pattern(_AIRCRAFT),
    "vehicle": _make_pattern(_TANKS + _ARMORED_VEHICLES + _ARTILLERY_AIR_DEFENSE + _LOGISTICS_VEHICLES + _NAVAL_MARINE),
    "personnel": _make_pattern(_PERSONNEL + _INFANTRY_WEAPONS),
    "uas": _make_pattern(_UAS),
    "pov": _make_pattern(POV_KEYWORDS),
    "negative": _make_pattern(NEGATIVE_KEYWORDS),
    "geo": _make_pattern(GEO_KEYWORDS),
}

def get_equipment_scores(title: str, description: str = "") -> tuple[dict, bool]:
    """Returns exactly the dictionary needed for the DB, and a boolean if it's worth scraping."""
    text = f"{title} {description}"
    scores = {
        "score_aircraft": len(PATTERNS["aircraft"].findall(text)),
        "score_vehicle": len(PATTERNS["vehicle"].findall(text)),
        "score_personnel": len(PATTERNS["personnel"].findall(text)),
        "score_uas": len(PATTERNS["uas"].findall(text)),
        "is_pov": 1 if PATTERNS["pov"].search(text) else 0
    }
    has_equipment = sum(v for k, v in scores.items() if k != "is_pov") > 0
    return scores, has_equipment

def is_negative_input(title: str, description: str = "") -> tuple[bool, str]:
    """Return (True, reason) if text describes negative/civilian rather than military action."""
    text = f"{title} {description}"
    match = PATTERNS["negative"].search(text)
    if match:
        return True, f"negative keyword '{match.group(1).lower()}'"
    return False, ""

def check_geo(title: str, description: str = "") -> Optional[str]:
    """Return first matched Ukraine/Russia geo keyword, or None."""
    text = f"{title} {description}"
    match = PATTERNS["geo"].search(text)
    return match.group(1).lower() if match else None

def is_pov_noise(scores: dict) -> bool:
    """True if clip is pure FPV drone noise — POV flag with no identifiable class."""
    return (
        scores.get("is_pov", 0) == 1 and
        scores.get("score_uas", 0) > 0 and
        scores.get("score_aircraft", 0) == 0 and
        scores.get("score_vehicle", 0) == 0 and
        scores.get("score_personnel", 0) == 0
    )