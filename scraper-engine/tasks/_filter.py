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

_UAS = [
    "drone", "uav", "fpv", "shahed", "geran", "lancet", "orlan", "bayraktar", 
    "tb2", "switchblade", "mavic", "baba yaga", "leleka", "valkyrie", "puma", 
    "poseidon", "shark", "zala", "supercam"
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
    "armored vehicle", "armoured vehicle"
]

_ARTILLERY_AIR_DEFENSE = [
    "artillery", "howitzer", "mortar", "mlrs", "m777", "pzh2000", "krab", 
    "caesar", "m109", "paladin", "dana", "zuzana", "archer", "bogdana", 
    "2s1", "gvozdika", "2s3", "akatsiya", "2s5", "giatsint", "2s7", "pion", 
    "2s19", "msta", "d-30", "msta-b", "grad", "bm-21", "uragan", "smerch", 
    "himars", "m270", "tos-1", "tos-1a", "solntsepyok", "patriot", "iris-t", 
    "nasams", "s-300", "s-400", "buk", "tor", "pantsir", "tunguska", "gepard"
]

_AIRCRAFT = [
    "helicopter", "ka-52", "alligator", "mi-8", "mi-17", "mi-24", "mi-28",
    "mi-35", "uh-60", "black hawk", "ah-64", "apache", "aircraft", "jet",
    "plane", "su-24", "su-25", "su-27", "su-30", "su-34", "su-35", "mig-29",
    "mig-31", "f-16", "a-50", "il-22", "tu-22", "tu-95", "tu-160",
    "glide bomb", "kab", "fab-500", "fab-1500", "fab-3000",
]

_NAVAL_MARINE = [
    "ship", "boat", "vessel", "usv", "sea drone", "magura", "magura v5", 
    "sea baby", "landing ship", "ropucha", "tapir", "corvette", "frigate", 
    "submarine", "kilo class", "buyan", "karakurt", "slava class", "moskva", 
    "raptor", "patrol boat", "bk-16"
]

_INFANTRY_WEAPONS = [
    "rpg", "atgm", "javelin", "nlaw", "stugna", "stugna-p", "kornet", 
    "fagot", "konkurs", "milan", "tow", "carl gustaf", "at4", "panzerfaust",
    "missile", "rocket launcher"
]

_PERSONNEL = [
    "soldier", "soldiers", "troops", "infantry", "sniper", "fighter", 
    "fighters", "combatant", "combatants", "marine", "marines", "spetsnaz", 
    "vdv", "sso", "kraken", "azov", "wagner", "storm-z", "kadyrovtsy", 
    "mercenary", "paratrooper"
]

# Combine all equipment lists
EQUIPMENT_KEYWORDS = (
    _UAS + _TANKS + _ARMORED_VEHICLES + _ARTILLERY_AIR_DEFENSE + 
    _AIRCRAFT + _NAVAL_MARINE + _INFANTRY_WEAPONS + _PERSONNEL
)

# Sort by length descending to ensure specific models match before broad terms
# e.g., "t-72b3" matches before "t-72"; "sea drone" before "drone"
EQUIPMENT_KEYWORDS.sort(key=len, reverse=True)
_eq_joined = "|".join(map(re.escape, EQUIPMENT_KEYWORDS))
EQUIPMENT_PATTERN = re.compile(rf"\b({_eq_joined})\b", re.IGNORECASE)


# ── 2. Impact & Aftermath (Negative Match) ─────────────────────────────
# Block on VISUAL STATE (fire, smoke, ruins) not on TARGET TYPE (refinery, bridge).
# "drone hits refinery" → valid; "smoke plume over refinery" → invalid.
#
# Omitted intentionally:
#   "explosion" / "detonation" — these ARE the moment of impact (FPV hit = explosion)
#   "crashed" — "drone crashed into tank" is the action itself
#   "destroyed" (standalone) — "T-72 destroyed by FPV" is valid action footage
#   "damaged" (standalone) — "BMP damaged by ATGM" is valid action footage
IMPACT_KEYWORDS = [
    # Aftermath states
    "aftermath",
    "ruins",
    "rubble",
    "wreckage",
    "debris",
    "remains",
    "crater",
    "crash site",
    "burning wreckage",
    "obliterated",
    "scorched",
    "charred",
    "smoldering",
    "incinerated",
    # Fire/smoke states (aftermath visual evidence)
    "fire",
    "flames",
    "in flames",
    "engulfed",
    "inferno",
    "blaze",
    "burning",
    "smoke",
    "smoke plume",
    "on fire",
    # Damage assessment language (editorial framing = not raw action footage)
    "bomb damage",
    "battle damage",
    "battle damage assessment",
    "post-strike",
    "war damage",
    "following the strike",
    "following the attack",
    "aftermath of",
    "result of",             # "result of the strike" = editorial aftermath framing
]

IMPACT_KEYWORDS.sort(key=len, reverse=True)
_impact_joined = "|".join(map(re.escape, IMPACT_KEYWORDS))
IMPACT_PATTERN = re.compile(rf"\b({_impact_joined})\b", re.IGNORECASE)


# ── 3. Geo Markers (Soft Verification) ─────────────────────────────────
GEO_KEYWORDS = [
    "ukraine", "ukrainian", "russia", "russian", "donetsk", "luhansk", 
    "zaporizhzhia", "kherson", "kharkiv", "kyiv", "mariupol", "bakhmut", 
    "avdiivka", "dnipro", "crimea", "donbas", "donbass", "wagner", "azov"
]

GEO_KEYWORDS.sort(key=len, reverse=True)
_geo_joined = "|".join(map(re.escape, GEO_KEYWORDS))
GEO_PATTERN = re.compile(rf"\b({_geo_joined})\b", re.IGNORECASE)


# ── Filter Functions ───────────────────────────────────────────────────

def check_equipment(title: str, description: str = "") -> tuple[bool, str]:
    """Return (True, matched_keyword) if text strictly names military equipment."""
    text = f"{title} {description}"
    match = EQUIPMENT_PATTERN.search(text)
    if match:
        return True, match.group(1).lower()
    return False, "no equipment keyword"


def is_infrastructure_strike(title: str, description: str = "") -> tuple[bool, str]:
    """Return (True, reason) if text describes impact/aftermath states rather than live action."""
    text = f"{title} {description}"
    match = IMPACT_PATTERN.search(text)
    if match:
        return True, f"impact/aftermath keyword '{match.group(1).lower()}'"
    return False, ""


def check_geo(title: str, description: str = "") -> Optional[str]:
    """Return first matched Ukraine/Russia geo keyword, or None."""
    text = f"{title} {description}"
    match = GEO_PATTERN.search(text)
    return match.group(1).lower() if match else None