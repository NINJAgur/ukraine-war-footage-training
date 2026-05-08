import pytest
from utils._filter import get_equipment_scores, is_negative_input, is_pov_noise, check_geo


# ── get_equipment_scores ──────────────────────────────────────────────

@pytest.mark.unit
def test_aircraft_keyword_match():
    scores, has_eq = get_equipment_scores("Ka-52 Alligator destroyed near Kherson")
    assert scores["score_aircraft"] > 0
    assert has_eq is True


@pytest.mark.unit
def test_vehicle_keyword_match():
    scores, has_eq = get_equipment_scores("T-72 tank column spotted advancing")
    assert scores["score_vehicle"] > 0
    assert has_eq is True


@pytest.mark.unit
def test_personnel_keyword_match():
    scores, has_eq = get_equipment_scores("Russian soldiers retreating under fire")
    assert scores["score_personnel"] > 0
    assert has_eq is True


@pytest.mark.unit
def test_uas_keyword_match():
    scores, has_eq = get_equipment_scores("FPV drone attack on convoy")
    assert scores["score_uas"] > 0
    assert has_eq is True


@pytest.mark.unit
def test_fpv_sets_is_pov():
    scores, _ = get_equipment_scores("FPV kamikaze strike on BMP")
    assert scores["is_pov"] == 1


@pytest.mark.unit
def test_is_pov_true_when_uas_no_class():
    # POV + UAS keyword, but no aircraft/vehicle/personnel score
    scores = {
        "is_pov": 1,
        "score_uas": 2,
        "score_aircraft": 0,
        "score_vehicle": 0,
        "score_personnel": 0,
    }
    assert is_pov_noise(scores) is True


@pytest.mark.unit
def test_is_pov_false_when_has_vehicle():
    scores = {
        "is_pov": 1,
        "score_uas": 1,
        "score_aircraft": 0,
        "score_vehicle": 1,
        "score_personnel": 0,
    }
    assert is_pov_noise(scores) is False


@pytest.mark.unit
def test_is_pov_false_when_no_pov_flag():
    scores = {
        "is_pov": 0,
        "score_uas": 3,
        "score_aircraft": 0,
        "score_vehicle": 0,
        "score_personnel": 0,
    }
    assert is_pov_noise(scores) is False


@pytest.mark.unit
def test_is_pov_false_when_no_uas():
    scores = {
        "is_pov": 1,
        "score_uas": 0,
        "score_aircraft": 0,
        "score_vehicle": 0,
        "score_personnel": 0,
    }
    assert is_pov_noise(scores) is False


@pytest.mark.unit
def test_empty_input_no_score():
    scores, has_eq = get_equipment_scores("", "")
    assert all(scores[k] == 0 for k in ("score_aircraft", "score_vehicle", "score_personnel", "score_uas"))
    assert scores["is_pov"] == 0
    assert has_eq is False


@pytest.mark.unit
def test_no_equipment_no_score():
    scores, has_eq = get_equipment_scores("Beautiful sunset over the fields")
    assert has_eq is False
    assert scores["score_aircraft"] == 0
    assert scores["score_vehicle"] == 0
    assert scores["score_personnel"] == 0


@pytest.mark.unit
def test_description_also_scored():
    scores, has_eq = get_equipment_scores("Generic title", "Ukrainian soldiers storm the trench")
    assert scores["score_personnel"] > 0
    assert has_eq is True


# ── is_negative_input ─────────────────────────────────────────────────

@pytest.mark.unit
def test_negative_civilian_match():
    is_neg, reason = is_negative_input("Civilians fleeing the city")
    assert is_neg is True
    assert "civilian" in reason.lower()


@pytest.mark.unit
def test_negative_aftermath_match():
    is_neg, reason = is_negative_input("Aftermath of the strike on the bridge")
    assert is_neg is True
    assert reason != ""


@pytest.mark.unit
def test_clean_military_title_not_negative():
    is_neg, reason = is_negative_input("T-72 tank destroyed by Ukrainian forces in Donetsk")
    assert is_neg is False
    assert reason == ""


@pytest.mark.unit
def test_negative_returns_false_for_clean_personnel():
    is_neg, _ = is_negative_input("Wagner troops advance under artillery cover")
    assert is_neg is False


@pytest.mark.unit
def test_negative_smoke_matches():
    is_neg, reason = is_negative_input("Smoke plume visible over refinery")
    assert is_neg is True
    assert "smoke" in reason.lower()


# ── check_geo ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_geo_match_ukraine():
    result = check_geo("Ukrainian artillery fires on Russian position")
    assert result is not None
    assert "ukrain" in result.lower()


@pytest.mark.unit
def test_geo_match_donetsk():
    result = check_geo("Heavy fighting reported in Donetsk region")
    assert result is not None
    assert result.lower() == "donetsk"


@pytest.mark.unit
def test_geo_no_match():
    result = check_geo("Tank spotted in undisclosed location")
    assert result is None


@pytest.mark.unit
def test_geo_match_in_description():
    result = check_geo("Combat footage", "Recorded near Bakhmut front line")
    assert result is not None
    assert "bakhmut" in result.lower()
