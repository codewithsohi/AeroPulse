from parser import (
    determine_flight_category,
    parse_metar_to_numeric,
    decode_metar_to_text,
)


def test_parse_standard_metar():
    raw = "KJFK 011200Z 24012G20KT 5000 BKN020 OVC040 28/24 Q1012"
    obs = parse_metar_to_numeric(raw)
    assert obs.wind_dir == 240.0
    assert obs.wind_speed == 12.0
    assert obs.wind_gust == 20.0
    assert obs.visibility_m == 5000.0
    assert obs.ceiling_ft == 2000.0  # Selects lowest of BKN020 and OVC040
    assert obs.temp_c == 28.0
    assert obs.dewpoint_c == 24.0
    assert obs.altimeter_hpa == 1012.0
    assert obs.flight_category == "MVFR"  # Ceiling 2000ft is MVFR (1000-3000ft)


def test_statute_miles_and_negative_temp():
    raw = "KJFK 011200Z 00000KT 10SM FEW050 M02/M05 A2992"
    obs = parse_metar_to_numeric(raw)
    assert round(obs.visibility_m) == 16093  # 10 miles to meters
    assert obs.ceiling_ft == 20000.0         # FEW is not a ceiling
    assert obs.temp_c == -2.0
    assert obs.dewpoint_c == -5.0
    assert obs.altimeter_hpa == 1013.21      # 29.92 inHg -> hPa
    assert obs.flight_category == "VFR"


def test_split_statute_miles_and_dewpoint_depression():
    # Split statute miles e.g. "1 1/2SM"
    raw = "KORD 011200Z 20010KT 1 1/2SM BR BKN015 10/08 A2995"
    obs = parse_metar_to_numeric(raw)
    assert obs.visibility_m == 2414.01       # 1.5 miles * 1609.34
    assert obs.ceiling_ft == 1500.0
    assert obs.flight_category == "IFR"      # Vis 1.5 SM is < 3 SM
    assert obs.dewpoint_depression == 2.0


def test_cavok_and_vfr():
    # CAVOK: Ceiling And Visibility OK
    raw = "EGLL 011200Z 27010KT CAVOK 18/10 Q1020"
    obs = parse_metar_to_numeric(raw)
    assert obs.visibility_m == 10000.0
    assert obs.ceiling_ft == 20000.0
    assert obs.flight_category == "VFR"


def test_vertical_visibility_and_lifr():
    # Vertical visibility e.g. VV002 (200 ft indefinite ceiling) + low vis
    raw = "KLAX 011200Z 00000KT 1/4SM FG VV002 08/08 A3000"
    obs = parse_metar_to_numeric(raw)
    assert obs.visibility_m == round(0.25 * 1609.34, 2)  # 0.25 * 1609.34 (~402.33m)
    assert obs.ceiling_ft == 200.0
    assert obs.flight_category == "LIFR"     # Ceiling < 500ft and vis < 1 SM


def test_wind_vector_decomposition():
    # Wind from 090 degrees at 10 knots -> u = -10.0, v = 0.0
    raw = "VIDP 011200Z 09010KT 5000 SCT030 25/20 Q1010"
    obs = parse_metar_to_numeric(raw)
    assert obs.wind_dir == 90.0
    assert obs.wind_speed == 10.0
    assert obs.wind_u == -10.0
    assert obs.wind_v == 0.0


def test_missing_dewpoint_and_vrb_wind():
    # Missing dew point "15//" and VRB wind
    raw = "KJFK 011200Z VRB05KT 9999 FEW030 15// Q1015"
    obs = parse_metar_to_numeric(raw)
    assert obs.wind_dir == 0.0
    assert obs.wind_speed == 5.0
    assert obs.visibility_m == 10000.0       # 9999 maps to 10000m
    assert obs.temp_c == 15.0
    assert obs.dewpoint_c is None
    assert obs.dewpoint_depression is None


def test_decode_metar_to_text():
    raw = "VIDP 050200Z 25009KT 3000 HZ NSC 28/16 Q1011 NOSIG"
    text = decode_metar_to_text(raw)
    assert "Indira Gandhi International Airport, Delhi" in text
    assert "IFR" in text
    assert "3,000 meters" in text


if __name__ == "__main__":
    tests = [
        ("Standard METAR", test_parse_standard_metar),
        ("Statute miles & sub-zero temps", test_statute_miles_and_negative_temp),
        ("Split statute miles & depression", test_split_statute_miles_and_dewpoint_depression),
        ("CAVOK & VFR rule", test_cavok_and_vfr),
        ("Vertical visibility & LIFR", test_vertical_visibility_and_lifr),
        ("Wind vector decomposition", test_wind_vector_decomposition),
        ("Missing sensor values & VRB", test_missing_dewpoint_and_vrb_wind),
        ("Plain-English text briefing", test_decode_metar_to_text),
    ]

    print("=" * 60)
    print("Running AeroPulse Test Suite...")
    print("=" * 60)
    for name, test_fn in tests:
        test_fn()
        print(f" [PASS] {name}")
    print("\nAll 8 test cases passed successfully!\n")