from parser import parse_metar_to_numeric

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

def test_statute_miles_and_negative_temp():
    raw = "KJFK 011200Z 00000KT 10SM FEW050 M02/M05 A2992"
    obs = parse_metar_to_numeric(raw)
    assert round(obs.visibility_m) == 16093  # 10 miles to meters
    assert obs.ceiling_ft == 20000.0         # FEW is not a ceiling
    assert obs.temp_c == -2.0
    assert obs.dewpoint_c == -5.0