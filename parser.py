import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetarObservation:
    station: str
    wind_dir: Optional[float]
    wind_speed: Optional[float]
    wind_gust: Optional[float]
    visibility_m: float
    temp_c: Optional[float]
    dewpoint_c: Optional[float]
    altimeter_hpa: Optional[float]
    ceiling_ft: float  # Lowest BKN or OVC; 20000.0 means unlimited/clear ceiling


def parse_metar_to_numeric(raw_metar: str) -> Optional[MetarObservation]:
    tokens = raw_metar.strip().split()
    if not tokens:
        return None

    if tokens[0] in ["METAR", "SPECI"]:
        tokens.pop(0)

    station_code = tokens.pop(0)

    # Cut off remarks
    if "RMK" in tokens:
        tokens = tokens[: tokens.index("RMK")]

    # Defaults
    wind_dir = None
    wind_speed = None
    wind_gust = 0.0
    vis_m = 10000.0
    temp_c = None
    dewpoint_c = None
    altimeter_hpa = None
    ceiling_layers = []

    for token in tokens:
        # Wind: 20009KT, 20009G18KT, or VRB05KT
        wind_match = re.match(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT$", token)
        if wind_match:
            d, s, g = wind_match.groups()
            wind_dir = 0.0 if d == "VRB" else float(d)
            wind_speed = float(s)
            wind_gust = float(g) if g else 0.0

        # Visibility: Meters (e.g., 3500, 9999)
        elif re.match(r"^\d{4}$", token):
            vis_m = float(token)

        # Visibility: Statute Miles (e.g., 10SM, 3/4SM, 1 1/2SM)
        elif "SM" in token:
            clean = token.replace("SM", "")
            if "/" in clean:
                num, den = clean.split("/")
                miles = float(num) / float(den)
            else:
                miles = float(clean)
            vis_m = miles * 1609.34

        # Cloud layers: FEW, SCT, BKN, OVC
        elif re.match(r"^(FEW|SCT|BKN|OVC)\d{3}", token):
            cover = token[:3]
            altitude_ft = float(token[3:6]) * 100.0
            # Aviation ceiling rule: broken (BKN) or overcast (OVC) only
            if cover in ["BKN", "OVC"]:
                ceiling_layers.append(altitude_ft)

        # Temperature / Dewpoint: 28/24 or M02/M05
        elif re.match(r"^(M?\d{2})/(M?\d{2})$", token):
            t, d = token.split("/")
            temp_c = -float(t[1:]) if t.startswith("M") else float(t)
            dewpoint_c = -float(d[1:]) if d.startswith("M") else float(d)

        # Altimeter: Q1012 (hPa) or A2992 (inHg)
        elif token.startswith("Q") and token[1:].isdigit():
            altimeter_hpa = float(token[1:])
        elif token.startswith("A") and token[1:].isdigit():
            # e.g., A2992 -> 29.92 inHg -> hPa
            in_hg = float(token[1:]) / 100.0
            altimeter_hpa = in_hg * 33.8639

    lowest_ceiling = min(ceiling_layers) if ceiling_layers else 20000.0

    return MetarObservation(
        station=station_code,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        wind_gust=wind_gust,
        visibility_m=vis_m,
        temp_c=temp_c,
        dewpoint_c=dewpoint_c,
        altimeter_hpa=altimeter_hpa,
        ceiling_ft=lowest_ceiling,
    )