import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

AIRPORTS = {
    # India
    "VIDP": "Indira Gandhi International Airport, Delhi",
    "VABB": "Chhatrapati Shivaji Maharaj International Airport, Mumbai",
    "VOBL": "Kempegowda International Airport, Bengaluru",
    "VOMM": "Chennai International Airport, Chennai",
    "VECC": "Netaji Subhash Chandra Bose International Airport, Kolkata",
    "VOHS": "Rajiv Gandhi International Airport, Hyderabad",
    "VOCI": "Cochin International Airport, Kochi",
    "VAPO": "Pune Airport, Pune",
    "VAAH": "Sardar Vallabhbhai Patel International Airport, Ahmedabad",
    "VOGO": "Dabolim Airport, Goa",
    "VIAR": "Sri Guru Ram Dass Jee International Airport, Amritsar",
    "VOTV": "Trivandrum International Airport, Thiruvananthapuram",
    # International Hubs
    "KJFK": "John F. Kennedy International Airport, New York",
    "KORD": "O'Hare International Airport, Chicago",
    "KLAX": "Los Angeles International Airport, Los Angeles",
    "KATL": "Hartsfield-Jackson Atlanta International Airport, Atlanta",
    "KSFO": "San Francisco International Airport, San Francisco",
    "EGLL": "Heathrow Airport, London",
    "LFPG": "Charles de Gaulle Airport, Paris",
    "EDDF": "Frankfurt Airport, Frankfurt",
    "OMDB": "Dubai International Airport, Dubai",
    "OTHH": "Hamad International Airport, Doha",
    "WSSS": "Singapore Changi Airport, Singapore",
    "RJTT": "Tokyo Haneda Airport, Tokyo",
    "VHHH": "Hong Kong International Airport, Hong Kong",
}

WEATHER_PHENOMENA = {
    "DZ": "Drizzle",
    "RA": "Rain",
    "SN": "Snow",
    "SG": "Snow grains",
    "IC": "Ice crystals",
    "PL": "Ice pellets",
    "GR": "Hail",
    "GS": "Small hail",
    "BR": "Mist",
    "FG": "Fog",
    "FU": "Smoke",
    "VA": "Volcanic ash",
    "DU": "Dust",
    "SA": "Sand",
    "HZ": "Haze",
    "SQ": "Squall",
    "FC": "Funnel cloud (Tornado)",
}

WEATHER_DESCRIPTORS = {
    "MI": "Shallow",
    "BC": "Patches of",
    "PR": "Partial",
    "DR": "Low drifting",
    "BL": "Blowing",
    "SH": "Showers of",
    "TS": "Thunderstorm with",
    "FZ": "Freezing",
}

CLOUD_COVER = {
    "FEW": "Few clouds",
    "SCT": "Scattered clouds",
    "BKN": "Broken clouds (Ceiling)",
    "OVC": "Overcast clouds (Ceiling)",
    "NSC": "Nil significant clouds",
    "SKC": "Sky clear",
    "CLR": "Clear skies",
    "NCD": "No cloud detected",
}

FLIGHT_CATEGORY_DESCRIPTIONS = {
    "VFR": "Visual Flight Rules (Ceiling > 3,000 ft and Visibility > 5 SM)",
    "MVFR": "Marginal VFR (Ceiling 1,000–3,000 ft or Visibility 3–5 SM)",
    "IFR": "Instrument Flight Rules (Ceiling 500–<1,000 ft or Visibility 1–<3 SM)",
    "LIFR": "Low IFR (Ceiling < 500 ft or Visibility < 1 SM - Hazardous)",
}


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
    ceiling_ft: float  # Lowest BKN, OVC, or VV; 20000.0 means unlimited/clear ceiling
    flight_category: str = "VFR"
    raw_metar: str = ""
    report_time: str = ""
    weather_phenomena: List[str] = field(default_factory=list)
    cloud_layers: List[str] = field(default_factory=list)

    @property
    def dewpoint_depression(self) -> Optional[float]:
        """Difference between temperature and dew point in Celsius."""
        if self.temp_c is not None and self.dewpoint_c is not None:
            return round(self.temp_c - self.dewpoint_c, 2)
        return None

    @property
    def wind_u(self) -> Optional[float]:
        """Zonal (East-West) wind component in knots (-speed * sin(dir))."""
        if self.wind_speed is not None and self.wind_dir is not None:
            rad = math.radians(self.wind_dir)
            return round(-self.wind_speed * math.sin(rad), 2)
        return None

    @property
    def wind_v(self) -> Optional[float]:
        """Meridional (North-South) wind component in knots (-speed * cos(dir))."""
        if self.wind_speed is not None and self.wind_dir is not None:
            rad = math.radians(self.wind_dir)
            return round(-self.wind_speed * math.cos(rad), 2)
        return None

    def to_summary(self) -> str:
        """Converts the observation into a comprehensive, human-readable aviation briefing."""
        airport_name = AIRPORTS.get(self.station, f"Airport {self.station}")
        vis_sm = round(self.visibility_m / 1609.34, 2)
        in_hg = (
            round(self.altimeter_hpa / 33.8639, 2)
            if self.altimeter_hpa is not None
            else "N/A"
        )
        cat_desc = FLIGHT_CATEGORY_DESCRIPTIONS.get(
            self.flight_category, self.flight_category
        )

        # Wind text
        if self.wind_speed == 0.0 or (self.wind_dir == 0.0 and self.wind_speed == 0.0):
            wind_text = "Calm (0 knots)"
        elif self.wind_dir == 0.0:
            wind_text = f"Variable at {self.wind_speed} knots"
        else:
            wind_text = f"{int(self.wind_dir)}° at {int(self.wind_speed)} knots"
        if self.wind_gust and self.wind_gust > 0:
            wind_text += f", gusting to {int(self.wind_gust)} knots"

        if self.wind_u is not None and self.wind_v is not None:
            u_dir = "East" if self.wind_u >= 0 else "West"
            v_dir = "North" if self.wind_v >= 0 else "South"
            wind_text += f" [u={self.wind_u} kt ({u_dir}), v={self.wind_v} kt ({v_dir})]"

        # Ceiling text
        if self.ceiling_ft >= 20000.0:
            ceiling_text = "Unlimited (No ceiling detected below 20,000 ft)"
        else:
            ceiling_text = f"{int(self.ceiling_ft):,} ft AGL"

        # Clouds
        clouds_text = (
            ", ".join(self.cloud_layers)
            if self.cloud_layers
            else "Sky clear / No significant clouds"
        )

        # Weather phenomena
        wx_text = (
            ", ".join(self.weather_phenomena)
            if self.weather_phenomena
            else "Nil significant weather reported"
        )

        lines = [
            "=" * 70,
            f" AEROPULSE AVIATION WEATHER BRIEFING: {self.station}",
            f" Location: {airport_name}",
            "=" * 70,
            f" Raw METAR:        {self.raw_metar}",
            f" Observation Time: {self.report_time or 'Zulu time not specified'}",
            f" Flight Category:  {self.flight_category} -> {cat_desc}",
            "-" * 70,
            f" * Wind Conditions:  {wind_text}",
            f" * Visibility:       {int(self.visibility_m):,} meters ({vis_sm} Statute Miles)",
            f" * Cloud Ceiling:    {ceiling_text}",
            f" * Cloud Layers:     {clouds_text}",
            f" * Weather Phenomena:{wx_text}",
            f" * Temperature:      {self.temp_c}°C" if self.temp_c is not None else " * Temperature:      N/A",
            f" * Dew Point:        {self.dewpoint_c}°C" if self.dewpoint_c is not None else " * Dew Point:        N/A",
            f" * Dewpoint Depr:    {self.dewpoint_depression}°C" if self.dewpoint_depression is not None else " * Dewpoint Depr:    N/A",
            f" * Altimeter (QNH):  {self.altimeter_hpa} hPa ({in_hg} inHg)" if self.altimeter_hpa is not None else " * Altimeter (QNH):  N/A",
            "-" * 70,
            " [ML Pipeline Status]: Tabular numeric features extracted for nowcasting.",
            "=" * 70,
        ]
        return "\n".join(lines)


def determine_flight_category(visibility_m: float, ceiling_ft: float) -> str:
    """
    Computes aviation flight rules category based on FAA/ICAO standards:
    - LIFR: Ceiling < 500 ft and/or Visibility < 1 SM (< 1609 m)
    - IFR:  Ceiling 500 to < 1000 ft and/or Visibility 1 to < 3 SM (1609 to < 4828 m)
    - MVFR: Ceiling 1000 to 3000 ft and/or Visibility 3 to 5 SM (4828 to 8047 m)
    - VFR:  Ceiling > 3000 ft and Visibility > 5 SM (> 8047 m)
    """
    vis_sm = visibility_m / 1609.34

    if ceiling_ft < 500.0 or vis_sm < 1.0:
        return "LIFR"
    elif ceiling_ft < 1000.0 or vis_sm < 3.0:
        return "IFR"
    elif ceiling_ft <= 3000.0 or vis_sm <= 5.0:
        return "MVFR"
    else:
        return "VFR"


def decode_weather_token(token: str) -> Optional[str]:
    """Decodes cryptic weather tokens like -RA, TSRA, +SN, FG, HZ, VCSH into plain English."""
    cleaned = token
    intensity = ""
    if cleaned.startswith("+"):
        intensity = "Heavy "
        cleaned = cleaned[1:]
    elif cleaned.startswith("-"):
        intensity = "Light "
        cleaned = cleaned[1:]
    elif cleaned.startswith("VC"):
        intensity = "In the vicinity: "
        cleaned = cleaned[2:]

    descriptor = ""
    for desc, desc_name in WEATHER_DESCRIPTORS.items():
        if desc in cleaned:
            descriptor = desc_name + " "
            cleaned = cleaned.replace(desc, "")
            break

    phenomena = []
    for phen, phen_name in WEATHER_PHENOMENA.items():
        if phen in cleaned:
            phenomena.append(phen_name)
            cleaned = cleaned.replace(phen, "")

    if descriptor or phenomena:
        combined_wx = " ".join(phenomena) if phenomena else ""
        return f"{intensity}{descriptor}{combined_wx}".strip()
    return None


def fetch_metar(icao: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Fetches raw live METAR for a given airport ICAO code.
    If an api_key is supplied, supports CheckWX API; otherwise defaults to
    NOAA's official keyless Aviation Weather API (aviationweather.gov).
    """
    station = icao.strip().upper()
    if not station:
        return None

    # 1. If an API key is provided (e.g. CheckWX)
    if api_key:
        try:
            url = f"https://api.checkwx.com/metar/{station}"
            headers = {"X-API-Key": api_key, "User-Agent": "AeroPulse/1.0"}
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].strip()
        except requests.RequestException:
            pass  # Fall back to NOAA

    # 2. Keyless NOAA Aviation Weather API (free worldwide METARs)
    noaa_url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=raw"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AeroPulseDecoder/1.0"
    }

    try:
        response = requests.get(noaa_url, headers=headers, timeout=10)
        response.raise_for_status()
        raw = response.text.strip()
        return raw if raw else None
    except requests.RequestException as err:
        print(f"[!] Network error while fetching METAR for {station}: {err}")
        return None


def parse_metar_to_numeric(raw_metar: str) -> Optional[MetarObservation]:
    raw_tokens = raw_metar.strip().split()
    if not raw_tokens:
        return None

    if raw_tokens[0] in ["METAR", "SPECI"]:
        raw_tokens.pop(0)

    if not raw_tokens:
        return None

    station_code = raw_tokens.pop(0)

    # Cut off remarks
    if "RMK" in raw_tokens:
        raw_tokens = raw_tokens[: raw_tokens.index("RMK")]

    # Normalize split fractional statute miles: e.g. ['1', '1/2SM'] -> ['1 1/2SM']
    tokens = []
    i = 0
    while i < len(raw_tokens):
        if (
            re.match(r"^\d+$", raw_tokens[i])
            and i + 1 < len(raw_tokens)
            and re.match(r"^(?:M|P)?\d+/\d+SM$", raw_tokens[i + 1])
        ):
            tokens.append(f"{raw_tokens[i]} {raw_tokens[i + 1]}")
            i += 2
        else:
            tokens.append(raw_tokens[i])
            i += 1

    # Defaults
    wind_dir = None
    wind_speed = None
    wind_gust = 0.0
    vis_m = 10000.0
    temp_c = None
    dewpoint_c = None
    altimeter_hpa = None
    ceiling_layers = []
    report_time = ""
    weather_phenomena_list = []
    cloud_layers_desc = []

    for token in tokens:
        # Timestamp: e.g. 051200Z (Day 05 at 12:00 Zulu)
        if re.match(r"^\d{6}Z$", token):
            day = token[0:2]
            hh = token[2:4]
            mm = token[4:6]
            report_time = f"Day {day} at {hh}:{mm} UTC (Zulu)"

        # CAVOK: Ceiling And Visibility OK
        elif token == "CAVOK":
            vis_m = 10000.0
            cloud_layers_desc.append("CAVOK (Visibility >= 10 km, no clouds below 5,000 ft)")

        # Wind: 20009KT, 20009G18KT, VRB05KT, 00000KT, or with MPS
        elif re.match(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)$", token):
            wind_match = re.match(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)$", token)
            d, s, g, unit = wind_match.groups()
            multiplier = 1.94384 if unit == "MPS" else 1.0
            wind_dir = 0.0 if d == "VRB" else float(d)
            wind_speed = round(float(s) * multiplier, 2)
            wind_gust = round(float(g) * multiplier, 2) if g else 0.0

        # Visibility: Meters (e.g., 3500, 9999, 0000)
        elif re.match(r"^\d{4}$", token):
            vis_m = 10000.0 if token == "9999" else float(token)

        # Visibility: Statute Miles (e.g., 10SM, 3/4SM, 1 1/2SM, M1/4SM, P6SM)
        elif "SM" in token:
            clean = token.replace("SM", "").replace("P", "").replace("M", "").strip()
            if " " in clean:
                whole, frac = clean.split(" ")
                num, den = frac.split("/")
                miles = float(whole) + (float(num) / float(den))
            elif "/" in clean:
                num, den = clean.split("/")
                miles = float(num) / float(den)
            else:
                miles = float(clean)
            vis_m = round(miles * 1609.34, 2)

        # Cloud layers: FEW, SCT, BKN, OVC (e.g. BKN020, OVC040CB, SCT030TCU)
        elif re.match(r"^(FEW|SCT|BKN|OVC)\d{3}", token):
            cover = token[:3]
            alt_hundreds = token[3:6]
            altitude_ft = float(alt_hundreds) * 100.0
            suffix = token[6:] if len(token) > 6 else ""
            desc = CLOUD_COVER.get(cover, cover)
            if suffix:
                desc += f" ({suffix})"
            cloud_layers_desc.append(f"{desc} at {int(altitude_ft):,} ft AGL")

            # Aviation ceiling rule: broken (BKN) or overcast (OVC) only
            if cover in ["BKN", "OVC"]:
                ceiling_layers.append(altitude_ft)

        # Vertical visibility (indefinite ceiling): VV002 (200 ft), VV/// (obscured)
        elif re.match(r"^VV\d{3}$", token):
            altitude_ft = float(token[2:5]) * 100.0
            ceiling_layers.append(altitude_ft)
            cloud_layers_desc.append(f"Vertical Visibility {int(altitude_ft):,} ft (Indefinite Ceiling)")
        elif token == "VV///":
            ceiling_layers.append(0.0)
            cloud_layers_desc.append("Vertical Visibility Obscured (Surface Ceil 0 ft)")

        # Temperature / Dewpoint: 28/24, M02/M05, 01/M01, or missing values like 15// or //10
        elif re.match(r"^(M?\d{2}|//?)/(M?\d{2}|//?)$", token):
            temp_match = re.match(r"^(M?\d{2}|//?)/(M?\d{2}|//?)$", token)
            t, d = temp_match.groups()
            if "/" not in t:
                temp_c = -float(t[1:]) if t.startswith("M") else float(t)
            if "/" not in d:
                dewpoint_c = -float(d[1:]) if d.startswith("M") else float(d)

        # Altimeter: Q1012 (hPa) or A2992 (inHg)
        elif token.startswith("Q") and token[1:].isdigit():
            altimeter_hpa = float(token[1:])
        elif token.startswith("A") and token[1:].isdigit():
            # e.g., A2992 -> 29.92 inHg -> hPa
            in_hg = float(token[1:]) / 100.0
            altimeter_hpa = round(in_hg * 33.8639, 2)

        # Weather Phenomena: TSRA, -RA, BR, HZ, FG, etc.
        else:
            decoded_wx = decode_weather_token(token)
            if decoded_wx:
                weather_phenomena_list.append(f"{decoded_wx} ({token})")

    lowest_ceiling = min(ceiling_layers) if ceiling_layers else 20000.0
    category = determine_flight_category(vis_m, lowest_ceiling)

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
        flight_category=category,
        raw_metar=raw_metar.strip(),
        report_time=report_time,
        weather_phenomena=weather_phenomena_list,
        cloud_layers=cloud_layers_desc,
    )


def decode_metar_to_text(raw_metar: str) -> str:
    """Decodes a raw METAR string into a full human-readable aviation report."""
    obs = parse_metar_to_numeric(raw_metar)
    if not obs:
        return "Error: Could not decode METAR report (empty or malformed)."
    return obs.to_summary()


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("        AEROPULSE: LIVE METAR FETCHER & AVIATION DECODER")
    print("=" * 70)

    api_key_env = os.environ.get("CHECKWX_API_KEY", None)

    # Allow passing ICAO directly via command line (e.g. python parser.py VIDP) or prompt
    if len(sys.argv) > 1:
        icao_input = sys.argv[1].strip().upper()
    else:
        prompt_text = "Enter Airport ICAO code (e.g., VIDP, KJFK, VABB, EGLL) [default: VIDP]: "
        icao_input = input(prompt_text).strip().upper()
        if not icao_input:
            icao_input = "VIDP"

    print(f"\n[+] Fetching live observation for {icao_input} from Aviation Weather API...")
    raw_report = fetch_metar(icao_input, api_key=api_key_env)

    if raw_report:
        obs = parse_metar_to_numeric(raw_report)
        print("\n" + obs.to_summary())
    else:
        print(f"[-] Could not find an active METAR report for airport '{icao_input}'.")
        print("    Please check the 4-letter ICAO code and your internet connection.")