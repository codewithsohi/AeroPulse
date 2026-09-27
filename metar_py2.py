import re
import requests

AIRPORTS = {
    "VABB": "Chhatrapati Shivaji Maharaj International Airport, Mumbai",
    "VIDP": "Indira Gandhi International Airport, Delhi",
    "VOBL": "Kempegowda International Airport, Bengaluru",
    "VOMM": "Chennai International Airport, Chennai",
    "VECC": "Netaji Subhash Chandra Bose International Airport, Kolkata",
    "VAPO": "Pune Airport, Pune",
    "KJFK": "John F. Kennedy International Airport, New York",
    "EGLL": "Heathrow Airport, London",
    "OMDB": "Dubai International Airport, Dubai",
}

WEATHER_PHENOMENA = {
    "DZ": "drizzle", "RA": "rain", "SN": "snow", "SG": "snow grains",
    "IC": "ice crystals", "PL": "ice pellets", "GR": "hail", "GS": "small hail",
    "BR": "mist", "FG": "fog", "FU": "smoke", "VA": "volcanic ash",
    "DU": "dust", "SA": "sand", "HZ": "haze", "SQ": "squalls", "FC": "funnel cloud",
}

WEATHER_DESCRIPTORS = {
    "MI": "shallow", "BC": "patches of", "PR": "partial", "DR": "low drifting",
    "BL": "blowing", "SH": "showers of", "TS": "thunderstorm with", "FZ": "freezing",
}

CLOUD_COVER = {
    "FEW": "few clouds",
    "SCT": "scattered clouds",
    "BKN": "broken clouds",
    "OVC": "overcast clouds",
    "NSC": "nil significant clouds",
    "SKC": "sky clear",
    "CLR": "clear skies",
}


def fetch_metar_no_key(icao: str) -> str:
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AviationMETARDecoder/1.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        raw = response.text.strip()
        return raw if raw else None
    except requests.RequestException as err:
        print(f"Network error: {err}")
        return None


def decode_metar_to_text(raw_metar: str) -> str:
    tokens = raw_metar.strip().split()
    if not tokens:
        return "Empty METAR string."

    if tokens[0] in ["METAR", "SPECI"]:
        tokens.pop(0)

    station_code = tokens.pop(0)
    station_name = AIRPORTS.get(station_code, f"Airport {station_code}")

    time_str = ""
    wind_str = ""
    vis_str = ""
    weather_list = []
    clouds_list = []
    temp_str = ""
    pressure_str = ""

    # Cut off remarks (RMK) so they don't corrupt the decoder
    if "RMK" in tokens:
        tokens = tokens[: tokens.index("RMK")]

    for token in tokens:
        # Time: 160430Z -> day 16 at 04:30 Zulu
        if re.match(r"^\d{6}Z$", token):
            day = token[0:2]
            hh = token[2:4]
            mm = token[4:6]
            time_str = f"reported on day {day} at {hh}:{mm} Zulu."

        # Wind: 20009KT or 20009G18KT
        elif re.match(r"^(\d{3}|VRB)\d{2,3}(G\d{2,3})?KT$", token):
            match = re.match(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT$", token)
            direction, speed, gust = match.groups()
            dir_text = "variable direction" if direction == "VRB" else f"{direction} degrees"
            if gust:
                wind_str = f"wind blowing from {dir_text} at {int(speed)} knots gusting up to {int(gust)} knots."
            else:
                wind_str = f"wind blowing from {dir_text} at {int(speed)} knots."

        # Visibility: 3500 (meters) or 10SM (statute miles)
        elif re.match(r"^\d{4}$", token):
            vis_str = f"visibility of {int(token)} meters."
        elif re.match(r"^(\d+SM|\d+/\d+SM)$", token):
            clean_sm = token.replace("SM", " statute miles.")
            vis_str = f"visibility of {clean_sm}"

        # Cloud layers
        elif re.match(r"^(FEW|SCT|BKN|OVC)\d{3}", token):
            cover = token[:3]
            altitude = int(token[3:6]) * 100
            clouds_list.append(f"{CLOUD_COVER.get(cover, cover)} at {altitude:,} feet")

        # Temperature / Dewpoint: 28/24 or M02/M05
        elif re.match(r"^(M?\d{2})/(M?\d{2})$", token):
            t, d = token.split("/")
            temp = int(t.replace("M", "-"))
            dew = int(d.replace("M", "-"))
            temp_str = f"temperature of {temp}°C with a dew point of {dew}°C."

        # Altimeter: Q1012 or A2992
        elif token.startswith("Q") and token[1:].isdigit():
            pressure_str = f"altimeter setting of {int(token[1:])} hPa"
        elif token.startswith("A") and token[1:].isdigit():
            val = float(token[1:]) / 100.0
            pressure_str = f"altimeter setting of {val:.2f} inHg"

        # Weather group
        elif token in WEATHER_PHENOMENA:
            weather_list.append(WEATHER_PHENOMENA[token])
        elif token in WEATHER_DESCRIPTORS:
            weather_list.append(WEATHER_DESCRIPTORS[token])

    # Assemble lines exactly matching the requested layout
    lines = [f"METAR for {station_name}"]

    if time_str:
        lines.append(time_str)
    if wind_str:
        lines.append(wind_str)

    # Combined visibility and present weather line
    vis_wx_line = vis_str
    if weather_list:
        wx_text = f"present weather: {', '.join(weather_list)}."
        vis_wx_line = f"{vis_str} {wx_text}".strip() if vis_str else wx_text
    if vis_wx_line:
        lines.append(vis_wx_line)

    if clouds_list:
        lines.append(f"clouds: {', '.join(clouds_list)}.")
    if temp_str:
        lines.append(temp_str)
    if pressure_str:
        lines.append(pressure_str)

    return "\n".join(lines)


def main():
    icao = input("Enter airport ICAO code (e.g., VABB, VIDP, KJFK): ").strip().upper()
    if not icao:
        print("No ICAO code entered.")
        return

    print(f"\nFetching METAR for {icao}...")
    metar_raw = fetch_metar_no_key(icao)

    if not metar_raw or "No data" in metar_raw:
        print(f"Could not retrieve METAR data for '{icao}'. Please verify the code.")
        return

    print(f"\n--- RAW METAR ---\n{metar_raw}")
    print("\n--- DECODED OUTPUT ---")
    print(decode_metar_to_text(metar_raw))


if __name__ == "__main__":
    main()