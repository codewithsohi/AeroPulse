import os
import requests
import pandas as pd
from datetime import datetime

def download_historical_metars(
    station_icao: str,
    start_year: int,
    end_year: int,
    output_dir: str = "data/raw"
) -> str:
    """
    Downloads historical hourly METAR strings for a given station from the NOAA/IEM archive.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{station_icao}_{start_year}_{end_year}.csv")
    
    if os.path.exists(out_file):
        print(f"[+] Found cached data: {out_file}")
        return out_file

    print(f"[+] Fetching NOAA archive for {station_icao} ({start_year} to {end_year})...")
    # IEM ASOS URL format returning raw METAR string and UTC timestamp
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station_icao}&data=metar&year1={start_year}&month1=1&day1=1"
        f"&year2={end_year}&month2=12&day2=31&tz=Etc%2FUTC&format=onlycomma"
        f"&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # The download contains header comments starting with '#'
    lines = [line for line in response.text.splitlines() if not line.startswith("#")]
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SUCCESS] Successfully saved {len(lines)} records to {out_file}")
    return out_file

if __name__ == "__main__":
    # Example: Download 3 years of data for KJFK (JFK New York) or VIDP (Delhi)
    download_historical_metars("KJFK", start_year=2021, end_year=2023)