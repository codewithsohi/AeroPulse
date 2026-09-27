# AeroPulse
Modular METAR parser and deep-learning pipeline for time-series aviation weather nowcasting, predicting the next hour's METAR parameters directly from historical observation sequences.

> A modular aviation weather toolkit that parses raw, unstructured METAR reports into clean tabular datasets and uses time-series machine learning models to forecast next-hour (t+1) weather observations and flight categories (VFR / MVFR / IFR / LIFR).

> ⚠️ **Project Status: Active Development**  
> The deterministic METAR decoder is fully functional. The tabular time-series feature engineering and ML forecasting models are actively being trained, evaluated, and tuned.

## Project Preview- 
<img width="992" height="930" alt="image" src="https://github.com/user-attachments/assets/7351daff-2ac4-46c5-899c-6e93f4b38538" />

## Why This Project?

Aviation safety and dispatch decisions rely heavily on **METAR** (Meteorological Aerodrome Report) observations:
1. **Raw reports are hard to parse:** Encoded tokens, international notation differences, variable remarks (`RMK`), and unscheduled updates (`SPECI`) make parsing tricky without dedicated cleaning.
2. **Current conditions change quickly:** Pilots and dispatchers frequently rely on the latest report, assuming persistence. Rapid convective development or shifting fog banks can quickly push an airfield below landing minimums.

**AeroPulse** bridges this gap: it turns messy string tokens into structured time-series data, generates lag features, and trains machine learning models to anticipate the **next observation state (t+1)** before it is published.

## Key Features

### 1. Deterministic METAR Decoder
- **Full Field Parsing:** Decodes station ICAO, observation timestamp, wind speed/direction/gusts, prevailing visibility, runway visual range (RVR), weather phenomena (e.g., `TSRA`, `BR`, `-SN`), cloud cover/bases, temperature, dew point, and altimeter setting (`QNH`).
- **Feature Derivation:** Automatically calculates flight rule categories (`VFR`, `MVFR`, `IFR`, `LIFR`), dew-point depression, and decomposes cyclical wind angles into continuous $u$ and $v$ vectors.
- **Clean Schema:** Emits structured Python dictionaries or Pandas DataFrames ready for analysis.

### 2. Time-Series Machine Learning Engine
- **Lag Feature Engineering:** Creates rolling temporal features (e.g., t-1, t-2, t-3 hour moving averages, rate of pressure change) to capture atmospheric trends without unnecessary deep learning complexity.
- **Dual Forecasting Tasks:**
  - **Regression:** Predicts continuous values for the next hour (temperature, dew point, altimeter, lowest cloud base).
  - **Classification:** Predicts whether the station will hold or transition between flight categories (e.g., VFR to IFR).
- **Interpretable Models:** Uses standard Scikit-Learn models and XGBoost to prioritize fast training, easy debugging, and high interpretability.

  ## Tech Stack

| Layer | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Core & Parsing** | `Python 3.10+` `re` `Dataclasses` | Deterministic token extraction and schema definitions |
| **Data & Feature Engineering** | `Pandas` `NumPy` | Data cleaning, time-series alignment, and rolling lag features |
| **Machine Learning** | `Scikit-Learn` `XGBoost` | Baseline models, regression, and classification |
| **Evaluation & Plotting** | `Matplotlib` `Seaborn` | Actual vs. predicted error curves and confusion matrices |
| **Testing** | `Pytest` | Edge-case unit tests for malformed METAR strings |

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone [https://github.com/codewithsohi/AeroPulse.git](https://github.com/codewithsohi/AeroPulse.git)
cd AeroPulse

# Create and activate a virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```
### 2. Quick Usage (Run Decoder)

Run the interactive terminal decoder to fetch and parse live METAR data by ICAO code:

```bash
python metar_py2.py

Enter airport ICAO code (e.g., VABB, VIDP, KJFK): VABB

Fetching METAR for VABB...

--- RAW METAR ---
METAR VABB 270530Z 29008KT 4000 HZ FEW020 SCT025 31/23 Q1011 NOSIG

--- DECODED OUTPUT ---
METAR for Chhatrapati Shivaji Maharaj International Airport, Mumbai
reported on day 27 at 05:30 Zulu.
wind blowing from 290 degrees at 8 knots.
visibility of 4000 meters. present weather: haze.
clouds: few clouds at 2,000 feet, scattered clouds at 2,500 feet.
temperature of 31°C with a dew point of 23°C.
altimeter setting of 1011 hPa
```

## Roadmap & Current Status

This repository is under active development. Below is the phased engineering roadmap:

| Phase | Key Deliverables & Features | Status |
| :--- | :--- | :---: |
| **Phase 1: Deterministic METAR Decoder** | • Live retrieval of raw METAR strings by ICAO identifier<br>• Tokenizer for wind, visibility, weather phenomena, clouds, temp/dew point, QNH<br>• Natural language output generator for human-readable summaries | `Completed` |
| **Phase 2: Feature Engineering & Preprocessing** | • Parse reports into tabular time-series format (`Pandas`)<br>• Transform wind speed/direction into continuous `u` and `v` vectors<br>• Generate rolling lag features (`t-1`, `t-2`, hourly deltas) | `In Progress` |
| **Phase 3: Machine Learning Weather Nowcasting** | • Baseline persistence model benchmark<br>• Train XGBoost / Random Forest regressors for next-step temp & dew point<br>• Build flight category transition classifier (`VFR` / `MVFR` / `IFR` / `LIFR`)<br>• Model evaluation using MAE, RMSE, and confusion matrices | `Planned` |
| **Phase 4: Dashboard & Deployment** | • Streamlit web app showing live airport conditions and forecasted trends | `Planned` |

---

## Contact & Author

Developed by **Sohi Kulkarni**
* **GitHub:** [@codewithsohi](https://github.com/codewithsohi)
* **LinkedIn:** [https://www.linkedin.com/in/sohi-kulkarni/](#)

