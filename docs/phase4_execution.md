# Phase IV execution

Run these commands from the repository root in the project environment. On Windows, use the existing repository bytecode cache by adding `-X pycache_prefix=.venv/pycache` between `python` and the script name.

```sh
python -m pip install -e '.[dev,modeling,neural,notebook]'
python scripts/phase4.py audit development
python scripts/phase4.py tune --device cpu
```

The weather work can run independently. The `stations` command reads both airport columns in the locked flight table and matches them against public IEM network GeoJSON files saved under `data/external/iem/metar/stations`. The URL format is `https://mesonet.agron.iastate.edu/geojson/network/<network>.geojson`. Use U.S. state networks and the territory networks that cover the cohort, including `MP__ASOS` for Saipan. The explicit PBI override is documented in `configs/g4_station_overrides.json`. Verified geographic matches use exact FAA/ICAO-derived identifiers and a 10 km sanity bound. Full-year report identities are checked before building G4.

```sh
python scripts/phase4.py stations
python scripts/phase4.py fetch
python scripts/phase4.py audit all
python scripts/phase4.py build-weather
python scripts/phase4.py fit XGB-base
python scripts/phase4.py fit XGB-weather
python scripts/phase4.py fit MLP-A --seed 6740
python scripts/phase4.py fit MLP-B --seed 6740
python scripts/phase4.py fit MLP-C --seed 6740
python scripts/phase4.py fit MLP-D --seed 6740
python scripts/phase4.py publish
python scripts/phase4_monitor.py
```

Repeat each MLP arm for seeds 6741 and 6742. The G1–G3 MLP arms require the selection lock and all-data audit. The G4 arms require the verified weather build as well. All primary holdout training IDs come from the November fold's training role. Only the test role is loaded for December.

Restart a command to reuse verified checkpoints. Development and final runs that stop before saving a complete fit restart that fit from its original seed. Before removing an interrupted writer lock, verify that the recorded process has stopped. The private attempt directories retain epoch curves and prior attempts. The notebook reads only the public aggregates and can be rerun with `scripts/phase4_build_notebook.py --execute <private-output.ipynb>`.

The Phase IV protocol supersedes the archived draft's four-feature and development-only scope. It uses exactly five G4 features at the target's origin and destination and a frozen 2025 final evaluation. The archive's 10-minute modeled latency is an approximation. Undated corrections are excluded, reports at the prediction boundary are ineligible, and source identity issues require explicit investigation. Weather and weighting results are exploratory given previous holdout exposure.

