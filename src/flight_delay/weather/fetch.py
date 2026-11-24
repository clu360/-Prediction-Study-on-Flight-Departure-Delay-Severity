"""Rate-limited, immutable fetch and verified reload of public station histories."""

from datetime import date, datetime, time, timezone
from http.client import IncompleteRead
import ssl
import time as clock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .metar import decode
from ..experiments.config import digest, file_hash
from ..experiments.results import atomic, exclusive, read_json, write_json

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def bounds(config):
    return tuple(datetime.combine(date.fromisoformat(config[key]), time(), timezone.utc)
                 for key in ("start", "end_exclusive"))


def request_parameters(station, start, end):
    params = [("station", station), ("data", "all"), ("tz", "Etc/UTC"),
              ("format", "onlycomma"), ("missing", "empty"), ("trace", "empty"),
              ("report_type", "3"), ("report_type", "4")]
    for index, day in enumerate((start, end), 1):
        params.extend((f"{key}{index}", str(getattr(day, key))) for key in ("year", "month", "day"))
    return params


class Archive:
    def __init__(self, root, start, end, version="v1"):
        self.start, self.end = start, end
        self.root = root / "raw" / version
        self.last_request = 0.0

    def files(self, station):
        signature = digest(request_parameters(station, self.start, self.end))[:12]
        base = self.root / f"{station}-{signature}"
        return base.with_suffix(".csv"), base.with_suffix(".json")

    def load(self, station, *, routine_and_special_only=False):
        csv_path, sidecar = self.files(station)
        record = read_json(sidecar)
        if (
            record["station"] != station
            or record["parameters"] != [list(pair) for pair in request_parameters(station, self.start, self.end)]
            or file_hash(csv_path) != record["sha256"]
            or csv_path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"Invalid cached weather bytes for {station}")
        reports, audit = decode(csv_path.read_bytes(), station, self.start, self.end)
        if audit != record["audit"]:
            raise ValueError(f"Changed weather parser evidence for {station}")
        if routine_and_special_only:
            reports, selected = decode(csv_path.read_bytes(), station, self.start, self.end,
                                       routine_and_special_only=True)
            record["causal_stream_audit"] = selected
        return reports, record

    def fetch(self, stations):
        with exclusive(self.root):
            for station in stations:
                raw, sidecar = self.files(station)
                if sidecar.exists():
                    self.load(station)
                    continue
                if raw.exists():
                    raise ValueError(f"Interrupted weather cache for {station}; preserve it and choose a new version")
                params = request_parameters(station, self.start, self.end)
                for attempt in range(5):
                    clock.sleep(max(0, 2 - (clock.monotonic() - self.last_request)))
                    self.last_request = clock.monotonic()
                    try:
                        url = f"{ENDPOINT}?{urlencode(params)}"
                        request = Request(url, headers={"User-Agent": "ISYE6740-flight-delay-research/1.0"})
                        with urlopen(request, timeout=45) as response:
                            contents, status = response.read(), response.status
                            headers = dict(response.headers.items())
                        break
                    except (URLError, TimeoutError, IncompleteRead, ssl.SSLError, ConnectionError) as exc:
                        if isinstance(exc, HTTPError) and exc.code not in (429, 503):
                            raise
                        if attempt == 4:
                            raise
                        print(f"{station}: retry {attempt + 1}/4 after {type(exc).__name__}", flush=True)
                        clock.sleep(min(30, 2 ** (attempt + 1)))
                _, audit = decode(contents, station, self.start, self.end)
                atomic(raw, lambda p: p.write_bytes(contents))
                record = dict(station=station, parameters=params, sha256=file_hash(raw),
                              retrieved_utc=datetime.now(timezone.utc).isoformat(), bytes=len(contents),
                              status=status, server_headers=headers, audit=audit,
                              validation="parsed" if audit["rows"] else "no_observations")
                write_json(sidecar, record)
                print(f"{station}: {audit['rows']} observations", flush=True)
