# Phase 1 preprocessing methodology

## Source and identity

The source is the user-supplied BTS Reporting Carrier exports for January–December 2025. Header aliases are explicitly enumerated in `discovery.py`; `OP_UNIQUE_CARRIER`, `OP_CARRIER_FL_NUM`, `TAIL_NUM`, and `FL_DATE` map to reporting airline, reporting flight number, tail number, and flight date. Unknown source fields do not become features. References: [BTS field dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) and [download selector](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ).

Every CSV is read with explicit string types and controlled null tokens. Dates must resolve to one 2025 month matching Year/Month and any recognizable full month filename. SHA-256 covers original source bytes (ZIP archive bytes for ZIP inputs); the ZIP member is separately recorded. Multiple CSV members for a month are an error. Exact duplicates use every raw column. Service conflicts use date, carrier, flight number, origin and destination before tail partitioning; tied same-tail schedules also create ambiguity. No raw rows are silently deduplicated.

Record IDs are `YYYY-MM-<20 hex SHA256 characters>-<zero-padded CSV row number>`. The full SHA-256 is in the manifest. Flight number is not a unique row ID. Missing tail values are preserved as null, while identifiers and airports are stripped and uppercased. Raw row number is zero-based, excluding the CSV header. Raw source filenames/month/row indexes remain local metadata.

## Time normalization

The airport map comes from pinned `airportsdata==20250909` ([upstream](https://github.com/mborsetti/airportsdata)); observed mapping, provenance, version, mapping hash, and unresolved codes are emitted. Pinned Polars/runtime uses its embedded IANA timezone rules, with per-date DST; Python tzdata is pinned separately and recorded. No fixed-offset or naive-datetime joins are performed.

0000 is start of the local day; 2400 on a scheduled departure means next-day midnight. Other 24xx or invalid minute values are null/flagged. Local FlightDate remains the cohort/fold date even for 2400. Scheduled arrival is departure UTC plus positive scheduled elapsed minutes. Destination CRSArrTime cross-checks the derived instant with a one-minute circular clock tolerance. Arrival clock mismatches are flagged; a predecessor with an inconsistent arrival schedule is excluded as `invalid_prior_schedule`.

Nonexistent spring-forward schedules are null. Fall-back schedules evaluate both folds: independent scheduled duration and destination arrival clock can select exactly one. If neither or both fit, the departure remains unresolved. A tail containing any unresolved schedule is conservatively excluded from links for the year (`invalid_timezone`) because physical schedule adjacency cannot be established; the events and labels remain retained.

Actual gate departure is schedule UTC + signed DepDelay. With usable TaxiOut, takeoff equals gate departure + nonnegative TaxiOut, provided the local WheelsOff agrees within the configured one minute. A missing/unusable WheelsOff can use the TaxiOut-derived value, explicitly flagged `derived`; inconsistent nonmissing WheelsOff does not fall back silently.

The supplied exports omit TaxiOut. Without TaxiOut, WheelsOff is localized on gate departure's local day and the neighboring days. Keep only a uniquely nearest candidate with nonnegative taxi. This is a conservative date-resolution rule: it rejects a closer negative interval rather than accepting an almost-day-long taxi on another date. DST-fold ambiguity and nonexistent same-day clocks are invalid. Actual 2400 is clock-face midnight with date resolved against gate departure. Taxi intervals above 180 minutes are flagged, never removed solely by that numerical flag. Very long taxi periods cannot be securely resolved without independent TaxiOut and may be rejected by nearest-date ambiguity.

Gate-time consistency is independently checked against DepTime. If the local clock from schedule UTC + DepDelay disagrees beyond the small tolerance, the locked formula remains unchanged for audit, but takeoff is invalid and the observation is excluded from the history event store. This specifically avoids accepting a one-hour DST error on a delayed leg crossing a transition. No arbitrary hour correction is applied. This guard uses a preceding flight's known departure observation for linkage; it does not use realized target outcomes to select predecessors. All eligible target labels remain retained.

## Cohorts and precedence

Target exclusions have stable priority: invalid status flags, canceled, diverted, missing/invalid/nonfinite departure delay. All remaining targets retain the signed label and severity, even if schedule, distance or tail is missing. The master event store retains every input row.

Each nonmissing tail is sorted by absolute scheduled departure and stable record ID. Shift exactly one master row to establish the predecessor. A tied or conflicting service schedule marks the ambiguous record and its immediate successor. The pipeline never searches backward past an invalid intermediate record. Stable primary reasons are assigned in code order: missing tail; unknown schedule/timezone; ambiguous rotation; no previous record; intervening canceled/diverted/invalid flags; missing predecessor delay; missing takeoff; airport mismatch; nonstrict schedule ordering; late prior takeoff; invalid prior arrival schedule; negative scheduled turnaround; turnaround above the configured 24-hour maximum; valid.

Scheduled turnaround is target scheduled departure minus prior scheduled arrival. Prior-to-target schedule gap instead measures departure-to-departure. Zero and 24 hours are included; prediction exactly at target scheduled departure is excluded. The 12-hour sensitivity uses the same links with inclusive 720-minute maximum. January's first observed aircraft leg receives `no_previous_record`; no 2024 connection is invented.

The intended prediction window ends before prior arrival and target departure. After selecting links, actual target gate departure and prior observed arrival (gate departure + actual elapsed time) are checked as **audit-only** facts. Flags do not silently change feature construction. Any violations must be disclosed and reviewed before Phase 2. No prior arrival outcome is an allowed predictor. History stores departure events, including valid observed departures from diverted flights; no historical rates are computed in Phase 1.

## Validation and resource handling

Synthetic tests exercise boundary labels, clocks, DST, cross-zone/overnight legs, predecessor guards, timing limits, leakage invariance, splits, input failures and deterministic reruns. The separate audit rereads every Parquet file, checks exact schema, UTC types, logical content hash, ID uniqueness, cohort membership, labels, folds and complete flow accounting. It reconstructs immediate adjacency from normalized master events in each tail bucket to validate same-tail continuity independently of the linkage output.

Normalization is bounded to 50,000-row batches of one source month; joins use 32 hash buckets. Caches include code/config/input/dependency fingerprints and file hashes. Code/config/source changes require a fresh output directory; raw inputs are rehashed after processing. Canonical sorted NDJSON values are SHA-256 hashed per output partition; Parquet compression and null-buffer bytes do not affect these logical hashes.

The execution report measures elapsed preprocessing/write time and Windows process peak working set when available; the separate read-back audit adds runtime. The local examples are a reproducible review sample, not a claim of human certification. Public reports contain aggregates only. Retrospective tail assignments, unreported ferry moves, missing 2024 buffers and schedule ambiguity limit operational linkage validity. Later work must preserve the three-month rolling/final-holdout protocol and explicitly approve any methodological changes.
