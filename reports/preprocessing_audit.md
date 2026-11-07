# Phase 1 preprocessing audit

Acceptance passed: **True**

## Record flow

```json
{
  "raw": 7001619,
  "flights_clean": 6879485,
  "flight_connections": 6879485,
  "modeling_cohort": 6461609,
  "history_departure_events": 6898735,
  "folds": 19697112,
  "excluded_targets": 122134,
  "valid_link_percentage": 93.92576624558379
}
```

## Class distributions

```json
{
  "flights_clean": [
    {
      "target_class": 0,
      "count": 5387042,
      "percentage": 78.30589062989453
    },
    {
      "target_class": 1,
      "count": 940795,
      "percentage": 13.675369595253134
    },
    {
      "target_class": 2,
      "count": 551648,
      "percentage": 8.018739774852333
    }
  ],
  "modeling_cohort": [
    {
      "target_class": 0,
      "count": 5199881,
      "percentage": 80.47347030747295
    },
    {
      "target_class": 1,
      "count": 883015,
      "percentage": 13.665559150979268
    },
    {
      "target_class": 2,
      "count": 378713,
      "percentage": 5.860970541547778
    }
  ]
}
```

## Link reasons

shape: (10, 3)
┌─────────────┬─────────────────────────────────┬─────────┐
│ link_status ┆ link_reason                     ┆ count   │
│ ---         ┆ ---                             ┆ ---     │
│ str         ┆ str                             ┆ u32     │
╞═════════════╪═════════════════════════════════╪═════════╡
│ invalid     ┆ airport_mismatch                ┆ 113768  │
│ invalid     ┆ ambiguous_rotation              ┆ 3038    │
│ invalid     ┆ gap_over_24h                    ┆ 94058   │
│ invalid     ┆ intervening_cancelled_or_diver… ┆ 76619   │
│ invalid     ┆ invalid_prior_schedule          ┆ 122     │
│ invalid     ┆ late_prior_takeoff              ┆ 62973   │
│ invalid     ┆ negative_scheduled_turnaround   ┆ 61162   │
│ invalid     ┆ no_previous_record              ┆ 6086    │
│ invalid     ┆ predecessor_missing_takeoff     ┆ 50      │
│ valid       ┆ valid                           ┆ 6461609 │
└─────────────┴─────────────────────────────────┴─────────┘

## Retrospective flags (do not select features)

{
  "audit_target_departed_before_prediction": 1,
  "audit_prior_arrived_before_prediction": 0,
  "audit_prior_arrival_after_target_departure": 14
}

12-hour sensitivity cohort: 6,157,416

## Acceptance checks

- twelve_months: True
- input_bytes_preserved: True
- raw_clean_accounting: True
- valid_labels: True
- one_connection_per_target_count: True
- cohort_matches_valid_link_count: True
- valid_cohort_timing_and_continuity: True
- fold_calendar_membership: True
- unique_keys: True
- connection_target_ids_match: True
- cohort_target_ids_match: True
- immediate_predecessor_same_tail: True
- history_observations_valid: True
- all_parquet_readable_schema_stable: True
- utc_schema: True
- logical_hashes_match: True
- no_stale_or_missing_partitions: True
- five_evaluation_airports_present: True
- review_examples_count: True
- review_examples_month_coverage: True
- target_exclusion_accounting: True

## Interpretation and limitations

Target departure and prior arrival audit failures are flagged, not used to silently change the schedule-selected cohort.
A Phase 2 operational deployment must decide how to handle retrospective violations of the intended forecast window.
WheelsOff without TaxiOut uses the uniquely nearest local calendar date, requires nonnegative taxi, and rejects unresolved DST folds.
Unobserved ferry movements, retrospective tail assignments, and December 2024 predecessors remain unobserved.
The 100 local examples are reproducibly sampled for human review; automatic checks do not constitute human certification.
Full schema: schema.json. Input hashes/coverage: input_manifest.json. Timezones: airport_timezones.json.
Normalization diagnostics: normalization_quality.json. Monthly airport/class counts and coverage tables are adjacent CSVs.