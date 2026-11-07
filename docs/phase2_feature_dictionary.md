# Phase II feature dictionary

One row per unchanged Phase I target_record_id. The full_features table contains one metadata ID column plus exactly twenty predictors; labels and other metadata are separate. Nulls are retained in source features and handled by training-fold-only imputation. All G3 outcome windows are [t−28 days,t); the schedule window is [t−7 days,t). alpha=20. The target's own outcome is explicitly excluded.

| Feature | dtype | Source | Formula | Availability | Group | Missing count | Missing fraction |
|---|---|---|---|---|---|---:|---:|
| `g1_carrier` | String | target_carrier | Reporting carrier | Before prediction | G1 | 0 | 0.00000000 |
| `g1_origin` | String | target_origin | Departure IATA code | Before prediction | G1 | 0 | 0.00000000 |
| `g1_dest` | String | target_dest | Destination IATA code | Before prediction | G1 | 0 | 0.00000000 |
| `g1_dep_hour_sin` | Float64 | target_crs_dep_local_hour | sin(2*pi*hour/24) | Before prediction | G1 | 0 | 0.00000000 |
| `g1_dep_hour_cos` | Float64 | target_crs_dep_local_hour | cos(2*pi*hour/24) | Before prediction | G1 | 0 | 0.00000000 |
| `g1_day_of_week` | Int8 | target_day_of_week | ISO weekday: Monday=1, Sunday=7; categorical | Before prediction | G1 | 0 | 0.00000000 |
| `g1_month` | Int8 | target_month | Month 1..12; categorical | Before prediction | G1 | 0 | 0.00000000 |
| `g1_crs_elapsed_min` | Float64 | target_crs_elapsed_min | Scheduled minutes, unchanged | Before prediction | G1 | 0 | 0.00000000 |
| `g1_distance_miles` | Float64 | target_distance_miles | Scheduled miles, unchanged | Before prediction | G1 | 0 | 0.00000000 |
| `g2_prior_origin` | String | prior_origin | Incoming leg origin, unchanged | At prediction | G2 | 0 | 0.00000000 |
| `g2_prior_dep_delay_min` | Float64 | prior_dep_delay_min | Signed prior departure delay, unchanged | At prediction | G2 | 0 | 0.00000000 |
| `g2_prior_crs_elapsed_min` | Float64 | prior_crs_elapsed_min | Prior scheduled minutes, unchanged | Before prediction | G2 | 0 | 0.00000000 |
| `g2_scheduled_turnaround_min` | Float64 | scheduled_turnaround_min | Target scheduled departure minus prior scheduled arrival, unchanged | Before prediction | G2 | 0 | 0.00000000 |
| `g2_minutes_until_crs_dep_at_prediction` | Float64 | minutes_until_target_crs_dep_at_prediction | Target scheduled departure minus prediction instant, unchanged | At prediction | G2 | 0 | 0.00000000 |
| `g3_airport_moderate_rate_28d` | Float64 | Valid prior departure observations | (M_a+20*p_global_mod)/(N_a+20) | Strictly earlier observations | G3 | 0 | 0.00000000 |
| `g3_airport_severe_rate_28d` | Float64 | Valid prior departure observations | (S_a+20*p_global_sev)/(N_a+20) | Strictly earlier observations | G3 | 0 | 0.00000000 |
| `g3_airport_mean_dep_delay_28d` | Float64 | Valid prior signed DepDelay | Airport signed-delay sum/N_a | Strictly earlier observations | G3 | 248 | 0.00003838 |
| `g3_airport_scheduled_volume_7d` | Int64 | Full normalized master schedule | Count origin schedules in [t-7 days,t), including eventual cancellations/diversions | Earlier scheduled instants | G3 | 0 | 0.00000000 |
| `g3_airline_airport_severe_rate_28d` | Float64 | Valid prior departure observations | (S_ac+20*airport_severe)/(N_ac+20) | Strictly earlier observations | G3 | 0 | 0.00000000 |
| `g3_airline_airport_count_28d` | Int64 | Valid prior departure observations | N_ac in [t-28 days,t) | Strictly earlier observations | G3 | 0 | 0.00000000 |

## Missingness and support

No global history yields null rates. With global but no airport history, airport rates use global rates and the airport mean stays null. With no airline-at-origin history, its severe rate equals the airport severe rate and its count is zero. Counts/volumes are nonnegative integers. Early January has no December 2024 buffer. No rows are dropped.

## Non-predictor contracts

`targets`: target_record_id, target_class (Int8, 0/1/2), target_dep_delay_min (Float64 signed minutes). Only target_class is y. Both outcome columns are forbidden in X.

`metadata`: IDs, local flight_date/source_month, origin/carrier, UTC prediction/scheduled times, timestamp/takeoff quality, support/fallback diagnostics and Phase I retrospective flags. All timestamps remain timezone-aware UTC. None of these metadata columns is automatically selected as a predictor.

Exact feature, target and metadata dtypes are in `reports/phase2/feature_schema.json`. Numeric distributions and categorical cardinalities for all twenty features are in `reports/phase2/feature_missingness.csv`; grouped missing counts and history coverage are adjacent CSVs. Unknown test categories use train-fitted all-zero one-hot blocks; numeric nulls use train-only medians (zero only for entirely null training columns).
