# Processed data dictionary

All stored columns use snake_case. UTC timestamps are timezone-aware microsecond timestamps; `source_month` uses target local FlightDate. Hive year/month directories are additional partition metadata. The exact machine-readable schema is `reports/schema.json`.

G1/G2 identify allowable information for future explicit feature views; this release does not construct model inputs. IDs, labels, metadata, audit outcomes and raw timestamps are excluded from those allowlists. History events are inputs to later causal G3 calculations, not current-flight predictors.

Labels and audited facts become known after the relevant outcome; they must never enter prediction feature matrices. Missing/invalid target labels are excluded. Missing predictor ingredients are retained and flagged until a later fold-local imputation policy. Valid cohort linkage/timing fields are required nonnull. Invalid connections keep nullable prior facts and a reason. No invented defaults are used.

## flight_connections

| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |
|---|---|---|---|---|---|---|
| `ambiguous_tail_rotation_flag` | Boolean | master same-tail schedule and service keys | Target schedule tie or conflicting service duplicate | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `audit_prior_arrival_after_target_departure` | Boolean | prior observed arrival, target actual departure | Prior arrival > target actual departure, retrospective only | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `audit_prior_arrived_before_prediction` | Boolean | prior observed arrival, prediction time | Prior observed arrival <= prediction, retrospective only | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `audit_prior_observed_arrival_utc` | Datetime(time_unit='us', time_zone='UTC') | prior gate departure, prior ActualElapsedTime | Prior gate departure + actual elapsed minutes; audit only | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `audit_target_departed_before_prediction` | Boolean | target actual gate time, prediction time | Target actual departure <= prediction, retrospective only | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `conflicting_service_duplicate_flag` | Boolean | date, airline, flight number, origin, dest | Multiple purported same-service records without deduplication | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `link_reason` | String | ordered linkage checks | Stable primary failure reason, or valid | metadata | Offline metadata; not a predictor | Required for every retained row |
| `link_status` | String | linkage rules | valid or invalid; exactly one decision per eligible target | metadata | Offline metadata; not a predictor | Required for every retained row |
| `prediction_time_utc` | Datetime(time_unit='us', time_zone='UTC') | prior WheelsOff / derived takeoff | Prior actual takeoff UTC + 1 minute | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_actual_gate_departure_utc` | Datetime(time_unit='us', time_zone='UTC') | prior schedule, prior DepDelay | Prior scheduled departure UTC + prior signed DepDelay | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_record_id` | String | master schedule record_id | Immediately preceding same-tail scheduled record | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_route_continuity_ok` | Boolean | prior Dest, target Origin | Standardized airports equal; null compares false | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_scheduled_arrival_utc` | Datetime(time_unit='us', time_zone='UTC') | prior schedule, prior CRSElapsedTime | Prior scheduled departure UTC + prior scheduled duration | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_scheduled_departure_utc` | Datetime(time_unit='us', time_zone='UTC') | prior schedule | Prior local scheduled departure localized to UTC | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_to_target_schedule_gap_min` | Float64 | target and prior scheduled departures | Target minus prior scheduled departure UTC, minutes | metadata | Before prediction (schedule/identity) | Required in valid cohort; nullable in invalid links |
| `prior_wheels_off_utc` | Datetime(time_unit='us', time_zone='UTC') | prior WheelsOff; optional prior TaxiOut | Resolve takeoff date against prior actual gate departure | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `scheduled_turnaround_min` | Float64 | target departure, prior scheduled arrival | Target scheduled departure UTC minus prior scheduled arrival UTC, minutes | G2 | Before prediction (schedule/identity) | Required in valid cohort; nullable in invalid links |
| `source_month` | Int8 | FlightDate | Target local calendar month | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `tail_has_unknown_order` | Boolean | master tail schedule | Any same-tail record has unresolved scheduled departure | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `takeoff_time_source` | String | prior takeoff resolution | parsed, derived, invalid; null if no predecessor | metadata | At prediction (prior departure and schedule) | Retain null/invalid flags; no imputation |
| `target_record_id` | String | record_id | Target record identity | metadata | Offline metadata; not a predictor | Required for every retained row |
| `timestamp_quality_flag` | String | target timestamp checks | Target schedule/takeoff quality category; metadata only | metadata | Retrospective target/event audit; not a predictor | Retain null/invalid flags; no imputation |

## flights_clean

| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |
|---|---|---|---|---|---|---|
| `cancelled` | Float64 | Cancelled | Numeric 0/1; invalid flags excluded from targets | metadata | Retrospective target/event audit; not a predictor | Retain null/invalid flags; no imputation |
| `crs_arr_time` | String | CRSArrTime | Source destination-local HHMM text; arrival cross-check | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `crs_dep_time` | String | CRSDepTime | Source local HHMM text; parsed minute clock validated in master | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `crs_elapsed_time` | Float64 | CRSElapsedTime | Finite positive scheduled duration in minutes | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `dest` | String | Dest | Strip and uppercase IATA code | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `distance` | Float64 | Distance | Finite miles, missingness retained and audited | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `diverted` | Float64 | Diverted | Numeric 0/1; diverted targets excluded; history retains departure observations | metadata | Retrospective target/event audit; not a predictor | Retain null/invalid flags; no imputation |
| `flight_date` | Date | FL_DATE / FlightDate | Parse originating scheduled local date | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `flight_number` | String | Flight_Number_Reporting_Airline / OP_CARRIER_FL_NUM | Identifier text; absence explicitly logged | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `origin` | String | Origin | Strip and uppercase IATA code | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `raw_row_number` | UInt32 | CSV position | Zero-based data row index, excludes header | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `raw_source_file` | String | source manifest | Original source filename, local only | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `raw_source_month` | Int8 | source manifest | Canonical input month | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `record_id` | String | source hash + month + row index | YYYY-MM-SHA256[:20]-CSV row index | metadata | Offline metadata; not a predictor | Required for every retained row |
| `reporting_airline` | String | Reporting_Airline / OP_UNIQUE_CARRIER | Reporting-carrier identifier | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `source_month` | Int8 | FlightDate | Target local calendar month | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `tail_number` | String | Tail_Number / TAIL_NUM | Strip/uppercase nonempty aircraft identifier | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `target_class` | Int8 | DepDelay | 0 if <15; 1 if [15,60); 2 if >=60 | label | After relevant event; audit/label/history only | Required for every retained row |
| `target_crs_arr_utc` | Datetime(time_unit='us', time_zone='UTC') | target_crs_dep_utc, CRSElapsedTime | Departure UTC + scheduled elapsed minutes | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `target_crs_dep_utc` | Datetime(time_unit='us', time_zone='UTC') | FlightDate, CRSDepTime, origin timezone | DST-aware local schedule to UTC | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `target_dep_delay_min` | Float64 | DepDelay | Finite signed departure delay; evaluation only | label | After relevant event; audit/label/history only | Required for every retained row |
| `timestamp_quality_flag` | String | target timestamp checks | Target schedule/takeoff quality category; metadata only | metadata | Retrospective target/event audit; not a predictor | Retain null/invalid flags; no imputation |

## folds

| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |
|---|---|---|---|---|---|---|
| `fold_id` | String | rolling-fold protocol | YYYY-MM of test month, Apr through Dec | metadata | Offline metadata; not a predictor | Required for every retained row |
| `is_final_holdout` | Boolean | test_month | True for November and December folds | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `role` | String | target local FlightDate | train for prior 3 months; test for following month | metadata | Offline metadata; not a predictor | Required for every retained row |
| `source_month` | Int8 | FlightDate | Target local calendar month | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `target_record_id` | String | record_id | Target record identity | metadata | Offline metadata; not a predictor | Required for every retained row |
| `test_month` | Int8 | rolling-fold protocol | Test calendar month 4..12 | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `train_months` | String | rolling-fold protocol | Comma-separated three immediately previous calendar months | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |

## history_departure_events

| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |
|---|---|---|---|---|---|---|
| `cancelled` | Float64 | Cancelled | Numeric 0/1; invalid flags excluded from targets | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `dep_delay` | Float64 | DepDelay | Finite signed historical departure delay in minutes | metadata (history outcome) | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `departure_severity` | Int8 | DepDelay | Same severity thresholds; historical observed event | metadata (history outcome) | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `diverted` | Float64 | Diverted | Numeric 0/1; diverted targets excluded; history retains departure observations | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `flight_date` | Date | FL_DATE / FlightDate | Parse originating scheduled local date | metadata | After relevant event; audit/label/history only | Required for every retained row |
| `gate_departure_clock_ok` | Boolean | actual gate UTC, DepTime | Derived local actual gate clock agrees within tolerance | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `observed_departure_time_utc` | Datetime(time_unit='us', time_zone='UTC') | schedule, DepDelay | Valid actual gate departure = scheduled departure UTC + signed delay | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `origin` | String | Origin | Strip and uppercase IATA code | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `record_id` | String | source hash + month + row index | YYYY-MM-SHA256[:20]-CSV row index | metadata | After relevant event; audit/label/history only | Required for every retained row |
| `reporting_airline` | String | Reporting_Airline / OP_UNIQUE_CARRIER | Reporting-carrier identifier | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `source_month` | Int8 | FlightDate | Target local calendar month | metadata | After relevant event; audit/label/history only | Required for every retained row |
| `timestamp_quality_flag` | String | target timestamp checks | Target schedule/takeoff quality category; metadata only | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |
| `valid_observation` | Boolean | history eligibility checks | True: operated departure with valid absolute time and finite delay | metadata | After relevant event; audit/label/history only | Retain null/invalid flags; no imputation |

## modeling_cohort

| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |
|---|---|---|---|---|---|---|
| `flight_date` | Date | FL_DATE / FlightDate | Parse originating scheduled local date | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `minutes_until_target_crs_dep_at_prediction` | Float64 | target schedule, prediction instant | Target scheduled departure UTC minus prediction time UTC, minutes | G2 | At prediction (prior departure and schedule) | Retain null/invalid flags; no imputation |
| `prediction_time_utc` | Datetime(time_unit='us', time_zone='UTC') | prior WheelsOff / derived takeoff | Prior actual takeoff UTC + 1 minute | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_crs_elapsed_min` | Float64 | prior CRSElapsedTime | Prior finite positive scheduled duration in minutes | G2 | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_dep_delay_min` | Float64 | prior DepDelay | Finite signed prior departure delay, minutes | G2 | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_dest` | String | prior Dest | Strip and uppercase prior destination IATA code | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_origin` | String | prior Origin | Strip and uppercase prior origin IATA code | G2 | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_record_id` | String | master schedule record_id | Immediately preceding same-tail scheduled record | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `prior_wheels_off_utc` | Datetime(time_unit='us', time_zone='UTC') | prior WheelsOff; optional prior TaxiOut | Resolve takeoff date against prior actual gate departure | metadata | At prediction (prior departure and schedule) | Required in valid cohort; nullable in invalid links |
| `scheduled_turnaround_min` | Float64 | target departure, prior scheduled arrival | Target scheduled departure UTC minus prior scheduled arrival UTC, minutes | G2 | Before prediction (schedule/identity) | Required in valid cohort; nullable in invalid links |
| `source_month` | Int8 | FlightDate | Target local calendar month | metadata | Before prediction (schedule/identity) | Required for every retained row |
| `takeoff_time_source` | String | prior takeoff resolution | parsed, derived, invalid; null if no predecessor | metadata | At prediction (prior departure and schedule) | Retain null/invalid flags; no imputation |
| `target_carrier` | String | Reporting_Airline / OP_UNIQUE_CARRIER | Strip and uppercase reporting-carrier code | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_class` | Int8 | DepDelay | 0 if <15; 1 if [15,60); 2 if >=60 | label | After relevant event; audit/label/history only | Required for every retained row |
| `target_crs_dep_local_hour` | Int8 | CRSDepTime | floor((parsed clock minutes mod 1440)/60) | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_crs_dep_utc` | Datetime(time_unit='us', time_zone='UTC') | FlightDate, CRSDepTime, origin timezone | DST-aware local schedule to UTC | metadata | Before prediction (schedule/identity) | Retain null/invalid flags; no imputation |
| `target_crs_elapsed_min` | Float64 | CRSElapsedTime | Finite positive scheduled duration in minutes | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_day_of_week` | Int8 | FlightDate | ISO weekday, Monday=1 .. Sunday=7 | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_dep_delay_min` | Float64 | DepDelay | Finite signed departure delay; evaluation only | label | After relevant event; audit/label/history only | Required for every retained row |
| `target_dest` | String | Dest | Strip and uppercase IATA code | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_distance_miles` | Float64 | Distance | Finite distance, miles; nulls retained/flagged | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_flight_number` | String | Flight_Number_Reporting_Airline / OP_CARRIER_FL_NUM | Identifier text, not a unique row ID | metadata | Offline metadata; not a predictor | Retain null/invalid flags; no imputation |
| `target_month` | Int8 | FlightDate | Local calendar month 1..12 | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_origin` | String | Origin | Strip and uppercase IATA code | G1 | Before prediction (schedule) | Retain null/invalid flags; no imputation |
| `target_record_id` | String | record_id | Target record identity | metadata | Offline metadata; not a predictor | Required for every retained row |
| `timestamp_quality_flag` | String | target timestamp checks | Target schedule/takeoff quality category; metadata only | metadata | Retrospective target/event audit; not a predictor | Retain null/invalid flags; no imputation |

## Accepted raw-to-canonical mapping

Source headers are compared after uppercasing and removing underscores/punctuation. Only the explicit aliases below are accepted; canonical intermediate names then map to processed columns above.

| Normalized source header | Canonical name |
|---|---|
| `ACTUALELAPSEDTIME` | `actual_elapsed_time` |
| `ARRDELAY` | `arr_delay` |
| `ARRTIME` | `arr_time` |
| `CANCELLED` | `cancelled` |
| `CRSARRTIME` | `crs_arr_time` |
| `CRSDEPTIME` | `crs_dep_time` |
| `CRSELAPSEDTIME` | `crs_elapsed_time` |
| `DAYOFWEEK` | `day_of_week` |
| `DEPDELAY` | `dep_delay` |
| `DEPTIME` | `dep_time` |
| `DEST` | `dest` |
| `DESTAIRPORTID` | `dest_airport_id` |
| `DISTANCE` | `distance` |
| `DIVERTED` | `diverted` |
| `DOTIDREPORTINGAIRLINE` | `dot_airline_id` |
| `FLDATE` | `flight_date` |
| `FLIGHTDATE` | `flight_date` |
| `FLIGHTNUMBERREPORTINGAIRLINE` | `flight_number` |
| `MONTH` | `month` |
| `OPCARRIERAIRLINEID` | `dot_airline_id` |
| `OPCARRIERFLNUM` | `flight_number` |
| `OPUNIQUECARRIER` | `reporting_airline` |
| `ORIGIN` | `origin` |
| `ORIGINAIRPORTID` | `origin_airport_id` |
| `REPORTINGAIRLINE` | `reporting_airline` |
| `TAILNUM` | `tail_number` |
| `TAILNUMBER` | `tail_number` |
| `TAXIOUT` | `taxi_out` |
| `WHEELSOFF` | `wheels_off` |
| `WHEELSON` | `wheels_on` |
| `YEAR` | `year` |
