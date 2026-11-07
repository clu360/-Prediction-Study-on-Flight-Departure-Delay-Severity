# Project guardrails

Phase II feature engineering is authorized by docs/phase2_specification.md. Do not implement classifiers, MLP training, model tuning, or the final project report.
Read docs/phase1_specification.md and preserve its research protocol.
Never commit source or processed flight records, local paths, secrets, or checkpoints.
Keep all input bytes unchanged. Use explicit predictor allowlists in future phases.
Never use arrival outcomes or realized target outcomes to select predecessors.
Retain cancellations and diversions in the master tail chain.
Use UTC-aware timestamps, local target dates for folds, and conservative ambiguity exclusions.
Run pytest and the independent artifact audit before claiming preprocessing success.
Aggregate public reports are permitted; linkage examples remain in ignored local data.
Preserve Phase I source data, linkages, folds, and the complete cohort unchanged.
Use exactly 9 G1, 5 G2, and 6 G3 features and the four explicit views.
Historical outcomes and scheduled-volume events must satisfy strict event_time < prediction_time, with inclusive lookback left boundaries.
Exclude a target's own outcome from its history even for retained retrospective anomalies; retain audit flags in metadata only.
Fit imputation, vocabularies and scaling on each fold's training rows only; freeze them during its test month.
