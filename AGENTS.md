# Project guardrails

Implement Phase 1 only until the team explicitly authorizes later phases.
Read docs/phase1_specification.md and preserve its research protocol.
Never commit source or processed flight records, local paths, secrets, or checkpoints.
Keep all input bytes unchanged. Use explicit predictor allowlists in future phases.
Never use arrival outcomes or realized target outcomes to select predecessors.
Retain cancellations and diversions in the master tail chain.
Use UTC-aware timestamps, local target dates for folds, and conservative ambiguity exclusions.
Run pytest and the independent artifact audit before claiming preprocessing success.
Aggregate public reports are permitted; linkage examples remain in ignored local data.
