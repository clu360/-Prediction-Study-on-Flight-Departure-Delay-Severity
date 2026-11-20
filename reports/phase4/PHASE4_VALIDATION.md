# Phase IV validation

**Status** Partial. The prescribed Phase IV frozen comparisons require real-data verification.

- The Atlanta one-day IEM CSV confirmed report type inputs, UTC observations, METAR text and required weather fields. Its SHA-256 is `01979796091bb577090732d59b92ae6bbf594c9277c7d41b9416adf5b9ec1b8d`. The parser retained 23 usable originals from 24 raw rows after excluding an undated correction.
- Single-day report checks supported PBI → DJT, HDN → HDN, ADK → PADK, BQN → TJBQ and SPN → PGSN. These confirm identity and request format for the sampled day. Full-year coverage is tracked separately.
- PyTorch 2.9.1 and the pinned XGBoost dependency are available on the local Windows CPU. No PACE job was submitted. A read-only SSH probe timed out.
- Phase IV boundary, weighting, probability, deterministic replay and checkpoint identity tests passed on synthetic inputs. Synthetic outputs provide implementation evidence only.

See the [progress journal](phase_4_progress.md) for the latest recorded checks and [execution guide](../../docs/phase4_execution.md) for reproduction commands.

