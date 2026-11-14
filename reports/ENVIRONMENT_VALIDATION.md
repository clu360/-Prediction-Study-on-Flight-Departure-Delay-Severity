# Environment repair and Phase III preparation

Reviewed November 13, 2026.

## Active environment

The active environment is the repository's ignored `.venv`, using Python 3.12.11. The older parent-level environment is superseded. The local parent VS Code workspace selects this interpreter by default and sets the pytest working directory to the repository.

Base dependencies and the development extra retain their existing exact versions. The optional `modeling` extra adds `xgboost==3.1.2` for the proposed classical-model comparison. Install with `python -m pip install -e '.[dev,modeling]'`.

Bytecode caches use `.venv/pycache`. This avoids stalled reads of pre-existing OneDrive-backed source caches. Local workspace terminal settings and the ignored `.env` configure the prefix; the README includes an explicit PowerShell command for other terminals.

## Completed verification

- All 13 explicitly pinned base/development/modeling dependencies match `pyproject.toml`.
- `pip check`: no broken requirements.
- Scientific imports, LogisticRegression, RandomForestClassifier and XGBoost imports pass.
- Editable package resolves to the current repository source.
- `flight-delay --help` and `pytest --version` launch successfully with the cache prefix.
- Full synthetic test suite: **55 passed in 22.43 seconds**.
- Git whitespace checks pass; environment and `.env` are ignored.

## Historical data-access limitation (resolved)

A fresh Phase I artifact audit failed with Windows `OSError: [Errno 22] Invalid argument`. Direct diagnostics then stalled opening `data/processed/audit/run_signature.json`; file attributes indicated an online-only OneDrive placeholder. The same read stalled outside the sandbox. Requesting local availability for the audit JSON files did not resolve access during this session.

Those initial failures were resolved using local, checksum-equivalent working storage under ignored `data/phase3_inputs`. All twelve original monthly CSV checksums, all 4,736 original normalized master-file byte hashes, and all 4,073 logical hashes for each feature, target and metadata table matched. A subsequent fresh independent Phase I/II validation passed on all 6,461,609 cohort rows, including all 19 Phase II acceptance checks. See [Phase III validation](phase3/PHASE3_VALIDATION.md) for current experiment status.

The environment-only checks above preceded model training; the full Phase III experiment and independent verification have since completed. The final PACE Linux release suite passed 112 tests in 178.11 seconds; the local Windows suite passed 111 tests with one symlink-privilege skip. Both environments passed `pip check`. Original source records, feature values, cohort membership, folds and research protocol are preserved. Restored storage is private and verified against the original checksums; artifact audits may refresh ignored diagnostic reports.
