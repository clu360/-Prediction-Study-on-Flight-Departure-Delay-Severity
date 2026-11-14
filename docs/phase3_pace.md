# Phase III on PACE

The PACE adapter distributes independent development fold/view bundles across Slurm jobs. It uses the existing data loaders, model factories, evaluation functions, ranking rules and final runner. The original Phase I/II source and artifacts remain unchanged. All 367 development fits and 13 final fits remain required; no fitting population or candidate grid is reduced. No MLP is included.

## Execution graph

1. A compute job creates the pinned Python environment and runs the full automated suite.
2. Another compute job reruns the independent Phase I/II audits, checks all 19 Phase II acceptance checks and prepares seven development baselines plus the Stage B manifest.
3. A 12-task array screens the complete grids: four views times April/July/October. Each task shares one sparse matrix bundle across its 22 learned candidates.
4. A coordinator verifies and merges all screening artifacts, then uses the existing equal-month aggregation and ranking functions to freeze at most two finalists per model/view.
5. A 16-task array completes the other four development months for each view. Each task evaluates at most six candidates and never loads a holdout target month.
6. A coordinator merges the finalists and invokes the original resumable tuner. All required fits are already present; this step generates the original selection evidence and selected backtests without additional fitting.
7. One final compute job invokes the original frozen final runner: 13 fresh August-October fits, each unchanged for both November and December. It independently verifies saved results, exports aggregates and executes the analysis notebook.

Every dependency uses Slurm `afterok`, including the complete arrays. Final scoring cannot begin before successful selection. Failures leave explicit incomplete jobs and preserved checkpoints.

## Storage and isolation

`scripts/package_phase3_pace.py inputs` archives the private working inputs and checks every available original byte checksum. Reads finish before a tar member is written; transient Windows sharing violations are retried. The source archive is created separately with `scripts/package_phase3_pace.py source`. It preserves actual source bytes, including line endings required by existing provenance. Neither archive contains the local virtual environment, SSH keys or local fitted-model output directories.

Use a new private scratch directory. Clone a Git bundle containing the implementation, then overlay the source archive to preserve exact working-file bytes and extract the input archive. Verify the archive SHA-256 values before extraction. Load PACE's Python 3.12 module for the launcher; all installation, audits and model computations run in allocated compute jobs.

```sh
module load python/3.12.5
python3 scripts/submit_phase3_slurm.py --account YOUR_PACE_ACCOUNT --qos inferno --concurrency 16
```

The account is an explicit command argument and stays in private workflow metadata. The current scientific configuration is `configs/phase3_pace.yaml`; its version records the batch-worker code hash. Each worker retains one outer fit and at most four estimator/BLAS threads, with a 16 GiB Slurm memory reservation and the original 8 GiB process guard. Global array concurrency is separate from that per-process outer-fit setting. The launcher caps concurrency at 16 tasks (64 allocated CPUs); Stage B itself has only 12 tasks.

Worker attempts have isolated result directories identified by Slurm job and restart IDs. A retry imports only completed records with matching run signatures, task-manifest hashes and artifact checksums. It never overwrites another attempt's in-progress files. The coordinator imports verified records with atomic metadata writes and checked prediction hardlinks/copies. Selection and final holdout locks remain authoritative.

Private job scripts and job IDs are in `data/model_outputs/phase3_pace/slurm/workflow.json`. The launcher refuses to overwrite an existing workflow; inspect and recover recorded jobs instead of submitting duplicate chains. The public `batch_execution.json` contains aggregate scheduling/provenance information without the account name or private filesystem paths. `squeue` and `sacct` establish actual job state; an interrupted SSH observation does not imply a failed job.

From the local workstation, `python scripts/check_phase3_pace.py --logs` reads the saved private connection metadata, checks Slurm state and completed checkpoint counts, and keeps the monitoring snapshot under ignored `data/`. It reuses the existing SSH identity and strict known-host verification; it does not submit or cancel jobs. Login load balancing can route separate observations to different login nodes, so the shared workflow/accounting records are the authoritative job evidence.

The free `embers` queue is preemptible; the user authorized the paid `inferno` account for this run. Account policy is described by [PACE](https://pace.gatech.edu/compute-and-storage-accounts-phoenix/), and array dependency semantics by [Slurm](https://slurm.schedmd.com/job_array.html). Report observed queue/runtime and any retries. Local and cluster experiment contexts remain separate; only the chosen completed cluster run supplies official study performance.

## Validation

Batch integration tests exercise all development shards, exact populations, a real two-candidate shortlist, the original selector with refitting disabled, interrupted-attempt reuse, holdout-month rejection and primary-lock enforcement. A scheduler test verifies successful-selection dependencies and bounded arrays. These are synthetic methodology checks; they are not study performance results. The cluster repeats the full suite and real-data audit before any large training job becomes eligible.

The first shared-filesystem upstream audit exceeded its 900-second guard before any model training started. The recovery wrapper verifies both transferred archives, extracts exact copies onto compute-node local storage, checks unchanged core-code/configuration fingerprints, and calls the original complete audit. It publishes the actual audit outcome and timing back to shared storage; a failed audit still blocks training. The PACE-only audit guard is now 3,600 seconds, with a two-hour scheduler reservation for staging, audits and baseline preparation. Model grids, populations, folds and final-holdout rules are unchanged.

`scripts/run_phase3_pace_staged.py` also stages development inputs and source onto node-local storage, checks archive SHA-256 values and identical scientific fingerprints, and calls the unchanged batch implementation. A directory symlink resolves its output path to the original shared result store, so all task manifests, primary holdout locks, attempt directories and atomic checkpoints remain durable. The original complete development gate still checks all required input hashes on each local copy. Archives containing fitted outputs are rejected. Temporary input copies are removed on exit; persisted results remain on shared storage. Selection and frozen final scoring continue through the original canonical runners.

For the live run, Stage B had already started against shared inputs and was preserved. Only its still-pending merge/shortlist coordinator and Stage C array were replaced after the three Linux staging checks passed. Stage C uses at most 15 four-CPU workers while a separate four-CPU reproducibility check may run concurrently, keeping that combined allocation within 64 CPUs. That check compares 53 completed independent local development fits against the cluster fits without refitting or using them in selection. Frozen final scoring requires both selection and reproducibility success. Private workflow metadata retains replaced job IDs; public execution metadata records the helper hashes and operational patch commits separately from the unchanged scientific source fingerprints.

The cross-platform probe subsequently found XGBoost prediction differences beyond 1e-7. The live verification workflow therefore uses four independent same-platform full-population repeats as its strict reproducibility gate, and preserves the complete 53-fit Windows/PACE comparison as an explicit diagnostic with numeric failures reported. Official fitting populations, candidate grids, seven-fold selection and frozen holdouts are unchanged. The four extra verification fits are separate from the 380 study fits. Fresh launchers schedule `seeded_repeat` after screening merge and reserve one concurrency slot while it can overlap Stage C; concurrency=1 runs that repeat before Stage C. The final job requires both selection and successful strict-repeat verification.

Temporary month-restricted hardlink facades must be created on the same node-local filesystem as staged inputs. The operational worker adapter supplies a local scratch location to the unchanged fold loader; durable models, predictions, locks and checkpoints remain shared. An initial independent-repeat attempt exposed an EXDEV link error before its learned-model fits. Regression checks now simulate this filesystem boundary for both staged workers and independent repeats. The protected scientific modules and batch-worker implementation remain unchanged.
