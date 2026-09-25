# Navigation and Semantic Regression Test Plan

This plan validates the navigation replan diagnostics and semantic detection
provenance added in the current change. It separates deterministic offline
checks from Isaac Sim behaviour that cannot be established by unit tests.

## Scope

The test campaign must answer five questions:

1. Does a changed goal invalidate a stale path while an unchanged goal keeps
   the current plan?
2. Does blocked-path checking continue past a zero-length waypoint segment and
   report why the route was considered blocked?
3. Do Open-Vocabulary and Isaac detections keep their source information after
   same-frame deduplication and later scene-graph merges?
4. Can `target_match_method` and `target_sources` be interpreted independently
   in a completed exploration run?
5. Does `open_vocab` reject an unavailable stack before Isaac starts and
   preserve valid objects when another candidate fails?

Pure-pursuit gains, velocity limits, waypoint spacing, and planner/runtime
collision geometry are not changed in this patch. They are observed during the
live runs below, but any tuning should be proposed and reviewed separately.

## Stage 1: Offline regression suite

Run the focused suite before every simulator run:

```bash
TEST_PYTHON="${ISAAC_PYTHON:-python}"
for test_file in \
  test_point_navigation_component.py \
  test_interaction_navigation_mapping.py \
  test_semantic_exploration_component.py \
  test_open_vocabulary_perception.py; do
  "$TEST_PYTHON" -m unittest discover -s tests -p "$test_file"
done
```

Then run the complete suite in the repository's configured development
environment:

```bash
"$TEST_PYTHON" -m unittest discover -s tests
```

Acceptance criteria:

- The focused suite passes.
- Any complete-suite failure is recorded with its missing dependency or asset;
  no failure in a touched module is accepted.
- `git diff --check` reports no whitespace errors.

## Stage 2: Geometry-only Isaac smoke test

Use a new output directory so results are not mixed with an earlier run:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --detection-mode isaac \
  --no-qwen \
  --target refrigerator \
  --record-dir grutopia/results/replan-regression-geometry \
  --map-output grutopia/results/replan-regression-geometry/final_map
```

Inspect `semantic_exploration_result`, `final_map.json`, and the generated
videos. Record:

- `plans`, `replans`, `replan_reasons`, `last_replan`, and planning failures;
- every `plan_history` entry's `reason`, `goal`, and path;
- repeated plans with identical paths, including their step intervals;
- visible oscillation, premature waypoint switching, collision, or prolonged
  zero velocity.

Acceptance criteria:

- The sum of `replan_reasons` equals `replans`.
- There is no sustained cooldown-frequency replan loop without corresponding
  blocked-path evidence.
- A `goal_changed` plan ends at the new goal rather than the previous goal.
- No new waypoint-following or velocity-tracking regression is visible.

## Stage 3: Fused semantic Isaac run

Start GroundingDINO and MobileSAM as documented in
`docs/en/get_started/lab-shared-environment.md`, then run all three detection
modes with separate output directories so their results can be compared:

```bash
curl -fsS http://127.0.0.1:12181/health
curl -fsS http://127.0.0.1:12183/health
```

```bash
for mode in isaac open_vocab hybrid; do
  $ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
    --gpu 0 \
    --perception-gpu 1 \
    --qwen-device cuda:2 \
    --detection-mode "$mode" \
    --target refrigerator \
    --record-dir "grutopia/results/replan-regression-semantic-$mode" \
    --map-output "grutopia/results/replan-regression-semantic-$mode/final_map"
done
```

`isaac` needs no perception services and is the ground-truth reference;
`open_vocab` must record `open_vocabulary_frames > 0` with a low
`open_vocabulary_failures`, and its scene-graph nodes must never carry an
`isaac` source; `hybrid` is expected to fuse both on the same node.

For the selected target and nearby scene-graph nodes, record:

- `semantic_detection_mode`, and that it matches the requested `--detection-mode`;
- `semantic_detection_effective_mode`, availability, and startup error for a
  degraded `hybrid` run;
- `target_match`, `target_match_method`, and `target_sources`;
- node label, confidence, embedding presence, observation count, point count,
  and `last_seen_step`;
- whether same-frame Isaac and Open-Vocabulary detections produce one
  observation with both sources rather than two observations;
- attempts, completed/rate-limited/empty frames, detections, partial failures,
  last labels, last success step, and the last error.

Acceptance criteria:

- `target_match` remains backward-compatible with `target_match_method`.
- `target_sources` reports `isaac`, `open_vocabulary`, or both according to the
  evidence that reached the selected node.
- A lexical match is not treated as proof of an Isaac-only detection.
- Deduplication does not double-count a same-frame, same-object observation.
- `isaac` reports `open_vocabulary_frames == 0`; `open_vocab` reports no `isaac`
  source on any node and no silent fallback when the services are down.

## Stage 4: Failure and repeatability checks

Run `hybrid` once with the Open-Vocabulary services unavailable and confirm that
the preflight reports the cause before the Isaac-only fallback continues. Then
run `open_vocab` the same way and confirm it exits before Isaac starts, with no
videos or empty 12000-step map presented as a valid run. Restart both services,
interrupt one after startup, and confirm three consecutive runtime failures
abort with the underlying exception. Then run the full semantic scenario at
least three times with separate output directories.

Compare across runs:

- success and terminal state;
- total plans and replans, grouped by reason;
- target node ID, match method, and sources;
- trajectory length, completion step, and minimum visible obstacle clearance.

Escalate for investigation if any run shows repeated identical replans, an
empty source list for a newly observed target, a source inconsistent with the
active perception path, or a navigation collision. Do not tune pure pursuit or
collision inflation until the triggering logs and video interval have been
preserved.

## Result record

For each live run, keep the command, commit hash, machine/GPU allocation,
service availability, result JSON, `final_map.json`, `final_map.npz`, videos,
and a short pass/fail note. A live issue is considered reproduced only when its
step number and corresponding replan reason or semantic node can be identified.

## Outcome and seam verification for new smoke runs

The semantic demo now gives each run a timestamped output directory by default.
Use an explicit fresh `--record-dir` when comparing named scenarios. Read
`run_summary.json` first: `target_found` only means perception found a node;
`status: succeeded` requires a lexically confirmed target, an emitted
terminal result, and a final robot position within `arrival_threshold_m` of
`approach_goal`. Embedding-only candidates are explored and rejected when
reached without confirmation. A stale
`status: running` means the process exited before its finalizer could record
an outcome. The detailed map remains in `final_map.json`; periodic stdout
contains only a compact progress event.

For a seam correctness smoke run, add `--verify-voronoi-incremental`. This
compares every successful seam splice with a full skeleton rebuild and falls
back to the full result if any cell differs. The option is intentionally
expensive and should stay off for ordinary runs. Check
`voronoi.incremental_verification_checks`,
`voronoi.incremental_verification_mismatches`,
`voronoi.incremental_seam_mismatches`, and
`voronoi.incremental_fallback_reasons` in `run_summary.json`.
Accept an incremental correctness run only when verification checks ran
and `incremental_verification_mismatches` is zero. Seam mismatches may be
resolved by expanding the window; inspect their count and the fallback reasons.
Compare `incremental_seconds - verification_seconds` with
`full_build_seconds` and the fallback fraction before tuning the window
threshold.

For the GRScene profile, `success_distance` is 1.10 m and is passed into
semantic exploration as the arrival threshold. This is measured in the XY
plane to the selected **approach goal**, not to the refrigerator center.
`run_summary.json` reports both distances and the margin to the threshold.
A result just inside 1.10 m should be described as entering the configured
arrival radius, not as touching or interacting with the object.

Open-vocabulary completion also requires repeated label evidence on the same
scene-graph node: at least eight observations matching the query, plus either
at least 60% matching label votes or eight repetitions of the same matching
phrase. Same-frame geometry fusion retains each distinct label as evidence,
even when a higher-confidence generic `door` wins the representative label.
A single high-confidence relabel of a door to "refrigerator" is insufficient.
`target_label_support` and
`rejected_provisional_targets` in the summary show this decision. The
thresholds are conservative smoke-run guards; they do not establish that the
perception model can reliably recognize the real refrigerator.

### 2026-09-23 live smoke evidence

- `smoke_isaac_verify_20260923_01` (Isaac labels, before the repeated-label
  guard): terminal success at step 2458, final XY distance 1.0994 m to the
  approach goal with a 1.10 m threshold. Its seam oracle ran 41 comparisons
  with zero full-skeleton mismatches; 46 seam mismatches were handled by
  window expansion or full rebuild. This verifies the configured arrival rule
  and this run's incremental skeletons, not physical object interaction.
- `smoke_open_vocab_confirmed_20260923_02` (before the repeated-label guard):
  falsely reported success at step 5400 after a briefly relabeled door node
  was treated as a refrigerator. The selected node had only five observations.
- `smoke_open_vocab_label_evidence_20260923_03` (with the guard): reached the
  6000-step limit with `target_confirmed: false` and `arrival_verified: false`.
  It processed 250 open-vocabulary frames with zero perception-service failures
  and rejected four provisional candidates. The prior false success did not
  recur. Its trajectory nevertheless passed within 1.118 m of the profile
  goal at step 4878 (1.289 m from the Isaac-labeled refrigerator center),
  then ended 1.828 m from the profile goal. This is an unconfirmed semantic
  outcome, not evidence that the robot never reached the refrigerator area.

### 2026-09-25 label-fusion rerun

`go2_open_vocab_label_fix_20260925_192219` completed at step 2365 with
`target_confirmed: true` and `arrival_verified: true`. It recorded 99
open-vocabulary frames and zero perception failures. The selected node's
representative label was still `door`, but its saved label counts were
`door: 100`, `refrigerator door: 15`, and `refrigerator: 1`. The first
100-step progress record with target confirmation was step 800. Final XY
distance to the approach goal was 1.0998 m against the 1.10 m threshold,
an arrival margin of just 0.0002 m. Distance to the profile reference goal
was 1.3561 m. Report this as a threshold-level arrival near the target;
stable parking and physical interaction were not verified. The run also
saved `groundingdino.mp4` and `groundingdino_detections.jsonl` so the model's
boxes can be distinguished from yellow Isaac ground-truth boxes.

The earlier `go2_semantic_exploration_open_vocab` run ended 1.430 m from the
profile goal; the Isaac-labeled success run ended 1.356 m away. Both paths
reached the same area. The label confirmation rule is a semantic guard and
must not be used to infer geometric proximity.
Future `run_summary.json` files record `profile_goal_distance_min_m`, its step,
and `profile_goal_distance_final_m` separately. The profile goal is used only
for after-run diagnosis, never as an input to exploration or online success.
