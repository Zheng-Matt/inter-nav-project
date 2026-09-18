# Navigation and Semantic Regression Test Plan

This plan validates the navigation replan diagnostics and semantic detection
provenance added in the current change. It separates deterministic offline
checks from Isaac Sim behaviour that cannot be established by unit tests.

## Scope

The test campaign must answer four questions:

1. Does a changed goal invalidate a stale path while an unchanged goal keeps
   the current plan?
2. Does blocked-path checking continue past a zero-length waypoint segment and
   report why the route was considered blocked?
3. Do Open-Vocabulary and Isaac detections keep their source information after
   same-frame deduplication and later scene-graph merges?
4. Can `target_match_method` and `target_sources` be interpreted independently
   in a completed exploration run?

Pure-pursuit gains, velocity limits, waypoint spacing, and planner/runtime
collision geometry are not changed in this patch. They are observed during the
live runs below, but any tuning should be proposed and reviewed separately.

## Stage 1: Offline regression suite

Run the focused suite before every simulator run:

```bash
python -m unittest \
  tests.test_point_navigation_component \
  tests.test_interaction_navigation_mapping \
  tests.test_semantic_exploration_component \
  tests.test_open_vocabulary_perception
```

Then run the complete suite in the repository's configured development
environment:

```bash
python -m unittest discover -s tests
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
- `target_match`, `target_match_method`, and `target_sources`;
- node label, confidence, embedding presence, observation count, point count,
  and `last_seen_step`;
- whether same-frame Isaac and Open-Vocabulary detections produce one
  observation with both sources rather than two observations;
- `open_vocabulary_frames` and `open_vocabulary_failures`.

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
the Isaac semantic fallback still completes without losing source attribution,
then run `open_vocab` the same way and confirm it reports the failure instead of
falling back. Then run the full semantic scenario at least three times with
separate output directories.

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
