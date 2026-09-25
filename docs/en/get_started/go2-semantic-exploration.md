# Go2 semantic Voronoi exploration

`grutopia/demo/go2_semantic_exploration.py` explores an initially unknown
occupancy map until Go2 finds and approaches a text-described object or area.

Install Isaac Sim, conda environments, and models with
[environment-setup.md](environment-setup.md) first. On this lab server, use
[lab-shared-environment.md](lab-shared-environment.md) instead.

## Architecture

The implementation adds three layers above the existing RGB-D/LiDAR map:

1. `SemanticVoronoiGraph` extracts and prunes a safe medial-axis skeleton from
   observed free space, compresses it into endpoint/junction/corridor nodes,
   detects free/unknown frontiers, and attaches persistent semantic objects to
   reachable topology nodes.
2. `OpenVocabularyPerception` sends RGB frames to GroundingDINO and MobileSAM,
   associates every mask pixel with its aligned world-space RGB-D point, and
   stores normalized CLIP embeddings. Isaac semantic boxes are used only in
   `isaac` and `hybrid`; `open_vocab` never receives ground-truth fallback.
3. `AdaptiveExplorationPlanner` scores frontier information gain, path cost,
   clearance, degree, extension, direction, visits, failures, and dead ends.
   When semantic evidence is available, an isolated local Qwen3-8B worker
   ranks the bounded JSON topology.

The local model never emits locomotion commands. It only ranks reachable
frontiers; A* and the Go2 RSL policy remain responsible for motion.

## Model setup

Use a separate model environment because Isaac Sim and Qwen3 require different
PyTorch / Transformers versions:

```bash
export QWEN3_PYTHON=/path/to/CapNav/bin/python

$QWEN3_PYTHON -m pip install -r requirements/semantic-exploration.txt
$QWEN3_PYTHON grutopia/demo/download_semantic_exploration_models.py
```

GroundingDINO and MobileSAM use these endpoints:

- `http://localhost:12181/gdino`
- `http://localhost:12183/mobile_sam`

Start them with `grutopia/demo/serve_semantic_perception.py` as shown in
[lab-shared-environment.md](lab-shared-environment.md), then verify both
`/health` endpoints. If an endpoint or CLIP weight is unavailable, `hybrid`
reports the failed preflight and continues with Isaac semantic labels;
`open_vocab` exits before Isaac starts. If Qwen3 is unavailable, frontier
selection continues in deterministic geometric mode.

`--detection-mode` chooses the detection source:

| Mode | Detections | Needs the services |
| --- | --- | --- |
| `isaac` | Isaac Sim semantic boxes only (ground truth) | no |
| `open_vocab` | GroundingDINO + MobileSAM only | yes |
| `hybrid` (default) | both, fused per object | yes, but degrades to `isaac` |

`open_vocab` never falls back to simulator labels. A ready stack that later
fails three consecutive RGB queries aborts the run with the underlying error
instead of silently producing an empty scene graph. A single bad candidate
mask is recorded as a partial failure without discarding other valid objects
from the same frame. Every scene-graph node records which sources agreed on it.

## Run

The default scene is the calibrated GRScenes home
`MV7J6NIKTKJZ2AABAAAAADA8_usd`. Download it first:

```bash
$ISAAC_PYTHON grutopia/demo/download_mv7_scene.py
$ISAAC_PYTHON grutopia/demo/download_go2_asset.py
$ISAAC_PYTHON grutopia/demo/download_go2_policy.py
```

Go2 starts on the official refrigerator path about 9.2 m away and explores
until it reaches the real scanned fridge.

On this lab server, `source /data20t/embodied/share/inter-nav/env.sh` already
sets `ISAAC_PYTHON`, `QWEN3_PYTHON`, and `PYTHONPATH`. Do not set
`CUDA_VISIBLE_DEVICES`.

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator
```

On any other machine:

```bash
export PYTHONPATH="$PWD"
export QWEN3_PYTHON=/path/to/CapNav/bin/python

$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator
```

The empty programmatic room is still available with `--scene programmatic`.

Disable unavailable model layers explicitly when testing the geometry:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --detection-mode isaac --no-qwen
```

Run the detector on its own, without any ground-truth labels:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --detection-mode open_vocab --qwen-device cuda:2
```

Outputs under the `--record-dir` include synchronized videos,
`run_summary.json`, `final_map.npz`, and `final_map.json`. Runs that actually
query the open-vocabulary detector also save `groundingdino.mp4` and
`groundingdino_detections.jsonl`. The GroundingDINO video contains model
query frames; yellow third-person boxes are Isaac ground truth. When
`--map-output` is omitted, its default is derived from `--record-dir`.
Omitting both options creates a new timestamped run directory. An explicitly
chosen directory must be new; the runner refuses to overwrite an earlier run.

Each run also writes a small analysis set:

- `manifest.json`: start time, Git revision and dirty flag, resolved run and
  scene settings, semantic thresholds, Qwen state, and available perception
  service model settings. A `null` random seed means this demo did not set one.
- `events.jsonl`: target selection/confirmation, goal changes, plans, planning
  failures, and Voronoi fallbacks, emitted only when they change.
- `trace.csv`: sampled position, heading, active goal distance, target node, and label
  evidence. Rows are appended at `--record-every` steps and at normal exit.
- `progress.json`: atomically updated at `--log-every` steps and on normal
  exit. It preserves the last recorded step if the process is killed before
  `final_map` and videos can be closed.
- `video_frames.jsonl`: zero-based frame numbers and simulation steps for the
  regular videos. Each GroundingDINO JSONL row has its own
  `video_frame_index` for `groundingdino.mp4`.

The final `run_summary.json` is written after artifact cleanup and includes
`completion_recorded` and `artifacts_complete`. A startup-only summary with
`status=running` does **not** establish that the process is still alive.
Summarize a directory of old and new runs with the standard-library script:

```bash
python grutopia/demo/summarize_semantic_runs.py grutopia/results \
  --output grutopia/results/semantic_runs.csv
```

The CSV marks startup-only records as `unfinished_record`, recovers the last
recorded step from progress, the old run log, or detector records, and leaves
missing historical metadata blank. Compare
runs only after checking scene, target, detection mode, Qwen state, code
revision, and seed. The local `/health` endpoints now expose GroundingDINO's
model/thresholds and MobileSAM's checkpoint path; restart existing perception
services to include those fields in future manifests.

Compact periodic progress prints only the step, state, position, target flags,
and goal distance. Saved final statistics distinguish `open_vocabulary_attempts`, completed
`open_vocabulary_frames`, rate-limited and empty frames, total detections,
partial candidate failures, the last successful step, last detected labels,
the last candidate errors, and the last frame exception. `target_found=false` with
`open_vocabulary_last_labels=["refrigerator"]` means the label was seen but has
not yet accumulated the two stable scene-graph observations required to lock
the navigation target.

`target_found` is provisional. `target_confirmed` requires at least eight
matching label observations on one node, plus either a matching-label fraction
of at least 60% or eight repetitions of the same matching phrase. Same-frame
geometry fusion preserves distinct labels as separate evidence, so a stable
`door` node can also accumulate `refrigerator door` observations. Success
additionally requires arrival within the profile's XY distance threshold of
the selected approach goal. Read `run_summary.json` before interpreting a
video or an old `final_map.json` as a completed run.

For `hybrid`, compare `semantic_detection_mode` with
`semantic_detection_effective_mode`. If preflight fell back to Isaac, the saved
statistics also contain `open_vocabulary_available=false` and the exact
`open_vocabulary_startup_error`; terminal output is not the only evidence.
