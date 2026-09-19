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
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration \
  --map-output grutopia/results/go2_semantic_exploration/final_map
```

On any other machine:

```bash
export PYTHONPATH="$PWD"
export QWEN3_PYTHON=/path/to/CapNav/bin/python

$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration \
  --map-output grutopia/results/go2_semantic_exploration/final_map
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

Outputs under the `--record-dir` include synchronized videos plus
`final_map.npz` and `final_map.json`. When `--map-output` is omitted, its
default is derived from `--record-dir` rather than from a fixed shared folder.

Progress and final statistics distinguish `open_vocabulary_attempts`, completed
`open_vocabulary_frames`, rate-limited and empty frames, total detections,
partial candidate failures, the last successful step, last detected labels,
the last candidate errors, and the last frame exception. `target_found=false` with
`open_vocabulary_last_labels=["refrigerator"]` means the label was seen but has
not yet accumulated the two stable scene-graph observations required to lock
the navigation target.

For `hybrid`, compare `semantic_detection_mode` with
`semantic_detection_effective_mode`. If preflight fell back to Isaac, the saved
statistics also contain `open_vocabulary_available=false` and the exact
`open_vocabulary_startup_error`; terminal output is not the only evidence.
