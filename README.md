# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

## Environment setup

On this lab server, `embodied` members should clone the repo and use the
shared conda / weights / scenes:

[docs/en/get_started/lab-shared-environment.md](docs/en/get_started/lab-shared-environment.md)

On any other machine, install Isaac Sim 4.2, the two conda environments,
scene/robot assets, and perception services by following
[docs/en/get_started/environment-setup.md](docs/en/get_started/environment-setup.md).

## Run Go2 navigation

On this lab server, after `source /data20t/embodied/share/inter-nav/env.sh`:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration \
  --map-output grutopia/results/go2_semantic_exploration/final_map
```

Do not set `CUDA_VISIBLE_DEVICES`. Use `$ISAAC_PYTHON`, not `python`.

On any other machine, from the repo root in the Isaac env:

```bash
export PYTHONPATH="$PWD"
export QWEN3_PYTHON=/path/to/miniconda3/envs/semexp/bin/python

python grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration \
  --map-output grutopia/results/go2_semantic_exploration/final_map
```

`--gpu` is Isaac Sim, `--perception-gpu` is CLIP, `--qwen-device` is the
Qwen3-8B worker. Pick free GPUs on your machine. Change `--target` to
`chair` or `plant` if you want a different object.

## Detection modes

`--detection-mode` selects which perception source feeds the semantic scene
graph. All three modes run the same mapping, Voronoi exploration, planning, and
locomotion stack; only the origin of the object detections changes.

| Command | Detections | Use it for |
| --- | --- | --- |
| `--detection-mode isaac` | Isaac Sim semantic boxes only (ground truth) | Geometry and planner debugging, an oracle upper bound on target recall, runs without any VLM service |
| `--detection-mode open_vocab` | GroundingDINO + MobileSAM only | Measuring what the robot actually perceives, with no ground-truth leakage |
| `--detection-mode hybrid` | Both, fused per object (default) | The full open-vocabulary exploration method |

**1. Isaac ground truth only** — one GPU, no GroundingDINO, no MobileSAM, no Qwen:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --detection-mode isaac \
  --no-qwen \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration_isaac
```

**2. Open vocabulary only** — requires the two HTTP services on the perception GPU:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --detection-mode open_vocab \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration_open_vocab
```

**3. Hybrid** — ground-truth labels fused with open-vocabulary detections
(this is the default, so `--detection-mode` may be omitted):

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --detection-mode hybrid \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration_hybrid
```

How the three modes differ at runtime:

- `isaac` ignores the open-vocabulary stack entirely; GroundingDINO is never
  queried, so `open_vocabulary_frames` stays `0`.
- `open_vocab` never falls back to simulator labels. If the services are
  unreachable the frame yields no detections and `open_vocabulary_failures`
  increments, so a degraded run cannot be mistaken for a good one. Configuring
  `open_vocab` without a perception stack is rejected at startup.
- `hybrid` merges both sources per object: a scene-graph node keeps every
  source that agreed on it and geometry comes from the denser observation.
  It degrades to Isaac-only when the services are unreachable.

Recorded runs identify their source, so results stay comparable: every
scene-graph node carries `sources` (`isaac`, `open_vocabulary`, or both) and
every statistics block reports `semantic_detection_mode`.

An empty programmatic room is available with `--scene programmatic`.

A finished run prints `semantic_exploration_result` with `"success": true`
and writes to `--record-dir`:

- `combined.mp4`, `robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`
- matching `*_preview.png`
- `final_map.json` (Voronoi graph, decisions, trajectory)
- `final_map.npz` (occupancy layers and skeleton)

For staged offline, geometry-only, and fused semantic validation, follow the
[navigation and semantic regression test plan](docs/navigation-semantic-regression-test-plan.md).

## Architecture

1. `SemanticVoronoiGraph` builds a safe medial-axis skeleton on observed free
   space, plus room-like regions and doorways.
2. `OpenVocabularyPerception` sends RGB to GroundingDINO + MobileSAM and
   stores CLIP embeddings. `--detection-mode` picks between these detections,
   the Isaac simulator labels, or a fusion of both.
3. `AdaptiveExplorationPlanner` ranks frontiers. A local Qwen3-8B worker may
   rerank the bounded topology JSON. The model never sends locomotion
   commands; A* and the Go2 RSL policy still move the robot.

## License

Simulation code follows the original GRUtopia MIT license. GRScenes assets
remain CC BY-NC-SA 4.0 and must be downloaded separately.
