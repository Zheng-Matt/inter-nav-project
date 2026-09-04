# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

## Environment setup

Install Isaac Sim 4.2, the two conda environments, scene/robot assets, and
perception services by following
[docs/en/get_started/environment-setup.md](docs/en/get_started/environment-setup.md).

## Run Go2 navigation

After that setup, from the repo root in the Isaac env:

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

Geometry-only smoke test (no VLM services, no Qwen, one GPU):

```bash
python grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 --no-open-vocabulary --no-qwen --target refrigerator
```

An empty programmatic room is available with `--scene programmatic`.

A finished run prints `semantic_exploration_result` with `"success": true`
and writes to `--record-dir`:

- `combined.mp4`, `robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`
- matching `*_preview.png`
- `final_map.json` (Voronoi graph, decisions, trajectory)
- `final_map.npz` (occupancy layers and skeleton)

## Architecture

1. `SemanticVoronoiGraph` builds a safe medial-axis skeleton on observed free
   space, plus room-like regions and doorways.
2. `OpenVocabularyPerception` sends RGB to GroundingDINO + MobileSAM and
   stores CLIP embeddings. Isaac labels are the fallback.
3. `AdaptiveExplorationPlanner` ranks frontiers. A local Qwen3-8B worker may
   rerank the bounded topology JSON. The model never sends locomotion
   commands; A* and the Go2 RSL policy still move the robot.

## License

Simulation code follows the original GRUtopia MIT license. GRScenes assets
remain CC BY-NC-SA 4.0 and must be downloaded separately.
