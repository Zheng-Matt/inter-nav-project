# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, semantic
Voronoi exploration, and the GRScenes MV7 refrigerator run.

The reference result is the successful `grscene_mv7_v26` run: Go2 starts in
`MV7J6NIKTKJZ2AABAAAAADA8_usd`, explores until it lexically matches
`refrigerator`, and stops about 1.2 m from the scanned fridge.

## What this repo contains

- `grutopia/core`: Isaac Sim / Gym runtime
- `grutopia_extension/interactive_navigation`: mapping, Voronoi graph, planner
- `grutopia_extension/robots/go2*`, `controllers/isaac_go2_ctrl.py`: Go2 policy
- `grutopia/demo/go2_semantic_exploration.py`: entry point
- download helpers for the MV7 scene, Go2 USD, locomotion checkpoint, and VLM weights

Large scene packs, old result videos, and unused robots (H1, G1, Franka, …)
are not included.

## Requirements

- Ubuntu 20.04 / 22.04
- NVIDIA GPU, driver 535+
- [Isaac Sim 4.2.0](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)
- A Python 3.10 environment that can import Isaac Sim (this repo was run with
  a conda env named `isaaclab`)
- A second Python environment for Qwen3 / GroundingDINO / MobileSAM (this repo
  was run with a conda env named `CapNav`)

Isaac Sim and Qwen3 need different PyTorch / Transformers versions. Keep them
separate.

```bash
git clone https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project

# Isaac / simulation environment
export ISAAC_PYTHON=/path/to/isaaclab/bin/python
$ISAAC_PYTHON -m pip install --no-deps -e .
$ISAAC_PYTHON -m pip install -r requirements/runtime.txt

# Isolated model environment
export QWEN3_PYTHON=/path/to/CapNav/bin/python
$QWEN3_PYTHON -m pip install -r requirements/semantic-exploration.txt
$QWEN3_PYTHON -m pip install flask huggingface-hub
```

## Assets

Nothing under `grutopia/assets/` is committed. Download only the files used by
the MV7 refrigerator experiment:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# GRScenes MV7 home + object metadata (~20 MB after extract)
$ISAAC_PYTHON grutopia/demo/download_mv7_scene.py

# Official Unitree Go2 USD
$ISAAC_PYTHON grutopia/demo/download_go2_asset.py

# isaac-go2-ros2 rough-terrain policy (SHA-256 verified)
$ISAAC_PYTHON grutopia/demo/download_go2_policy.py

# CLIP, GroundingDINO, MobileSAM, Qwen3-8B
$QWEN3_PYTHON grutopia/demo/download_semantic_exploration_models.py
```

Expected layout:

```text
grutopia/assets/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation_preview.usda
grutopia/assets/benchmark/meta/MV7J6NIKTKJZ2AABAAAAADA8_usd/object_dict.json
grutopia/assets/robots/go2/isaaclab_go2.usd
grutopia/assets/robots/go2/policy/move_by_speed/rough_model_7850.pt
```

If HuggingFace is blocked, copy those paths from a machine that already has
InternUtopia / GRScenes. The scene license is
[CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenRobotLab/GRScenes).

## Perception services

Start GroundingDINO and MobileSAM before the Isaac run. Default ports match
the Agent VLM endpoints used by `OpenVocabularyPerception`:

```bash
$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py grounding-dino \
  --host 127.0.0.1 --port 12181 --device cuda:1

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py mobile-sam \
  --host 127.0.0.1 --port 12183 --device cuda:1 \
  --mobile-sam-checkpoint grutopia/assets/models/mobile_sam.pt
```

If the services are down, exploration still runs with Isaac semantic labels.
If Qwen3 fails to start, frontier ranking falls back to geometric scoring.

## Reproduce `grscene_mv7_v26`

From the repo root, with no display:

```bash
export PYTHONPATH="$PWD"
export QWEN3_PYTHON=/path/to/CapNav/bin/python

mkdir -p grutopia/results/go2_semantic_exploration/grscene_mv7_v26

$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 2 \
  --perception-gpu 1 \
  --qwen-device cuda:3 \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration/grscene_mv7_v26 \
  --map-output grutopia/results/go2_semantic_exploration/grscene_mv7_v26/final_map
```

`--gpu` / `--perception-gpu` / `--qwen-device` are machine-local. Pick free
GPUs; the defaults in the script are only a starting point.

The reference run finished successfully at step 2458:

- `success: true`
- `target_query: refrigerator`
- `target_match: lexical`
- robot near `[9.30, -0.75, 0.52]`, fridge near `[10.53, -1.70, 1.01]`

Outputs in the record directory:

- `combined.mp4`, `robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`
- matching `*_preview.png`
- `final_map.json` (Voronoi graph, decisions, trajectory)
- `final_map.npz` (occupancy layers and skeleton)

Offline checks that do not need Isaac:

```bash
$ISAAC_PYTHON -m unittest discover -s tests -p 'test_*semantic*'
$ISAAC_PYTHON -m unittest tests.test_go2_navigation_support tests.test_point_navigation_component

$ISAAC_PYTHON grutopia/demo/evaluate_semantic_exploration.py \
  --map-prefix grutopia/results/go2_semantic_exploration/grscene_mv7_v26/final_map \
  --target-label refrigerator
```

Geometry-only smoke test:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --no-open-vocabulary --no-qwen --target refrigerator
```

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
