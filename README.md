# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

Follow the sections below in order. After step 8 you should have videos and a
`final_map` under `grutopia/results/go2_semantic_exploration/`.

## 1. Hardware and software

| Item | Requirement |
| --- | --- |
| OS | Ubuntu 20.04 or 22.04 |
| GPU | NVIDIA RTX 2070 or newer. Full semantic mode is comfortable on 2–3 GPUs (24 GB class). One GPU is enough for the geometry-only smoke test. |
| Driver | 535 or newer (535.129.03 is the Isaac Sim 4.2 recommendation) |
| RAM | 32 GB or more |
| Disk | about 25 GB for Qwen3-8B + GroundingDINO + CLIP + MobileSAM, plus Isaac Sim |
| Simulator | [Isaac Sim 4.2.0](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) only. Do not use 4.1 or 4.5. |
| Python | 3.10, provided by Isaac Sim |
| Conda | Miniconda or Anaconda |

Isaac Sim and Qwen3 need different PyTorch / Transformers versions. This repo
uses **two conda environments**:

- `grutopia`: Isaac Sim + this package (simulation, mapping, Go2 policy, CLIP)
- `semexp`: GroundingDINO, MobileSAM, Qwen3-8B worker

## 2. Install Isaac Sim 4.2.0

1. Install the NVIDIA driver and verify `nvidia-smi` works.
2. Install [Omniverse Isaac Sim 4.2.0 (workstation)](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).
3. Confirm the install contains `isaac-sim.sh` and `python.sh`. The default
   path looks like:

```text
~/.local/share/ov/pkg/isaac-sim-4.2.0
```

If Isaac already runs from an Isaac Lab conda env (`isaaclab`), you can skip
section 4 and point `ISAAC_PYTHON` at that env's `python`. It must still be
Isaac Sim **4.2**.

## 3. Clone this repository

```bash
git clone https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project
```

All later commands are run from this directory.

## 4. Create the Isaac / simulation environment

This links a conda env to Isaac Sim and installs the Python package.

```bash
# Needs conda on PATH. The script asks for the Isaac Sim folder
# (the directory that contains isaac-sim.sh) and the env name.
./setup_conda.sh
```

Accept the default env name `grutopia`, or type another name. Then:

```bash
conda activate grutopia
python -m pip install -r requirements/runtime.txt

# Confirm Isaac Sim imports
python -c "import omni.isaac.kit; print('isaac ok')"
```

If you already have a working Isaac 4.2 Python (for example Isaac Lab):

```bash
export ISAAC_PYTHON=/path/to/isaaclab/bin/python
$ISAAC_PYTHON -m pip install --no-deps -e .
$ISAAC_PYTHON -m pip install -r requirements/runtime.txt
```

In the commands below, `python` means the Isaac env. You can also set:

```bash
export ISAAC_PYTHON="$(which python)"   # after conda activate grutopia
```

## 5. Create the model environment

Do **not** install these packages into the Isaac env.

```bash
conda create -y -n semexp python=3.10
conda activate semexp
python -m pip install -r requirements/semantic-exploration.txt
```

`requirements/semantic-exploration.txt` installs recent `torch`,
`transformers`, MobileSAM, Flask, and Hugging Face Hub. If the default PyTorch
wheel does not match your CUDA driver, install a CUDA build from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then rerun the
requirements file.

```bash
export QWEN3_PYTHON="$(which python)"   # after conda activate semexp
```

Keep this variable set in every terminal that starts a model service or the
Go2 demo.

## 6. Download assets and models

Nothing under `grutopia/assets/` is committed. From the repo root:

```bash
conda activate grutopia
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# GRScenes home used by the default profile (~ tens of MB after extract)
python grutopia/demo/download_mv7_scene.py

# Official Unitree Go2 USD (needs Isaac / Nucleus access)
python grutopia/demo/download_go2_asset.py

# isaac-go2-ros2 rough-terrain policy (SHA-256 verified)
python grutopia/demo/download_go2_policy.py
```

```bash
conda activate semexp
# CLIP, GroundingDINO, MobileSAM, Qwen3-8B (about 20 GB)
python grutopia/demo/download_semantic_exploration_models.py
```

If Hugging Face rate-limits you, run `huggingface-cli login` in `semexp`
first. If Nucleus cannot fetch the Go2 USD, place `isaaclab_go2.usd` at the
path below yourself.

Expected layout:

```text
grutopia/assets/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation_preview.usda
grutopia/assets/benchmark/meta/MV7J6NIKTKJZ2AABAAAAADA8_usd/object_dict.json
grutopia/assets/robots/go2/isaaclab_go2.usd
grutopia/assets/robots/go2/policy/move_by_speed/rough_model_7850.pt
grutopia/assets/models/mobile_sam.pt
```

The scene license is
[CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenRobotLab/GRScenes).

## 7. Start perception services

Open two terminals in the **model** env. Pick a free GPU (example: `cuda:1`).

```bash
cd /path/to/inter-nav-project
conda activate semexp
export QWEN3_PYTHON="$(which python)"

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py grounding-dino \
  --host 127.0.0.1 --port 12181 --device cuda:1
```

```bash
cd /path/to/inter-nav-project
conda activate semexp
export QWEN3_PYTHON="$(which python)"

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py mobile-sam \
  --host 127.0.0.1 --port 12183 --device cuda:1 \
  --mobile-sam-checkpoint grutopia/assets/models/mobile_sam.pt
```

Wait until each process prints that Flask is running. Optional check:

```bash
curl -s http://127.0.0.1:12181/health
curl -s http://127.0.0.1:12183/health
```

If these services are down, exploration still runs with Isaac semantic labels.
If Qwen3 fails to start, frontier ranking falls back to geometric scoring.

## 8. Run Go2 navigation

Third terminal, **Isaac** env, repo root. Use GPUs that are free on your
machine. The flags below assume:

- `--gpu 0`: Isaac Sim
- `--perception-gpu 1`: CLIP inside the Isaac process (same GPU as the HTTP services is fine)
- `--qwen-device cuda:2`: Qwen3-8B worker spawned by the demo

```bash
cd /path/to/inter-nav-project
conda activate grutopia
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

On a machine with no display the script already defaults to `--headless`.
Change `--target` to another object name, such as `chair` or `plant`.

Geometry-only smoke test (no VLM services, no Qwen, one GPU):

```bash
python grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 --no-open-vocabulary --no-qwen --target refrigerator
```

An empty programmatic room is available with `--scene programmatic`.

## 9. What success looks like

The process prints JSON lines. A finished run ends with
`semantic_exploration_result` and `"success": true` when Go2 reaches the
matched target. Files under `--record-dir`:

- `combined.mp4`, `robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`
- matching `*_preview.png`
- `final_map.json` (Voronoi graph, decisions, trajectory)
- `final_map.npz` (occupancy layers and skeleton)

Offline checks that do not need Isaac:

```bash
conda activate grutopia
export PYTHONPATH="$PWD"

python -m unittest discover -s tests -p 'test_*semantic*'
python -m unittest tests.test_go2_navigation_support tests.test_point_navigation_component

python grutopia/demo/evaluate_semantic_exploration.py \
  --map-prefix grutopia/results/go2_semantic_exploration/final_map \
  --target-label refrigerator
```

## Architecture

1. `SemanticVoronoiGraph` builds a safe medial-axis skeleton on observed free
   space, plus room-like regions and doorways.
2. `OpenVocabularyPerception` sends RGB to GroundingDINO + MobileSAM and
   stores CLIP embeddings. Isaac labels are the fallback.
3. `AdaptiveExplorationPlanner` ranks frontiers. A local Qwen3-8B worker may
   rerank the bounded topology JSON. The model never sends locomotion
   commands; A* and the Go2 RSL policy still move the robot.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `omni.isaac` import fails | Isaac Sim 4.2 is not sourced. `conda activate grutopia` after `./setup_conda.sh`, or use Isaac's `python.sh`. |
| `GRScenes navigation USD not found` | `download_mv7_scene.py` did not finish. Confirm the `.usda` path in section 6. |
| Go2 USD missing | Run `download_go2_asset.py` from the Isaac env, or copy `isaaclab_go2.usd` into `grutopia/assets/robots/go2/`. |
| CUDA OOM | Give Isaac, the two HTTP services, and Qwen different GPUs. Or run `--no-qwen` / `--no-open-vocabulary`. |
| Qwen worker never starts | `QWEN3_PYTHON` must be the `semexp` interpreter, not Isaac Python. |
| No detections | GroundingDINO / MobileSAM not listening on `12181` / `12183`. |

## License

Simulation code follows the original GRUtopia MIT license. GRScenes assets
remain CC BY-NC-SA 4.0 and must be downloaded separately.
