# Lab shared environment (this server)

If your account is in the `embodied` group on this machine, **do not** install
Isaac Sim or download models yourself. Code still comes from GitHub; conda,
weights, and scenes are already on the shared disk.

People who are not on this server should follow
[environment-setup.md](environment-setup.md) instead.

Shared paths (group `embodied`, read/execute only):

| What | Path |
| --- | --- |
| Isaac / simulation Python | `/data20t/embodied/share/inter-nav/envs/isaaclab` |
| Qwen / perception Python | `/data20t/embodied/share/inter-nav/envs/CapNav` |
| Hugging Face cache | `/data20t/embodied/share/inter-nav/huggingface` |
| Scenes, Go2 USD, MobileSAM | `/data20t/embodied/share/inter-nav/assets` |

Do not `pip install` or `conda install` into those two environments.

`source env.sh` sets interpreters, `HF_HOME`, `PYTHONPATH`, and a
`grutopia/assets` symlink. It does **not** `conda activate`, start Isaac, pick
GPUs, or start GroundingDINO / MobileSAM. Source it again in every new
terminal.

## 1. Clone the code

```bash
git clone https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project
```

Use your own clone. Do not run from someone else's working tree.

## 2. Load the shared environment

```bash
source /data20t/embodied/share/inter-nav/env.sh
```

This sets:

- `ISAAC_PYTHON` → shared `isaaclab`
- `QWEN3_PYTHON` → shared `CapNav`
- `HF_HOME` → shared Hugging Face cache (Qwen3-8B, CLIP, GroundingDINO)
- `PYTHONPATH` → the current repo
- `grutopia/assets` → symlink to the shared scenes if that folder is missing

Scene USDA files use relative paths, so they do not depend on `/data/zkj`.

Check that you are in `embodied` (`groups`) and that a GPU is free
(`nvidia-smi`).

Do **not** set `CUDA_VISIBLE_DEVICES`. Isaac Sim then mis-numbers GPUs and
fails with `No device could be created`. Select cards only with `--gpu`,
`--perception-gpu`, and `--qwen-device`.

Use `$ISAAC_PYTHON` / `$QWEN3_PYTHON`. Do not run `python` unless that
interpreter is already the shared Isaac env.

A display is not required. Headless runs still write `combined.mp4` and the
other videos. The first full GRScenes load can take several minutes while
Isaac compiles house materials.

## 3. Geometry-only smoke test (one GPU)

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 --detection-mode isaac --no-qwen --target refrigerator
```

This does not need Qwen, CLIP, or the HTTP perception services. `--detection-mode`
takes `isaac` (ground truth only), `open_vocab` (VLM detections only), or
`hybrid` (both, the default).

## 4. Full semantic demo

Pick three free GPUs (example: `0`, `1`, `2`).

Start GroundingDINO and MobileSAM in two other terminals. They are required for
`open_vocab`; only `hybrid` may degrade to Isaac semantic labels when they are
down.

```bash
cd /path/to/your/inter-nav-project
source /data20t/embodied/share/inter-nav/env.sh

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py grounding-dino \
  --host 127.0.0.1 --port 12181 --device cuda:1
```

```bash
cd /path/to/your/inter-nav-project
source /data20t/embodied/share/inter-nav/env.sh

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py mobile-sam \
  --host 127.0.0.1 --port 12183 --device cuda:1 \
  --mobile-sam-checkpoint grutopia/assets/models/mobile_sam.pt
```

In a third terminal, verify readiness before launching Isaac:

```bash
curl -fsS http://127.0.0.1:12181/health
curl -fsS http://127.0.0.1:12183/health
```

Then run the demo:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator \
  --record-dir grutopia/results/go2_semantic_exploration \
  --map-output grutopia/results/go2_semantic_exploration/final_map
```

`--gpu` is Isaac Sim, `--perception-gpu` is CLIP, `--qwen-device` is the
Qwen3-8B worker. Change `--target` to `chair` or `plant` if you want.

A finished run prints `semantic_exploration_result` with `"success": true`
and writes videos plus `final_map.json` / `final_map.npz` under
`--record-dir`.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Permission denied on `/data20t/embodied/share` | Your account is not in `embodied`. Ask an admin to add you, then log in again. |
| `grutopia/assets` missing | Run `source /data20t/embodied/share/inter-nav/env.sh` from the repo root. |
| Hugging Face tries to download again | `HF_HOME` is not set. Source `env.sh` in that terminal. |
| `QWEN3_PYTHON` points at Isaac Python | Source `env.sh` again. Qwen must be the `CapNav` interpreter. |
| `No device could be created` / CUDA bad state | Unset `CUDA_VISIBLE_DEVICES`. Pass `--gpu N` only. |
| CUDA OOM | Give Isaac, perception, and Qwen different free GPUs. |
| `open_vocab requires ready ... before the run starts` | Start both HTTP services, check ports `12181` and `12183`, and confirm the CLIP weights are visible through `HF_HOME`. |
| `open-vocabulary perception failed 3 consecutive frames` | Read `open_vocabulary_last_error` in the progress/error output; check the named service and GPU before retrying. |
| First launch is slow | Full GRScenes material compile takes several minutes. Later runs are faster. |
