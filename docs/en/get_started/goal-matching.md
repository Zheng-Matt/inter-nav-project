# Open-vocabulary goal matching

Use `--goal-matching robust` (the default) for category synonyms and complete
object descriptions. Examples include `refrigerator`, `fridge`, `white fridge`,
and `a large appliance used to keep food cold`. A category label alone does not
prove a color, property, or spatial relationship.

The pipeline stays RGB → GroundingDINO proposals → MobileSAM masks and RGB-D
positions → Qwen-VL classification/description verification → scene graph →
target confirmation → reachable approach point. Exact aliases are shared across
classifiers. The original Qwen path already supported `fridge`; robust matching
also replaces permissive substring matches and preserves descriptive attributes.

Qwen-VL reports the observed category/color before evaluating the full query.
Invalid or missing scores, unknown objects and unverifiable relationships do not
confirm a description. Evidence is kept separately for each object and query;
repeated detections from one capture step do not add extra votes. Older Qwen
services can still classify categories but cannot confirm descriptions.

## Configuration

| Navigation option | Default | Purpose |
| --- | --- | --- |
| `--goal-matching` | `robust` | Select `legacy` for the original matching path. |
| `--description-verification` | enabled | `--no-description-verification` disables verification without accepting unverified descriptions. |
| `--goal-match-confidence` | `0.80` | Minimum Qwen description score; this is not a calibrated 80% probability. |
| `--goal-node-confidence` | `0.10` | Minimum object detection confidence. |
| `--goal-embedding-threshold` | `0.24` | Minimum CLIP cosine similarity for a provisional target. |
| `--goal-embedding-margin` | `0.03` | Required lead over the second candidate. |
| `--qwen-vl-crop-mode` | `masked` | Replace pixels outside the SAM foreground with neutral gray; `context` retains background. |
| `--semantic-ground-clearance` | `0.02` m | Reject candidates whose height is consistent with the known ground surface; `0` disables this guard. |

Descriptions need two positive observations before selecting an approach target,
and eight positive observations with at least 60% support for final confirmation.
CLIP similarity can guide another observation but cannot alone confirm a full
description. Ground filtering is disabled when ground height is unknown and
needs further validation for low objects and uneven floors. Legacy mode keeps
the original crop and matching behavior.

Start perception services using the [Qwen-VL setup notes](../../../grutopia/demo/QWEN_VL_SEMANTICS.md)
and [environment guide](lab-shared-environment.md). The standard lab Isaac launch
uses physical `--gpu` numbers without `CUDA_VISIBLE_DEVICES`; archived commands
describe historical experiments and are not general launch scripts.

```bash
source /data20t/embodied/share/inter-nav/env.sh
"$ISAAC_PYTHON" grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 --perception-gpu 1 --qwen-device cuda:2 \
  --target 'a large appliance used to keep food cold' \
  --detection-mode open_vocab --semantic-classifier qwen-vl --goal-matching robust
```

Select free devices and check all service health endpoints first. Results,
videos and full predictions belong in a new `grutopia/results/<run-name>/`
directory. Existing record directories are refused. Model weights and scene
assets stay in the configured asset storage.

## CPU checks and frozen baseline

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/goal-matching-test.txt
.venv/bin/python -m unittest discover -s tests -p 'test_goal*.py'

baseline_dir=$(mktemp -d /tmp/inter-nav-main.XXXXXX)
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$baseline_dir"
.venv/bin/python grutopia/demo/evaluate_goal_matching.py --baseline-root "$baseline_dir"
```

The default dataset contains 30 scripted contracts with preset labels/scores.
The historical comparison was 14/30 correct for the frozen baseline and 30/30
for robust matching. It measures matching rules, not actual model accuracy.
Go2 policy checks additionally require PyTorch/Isaac assets; an older G1 camera
check references a missing profile and is a separate known limitation.

## Real model inputs and reproduction

`tests/fixtures/goal_matching/clean_real_fridge/` contains context crops;
`foreground_real_fridge/` contains masked crops, masks and four raw RGB frames.
Five identical context images are shared through relative paths to the clean
dataset. The two `floor_000240.png` versions have different pixels and remain
separate. Both datasets keep development and held-out case manifests.
The foreground dataset's [source records](../../../tests/fixtures/goal_matching/foreground_real_fridge/mask-provenance.json)
retain input hashes and segmentation provenance.

With an independent, healthy Qwen-VL service, evaluate the saved inputs:

```bash
source /data20t/embodied/share/inter-nav/env.sh
export PYTHONPATH=.
R="grutopia/results/goal_matching_repeat_$(date +%Y%m%d_%H%M%S)"
mkdir "$R"
mkdir "$R/baseline-main"
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$R/baseline-main"
"$ISAAC_PYTHON" grutopia/demo/evaluate_goal_matching.py \
  --dataset tests/fixtures/goal_matching/foreground_real_fridge/development.json \
  --baseline-root "$R/baseline-main" --live-qwen-url http://127.0.0.1:12185/classify \
  --qwen-vl-crop-mode masked --record-dir "$R/development"
# Repeat with heldout.json and a new record directory.
```

Each live case uses one observation to isolate matching, so it does not test
the production eight-observation confirmation rule. `recorded_fridge_smoke.json`
is the smaller single-image smoke input. For the original context comparison,
use the clean dataset and `--qwen-vl-crop-mode context`.

Optional input reconstruction, from the repository root with assets and SAM
available (the selected `R` must already exist):

```bash
cp docs/reports/assets/goal-matching-experiment-20261009/near-start-profile.json "$R/near-start-profile.json"
"$ISAAC_PYTHON" grutopia/demo/build_goal_matching_scene.py "$R"
"$ISAAC_PYTHON" grutopia/demo/build_goal_matching_dataset.py "$R" \
  --mobile-sam-url http://127.0.0.1:12183/mobile_sam
```

The scene helper references one refrigerator from the original GRScene, with
its geometry/material/transform and neutral lighting. This is a single-asset
smoke scene, not the complete house. Helpers refuse to overwrite their outputs.
Use the existing Go2 exploration profile for full-house runs.

## Verified results and remaining work

The historical foreground comparison scored 20/20 in each of two groups
(baseline 15/20 and 14/20). All inputs came from one refrigerator trajectory;
they do not establish performance on unseen objects or independent scenes.
Full outputs and latency measurements remain in the [experiment archive](../../reports/assets/ARCHIVED_GOAL_MATCHING_RESULTS.md).

The [single-asset navigation summary](../../reports/assets/goal-matching-mask-20261009/minimal-summary.json)
records confirmed arrival at step 749, with 32 perception frames and no service
failures. Arrival was 1.0955 m from the approach point under a 1.1 m rule; it does
not establish manipulation success. That run also logged PhysX device errors.
The subsequent device configuration change has no successful GPU validation,
and complete-house description navigation remains unverified.

Further evaluation needs multiple objects/colors, occlusion and confusing
instances, score calibration, uneven ground, repeated latency measurements,
and full-house navigation. Cleanup does not change these evidence limits.
