# Unitree Go2 point navigation

`grutopia/demo/go2_point_navigation.py` reuses the fused RGB-D/LiDAR map,
online A* planner, velocity tracking, terminal checks, map persistence, and
synchronized video recorder with a Unitree Go2 locomotion layer.

## Assets

```bash
$ISAAC_PYTHON grutopia/demo/download_go2_asset.py
$ISAAC_PYTHON grutopia/demo/download_go2_policy.py
```

The runner prefers NVIDIA Isaac Lab's official packed Go2 USD
(`isaaclab_go2.usd`). The default locomotion actor is the Isaac Lab
rough-terrain RSL-RL policy from
[isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)
(`rough_model_7850.pt`). The downloader verifies byte length and SHA-256.

## Run and record

```bash
$ISAAC_PYTHON grutopia/demo/go2_point_navigation.py \
  --headless --gpu 0 \
  --record-dir grutopia/results/go2_navigation \
  --map-output grutopia/results/go2_navigation/final_map
```

Recording is enabled by default. The output directory contains
`robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`, `combined.mp4`,
matching previews, and `final_map.npz` / `final_map.json`.
