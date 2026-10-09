# GPU 0 smoke test：部分通过（2026-10-09）

本次实际连接服务器，测试上一轮提交 `7186a7dff66a5084feb2de5d3ca7b1623aa6c016`，分支为 `improve-open-vocabulary-goal-matching`。三个要求的目标表达在单张真实录制图像上通过；颜色反例、八帧生产确认和导航到达未通过。不能据此声称目标匹配已全面可靠。

## 版本、隔离与保存路径

- 本地代码与服务器独立 clone `/data/zzx/inter-nav-goal-matching-smoke` 的 HEAD 一致。通过 Git bundle 同步，运行过程中没有修改源码。
- 原服务器工作目录 `/data/zzx/inter-nav-project` 保持在 `f7eab5948b30855776e3ce59b5a24d1ea6af5fdd`，保留其三条未推送提交和素材链接。GitHub main 不变。
- 所有新 GPU 进程设置 `CUDA_VISIBLE_DEVICES=0`。Qwen 使用独立端口 `127.0.0.1:12186`，没有向其他现有模型服务发推理请求，也没有启停其他任务。
- GPU 0 为 RTX 4090 D 24 GB；初始 240 MiB、利用率 0%。其中约 214 MiB 来自已有仿真进程的 CUDA 上下文，没有终止该进程。
- 复用已有模型和解释器；没有安装到共享环境或下载权重。模型与仿真分阶段运行，Qwen 退出后才启动 Isaac。
- 原始文件保存在各自 clone 的 `grutopia/results/goal_matching_smoke_20261009_1425/`；该目录已取回本地。日志、图像、视频和临时脚本不进 Git。小型审核快照在本报告的 `assets/goal-matching-smoke-20261009/`；可复现单图样例在 `tests/fixtures/goal_matching/recorded_fridge_smoke.{json,png}`。

## 实际结果

Qwen3-VL-8B-Instruct 对一个人工检查的深色双门冰箱图像作五种查询。比较工具从冻结 main 直接导入原始匹配代码；使用同一个新服务、同一图像，原始路径不发送描述核验参数。单图比较配置为一次证据即可确认，生产导航仍保留八票门槛。

| 查询 | 原始 main | 新匹配 | 新匹配判定 |
| --- | --- | --- | --- |
| `refrigerator` | 冰箱 | 冰箱 | PASS |
| `fridge` | 冰箱 | 冰箱 | PASS |
| `a large appliance used to keep food cold` | 未匹配 | 冰箱，分数 0.9 | PASS |
| `a red fridge` | 未匹配 | 错误接受冰箱，分数 0.9 | FAIL |
| `a chair with armrests` | 未匹配 | 分数 null，未匹配 | PASS |

| 单图指标（仅五项 smoke） | 原始 main | 新匹配 |
| --- | ---: | ---: |
| 正确案例 | 4/5 | 4/5 |
| 三个正例召回 | 2/3 | 3/3 |
| 两个无目标查询误匹配 | 0/2 | 1/2 |
| 请求中位延迟 | 241.2 ms | 391.6 ms |
| 温热匹配中位耗时，不含模型请求 | 19.6 μs | 18.3 μs |
| HTTP/解析异常 | 0 | 0 |

只有一个实例、三个正查询和两个负查询，不能解释成真实数据集准确率。比较工具没有逐项保存请求延迟，也没有区分类别和描述请求的延迟分布；表中仅报告实际记录的整体中位数，不能推断完整导航延迟。

八帧回放用同一个功能描述，经过实际 HTTP 客户端、真实模型、场景图和生产匹配配置。七帧返回 `refrigerator / 0.9`，一帧返回 `cabinet / 0`；目标进入候选状态，但支持票为 7，未达到默认 8 票，因此最终确认 FAIL。没有重复计票、降低门槛或将类别票当成描述证据。

程序化 Isaac 场景以 `fridge` 为目标，运行完整的传感器、构图、目标选择、规划、Go2 控制及终止判定。为限定单卡短测试，感知使用 `--detection-mode isaac`，没有完整运行 DINO + SAM + Qwen 的在线视觉导航。结果：

- 目标 `object:refrigerator:0` 被正确找到并确认，标签支持 20；规划 2 次，失败 0。
- 第 479 步到达运行上限：`global_step_limit`。最终距接近点 0.8055 m，门槛 0.7000 m，差 0.1055 m；到达 FAIL。
- `run_summary.json` 的语义退出码为 2，但原 CLI 的 `main()` 返回值未传给进程退出码，因此系统进程退出码为 0。判定以结果事件和 run summary 为准，不能仅检查 shell 退出码。本次没有修改这个既有 CLI 行为。

## 耗时与资源

| 阶段 | 实测墙钟时间 | GPU 0 总显存采样峰值 |
| --- | ---: | ---: |
| 独立 Qwen 服务完整生命周期，包含等待/请求 | 183.660 s | 17,685 MiB |
| 原始/新方法十次请求及比较工具 | 6.164 s | 17,659 MiB |
| 八帧描述回放 | 4.117 s，内部回放计时 3.597 s | 17,661 MiB |
| 最小 Isaac 导航 | 48.391 s | 4,665 MiB |

每秒采样一次 `nvidia-smi`，峰值是整卡总占用，包含原有约 240 MiB 上下文；不等于本进程分配量，也不能捕捉所有瞬时峰值。18 次真实模型请求完成后，独立 Qwen 服务按计划终止；导航进程随后正常退出，GPU 0 恢复到 240 MiB、利用率 0%。没有训练。

模型环境：Python 3.10.20、PyTorch 2.5.1+cu118、Transformers 5.15.0、NumPy 1.26.4、OpenCV 4.11.0、Flask 3.1.3、驱动 550.107.02。比较工具/Isaac 使用共享 isaaclab Python 3.10.19。服务器目标匹配回归测试 17/17 通过，耗时 0.395 s。

## 样本与异常的限制

历史样本来自 `qwen_vl_refrigerator_20260927_131201`，原录制提交 `c3e25392f30afa73373a7434a343f7dd08b646a7`，其工作目录当时有未提交修改。step 1800 / frame 94 的图像由本次更新代码裁切并重新调用模型推理，没有复用历史分类结果作为新预测。

历史 RGB 视频已有渲染的检测框和标签。单图比较排除了类别文字和标题，红色/紫色框线仍保留；裁切 margin 从生产默认值改为 0，并裁去上部标签区域，所以该图像与生产上下文 crop 不完全一致。图像 SHA-256 为 `b4235feffad37df5b7652655f246a593465aa616cf6a6291dd914aa91ac9bf43`。八帧 contact sheet 检查发现部分帧仍有 `unknown` 或分数字样；八帧回放只能作为工程链路记录，不能作为洁净视觉评估。

`a red fridge` 的误接受可能受框线颜色、裁切上下文或模型属性判断影响，当前证据不能区分原因。不能仅提高分数门槛解决：同图正确的功能描述与错误的红色描述均得到 0.9。下一轮需要无标签原始 RGB、多颜色正/负实例，以及可核验的属性证据和分数校准。

Isaac 日志存在 `CUDA_VISIBLE_DEVICES` 提醒、非可见 GPU 的 “CUDA being in bad state” 跳过提示、warmup 的 `Lidar Sensor does not exist`、深度非有限值警告及 headless/关闭插件警告。本次 GPU 表只将物理 GPU 0 标为 Active，应用成功启动并运行 480 步，没有 OOM、traceback 或设备创建失败。根据用户要求保留 `CUDA_VISIBLE_DEVICES=0`；这次成功启动不说明该设置在其他 Isaac 环境均安全。导航短运行未到达的原因尚未定位，不能把 warmup 警告直接认定为根因。

比较工具还暴露了元数据问题：`git rev-parse` 在 repo 内的冻结归档目录中向上找到父 repo，将原始基线错误标为 `7186a7d`。已修正工具，只有目录自身存在 `.git` 才读取 revision；归档返回 null。冻结基线实际来自 `git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb`，且原始 worker 的 matcher 源码 SHA-256 与该提交逐字节一致。原始日志保留，审核快照附带显式更正说明。修改后 17 项本地测试通过，30 条 CPU 案例仍为原始 14/30、新实现 30/30；这项修正没有改变导航或模型匹配逻辑。

## 确切配置与复现

实际运行的命令数组、环境变量、提交、开始/结束 UTC 时间、系统退出码与显存采样写入 `qwen_service.resource.json`、`comparison.resource.json`、`replay.resource.json`、`navigation.resource.json`。对应 `.log` 和 `.gpu.jsonl` 为原始日志与逐秒记录。汇总快照见 [resources.json](assets/goal-matching-smoke-20261009/resources.json)。

在独立 clone 根目录，复用共享环境；先再次检查 GPU 0 和端口，不覆盖旧输出。下面是实际模型配置和比较命令，`R` 改用新目录便于复现：

```bash
source /data20t/embodied/share/inter-nav/env.sh
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R="grutopia/results/goal_matching_repeat_$(date +%Y%m%d_%H%M%S)"
mkdir "$R"
mkdir "$R/baseline-main"
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$R/baseline-main"

# 独立终端启动；完成后只关闭本次服务。
/data20t/embodied/wzj/internnav_data/envs/qwen35_caption/bin/python -u \
  grutopia/demo/serve_semantic_perception.py qwen-vl \
  --host 127.0.0.1 --port 12186 --device cuda:0 \
  --qwen-vl-model /data20t/embodied/wzj/internnav_data/checkpoints/Qwen3-VL-8B-Instruct

curl -fsS http://127.0.0.1:12186/health
"$ISAAC_PYTHON" grutopia/demo/evaluate_goal_matching.py \
  --dataset tests/fixtures/goal_matching/recorded_fridge_smoke.json \
  --baseline-root "$R/baseline-main" --record-dir "$R/live-comparison" \
  --live-qwen-url http://127.0.0.1:12186/classify --matching-repeats 20
```

模型配置为 max pixels 147456、类别生成预算 32 tokens、描述至少 64 tokens、非随机生成。描述门槛 0.80。比较工具是指标记录器，退出 0 不意味着每条案例正确。

模型退出后，实际最小导航命令为（本次外层 supervisor 强制 180 s 上限）：

```bash
"$ISAAC_PYTHON" -u grutopia/demo/go2_semantic_exploration.py \
  --scene programmatic --start 10 -2.6 0.40 --goal 12 -2.6 0.40 \
  --gpu 0 --perception-gpu 0 --qwen-device cuda:0 --headless \
  --detection-mode isaac --no-qwen --target fridge --goal-matching robust \
  --max-steps 480 --mapping-warmup-steps 80 --record-every 80 --rendering-interval 4 \
  --record-dir "$R/navigation"
```

下一步先用无标签 RGB 核查属性误匹配，再进行在线 open_vocab 描述导航。需要单独检查短场景控制器/路径跟踪及 CLI 退出码，并在新的独立输出目录中适当延长导航预算。此次没有通过改变起点或放宽到达/确认门槛补造成功结果。

审核明细：[单图对照](assets/goal-matching-smoke-20261009/live-comparison.json)、[八帧回放](assets/goal-matching-smoke-20261009/replay.json)、[导航结果](assets/goal-matching-smoke-20261009/navigation-summary.json)、[汇总](assets/goal-matching-smoke-20261009/summary.json)。
