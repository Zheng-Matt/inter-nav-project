# Open-vocabulary goal matching：GPU 5/7 续测与在线验证（2026-10-09）

[Draft PR #1](https://github.com/Zheng-Matt/inter-nav-project/pull/1)。最终核验提示的最小真实资产在线导航 **PASS**：第 767 步到达，23 次描述正证据，32 次在线感知无失败。完整房屋场景未完成初始化；本报告不把单物体 smoke 等同于完整导航基准。

## 代码、路径与资源隔离

- 独立分支：`improve-open-vocabulary-goal-matching`；服务器独立 clone：`/data/zzx/inter-nav-goal-matching-smoke`。
- 最初续测运行提交：`0eeb29b006bee4c6eefb823a060892a164a6e582`。
- 最终视觉核验/导航运行提交：`b9c732bc4da8ffd2aa4c00b74e07522093343f32`；其后的 `168d622` 只更新测试中的生成预算契约，未修改运行逻辑。
- 冻结原始匹配器：`2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb`。裁剪比较工具独立进程导入该归档，matcher 源码 SHA-256 已与 `git show` 逐字节核对。归档无 `.git`，日志中的 `baseline_revision=null` 正确；快照另记录冻结提交。
- 原服务器 `/data/zzx/inter-nav-project` 保持 `f7eab5948b30855776e3ce59b5a24d1ea6af5fdd`，没有改动其 main；原有未跟踪 assets 链接保留。
- GPU 0 仿真；GPU 5 独立 DINO/SAM；GPU 7 独立 Qwen3-VL。分别设置 `CUDA_VISIBLE_DEVICES=0/5/7`，模型进程内设备为 `cuda:0`。专用端口为 12187、12188、12186；未使用其他任务的模型服务。
- 模型、素材、环境复用共享目录；未安装依赖、训练或修改共享资源。原始输出保存在 `grutopia/results/goal_matching_experiment_20261009_152404/`，本地也保存完整副本。报告小型快照在本报告的 assets 子目录；可复测裁剪在 `tests/fixtures/goal_matching/clean_real_fridge/`。已有输出目录不覆盖。保存的子场景复现 helper 另增加已有文件检查；这是文件保护，不改变已运行场景。

新增可配置 DINO/SAM URL 和默认关闭的 `--record-raw-rgb`。原始 RGB 在映射采样步写入 `raw_rgb/frame_XXXXXX.png`，不带检测框、类别或分数，已有帧拒绝覆盖；旧录制行为仍可使用。后续核验提示要求先生成 `observed_object`（可见类别、主表面颜色），再判断完整目标；描述生成上限至少 128 tokens，普通类别仍为 32。结果摘要使用组件的实际匹配方式，修正描述核验被记成 lexical 的问题。没有调整 0.80 分数门槛、两帧选择、八帧确认或到达距离。

## 无标注 RGB 的真实模型比较

同一真实 GRScene 冰箱实例，两个开发帧（24、192），人工标注 8 个正例、12 个负例。使用生产 `_qwen_context_crop()`：保留真实上下文、默认 0.15 边距，只添加候选框的一像素红边。正例覆盖 refrigerator、fridge、功能描述、灰棕色描述；负例覆盖红/蓝/白色、椅子、床旁关系，以及主要覆盖地面的候选框。真值由图像和资产核验确定，未使用模型预测作为标签。

比较配置是一帧即可选择/确认，专门隔离匹配器；不是生产八帧确认结果。

| 开发集指标 | 冻结原始匹配器 | 初版描述核验 | 最终描述核验 |
| --- | ---: | ---: | ---: |
| 正确案例 | 15/20 | 15/20 | 18/20 |
| 正例正确匹配 | 4/8 | 8/8 | 8/8 |
| 无目标误匹配 | 1/12 | 5/12 | 2/12 |
| 请求延迟中位数 | 234.1 ms | 473.6 ms | 637.6 ms |

原始列为最终提示复测中的类别请求（初次原始请求中位数 244.8 ms）。最终开发集测量时导航已结束；初版测量与 legacy 导航同时进行，延迟存在请求排队影响。延迟包含 HTTP、图像处理、排队与推理，是全部请求中位数，不是纯描述推理速度。

初版真实模型将红色描述判为 1.0/0.9、白色描述一次判为 0.9。正确的功能描述也得到相同高分，提高分数门槛不能区分。先生成可见外观后，本组颜色反例均得到 0.0；但是 frame 192 的地面候选仍被判为冰箱，类别查询和功能描述各误接受一次。因此新功能提高了描述召回，**没有证明其整体误匹配率优于原始方法**。

另用未参与提示调整的帧 96、240 和改写描述 `an appliance that keeps food cold`、`the fridge` 复测：

| 保留帧指标 | 冻结原始匹配器 | 最终核验 |
| --- | ---: | ---: |
| 正确案例 | 14/20 | 20/20 |
| 正例正确匹配 | 2/8 | 8/8 |
| 无目标误匹配 | 0/12 | 0/12 |
| 请求延迟中位数 | 282.3 ms | 757.0 ms |

这轮与最终在线导航共用本任务 Qwen 服务，串行服务可能排队，不能作为独立延迟结论。保留帧仍来自同一物体、同一短轨迹；不是未见物体、不同房间或独立数据集上的泛化测试。开发集用于调整提示，18/20 也不是无偏估计。

审核快照：[调整前](assets/goal-matching-experiment-20261009/development-before.json)、[调整后](assets/goal-matching-experiment-20261009/development-after.json)、[保留帧](assets/goal-matching-experiment-20261009/heldout.json)。其中保留所有查询、真值、预测、模型原文、图像 hash、配置和源码 hash。

## 在线导航和仿真结果

真实资产测试从原 GRScene 引用冰箱的原几何、材质和位置变换，使用独立 USD 子场景、标准平地和中性 DomeLight；不使用仿真语义标签兜底。源文件和 USD prim 的 hash/变换见 [provenance](assets/goal-matching-experiment-20261009/minimal-scene-provenance.json)。起点为 `(8.5, -1.76, 0.55)`，功能描述相同，每次最多 1200 步，原 profile 的到达距离为 1.1 m。记录 profile 的 benchmark goal 不作为在线目标选择输入。

| 同配置真实资产运行 | 结果 | 停止步 | 墙钟时间 | 描述正证据 | 在线感知/失败 |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy 模式 | FAIL，未确认目标 | 1199 | 126.767 s | 0 | 50 / 0 |
| 初版 robust | PASS | 767 | 103.808 s | 25 | 32 / 0 |
| 最终 robust | PASS | 767 | 114.243 s | 23 | 32 / 0 |

导航对照使用当前 runner 的 `--goal-matching legacy` 模式隔离旧匹配逻辑；并非运行整套冻结 main 的导航程序。裁剪比较才直接导入冻结 main。所有运行是单次轨迹，没有随机种子，也没有统计显著性结论。

最终运行接近点 `(10.15, -1.65, 0.5608)`，结束位置 `(9.1634, -1.1663, 0.5112)`，距接近点 1.0988 m，满足既有 1.1 m 规则，余量仅 0.0012 m；距物体中心 1.4636 m。它验证既有接近点到达定义，不表示贴近物体中心或抓取成功。生产两次选择、八次确认保留，目标节点记录 23 个正分数；两个计划、一次重规划、零规划失败。

最终图中仍有地面误检测产生的第二个 refrigerator 节点，中心高度约 0.153 m、只有一次正证据，没有达到选择/确认数量门槛。实际选中的 refrigerator 节点位于约 `(10.535,-1.678,0.700)`。数量门槛此次挡住了孤立伪候选，但重复的视觉错误仍可能确认错误目标，不能认为已消除背景混淆。

[最终结果](assets/goal-matching-experiment-20261009/online-final-summary.json)、[legacy 结果](assets/goal-matching-experiment-20261009/online-legacy-summary.json)。初版 summary 将描述路径写为 lexical 的记录错误已在最终代码修正，原始文件保留。

另外，上一轮程序化几何场景保持相同起点、目标和 0.7 m 阈值，只把最大步数从 480 延至 2400，实际第 909 步 PASS，171.184 s、GPU 0 峰值 4474 MiB。因此 480 步失败与步数预算不足相符；该实验使用仿真类别，只验证运动/规划，不验证视觉匹配。

完整房屋场景在 300 s 外层预算内未完成初始化，317.906 s 后清理结束，进程 -9；尚无在线感知帧或主循环结果。不能判为目标匹配失败或成功，完整房屋导航仍未验证。

## 资源、异常与回归

最终在线实验每秒采样的整卡峰值：GPU 0 **4494 MiB**、GPU 5 **4062 MiB**、GPU 7 **17485 MiB**。这包含同卡上下文，不等于各进程分配，也不能捕捉所有瞬时峰值。仿真和本任务服务均已停止，GPU 0/5/7 恢复 **18/19/19 MiB，利用率均 0%**；未终止其他用户任务。精确命令数组、起止 UTC、提交、超时和显存采样保存在原始 `.resource.json` / `.gpu.jsonl`，小型汇总见 [resources](assets/goal-matching-experiment-20261009/resources.json)。

复用已有 Qwen3-VL-8B-Instruct、qwen35_caption Python、CapNav DINO/SAM 环境和 isaaclab Python，均 offline/local-files-only。具体模型路径、软件版本沿用 [GPU 0 smoke 环境记录](assets/goal-matching-smoke-20261009/model-environment.json)。模型生成不采样。

服务器最终回归：17 个目标/服务测试、2 个原始 RGB 测试、3 个运行记录测试全部通过，合计 22/22。本地同组也通过。完整本地 suite 为 158 项，156 通过，2 个既有环境错误（缺 torch、缺 G1 profile），与初版/冻结基线所述错误相同。差异空白与修改模块编译检查通过。测试输出见 assets 中的 server/local regression 日志。

Isaac 仍有 headless、传感器预热、非有限深度和插件关闭警告，没有 OOM 或此次运行 traceback。legacy 语义结果 exit_code=2，但外层观测到进程 0；CLI 源码确实执行 `sys.exit(main())`，原报告把原因归为没有传递返回值是不准确的。日志停在 Simulation App Shutting Down，返回值冲突的关闭原因尚未定位；自动判定必须读取最终 summary 和 arrival_verified，不能只用进程退出码。

## 可复现流程

独立 clone 根目录，检查空闲设备及专用端口；使用新的 `R`，避免覆盖旧结果。先按实验室规范挂载共享 assets，再准备原 profile 副本（本次只改 name/start），生成单资产子场景：

```bash
source /data20t/embodied/share/inter-nav/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R="grutopia/results/goal_matching_repeat_$(date +%Y%m%d_%H%M%S)"
mkdir "$R"
cp docs/reports/assets/goal-matching-experiment-20261009/near-start-profile.json "$R/near-start-profile.json"
"$ISAAC_PYTHON" docs/reports/assets/goal-matching-experiment-20261009/build-minimal-scene.py "$R"
```

三个独立服务分别在各自终端启动，只关闭本次进程：

```bash
CUDA_VISIBLE_DEVICES=5 "$QWEN3_PYTHON" grutopia/demo/serve_semantic_perception.py grounding-dino --port 12187 --device cuda:0
CUDA_VISIBLE_DEVICES=5 "$QWEN3_PYTHON" grutopia/demo/serve_semantic_perception.py mobile-sam --port 12188 --device cuda:0
CUDA_VISIBLE_DEVICES=7 /data20t/embodied/wzj/internnav_data/envs/qwen35_caption/bin/python grutopia/demo/serve_semantic_perception.py qwen-vl \
  --port 12186 --device cuda:0 --qwen-vl-model /data20t/embodied/wzj/internnav_data/checkpoints/Qwen3-VL-8B-Instruct
```

检查三项 `/health` 后，以下是最终导航的普通 CLI 等价命令；实际执行使用只记录动作的 diagnostic wrapper 和 240 s supervisor，它们一并保存于 assets，未修改动作或目标决策。`--no-qwen` 关闭文本探索后端，视觉分类仍由显式 Qwen-VL HTTP 服务执行。

```bash
CUDA_VISIBLE_DEVICES=0 "$ISAAC_PYTHON" -u grutopia/demo/go2_semantic_exploration.py \
  --scene grscene --profile "$R/minimal-profile.json" --gpu 0 --perception-gpu 5 --qwen-device cuda:7 \
  --headless --no-qwen --detection-mode open_vocab --semantic-classifier qwen-vl \
  --grounding-dino-url http://127.0.0.1:12187/gdino --mobile-sam-url http://127.0.0.1:12188/mobile_sam \
  --qwen-vl-url http://127.0.0.1:12186/classify --qwen-vl-max-candidates 4 \
  --target 'a large appliance used to keep food cold' --goal-matching robust \
  --max-steps 1200 --mapping-warmup-steps 80 --record-every 80 --record-raw-rgb --record-dir "$R/online-robust"
# 对照改成 --goal-matching legacy，并使用另一个尚不存在的 record-dir。
```

裁剪真值和 PNG 已随分支保存，可不重新仿真直接复测：

```bash
mkdir "$R/baseline-main"
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$R/baseline-main"
"$ISAAC_PYTHON" grutopia/demo/evaluate_goal_matching.py \
  --dataset tests/fixtures/goal_matching/clean_real_fridge/development.json \
  --baseline-root "$R/baseline-main" --live-qwen-url http://127.0.0.1:12186/classify --record-dir "$R/development"
# 以 heldout.json 和新的 record-dir 再测保留帧。
```

待完成：完整房屋初始化后的在线验证、多物体/不同颜色/独立场景数据集、遮挡与同类候选混淆、描述分数校准、背景候选聚焦或 mask 约束，以及 Isaac 关闭时语义失败和进程退出码不一致的定位。当前结果适合继续审核 draft，不支持直接合并或宣称一般导航性能提升。
