# Open-vocabulary goal matching：前景掩码与地面过滤（2026-10-09）

[Draft PR #1](https://github.com/Zheng-Matt/inter-nav-project/pull/1)。本轮针对[上一轮报告](open-vocabulary-goal-matching-online-2026-10-09.md)中“地面框内有冰箱背景，Qwen 将背景当目标”的实测问题，复用已有 SAM 掩码隔离候选前景，并在已知平地场景用深度点过滤地面候选。两组真实模型裁剪比较均为 20/20，最小场景到达验证通过，但仿真日志有 GPU 设备编号错误；修正编号后的实机验证和完整房屋导航尚未完成。

## 实现与兼容范围

- robust + Qwen-VL 默认 `qwen_vl_crop_mode=masked`：将 SAM 掩码外像素置为中性 RGB `(127,127,127)`，保留候选框、0.15 边距、384 最大边长及前景颜色。复用已有分割结果，没有额外模型请求。`--qwen-vl-crop-mode context` 可做上下文裁剪对照；legacy 和 CLIP 路径保留原行为。
- 已知世界坐标地面高度时，robust + Qwen-VL 先检查候选有效深度点的高度：若第 95 百分位不高于 `ground_height + semantic_ground_clearance`，在分类、类别投票和目标证据前拒绝该候选。默认 clearance 为 0.02 m；设为 0 关闭，API 中地面高度未知 (`None`) 也不启用。GRScene 的人工平地顶部为 0.15 m，判断使用相对地面高度，不把世界坐标 0 当所有场景的地面。
- 过滤是几何判断，不使用仿真物体类别。拒绝记录含框、来源、原因；最终代码另记录高度 p95、阈值和有效点数。实际 GPU 运行发生在增加这三个诊断字段之前，因此原日志不含这些数字。
- 保留已有同义词规范化、描述的可见外观核验、可选 embedding 回退、置信度/候选差距配置和查询独立的跨帧证据。生产分数门槛 0.80、两帧选择、八帧且至少 60% 正证据确认均未放宽。
- 新增 `--physics-gpu` 可单独指定物理引擎 CUDA 编号，默认仍沿用原 `--gpu`。当 `CUDA_VISIBLE_DEVICES=3` 时，CUDA 中可见的设备编号是 0；渲染设备仍需单独按实际环境选择。这依据 [NVIDIA CUDA 环境变量文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)，修正后的 Isaac 运行仍待验证。

掩码本身不能消除所有误分类：本轮地面掩码的类别查询被模型称为 table，功能查询则拒绝。几何过滤防止平地候选成为 refrigerator 或 table 等任意目标的证据。低矮物体、地毯、错误地面高度、台阶以及分割/深度错误可能导致漏检，需关闭过滤或使用合适配置；尚未实现局部地形估计。

## 版本、保存位置与输入核验

独立分支 `improve-open-vocabulary-goal-matching`，独立服务器目录 `/data/zzx/inter-nav-goal-matching-smoke`。原服务器仓库 `/data/zzx/inter-nav-project` 未修改，保留 main `f7eab5948b30855776e3ce59b5a24d1ea6af5fdd`。GitHub main 基线为 `2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb`，没有合并。

| 阶段 | 确切提交 | 作用 |
| --- | --- | --- |
| 掩码输入生成 | `542217912af1254757b242abc2b6faf0e5dc29af` | 使用原裁剪 helper 和真实 SAM 服务，掩码外先置灰 |
| 开发集比较 | `07120287995f15f2448fd366c4908a335edc874b` | 评估器新增 masked/context 对照 |
| 保留帧、最小场景、完整房屋尝试 | `bd2bda95abc9827519ee1a1b5ad90ffd098b10ef` | 生产前景裁剪与地面过滤 |
| 修正设备编号后的启动尝试 | `dc6bff5b9532e93a66494f8f2cba45541bc0310c` | 可配置 physics_gpu；模型服务已停止，未进入仿真 |
| 最终源码回归 | `0411194617c6b98887ed23aae44b89aa38b2a609` | 仅增加几何过滤数值诊断字段，未重新运行 GPU 模型 |

完整原始日志、命令、显存采样、视频与运行脚本保存在双方独立 clone 的 `grutopia/results/goal_matching_mask_20261009_201020/`（忽略目录）。小型审核快照在 `docs/reports/assets/goal-matching-mask-20261009/`；4 张无标注 RGB、6 个上下文/前景裁剪、真实二值 SAM 掩码和人工真值在 `tests/fixtures/goal_matching/foreground_real_fridge/`。不覆盖旧结果，不将模型权重或运行视频提交到代码仓库。

[输入来源](assets/goal-matching-mask-20261009/mask-provenance.json)记录原始 RGB、二值掩码 SHA-256 和候选框。服务器 OpenCV 4.11.0 下，最终生产 helper 用相同 RGB + 记录的 SAM 掩码生成的六个裁剪，均与实际推理输入逐像素一致。macOS OpenCV 5.0.0 下，其中一个缩放裁剪有 5/157824 个颜色通道相差 1，其余一致；未替换实际实验输入。[核验记录](assets/goal-matching-mask-20261009/crop-verification.json)。

## 真实模型比较

使用同一真实 GRScene 冰箱实例、无标注 RGB，以及现有 MobileSAM vit_t / Qwen3-VL-8B-Instruct；模型权重、offline 环境和生成设置沿用上一轮。手工标签来自图像及资产核验，未用模型预测作为真值。开发帧 24/192，保留帧 96/240；每组 8 正例、12 负例，包括 refrigerator/fridge、功能描述、灰棕色属性描述、错误颜色、椅子、床旁关系和地面候选。

冻结 main matcher 由独立进程导入，源码 hash 与 `git show` 核对；baseline 的模型输入为原上下文裁剪，新方法输入为前景裁剪。比较同时包含之前的描述核验改进与本轮掩码，**不能将全部差值归因于掩码**。上一轮同组开发帧上下文 robust 为 18/20、误匹配 2/12，本轮前景 robust 为 20/20、误匹配 0/12；两轮独立请求，不是多次随机重复的控制实验。裁剪评估不运行几何过滤。

| 组别 / 方法 | 正确案例 | 正例召回 | 无目标误匹配 | 服务请求中位数 |
| --- | ---: | ---: | ---: | ---: |
| 开发帧 / 冻结 main + context | 15/20 | 4/8 | 1/12 | 239.7 ms |
| 开发帧 / robust + masked | 20/20 | 8/8 | 0/12 | 630.5 ms |
| 保留帧 / 冻结 main + context | 14/20 | 2/8 | 0/12 | 240.0 ms |
| 保留帧 / robust + masked | 20/20 | 8/8 | 0/12 | 698.6 ms |

服务错误均为 0；两组总墙钟时间分别 17.537 s、18.554 s。延迟包含 HTTP、图像处理、排队与推理，各类别/描述请求混合统计；增加约 2.6–2.9 倍，不能称为推理加速。warm matcher 中位数约 11–19 μs，样本不足以推断该微小差异的性能价值。

这两组均来自同一物体短轨迹。保留帧用于上一轮描述提示验证，本轮掩码选型参考了开发集，不能视为未见物体泛化测试。评估器设一帧即可选择/确认，以隔离单次匹配；不等同于生产八帧确认结果。[开发完整结果](assets/goal-matching-mask-20261009/development-mask.json)、[保留帧完整结果](assets/goal-matching-mask-20261009/heldout-mask.json)保留配置、每个查询/预测、原文和源码/图像 hash。

## 在线导航、异常和资源

最小场景沿用上一轮冰箱原几何/材质/位置及中性平地，起点 `(8.5,-1.76,0.55)`、目标 `a large appliance used to keep food cold`、1200 步上限和 1.1 m 到达阈值不变，不使用仿真语义兜底。DINO/SAM 与仿真使用物理 GPU 3，Qwen 使用 GPU 6；用户提及的 5/7 在本次检查时已经忙。

- **目标匹配与到达行为 PASS**：第 749 步、墙钟 93.690 s，24 条描述正证据，32 个 open_vocab 感知帧、0 感知失败，两次计划、一次重规划、0 规划失败。
- 19 次平地候选在模型分类前拒绝，32 次分类请求；最终没有上一轮地面产生的伪 refrigerator 节点。另有两次 cabinet 分类，目标描述均为 0 分。[候选分析](assets/goal-matching-mask-20261009/candidate-analysis.json)。上一轮 32 帧有 51 次分类，本轮减少到 32；不同设备/轨迹/录制频率使墙钟时间不可用作因果速度对照。
- 结束位置距接近点 1.0955 m，满足既有 1.1 m 规则，余量 0.0045 m；距物体中心 1.4905 m。验证接近点到达，不验证抓取或贴近物体中心。[summary](assets/goal-matching-mask-20261009/minimal-summary.json)。
- **存在设备配置错误**：`CUDA_VISIBLE_DEVICES=3` 配合旧 `--gpu 3` 使 PhysX 报 CUDA error 101 / invalid device ordinal，无法创建 CUDA context。主循环仍执行并到达，但没有证据确认物理引擎的实际回退模式；不能称为无错误 GPU 物理验证。
- 原始完整房屋从默认起点 `(2.1091,-0.780335,0.55)` 启动，同样遇到设备编号错误，尚无主循环/感知结果。诊断后主动停止本任务进程，实际 231.669 s、进程 -9；并非跑完或耗尽 1200 s 预算，结果 **未评估**。
- 加入 `--physics-gpu 0` 后的最小场景启动在 1.062 s 内因 SAM 12188 connection refused 退出 2；服务此前已被资源保护停止，未进入 Isaac，不能验证设备修正，也不是目标匹配失败。[启动 summary](assets/goal-matching-mask-20261009/preempted-summary.json)、[诊断摘录](assets/goal-matching-mask-20261009/runtime-diagnostics.txt)。修正后的完整房屋脚本保存但未执行。

最小运行整卡采样峰值为 GPU 3 **8805 MiB**、GPU 6 **17664 MiB**，包含模型、渲染和已有上下文，不等于各进程独立分配。资源检查起初发现少量共享上下文，已停止初次服务并改用外来进程显存门槛。后续其他任务在 GPU 3 分配超过 512 MiB 时保护机制停止本任务 SAM/Qwen；清理剩余 DINO 后本任务 GPU 进程为 0，未终止其他任务。完整命令数组、CVD、提交、时间、退出码和采样见 [resources](assets/goal-matching-mask-20261009/resources.json)，触发见 [preemption](assets/goal-matching-mask-20261009/resource-preemption.json)。没有训练、依赖安装或共享权重修改。

## 回归与复现

服务器最终 CPU 回归 **51/51 通过**：6 个新前景/地面测试、10 个 Qwen 感知测试、17 个 matcher/服务测试、13 个开放词汇感知测试、2 个原始 RGB 测试和 3 个运行记录测试。本地完整 suite 为 **164 项，162 通过，2 个既有环境错误**（缺 torch、缺 G1 profile），没有把它写成全量通过。[服务器输出](assets/goal-matching-mask-20261009/server-regression.txt)、[本地输出](assets/goal-matching-mask-20261009/local-regression.txt)。模块编译、CLI 配置与空白检查通过；新 physics_gpu 尚无成功 GPU 运行。

在独立 clone 根目录先检查 GPU/专用端口，以新的结果目录运行。下面是可复现流程，**不是声称已重新执行的命令**：

```bash
source /data20t/embodied/share/inter-nav/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
R="grutopia/results/goal_matching_repeat_$(date +%Y%m%d_%H%M%S)"
mkdir "$R"
mkdir "$R/baseline-main"
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$R/baseline-main"
# 独立 Qwen 服务已在 12186 启动且健康后，使用已保存的真实输入：
"$ISAAC_PYTHON" grutopia/demo/evaluate_goal_matching.py \
  --dataset tests/fixtures/goal_matching/foreground_real_fridge/development.json \
  --baseline-root "$R/baseline-main" --live-qwen-url http://127.0.0.1:12186/classify \
  --qwen-vl-crop-mode masked --record-dir "$R/development"
"$ISAAC_PYTHON" grutopia/demo/evaluate_goal_matching.py \
  --dataset tests/fixtures/goal_matching/foreground_real_fridge/heldout.json \
  --baseline-root "$R/baseline-main" --live-qwen-url http://127.0.0.1:12186/classify \
  --qwen-vl-crop-mode masked --record-dir "$R/heldout"
# 可选：独立 SAM 12188 健康后，从提交的原始 RGB 重建掩码输入：
"$ISAAC_PYTHON" docs/reports/assets/goal-matching-mask-20261009/build-masked-dataset.py "$R"
```

以 `--qwen-vl-crop-mode context` 可复测当前 robust 提示的上下文对照。原型生成脚本及实际命令在原始结果目录保留；报告 helper 经过整理，默认使用提交的 4 张 raw RGB，并调用已核验相同的生产掩码 helper。SAM 服务需单独启动，不能调用他人服务。

模型启动、共享 assets、单资产场景构建沿用[上一轮复现步骤](open-vocabulary-goal-matching-online-2026-10-09.md#可复现流程)，模型服务内部 `cuda:0` 对应各自 CVD 的物理卡。本次真正执行的导航命令与 profile 路径在 resources 中逐项保存。下一轮在独立空闲卡上先验证 `CUDA_VISIBLE_DEVICES=3`、`--gpu 3 --physics-gpu 0` 的 Isaac 初始化，再重复最小场景，最后从原始起点跑完整房屋；具体物理编号按当时资源替换。仍需多物体/多房间和负例验证、低物体与非平地评估，以及同设备重复延迟测量。
