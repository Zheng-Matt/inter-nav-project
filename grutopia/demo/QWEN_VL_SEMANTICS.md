# Qwen-VL 物体分类接入（2026-09-27）

来源：[Kitty051201 / why-tvtest-20260927](https://github.com/Kitty051201/inter-nav-project/tree/why-tvtest-20260927)，参考提交 `0514c06`。本次按当前项目结构移植视觉分类链路，保留已有终点节点与实验记录工作。

## 原来有什么问题，如何改进

| 原来有什么问题 | 如何改进 |
| --- | --- |
| DINO 提示词及其输出短语承担物体类别判断；矩形屏幕候选可能对应电视、显示器、画框或镜子。 | DINO 只提供候选框，MobileSAM 提供 mask，Qwen3-VL-8B-Instruct 对框内物体分类。类别为 unknown 时记录判断，禁止写成目标证据。 |
| 环境词与目标词混在一个长提示词中；目标框可能被大量背景候选挤占。 | 对同一 RGB 分别做环境和目标两次 DINO 查询。目标 television 使用 `monitor . screen`；这两个词只生成候选，最终类别由 Qwen 判断。 |
| 对分割后去掉背景的裁剪分类，会损失上下文。 | Qwen 读取保留真实背景、增加 15% 边距并画红框的 RGB 裁剪；mask 与 RGB-D 仍负责物体世界坐标。裁剪最长边限制为 384。 |
| 重叠候选增加推理开销，同一帧可能反复增加节点证据。 | 目标候选优先，IoU ≥ 0.8 去重，最多分类 12 个候选。Qwen 来源同一节点同一帧最多计一次。 |
| 不同类别可能因图像特征相似而被合并。 | Qwen 类别仅按明确类别与精确别名关联节点；monitor 不等于 television。保留旧模式的标签融合逻辑。 |
| 视频只显示 DINO 短语，无法看见 Qwen 实际判断。 | 同步显示 `DINO 标签 -> Qwen 类别`，当前查询目标标红。现有 `groundingdino_detections.jsonl` 增加 classified：包含类别、模型、输出文本及是否作为地图证据。unknown 也记录。 |

## 与当前终点节点工作的关系

本次沿用当前终点标签累计证据要求和到达距离阈值。**没有加入接近目标后再收集两帧的复核门槛，也没有移植新的不可达切换机制。** Qwen 分类决定每帧的物体类别，既有终点逻辑负责判断证据是否足够、机器人是否到达。

原有文本 Qwen3-8B 仍负责探索前沿评分；新增 Qwen3-VL-8B-Instruct 服务专门看图分类。`--no-qwen` 关闭前者，不会关闭视觉分类。

## 启动

DINO 与 MobileSAM 继续使用现有 12181、12183 服务。Qwen-VL 使用独立 Python 环境，避免升级旧感知环境影响 DINO。该环境必须能够导入 `transformers.Qwen3VLForConditionalGeneration`，并已安装本地 Qwen3-VL-8B-Instruct 权重；服务不会自动下载模型。模型用法参考[官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

```bash
# 将变量设为实际的独立环境与本地权重目录
$QWEN_VL_PYTHON grutopia/demo/serve_semantic_perception.py qwen-vl \
  --qwen-vl-model "$QWEN_VL_MODEL" --device cuda:3 --port 12185

curl -fsS http://127.0.0.1:12185/health

$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --detection-mode open_vocab --semantic-classifier qwen-vl \
  --target television --gpu 0 --qwen-device cuda:2
```

默认演示使用 `open_vocab` 与 `qwen-vl`。启动前检查三个模型服务。显式选择 `--semantic-classifier clip` 可使用原有 DINO 标签与 CLIP 特征链路。`--detection-mode isaac` 保留纯真值模式。

## 记录与验证边界

继续使用既有独立运行目录、run manifest、summary、events、trace、视频帧索引及目标地图标记；分类结果写入现有检测 JSONL，控制台不逐框输出。

测试覆盖两个提示词、目标优先去重、mask 三维坐标、类别来源、unknown 记录及过滤、限频不重放旧结果、服务预检查、不同类别不混合及终点原有证据要求。均为离线接口测试；尚未运行真实 Qwen-VL 权重或新的 Isaac 导航实验，不能据此宣称导航成功率提升。
