# Open-vocabulary goal matching：实现与验证（2026-10-09）

本次实现已经完成 CPU 验证，并准备为独立分支 `improve-open-vocabulary-goal-matching` 建立 draft PR。用户在执行过程中要求显卡繁忙时先改代码、稍后测试，因此本报告不包含新的真实模型推理或 Isaac 导航实验，也不宣称真实目标准确率或导航成功率提升。

## 现有调用链与基线

基线为 GitHub `main` 的 `2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb`。在独立本地副本中工作；服务器 `/data/zzx/inter-nav-project` 当时存在三条未推送提交，保留该工作目录的现状。

调用链为：

1. `grutopia/demo/go2_semantic_exploration.py` 读取目标与运行配置。
2. `go2_navigation_runner.py` 创建感知、探索组件及运行记录。
3. `OpenVocabularyPerception`：RGB → DINO 候选 → SAM mask / RGB-D 世界坐标 → Qwen-VL 类别，或旧 CLIP 图像 embedding。
4. `mapping_runtime.py` 按帧融合检测；`SceneGraphMap` 按物体关联位置、类别票数和可选 embedding。
5. `SemanticExplorationComponent._best_target_node()` 选目标，`target_confirmed` 核查证据，随后选择可达接近点；`evaluate()` 同时检查确认状态与到达距离。
6. `SemanticRunArtifacts`、`final_map.json` 和检测 JSONL 保存配置、证据和结果。

原始 Qwen 路径已经支持精确 `fridge → refrigerator` 等别名，不能把它描述成完全没有同义词支持。问题主要在于 CLIP 路径的双向子串比较、两条路径对完整描述的处理，以及旧 embedding 匹配缺少候选间隔门槛。Qwen 将目标字符串加进允许类别列表，但用 `white fridge` 请求、得到 `refrigerator` 类别时仍无法确认完整描述。CLIP 则可能仅凭 `refrigerator` 出现在 `white refrigerator` 中就确认，忽略颜色。

## 研究与实现选择

[CLIP 官方实现](https://github.com/openai/CLIP)使用图像与文本特征的余弦相似度。本次复用现有 CLIP encoder，补足向量归一化、非有限/零向量检查与第一、第二候选的间隔控制；保留每任务一次的文本特征缓存。没有把余弦分数当成概率。

[Qwen3-VL 官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)给出同一请求中输入图像和文字的调用方式。本次选择在已有红框上下文 crop 的分类请求中同时核验目标描述，复用已部署模型，无需额外 encoder 或逐候选第二次推理。DINO 仍负责候选定位；其自然语言查询能力来自[官方 GroundingDINO 实现](https://github.com/IDEA-Research/GroundingDINO)，候选分数不充当目标身份确认。

实现集中在已有感知 → 场景图 → 目标确认链：

- `goal_matching.py` 区分类别查询与描述查询，统一大小写、空格、精确别名和常见请求前缀；支持 `|` 类别备选。多词未知短语保守按描述处理，不自动丢弃属性。
- 类别目标以精确类别/别名匹配替代任意子串。旧 DINO 的 `refrigerator door` 标签采用明确兼容规则；没有泛化成任意包含 refrigerator 的标签。
- 描述保留完整文本。Qwen HTTP 请求增加可选 `target_query`，响应增加 `goal_match_score` 和 `goal_verification_status`。类别与描述满足程度分别判断；`unknown`、非法 JSON、布尔/字符串/NaN 分数、无法从 crop 验证的属性或房间关系都不能提供正证据。
- `SemanticDetection` 增加可选的查询和分数字段。场景图按“物体节点 + 查询”累计核验分数，一个捕获步最多计一次；同帧相冲突分数保守取低值。混合检测保留 Qwen 核验来源，不让 Isaac 类别票自动成为属性证据。
- 描述默认至少两次正核验才能作为目标接近；最终确认需要至少八次分数达标的核验，且达标占比至少 60%。类别原有重复标签确认规则保留；描述不使用这个绕过比例的规则。
- 缺少描述响应字段的旧服务继续产生类别检测和观察候选，但不能确认描述。关闭描述核验同样不会退化成“忽略属性”的成功条件。CLIP embedding 默认仍仅用于临时观察，不能单独确认完整描述。
- 查询类型、核验支持数、embedding 分数/间隔/失败类型进入统计。类别证据与描述分数分别保存到 `final_map.json`；检测 JSONL 保留原始模型输出及每框核验状态。

## 配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--goal-matching` | `robust` | `legacy` 保留原始匹配路径，用于对照 |
| `--description-verification` | 开启 | 完整描述核验；可用 `--no-description-verification` 关闭 |
| `--goal-match-confidence` | 0.80 | 接受 Qwen 描述分数的门槛 |
| `--goal-node-confidence` | 0.10 | 目标节点最低检测置信度 |
| `--goal-embedding-threshold` | 0.24 | CLIP 临时目标最低余弦相似度 |
| `--goal-embedding-margin` | 0.03 | 至少两个候选时，第一名与第二名的最低间隔 |

运行 manifest 会记录嵌套 `GoalMatchingConfig`，Qwen 健康响应新增 `goal_verification` 能力字段。类别请求保持原来的生成预算；描述请求生成预算至少 64 tokens，以容纳两个 JSON 字段。

## 已完成验证

环境：macOS arm64、Python 3.13.13；NumPy 2.4.4、SciPy 1.18.1、requests 2.34.2、OpenCV headless 5.0.0.93、Flask 3.1.3、Pillow 12.2.0。

- 新增 17 项 CPU 检查通过，覆盖别名、属性/功能描述、请求/响应、同类别不同实例、重复帧、查询隔离、冲突融合、embedding 缓存/无效值/间隔、旧服务回退、八帧描述确认、导航接口及最终证据保存。
- 全量 unittest：156 项中 154 通过，2 个环境错误。原始 main 在同一环境为 139 项中 137 通过，出现相同错误：缺少 `torch` 导致 Go2 policy 测试模块不能导入，缺少 `grutopia/demo/profiles/g1_grscene_mv7_navigation.json`。未为本次匹配修改补造该场景文件。
- 修改模块编译检查、差异空白检查通过；导航 CLI `--help` 已检查，新参数可见。没有启动 Isaac 或模型。

### 手工接口案例对照：不是模型准确率

`tests/fixtures/goal_matching/contracts.json` 含 30 个人工设计案例：18 个存在目标、12 个无可确认目标。类别和 VLM 分数是预设输入。比较工具在独立进程中直接导入原始 main 的代码，避免用新实现模拟原始基线。评估配置采用一次观察即可选取/确认，仅隔离匹配规则；生产配置仍为两次选择、八次确认。

| 规则层指标 | 原始 main | 新匹配 |
| --- | ---: | ---: |
| 正确案例 | 14/30 | 30/30 |
| 存在目标时正确匹配 | 7/18 | 18/18 |
| 无目标时误匹配 | 5/12 | 0/12 |
| 温热匹配开销中位数 | 1.750 μs | 2.000 μs |
| 温热匹配开销 P95 | 3.708 μs | 3.208 μs |

这些数字只能说明接口规则在设计案例上的表现。它们不包含真实 Qwen/CLIP/DINO 推理，不评估描述分数可靠性，也不是目标导航准确率。微秒级开销只测 1–2 个物体节点上的匹配方法，排除了构图、网络与模型推理；不能推断端到端延迟。

运行明细见[CPU 对照快照](assets/goal-matching-20261009/cpu-comparison.json)。原始运行文件位于忽略目录 `grutopia/results/goal_matching_cpu_20261009_02/comparison.json`。数据 SHA-256 为 `a976b443a5f167529ef257ec4ab279f145e30b922db465e8e89008d048686012`。

## 保存路径与 CPU 复现

代码使用自己的独立 clone；共享 conda、模型和素材目录遵循[实验室环境规范](../en/get_started/lab-shared-environment.md)，不向共享环境安装依赖。测试样本在 `tests/fixtures/goal_matching/`，说明在 `docs/reports/`，报告用的小型快照在本报告的 `assets/` 子目录。原始输出、视频、大型图像与模型不进提交，放在 repo 的 `grutopia/results/<独立运行名>/`；已有运行目录不覆盖。

在 repo 根目录，用隔离环境执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/goal-matching-test.txt
.venv/bin/python -m unittest discover -s tests -p 'test_goal*.py'

baseline_dir=$(mktemp -d /tmp/inter-nav-main.XXXXXX)
git archive 2a629e91e0e25dcb9aeb4ff5ee126b397d9ebedb | tar -x -C "$baseline_dir"
.venv/bin/python grutopia/demo/evaluate_goal_matching.py \
  --baseline-root "$baseline_dir"
```

工具默认创建 `grutopia/results/goal_matching_<时间戳>/comparison.json`，可显式传 `--record-dir`，但必须是尚不存在的目录。输出包含案例预测、分母、完整 matcher 配置、数据 hash、来源和 Python 版本。独立归档没有 `.git` 时 revision 字段为空；冻结基线的 commit 由上面的 `git archive` 命令确定。

## GPU 空闲后的真实实验流程（尚未执行）

服务器 SSH 已安全认证并完成只读预检；三个既有服务当时就绪，但显卡显存普遍紧张。根据用户后续指示，本次没有追加模型请求、启动新服务或仿真，也没有修改原服务器工作目录。

1. 在自己的分支副本中加载 `/data20t/embodied/share/inter-nav/env.sh`。使用已有 Qwen-VL 独立解释器和本地权重，等 GPU 空闲后启动这次代码的 Qwen-VL 服务。可用独立端口 12186，并向导航传 `--qwen-vl-url http://127.0.0.1:12186/classify`；先检查健康响应 `goal_verification=true`。不要直接重启别人的服务或修改共享环境。
2. 从现有录制的 RGB 和框标注制作人工核验的红框上下文 crop，保持与 `_qwen_context_crop()` 一致的边距、RGB 顺序和尺寸。正/负样本包含 fridge/refrigerator、功能描述、颜色匹配/冲突、同类不同物体、电视/显示器、不可见房间关系和遮挡。对每个查询独立标注应匹配的候选 ID，不能把模型输出当作真值。
3. 在同一服务、相同图像和配置上运行 `evaluate_goal_matching.py --dataset <人工标注文件> --baseline-root <冻结main> --live-qwen-url http://127.0.0.1:12186/classify`。它分别请求原始类别分类与完整描述核验，报告匹配正确率、无目标误匹配率、precision、服务错误、匹配开销和请求中位延迟；保存图像 hash 与完整模型输出。真实样本文件中每个候选需有相对文件路径 `crop`。该模式只接受 Qwen 案例，且评估一次观察，不代表多帧导航确认。
4. 再以同一 GRScene profile、各自独立输出目录，比较 `--goal-matching legacy` 与 `robust` 的完整导航。先使用 `--no-qwen` 固定前沿规则，再单独评估文字 Qwen 前沿策略。每目标至少多次运行，记录随机性/服务状态；同时比较匹配正确率、错目标到达、导航成功、路径长度和请求延迟。main 当前未固定仿真随机种子，单次结果不能作显著性结论。

人工标注文件格式示例（实际 crop 需先保存并人工核验）：

```json
{
  "schema_version": 1,
  "provenance": "human annotations of recorded scene crops",
  "cases": [{
    "id": "white-fridge-scene-01",
    "query": "white fridge",
    "classifier": "qwen-vl",
    "expected": "white",
    "candidates": [
      {"id": "white", "label": "refrigerator", "crop": "crops/white-fridge.png"},
      {"id": "black", "label": "refrigerator", "crop": "crops/black-fridge.png"}
    ]
  }]
}
```

## 未验证与风险

- Qwen 的描述分数是模型自报判断，未经校准；0.80 是可配置初值，需真实正/负数据检验，不能解释成 80% 的统计准确率。
- 描述路径仍受现有分类词表约束。没有列出的对象若被判为 unknown，就不能确认；新颖多词类别、语言解析及更大开放词表尚需扩展和实测。
- 红框 crop 保留有限上下文，房间身份、远距离空间关系、材质或隐藏属性可能无法验证；此时应继续观察/探索。
- 属性证据依赖既有物体关联（0.75m 距离和类别）。非常接近的同类实例仍可能合并，暂未重做实例跟踪。
- 描述请求增加文本与生成预算，是否改善或损害模型分类、是否超时、端到端延迟及真实导航表现均待 GPU 实验。
- 尚未在完整 Isaac/模型环境中验证此次运行代码；已有 CPU 检查与模拟导航接口不能替代真实实验。
