# 两次 GroundingDINO + MobileSAM + Qwen3-VL 语义分类

此版本所在项目：`/data/why/inter-nav-tvtest`。代码已修改，尚未运行测试、仿真或模型推理。

默认 Qwen-VL 模式下，每次实际感知更新对同一张 RGB 调用两次 GroundingDINO。
第一遍只用环境词，第二遍只用固定的目标搜索词；两次结果先合并，再去重、分割和分类。
Qwen 只负责物体分类，不参与生成第二次词表。

当前目标为 television（包括 TV 等已有同义词）时，两次 caption 分别为：

```text
context: chair . table . sofa . door . cabinet . bed . lamp . refrigerator . plant . person . picture . mirror . window . bookshelf . curtain . microwave
target:  monitor . screen
```

电视目标的第二遍不附加 television、tv 或 display，保持已完成的单图对照实验中的小词表。
目标词表定义在 `open_vocabulary_perception.py` 的 `TARGET_SEARCH_TERMS`：
television → monitor/screen；refrigerator → fridge/refrigerator；chair → chair/armchair/stool；sofa → sofa/couch。
其他目标默认用规范化后的目标名，不向 Qwen 请求扩展词。除电视外的固定组合尚未验证召回效果。
第一遍自动排除当前目标和第二遍搜索词；因此电视目标恰好使用上面的 16 个环境词。
这些搜索词不代表最终类别映射，monitor/screen 不会被直接改成 television。

流程：同一 RGB → GroundingDINO 环境/目标两遍候选框 → 合并、高重叠框去重和数量限制 → MobileSAM mask →
有效深度/三维点 → Qwen3-VL 上下文裁图分类 → 三维语义节点。
mask 仍用于三维定位；Qwen3-VL 的裁图保留真实背景，各边扩展 bbox 尺寸的 15%，
并用红框标出要分类的原始区域。不把 mask 外背景抹黑。

最终类别来自 Qwen3-VL 的受约束 JSON 输出，不把 screen/monitor 直接当作 television。
基础分类集合含电视、显示器、画、镜子、窗户以及常见家具，并加入规范化的当前目标类别。
unknown 或非法模型输出不写入语义地图；服务错误显式抛出，不回退到 DINO 类别或仿真标签。
默认 `--semantic-source model`。显式选择 mixed 会启用已有仿真语义混合模式。

## 开销控制

- 每次非缓存感知更新调用两次 DINO，仍共用原来的 12181 服务，不启动第二份检测模型。
- 若调用者显式设置 include_context=False，则只进行目标遍；旧的 clip 模式保持单遍。
- 缓存键包含两次 caption 和最终目标，修改目标后不会误用上次分类结果。
- Qwen-VL 模式不运行 CLIP 图像特征和标签复核。
- IoU >= 0.80 的高度重叠框去重；每轮最多分类 12 个候选（`--qwen-vl-max-candidates`）。
- 严格按 proposal_source 排序：target 遍先处理，再处理 context 遍；各遍内部按 DINO 分数降序。
- 若两遍框高度重叠，优先保留 target 遍；target 遍先占候选名额，剩余名额才给 context 遍。
- target 遍自身去重后若仍超过 12 个，则其内部按分数截断；普通环境框不会挤掉目标遍框。
- 裁图最长边 384；VL 服务图像像素预算 384×384，最多生成 32 个新 token。
- 相比单遍，DINO 请求数增加一倍，但 SAM/VL 在合并去重后才运行，仍受同一候选上限约束。
- 候选上限会降低低优先级物体的覆盖率，日志记录被省略的框；实际延迟尚未测试。
- DINO 和客户端置信度阈值没有调低。`confidence` 仍是 DINO 框分数，不是 VL 分类概率。

## 服务与参数（仅供之后手动启动，本次没有执行）

现有 GroundingDINO/MobileSAM 服务继续使用 12181/12183。新增视觉分类服务使用 12185。
它独立于原来的 Qwen3 文本探索评分器，`--no-qwen` 不会关闭视觉分类。

需要已有的 Qwen3-VL-Instruct 本地权重和支持 Qwen3-VL 的 transformers 环境。
默认使用本地模型目录 `/data20t/embodied/wzj/internnav_data/checkpoints/Qwen3-VL-8B-Instruct`；可用 `--qwen-vl-model` 显式覆盖。
加载使用 `local_files_only=True`，不会自动下载权重。已只读确认配置、索引和四个权重分片存在，未加载模型验证。
代码参照 [Qwen3-VL 官方模型示例](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

在选定可用 GPU 后，可手动启动服务（下面的变量需要自己设置）：

```bash
cd /data/why/inter-nav-tvtest
VL_MODEL="/data20t/embodied/wzj/internnav_data/checkpoints/Qwen3-VL-8B-Instruct"
"$VL_PYTHON" -u grutopia/demo/serve_semantic_perception.py qwen-vl \
  --host 127.0.0.1 --port 12185 --device "cuda:$VL_GPU" \
  --qwen-vl-model "$VL_MODEL"
```

`VL_PYTHON` 可指向支持 Qwen3-VL 的独立环境，不需要升级 Isaac Sim 环境。
服务的 `--vl-max-pixels`、`--vl-max-new-tokens` 可单独配置。

在原有导航命令中使用：

```text
--open-vocabulary --semantic-source model --semantic-classifier qwen-vl
--qwen-vl-url http://localhost:12185/classify
--qwen-vl-timeout 60 --qwen-vl-max-candidates 12
--no-qwen --target television
```

上述是参数片段，不是完整启动命令。旧的 `--label-refinement` 参数只作用于
`--semantic-classifier clip`，不会覆盖 Qwen-VL 的答案。

## 排查记录

`GDINO_REQUEST` / `GDINO_RESPONSE` / `GDINO_PARSED` 保留，并增加 pass=context/target。
`VL_CLASSIFIED` 记录来源 pass、原检测类别、最终类别、模型名、原始分类回答和框。
`SEMANTIC_DROP` 增加 duplicate_bbox、vl_candidate_limit、vl_unknown、empty_bbox 原因。
`label_refinement.jsonl` 保留 label_source、classifier_model、classifier_response、raw_label 和 final_label，
新增 proposal_source，标记最终记录来自 context 还是 target 遍。
两遍检测改造保留三维定位和严格标签匹配；后续导航保护改动见下节。

## 2026-09-24：目标切换与局部误识别保护（未运行测试）

本次仅修改 inter-nav-tvtest，不增加模型调用次数，不使用仿真标签来纠正模型。

1. `mapping_runtime.py` 记录每条路径对应的目标坐标。坐标变化立即丢弃旧路线；
   `semantic_exploration_component.py` 在目标节点改变时也主动清除路线，即使坐标相同。
   同一节点的位置相对上次规划累积移动 0.35 m 时更新接近点并清除路线。
   下一次移动动作重新规划，日志增加 `NAV_PLAN_INVALIDATED` 和 requested_goal。
2. 保留当前目标的同票优先权：候选观测次数相同，不再仅因 DINO 框分数略高而切换。
   到达接近点的既有成功距离内，先停止平移并转向目标。
   初版要求在到达、选中该节点之后，再有两个不同的新观测帧支持同一节点，才允许成功。
   初版两帧之间及成功时的最近证据最大间隔为 96 仿真步；离开到达范围、目标切换或位置重规划会重置确认。
   上述确认时机和证据有效期已由下文“接近途中确认”更新替代。
   240 步内无法确认则放弃本轮候选，冷却 480 步，并继续选择其他目标或探索。
   使用原 RGB 更新频率，不额外请求模型，但确认等待可能增加一次任务的总时长。
   这些默认值在 `SemanticExplorationConfig` 中设置，尚未进行运行验证。
3. `mapping.py` 对匹配到同一节点、同一 capture step 的多个框只融合一次，
   同帧先处理 DINO 分数较高者；重复使用缓存观测也不增加 observations、位置均值或分数均值。
   保留既有“同类别且 XY 距离小于 0.75 m”的关联规则，因此不能保证所有家具碎片都归为同一真实物体。
   mixed 模式的仿真观测由运行层补上当前 step；model 模式不引入仿真观测。

Qwen 分类后保留轻量检查，仅针对 monitor/television 的画面边缘细长框：

- 框触及画面边缘、长宽比至少 4，且长边覆盖对应画面尺寸至少 80%：
  `clipped_screen_strip`。只依据框的形状和画面裁切情况，不依据附近是否存在某类家具。

已按用户要求移除 monitor/television 与 refrigerator/freezer 的类别配对、框/mask 包含过滤、
相关阈值和父框日志字段。重叠本身不再触发暂缓入图，冰箱仍保留在正常检测与分类词表中。
后续改动应优先采用通用机制，不为提高单次测试结果添加未经说明的对象配对或场景特化规则。

上述候选保留原始模型标签和可视化框，但 `semantic_eligible=false`，不写入本帧语义地图证据。
因此“视频里仍有 monitor 框”不等于它已进入地图或通过导航确认。
`SEMANTIC_HOLD` 和 `label_refinement.jsonl` 记录暂缓原因，后者还记录三维中心；
`TARGET_GUARD` 记录目标切换、到达待确认、新帧确认和超时，最终 JSON 保存 target_events。

该规则不删除历史节点，也不全面修正矩形物体分类；真实但被遮挡/裁切的屏幕也可能暂缓采纳，
需要后续正常视角提供证据。固定形状阈值、重复观测均不能保证语义正确。
本次只进行了源文件差异与部署校验，没有运行测试、仿真、模型推理或语法编译检查。

## 2026-09-25：不可达候选退出（未运行测试）

按用户要求，本次不调整确认距离、两帧确认规则、240 步确认等待和 480 步确认失败冷却。
修改限于 semantic_exploration_component.py、mapping_runtime.py 和本说明。

- 接近点必须已观测、空闲，并在膨胀栅格上与机器人连通。连通性使用与 A* 相同的八邻域，
  保留 A* 对未知通行格的既有策略；机器人落入膨胀格时沿用规划器的最近空闲起点处理。
  这是栅格可达性预筛，不保证实际执行一定能到达。
- 保留原接近距离和搜索范围，在该范围内挑选连通的接近点。找不到则记录
  `navigation_unreachable`，排除本轮候选，下一步选择其他目标；没有候选时继续探索。
  不再把未验证的原始期望点作为兜底目标。
- 一次完整规划（Voronoi 及 A* 回退）失败即更换接近点，备选点距已经失败的点至少 0.30 m。
  同一次候选导航最多失败 3 个接近点；没有不同的可用点时可以提前退出。
- 有路径但停滞也计为一次接近点失败：默认 480 仿真步内，剩余路径未减少至少 0.20 m。
  重规划时重置路径长度基准，但只有机器人发生至少 0.20 m 位移才更新进展计时，
  防止原地重规划不断续期。到达后的确认仍完全使用原机制，不计入该行进监测。
  这组新参数不是确认失败冷却时间；它们位于 SemanticExplorationConfig，尚未运行调优。
- 不可达记录独立存入 `unreachable_targets`；节点、标签和观测次数保留，
  新增语义票数及时间流逝本身均不能恢复资格。每个既有 frontier_selection_interval
  （默认 160 步）最多复查一次：有距先前可达接近点集合至少 0.30 m 的新可达点，
  或节点位置变化至少原有的 0.35 m 且存在可达点，才解除本次几何排除。
  确认失败的既有冷却记录仍单独生效，本次不修订其重新准入规则。

日志沿用 `[TARGET_GUARD]`，新增：

- `navigation_attempt_failed`：原因 planning_failed / no_navigation_progress、尝试次数、失败接近点；
  规划失败还记录底层 PlanningError 的文本，便于区别无路、端点超界等情况。
- `navigation_unreachable`：本轮候选退出，附失败接近点列表和可达接近点数量。
- `navigation_candidate_reopened`：候选几何条件改善，记录 candidate_node 和恢复原因。

统计与最终 JSON 增加不可达候选摘要、当前接近点失败次数和最后进展步数；上述事件也保存到 target_events。
只增加 CPU 栅格连通性和路径进展检查，并缓存连通结果；没有新增感知模型调用，
没有引入物体类别配对或固定节点编号。该机制不纠正模型分类，也不保证重新选中的节点一定正确。
本次未运行应用、仿真、模型、测试或语法编译检查，仅阅读源文件差异并进行文件部署校验。

## 2026-09-25：接近途中确认，到达后成功（未运行测试）

此节记录初版；确认距离、触边处理及进度续期已由下文 2026-09-26 更新替代。

视觉证据与到达等待分开计时，避免接近后目标被画面裁切，丢失途中已取得的确认。

- 仍需选中当前目标之后的两个不同新观测帧。选中之前的历史 observations、
  选择当帧和缓存感知结果不计数。两帧完成前的最大间隔仍为 96 仿真步。
- 默认在机器人距物体节点的 XY 距离不超过 3.0 m 时采集确认。该值不是到达阈值；
  到达仍使用原 profile 的 success_distance，相对于已规划的接近点计算。
- 纯模型模式复用现有检测的 bbox、图像尺寸、三维位置、capture step 和标签。
  当前节点必须在此步获得新观测，且检测按同类别、最近 XY 距离小于 0.75 m
  关联到当前节点。仅有另一个同类别物体被识别不能确认当前目标。
  该关联沿用现有地图的空间假设，不是物体实例身份的真值保证。
- 确认框距任一画面边缘必须大于 2 像素；宽、高分别至少占图像对应尺寸的 2.5%，
  面积至少占图像面积的 0.25%。这些通用检查只影响成功确认，不删除语义节点、
  不修改分类结果或视频框，也不能保证模型分类正确。
- 两帧完成后，证据默认保留 960 仿真步；后续符合条件的新观测可以续期。
  到达原成功范围且证据仍有效即可成功，不要求在已经裁切的近景重新采集两帧。
  证据过期后重新收集两帧；离开到达范围仅重置到达等待，不清除有效视觉证据。
- 目标切换、位置变化达到既有 0.35 m 重规划阈值、接近点导航失败或候选退出，
  都清空确认，并要求变化之后的新证据。证据不能跨目标或跨失败重试沿用。
- 到达但没有有效确认时仍停止平移并朝向目标，沿用 240 步等待及 480 步冷却。
  本次不新增自动后退、相机俯仰或观察点规划，也不改冷却后的候选重新准入。
- mixed 模式仍允许其既有场景图语义作为确认来源；上述模型框质量检查用于 model 模式。
  model 模式没有引入仿真语义。没有增加模型调用或识别频率。

参数位于 SemanticExplorationConfig：target_confirmation_valid_steps、
target_confirmation_observation_distance、target_confirmation_min_box_area、
target_confirmation_min_box_span、target_confirmation_edge_margin。均为未经运行调优的默认值。

日志仍使用 TARGET_GUARD：fresh_confirmation 增加 phase=approach/arrival、bbox、
object_distance、valid_until；visual_confirmation_ready 表示视觉证据已齐，尚不代表到达；
confirmation_expired 表示证据过期；confirmation_observation_ignored 说明距离、框裁切、
框太小或空间关联不符等原因。arrival_pending 增加 visual_confirmed。
统计及最终 JSON 增加 target_visual_confirmed、target_confirmation_valid_until 和相关参数。

如果接近途中始终没有再次识别出目标，本机制仍不会成功；例如既往 plant 运行仅在
8400/8424 步识别后便再无 plant，新机制不能凭这两次选择前/选择当帧的记录直接判成功。
本次仅进行源文件差异检查和备份部署，没有运行应用、仿真、模型、测试或语法编译。

## 2026-09-26：放宽途中确认范围，并按接近进展保留证据（未运行测试）

- 采集新确认证据的默认 XY 距离范围改为距目标节点 1.0～5.0 m（包含边界），
  分别由 target_confirmation_min_observation_distance 和
  target_confirmation_observation_distance 控制。到达成功距离仍取原配置，未改变。
- 不再因任意一条边触及画面边缘而返回 clipped_box。普通触边框可参与确认，
  fresh_confirmation 会记录 box_touches_edge；边缘距离仍用 2 像素作标记。
- 保留过小框检查。对于触边、长宽比至少 4，且长边覆盖对应画面尺寸至少 80%、
  同时在长边方向被画面截断的片段，返回 clipped_edge_strip。这是统一的确认质量规则，
  不依赖目标类别；感知模块中原有 monitor/television 边缘细长框保护也保留。
  此次不改变模型标签、图节点、原始检测框或模型调用频率。
- 两个不同的新帧、96 步组帧间隔、当前同一目标的空间关联均保留。
  视觉确认本身仍不代表导航成功；必须到达原接近范围。
- 完成两帧确认后，仍有 960 步的有限有效期，但持续接近允许续期：
  相对于确认时固定的目标位置，机器人距离比已经达到的最佳距离再减少至少
  target_navigation_min_progress（现有默认 0.20 m），有效期更新为当前步 + 960。
  此动作只保留已完成的证据，不增加视觉观测次数或改写最后视觉观测步数。
  它允许进入 1 m 内、目标被裁切或临时识别消失后，继续使用仍有效的途中证据。
- 仅有时间流逝、原地抖动、在旧位置之间往返、目标中心小幅漂移或重规划，不会续期。
  已过期证据不能靠运动恢复，仍须重新取得两帧；新的合格识别可以刷新证据。
  目标切换、明显位置变化、导航接近点失败及候选退出仍按原逻辑清空确认。
- 增加 confirmation_progress_retained 日志，记录到证据位置的距离和新的 valid_until。
  confirmation_observation_ignored 现在附带实际 object_distance 及允许的距离范围；
  太近的原因为 inside_minimum_observation_range，超过上限仍为 outside_observation_range。

此次只修改 semantic_exploration_component.py 和本说明；未加入自动后退或相机俯仰，
也未修改确认超时、冷却、候选排序或不可达退出机制。新默认值未经过运行验证，
只完成源文件差异检查、原文件备份与部署后的哈希核对，没有运行测试、仿真或推理。

## 2026-09-26：探索阶段停滞退出（未运行测试）

- 语义目标退出后，前往 frontier 也接入导航失败处理：规划失败立即记一次失败；
  连续 frontier_navigation_stall_steps（默认 480）步无有效路线进展也记一次失败。
  复用已有路径剩余距离，进展阈值沿用 target_navigation_min_progress（0.20 m）。
  路线重规划本身不算进展；无可用路径时同样有 480 步限制。
- 失败后清除旧路线，下次更新重新选择边界。失败点在
  frontier_retry_cooldown_steps（默认 480）步内不参与选择，优先尝试其他点。
  沿用已有失败降分和 blacklist_failure_threshold（默认 2 次）加入黑名单。
  这不是判定整个方向永久不可达，而是本次运行中对该 frontier ID 的失败记忆。
  冷却结束仅重新取得候选资格；黑名单不会因时间经过自动清除。
- 每 160 步重复选中同一边界不会重置导航停滞计时。抵达边界后立即重新选择。
- 另设不随边界切换或重规划重置的探索位移计时：以累计 XY 位移至少 0.20 m
  更新锚点；连续 exploration_stationary_steps（默认 1440）步未达到该位移，
  返回 failure_reason=exploration_stalled。它检测原地停滞，不保证检测大范围往返。
- 没有可选边界时保留原地扫描和周期重选；连续
  exploration_no_frontier_steps（默认 960）步仍没有可选目标，返回
  failure_reason=no_available_frontiers。这只表示当前地图及筛选条件下无候选，
  不声称整个场景已经探索完。发现可选目标会结束这一段无候选等待。
- 上述位移/无候选计时只覆盖探索阶段；进入语义目标导航时重置。
  语义目标的标签、候选排序、确认距离、确认帧数和确认冷却均未修改。
- 新日志前缀 FRONTIER_GUARD，事件包括 selected、navigation_attempt_failed、
  scan_without_frontier、exploration_failed；记录边界 ID、位置、失败次数、
  retry_after、blacklisted 和退出原因。最终 JSON 的 semantic_exploration
  增加 frontier_events，统计增加探索退出状态。终止走原 runner 的结果与保存流程。

这些参数都是仿真步数，不是秒；max_steps=0 时停滞退出仍然有效。
只复用已有定位、路径和 frontier 候选，不新增模型调用或地图重建。
本次不处理误标签或物理卡住的根本原因，也没有自动后退动作。
仅静态审阅代码差异及部署哈希核对，未运行测试、仿真、推理或语法编译。
