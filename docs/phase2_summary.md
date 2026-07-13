# 第二阶段牙弓 Curve 预测与缺牙 Landmark 补全任务总结

## 1. 阶段任务背景

本阶段工作的核心目标是：基于 Teeth3DS+ / 3DS+ 全牙弓扫描数据，在存在模拟缺牙、拔牙或 landmark 缺失的情况下，利用深度学习模型从残余牙齿上下文中预测缺失牙位的关键几何点，并进一步生成符合牙弓形态的三维 curve。

该任务服务于数字牙科 CAD、义齿设计和自动排牙等下游场景。临床缺牙病例中，系统往往无法直接获得完整牙列的空间基准。如果能够通过 AI 补全缺失牙位的几何参考点，并生成稳定的牙弓 curve，就可以为后续的空间对齐、支架设计、排牙路径和正畸/修复规划提供几何约束。

本阶段代码共包含六个主要文件夹，阅读顺序及其角色如下：

| 文件夹 | 主要作用 | 阶段定位 |
|---|---|---|
| `curve` | 早期中心点预测和曲线生成方案 | 第一版可运行原型 |
| `curve_new` | 四 landmark 缺牙预测、固定参考曲线、误差归因分析 | 从中心点升级到 landmark 级别 |
| `curve_new_pipeline2` | Pipeline 2，两点式 curve anchor 定义 | 当前上牙弓效果较好的方案 |
| `curve_new_pipeline1` | Pipeline 1，前牙 Mesial/Distal，后牙 buccal cusp | 尝试用咬合面/颊侧尖点描述 curve |
| `curve_new_pipeline1_new_posterior` | Pipeline 1 后牙改进版旧实现，加入后牙 cusp 筛选与镜像补点 | 其中后牙镜像平面法向量获取方法有误，不作为最终依据 |
| `curve_new_pipeline1_new_posterior_new` | Pipeline 1 后牙改进版修正版，修正后牙镜像平面法向量获取逻辑 | 当前后牙 cusp 镜像补点说明应以该版本为准 |

这些文件夹的共同核心思想是：将完整牙弓映射为固定长度为 16 的 FDI 序列，利用 Transformer 在序列上下文中补全被遮挡的牙位，再基于预测/真实几何点生成牙弓 curve。

## 2. 统一数据表示与基础框架

所有版本都围绕固定 16 牙位序列展开：

- 上颌顺序：`18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28`
- 下颌顺序：`48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38`

每个牙位在序列中占一个 token。模型输入会包含：

- 当前可见几何坐标；
- 牙位是否缺失或被模拟遮挡的 missing flag；
- 上颌/下颌 jaw flag；
- 位置 embedding；
- jaw embedding，用于区分上下颌的解剖模式。

数据从三类文件中提取：

- `.obj` 牙齿网格；
- tooth label JSON，用于将网格顶点归属到 FDI 牙位；
- keypoint JSON，用于读取 Mesial、Distal、OuterPoint、InnerPoint、Cusp 等 landmark。

landmark 坐标会通过 KDTree 匹配到最近网格顶点，再利用该顶点 label 判断所属牙位。重复标注会做平均。无效、不完整或无法构造目标点的牙位会置零，并通过 mask 排除。

## 3. `curve`：中心点补全与早期曲线生成

`curve` 是本阶段最早的可运行方案。它不直接预测每颗牙的多个 landmark，而是先从每颗牙的 4 个核心解剖点计算几何中心：

- Mesial；
- Distal；
- OuterPoint / Buccal；
- InnerPoint / Lingual。

每颗牙的中心点定义为这些 landmark 的三维平均值，因此每个牙位只输出一个 `(x, y, z)` 坐标。模型输入形状近似为 `(16, 5)`：

```text
X, Y, Z, is_missing, jaw_type
```

训练阶段采用动态 masking：随机遮挡若干真实存在的牙位，将其坐标置零，并只在被遮挡牙位上计算 loss。这个设计使模型学习“根据残余牙齿上下文恢复缺牙位置”，而不是简单复读已知输入。

早期几何后处理使用二次多项式/抛物线拟合。`postprocess.py` 中将牙弓点投影到 `X-Y` 和 `X-Z` 平面，分别拟合二次多项式，再生成平滑的 U 形曲线。这个方法的优点是强行保证整体平滑，减少局部外弧波动；缺点是它对牙弓真实三维形态的表达能力有限，并且把牙弓简化为特定坐标方向下的函数关系。

此外，早期 `app.py` 中会把预测出的缺牙中心点加入 `final_centers_for_curve` 再拟合 curve。也就是说，用户选择不同缺牙 mask 后，曲线本身会随预测结果变化。这对交互展示直观，但从业务语义上会混淆“固定参考 curve”和“模型预测结果”。

## 4. `curve_new`：四 landmark 预测与固定参考曲线

`curve_new` 将任务从单中心点预测升级为四 landmark 预测。每颗牙不再只有一个中心，而是完整预测：

```text
Mesial, Distal, OuterPoint, InnerPoint
```

核心张量形状变为：

```text
(16, 4, 3)
```

模型输入维度对应变为：

```text
12 coordinates + 1 missing flag + 1 jaw flag = 14
```

模型结构仍然是 Transformer Encoder，但相较早期版本更完整：

- embedding size 为 128；
- 6 层 Transformer encoder；
- 4 个 attention heads；
- jaw embedding 独立注入；
- 输出为每个牙位 4 个三维 landmark。

训练 loss 也从单点误差扩展为多项约束：

- point Smooth L1 loss：直接约束四个 landmark 坐标；
- center Smooth L1 loss：约束四点平均得到的牙齿中心；
- intra-tooth geometry loss：约束四个 landmark 两两距离，保持牙冠内部几何结构。

`curve_new` 的一个关键改进是固定参考曲线规则。系统在 Streamlit 用户选择 mask 之前，先用所有初始可用 ground-truth 牙齿中心生成 reference curve。之后无论用户遮挡哪些牙，curve 都不再重算，也不会使用预测结果作为曲线 anchor。

这解决了早期版本的问题：预测点不应改变参考曲线本身。预测结果应该用于评估 AI 补全能力，而 reference curve 应该代表该病例原始完整牙弓的几何基准。

`curve_new` 还加入了误差归因分析。中心误差被分解为四个 landmark 误差向量在最终中心偏移方向上的带符号投影。这个分解满足四项贡献相加严格等于中心误差。分析结论是：单个样本中主导误差的 landmark 会随牙位、遮挡上下文、误差方向和标注噪声变化，不存在一个在所有样本中固定主导中心误差的 landmark。

## 5. 从四 landmark 到两点 curve anchor 的动机

实际牙弓 curve 并不一定适合由“牙齿中心点”单独表示。尤其在上下牙弓咬合关系中，上颌和下颌的目标 curve 并不是同一条几何线：

- 上颌 curve 更适合沿特定解剖侧点或接触侧点组织；
- 下颌 curve 更接近咬合接触或后牙功能尖相关的轨迹；
- 单纯牙齿中心点可能过于中性，无法表达上下牙弓咬合时两条曲线的实际差异。

因此，后续 pipeline 不再使用四 landmark 的中心作为 curve anchor，而是为每颗牙定义两个点，直接将这两个点作为 B-Spline anchor。这样每颗牙贡献两个有解剖意义的点，整条 curve 的 anchor 数从 `N` 个牙齿中心变为 `2N` 个点。

统一张量形状变为：

```text
(16, 2, 3)
```

模型输入维度变为：

```text
6 coordinates + 1 missing flag + 1 jaw flag = 8
```

训练、验证和 UI 都围绕两点展开：

- point loss：预测两点坐标；
- midpoint loss：约束两点平均值；
- geometry loss：约束两点之间距离；
- UI 指标：mean point error 和 midpoint error。

## 6. `curve_new_pipeline2`：当前上牙弓较优的两点方案

Pipeline 2 是当前更适合上牙弓 curve 的方案。它的核心不是修改模型，而是重新定义每颗牙的两个目标点。

牙齿被分为前牙和后牙：

- 前牙：上颌 `13,12,11,21,22,23`，下颌 `43,42,41,31,32,33`
- 后牙：其余磨牙与前磨牙

Pipeline 2 的点定义如下：

| 牙齿类型 | Point0 | Point1 |
|---|---|---|
| 前牙 | `(InnerPoint + Mesial) / 2` | `(InnerPoint + Distal) / 2` |
| 后牙 | Mesial | Distal |

这一设计的含义是：

- 前牙不直接使用 Mesial/Distal，而是用 InnerPoint 与近远中点的中点，使 anchor 更偏向内侧曲线；
- 后牙使用 Mesial/Distal，避免 Cusp 高度和咬合面尖点带来的不稳定；
- 两个点均带有近远中语义，因此可以稳定排序。

`app.py` 中的 `collect_curve_anchor_points()` 会根据 FDI 象限调整 Mesial/Distal 方向，使相邻牙齿接触侧在 anchor 序列中保持连续：

- 右侧象限沿远中到近中；
- 左侧象限沿近中到远中。

曲线生成使用 `generate_bspline_curve()`，它是参数化 3D B-Spline 插值，而不是早期抛物线拟合。该方法使用 chord-length parameterization，不假设牙弓一定是 `y=f(x)`，因此能适应扫描坐标系和三维姿态变化。

Pipeline 2 的优势在于：

- 目标点稳定，均来自较可靠的 Mesial/Distal/InnerPoint；
- 避免后牙 Cusp 点数量不一致、颊舌侧判定不稳定的问题；
- anchor 顺序有明确近远中语义；
- 对上牙弓 curve 已经能得到较优秀结果。

## 7. `curve_new_pipeline1`：基于前牙近远中点与后牙 buccal cusp 的方案

Pipeline 1 的目标是让 curve 更接近咬合或颊侧功能区域。因此它采用：

| 牙齿类型 | Point0 | Point1 |
|---|---|---|
| 前牙 | Mesial | Distal |
| 后牙 | 两个 buccal Cusp |

后牙的两个点通过 `Cusp` 和 `OuterPoint` 构造。代码会读取后牙的 Cusp 标注，并使用 `OuterPoint` 判断颊侧方向。早期说明中提到“选择距离 OuterPoint 最近的两个 cusp”，实际代码中进一步升级为基于投影的 buccal side 筛选：以牙齿 mesh center 到 OuterPoint 的方向作为 buccal axis，将每个 cusp 投影到该方向上，选取更靠颊侧的两个 cusp。

这个方案的问题主要集中在后牙：

1. 后牙 Cusp 标注数量不固定。有些牙可能只有一个可靠 buccal cusp，有些会混入 lingual cusp。
2. Cusp 位于咬合面，Z 方向高度起伏明显，不同牙位差异很大。
3. Cusp 的解剖语义不如 Mesial/Distal 稳定，两个点的顺序也缺少天然近远中语义。
4. B-Spline 是插值曲线，会穿过 anchor；一旦后牙 Cusp anchor 偏离，就容易导致局部曲线抖动或后牙段异常弯折。
5. 下牙弓后牙咬合面形态复杂，buccal cusp 的选择更容易影响整体 curve。

因此，Pipeline 1 的问题并不是 Transformer 主体结构造成的，而是后牙 anchor 定义本身存在不稳定性。模型即使学习到了合理上下文，也会被不稳定、不连续或语义不一致的后牙目标点牵制。

## 8. `curve_new_pipeline1_new_posterior_new`：后牙 Cusp 镜像补点修正版

`curve_new_pipeline1_new_posterior_new` 是对 Pipeline 1 后牙问题的补救版本。它保留 Pipeline 1 的整体思路：前牙用 Mesial/Distal，后牙仍试图使用 buccal cusp，但新增了后牙 cusp 筛选和合成逻辑。相比旧目录 `curve_new_pipeline1_new_posterior`，该版本修正了单 cusp 镜像补点时镜像平面法向量的获取方式；旧目录中通过 tooth mesh 和 buccal axis 估计 lateral axis 的方法是错误实现，不应继续作为方法说明依据。

该版本新增的关键函数包括：

- `_select_reliable_buccal_cusps()`：筛选两个可靠的 buccal cusp；
- `_select_most_buccal_cusp()`：当无法获得两个可靠 cusp 时，选择最靠颊侧的单个 cusp；
- `_estimate_lateral_axis()`：使用该牙的 Distal - Mesial 连线作为镜像平面法向量，并归一化为 lateral axis；
- `_mirror_single_buccal_cusp()`：镜像平面经过 tooth mesh center，法向量为 Mesial-Distal 连线；将单个可靠 buccal cusp 沿该法向量翻转，合成第二个 buccal cusp。

该方法的工程动机是：如果后牙只有一个可靠 buccal cusp，则利用该牙已有的 Mesial/Distal 标注确定近远中方向，把 tooth mesh center 作为镜像平面经过点，构造第二个对侧 buccal 点，避免该牙因点数不足而被排除。实现上，当只获得一个可靠 cusp 时，代码会额外要求该后牙同时存在 Mesial 和 Distal 标注；若缺少这两个点，则不进行镜像补点。

但是这个补救方案仍然无法完全解决后牙问题：

- 合成点依赖 mesh center、OuterPoint、Mesial/Distal 标注和单 cusp 选择质量；
- 镜像假设牙冠近似对称，但真实后牙形态并不总是满足；
- 如果 OuterPoint 或 Cusp 标注本身有噪声，合成点会继承甚至放大这种误差；
- 合成点不是人工真实 landmark，而是几何推断点，会引入额外定义偏差；
- 下牙弓后牙段对 Cusp 选择尤其敏感，局部错误会直接影响插值 curve。

因此，该版本更像是对 Pipeline 1 数据可用性的增强，而不是从根本上重新定义稳定的 curve anchor。它可以减少“后牙点不足”的情况，并且相对旧实现避免了 mesh-derived lateral axis 带来的镜像方向错误，但不一定能保证后牙 curve 质量稳定。

## 9. 模型训练策略对比

早期 `curve` 使用随机遮挡 1 到 3 颗牙。`curve_new` 使用随机遮挡 1 到 5 颗有效牙。后续 pipeline 版本进一步改为连续遮挡，模拟临床上连续缺牙更常见的场景。

Pipeline 训练策略：

- 每个训练样本随机选择连续有效牙位窗口；
- 遮挡数量 `K` 通常为 3 到 6；
- 至少保留 4 颗有效牙作为上下文；
- 如果样本无法满足条件，则不产生 masked loss；
- 训练集按 patient-level split 划分，避免同一患者上下颌泄漏到不同集合。

这种连续遮挡比离散随机遮挡更贴近任务目标，因为牙弓 curve 补全往往需要在连续缺失区间中恢复局部形态，而不是只补零散单牙。

## 10. Curve 生成策略演进

本阶段 curve 生成经历了三次明显变化：

### 10.1 早期：预测中心点参与抛物线拟合

`curve` 中，用户 mask 牙位后，模型预测中心点，预测点会进入 curve fitting。曲线会随预测结果变化。该方案适合演示“AI 补全后生成曲线”，但参考曲线语义不够稳定。

### 10.2 中期：固定 ground-truth 中心点 B-Spline

`curve_new` 中，reference curve 在 mask 前由所有 ground-truth 牙齿中心生成，不再受用户 mask 和预测结果影响。这明确区分了“真实参考曲线”和“预测结果”。

### 10.3 后期：固定 ground-truth 两点 anchor B-Spline

Pipeline 1/2 中，curve 不再由牙齿中心生成，而是由每颗牙的两个目标点生成。有效牙位按 FDI 顺序展开为 `(2N, 3)` anchor 序列，再输入参数化 B-Spline。

这一步是本阶段最关键的概念升级：curve 的质量不再只取决于模型预测误差，也高度依赖 anchor 的解剖定义是否稳定、连续、符合上下牙弓咬合需求。

## 11. 上下牙弓 Curve 差异与当前结论

用户提出的关键观察是：上下牙弓咬合时，两条 curve 实际上不同。因此，不能简单地用同一种“牙齿中心线”同时解释上牙弓和下牙弓。

结合代码和 pipeline 设计，当前可以总结为：

1. 上牙弓使用 Pipeline 2 的点定义已经能获得较优秀 curve。其原因是 anchor 主要来自 Mesial/Distal 与 InnerPoint 推导点，标注稳定，点序明确，局部曲线连续性较好。
2. 下牙弓使用 Pipeline 1 或 Pipeline 1 posterior 修正版时，后牙段仍存在难处理问题。根源主要是后牙 Cusp 点本身的标注、数量、颊舌侧筛选、点序和咬合面高度波动不稳定。
3. Pipeline 1 posterior 修正版的 Mesial-Distal 镜像补点能提高后牙点构造成功率，但没有从根本上消除后牙 Cusp 定义不稳定的问题；旧版 mesh-derived lateral axis 方法不作为有效方案。
4. 后牙问题不是单纯模型容量不足，而是 curve anchor 选择和解剖语义定义的问题。

因此，第二阶段最重要的技术认识是：牙弓 curve 预测任务中，“预测模型”和“curve anchor 定义”同等重要。一个稳定的 anchor 定义往往比复杂的后处理补救更关键。

## 12. 各文件夹核心贡献总结

### `curve`

- 建立了固定 16 牙位序列；
- 使用 4 landmark 平均得到牙齿中心；
- 实现了 Masked Transformer 缺牙中心点预测；
- 使用抛物线拟合生成平滑 U 形 curve；
- 完成 Streamlit + Plotly 3D 可视化原型。

局限：

- 只预测中心点，无法表达上下牙弓 curve 差异；
- 预测点参与曲线拟合，reference curve 不固定；
- 抛物线拟合表达能力有限。

### `curve_new`

- 升级为四 landmark 预测；
- 加入患者级划分、归一化、3D augmentation；
- 加入 point / center / geometry 多目标 loss；
- 固定 reference curve，不再由预测结果改变；
- 增加 landmark 对中心误差的投影归因分析。

局限：

- curve 仍基于牙齿中心点；
- 中心线难以表达上下牙弓咬合时不同的曲线目标。

### `curve_new_pipeline2`

- 将目标改为每牙两个点；
- 前牙使用 Inner-Mesial / Inner-Distal 中点；
- 后牙使用 Mesial / Distal；
- 使用连续缺牙 masking；
- 用 ordered two-point anchors 生成固定 B-Spline；
- 当前上牙弓 curve 结果较好。

局限：

- 对下牙弓是否最优仍需要独立验证；
- 点定义偏向近远中/内侧参考，不一定完全覆盖下牙弓功能尖轨迹。

### `curve_new_pipeline1`

- 前牙使用 Mesial / Distal；
- 后牙尝试使用两个 buccal cusp；
- 目标更接近咬合面或功能尖相关 curve；
- 引入后牙 Cusp 与 OuterPoint 的颊侧筛选逻辑。

局限：

- 后牙 Cusp 不稳定；
- 点数、点序、颊舌侧判断存在不确定性；
- B-Spline 插值会放大异常 anchor 对局部曲线的影响。

### `curve_new_pipeline1_new_posterior_new`

- 在 Pipeline 1 基础上增加后牙可靠 buccal cusp 筛选；
- 对单个可靠 cusp 的后牙进行镜像补点，镜像平面经过 tooth mesh center，法向量使用 Mesial-Distal 连线；
- 旧目录 `curve_new_pipeline1_new_posterior` 中通过 mesh 估计 lateral axis 的方法有误，修正版以 Mesial/Distal 标注确定镜像方向；
- 提供 `visualize_cusp_mirror.py` 辅助检查后牙几何选择。

局限：

- 镜像点是几何合成，不是真实标注；
- 强依赖 OuterPoint、mesh center、Mesial/Distal 标注和可靠 cusp 选择；
- 仍不能完全解决下牙弓后牙 curve 不稳定问题。

## 13. 可用于 Final Report 的核心表述

本阶段工作从最初的牙齿中心点补全，逐步演进到 landmark 级补全和两点式 curve anchor 预测。早期方案证明了 Masked Transformer 能够根据残余牙齿上下文恢复缺失牙位，并生成较平滑的牙弓曲线；后续 `curve_new` 将预测目标细化为四个 landmark，并引入固定 ground-truth reference curve 与误差归因分析，使评估语义更加清晰。

进一步实验表明，仅用牙齿中心点难以表达上下牙弓咬合时不同的 curve 形态。因此，Pipeline 1 和 Pipeline 2 将每颗牙的 curve 表示改为两个解剖点，并用 `(2N, 3)` ordered anchors 生成三维参数化 B-Spline。Pipeline 2 通过前牙内侧中点和后牙 Mesial/Distal 构造稳定 anchor，在上牙弓上取得较好的 curve 表现。Pipeline 1 尝试用后牙 buccal cusp 描述更接近咬合面的曲线，但由于后牙 Cusp 标注数量、颊舌侧判定和点序不稳定，下牙弓后牙段仍然较难处理。Pipeline 1 posterior 修正版通过 Mesial-Distal 镜像平面补点缓解了点数不足问题，但不能从根本上消除后牙 Cusp anchor 的不稳定性；旧版通过 mesh 估计 lateral axis 的镜像方向获取方法是错误实现。

最终，本阶段的主要结论是：牙弓 curve 预测并不是单纯的模型回归问题，curve anchor 的解剖定义决定了模型学习目标和 B-Spline 曲线质量。对于上牙弓，Pipeline 2 的 anchor 定义已经较稳定；对于下牙弓，后牙区域仍需要进一步寻找更可靠、更连续且更符合咬合关系的目标点定义。
