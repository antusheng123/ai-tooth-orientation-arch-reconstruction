# 第一阶段工作总结：AI Single Tooth Orientation Prediction

## 1. 阶段定位与项目目标

本阶段对应实习项目 **NDCS Project 1 - AI single Tooth Orientation Prediction** 的第一阶段，主要目标是完成从原始 3D 牙科扫描数据到可训练 AI 模型原型的完整准备工作，并建立初步的模型训练与可视化验证流程。

根据任务书，项目最终希望开发一个 AI 系统，能够对给定的单颗牙齿 3D 模型自动判断其主要方向表面，包括：

- Mesial：近中面
- Distal：远中面
- Lingual / Inner：舌侧 / 内侧
- Buccal / Outer：颊侧 / 外侧

任务书中还提到最终系统需要支持集成与可视化，即模型输出应能在工具或界面中以不同颜色标注牙齿表面，方便人工验证和后续工作流使用。因此，本阶段工作不仅包括数据集构建，也包含了模型原型、训练脚本和可视化验证界面。

需要说明的是，任务书中有一段关于 web-based 2D denture design modules 的描述，与本项目标题及 desired outcome 不完全一致。结合项目名称、工作描述和最终交付目标，本阶段实际围绕 **3D 单牙方向识别 / 表面朝向预测** 展开。

## 2. 第一阶段总体成果

第一阶段已经完成了以下核心工作：

1. 梳理 3DS+ / 3DTeethLand / Teeth3DS 相关原始牙科数据资产。
2. 将全颌 3D 牙弓模型按 FDI 牙位编号自动分割为独立单牙模型。
3. 将单牙几何与方向地标统一到同一坐标系，并保存为训练所需文件。
4. 统计单牙数量和四个核心方向地标的完整度。
5. 按病人 / 牙弓级别进行 train / test 划分，避免同一病人的牙齿同时出现在训练集和测试集。
6. 将单牙网格进一步采样为带法向量的 6D 点云。
7. 实现一个基于 Point Transformer 思路的单牙地标回归模型。
8. 实现支持缺失标注 mask 的训练与评估流程。
9. 实现 Streamlit + Plotly 的 3D 可视化原型，用于加载单牙模型、运行预测并按方向着色。
10. 根据中期汇报记录，当前最佳模型在 held-out validation set 上达到 **Mean Euclidean Error (MEE) = 1.1636 mm**，低于牙科 landmark detection 中常用的 2.0 mm 临床可接受阈值。
11. 在 Streamlit 可视化中补充单牙与批量评估指标，包括 per-landmark error、MEE、Success Rate@1mm、Success Rate@2mm 和 Success Rate@3mm。

从阶段边界来看，本阶段已经从“数据准备”推进到了“可运行 AI 原型验证”。后续阶段可以在此基础上继续提升模型精度、完善评估指标，并将输出从地标点预测进一步稳定映射到表面区域预测。

## 3. 数据来源与原始结构

本阶段使用的数据来源为 3DS+ 相关牙科扫描数据集，包括带标签的训练牙弓和未标注测试牙弓。已整理的信息如下：

- 原始训练规模：240 个带标签训练牙弓。
- 原始测试规模：100 个未带标签测试牙弓。
- 每个训练样本包含：
  - 全颌 `.obj` 网格模型。
  - 带顶点级牙齿标签的 `.json` 文件，其中 labels 数组用于标识每个顶点所属牙位。
  - 带方向地标的 `__kpt.json` 文件，其中包含 Mesial、Distal、InnerPoint、OuterPoint、FacialPoint 等地标。

本阶段重点使用训练集中同时具备网格、顶点标签和地标标注的数据。由于任务目标是单牙方向预测，因此第一步不是直接训练全颌模型，而是先把全颌牙弓转换为独立单牙样本。

## 4. 单牙分割与数据集构建

### 4.1 分割脚本

单牙分割逻辑主要位于：

- `data preparation部分/seg_for_single_teeth.py`

该脚本的主要输入包括：

- 分割数据根目录：包含 `.obj` 与顶点标签 `.json`。
- 地标数据根目录：包含 `__kpt.json`。
- 输出目录：用于保存分割后的单牙数据。

脚本按 `upper` 和 `lower` 两类牙弓遍历病人文件夹，对每个牙弓执行以下流程：

1. 加载全颌 `.obj` 网格。
2. 读取顶点标签数组 `labels`。
3. 读取地标文件中的方向关键点。
4. 检查顶点数量和标签数量是否一致。
5. 使用 KDTree 将地标坐标绑定到最近网格顶点，再根据该顶点标签判断该地标属于哪一颗牙。
6. 遍历所有非 0 的牙位标签，将每颗牙单独提取为一个 mesh。
7. 过滤顶点数量过少的异常碎片。
8. 对单牙 mesh 和地标进行中心化归一化。
9. 保存单牙 `.off`、对应地标 `.json` 和距离图特征 `_f.npy`。

### 4.2 基于顶点标签的单牙提取

分割过程使用 FDI 牙位编号作为牙齿 ID。对于每颗牙，脚本通过：

- `vertex_mask = labels == tooth_id`
- `face_mask = np.all(vertex_mask[mesh.faces], axis=1)`

筛选出所有三个顶点都属于该牙位的面片，从而得到干净的单牙子网格。该方法相比按空间距离裁剪更稳定，因为它直接依赖原始数据提供的顶点级语义标签。

### 4.3 坐标归一化

为了减少牙齿在口腔全局空间中的绝对位置影响，每颗牙分割后会计算自身几何中心：

```python
centroid = tooth_mesh.vertices.mean(axis=0)
tooth_mesh.vertices -= centroid
```

对应地标坐标也执行相同平移：

```python
norm_coord = np.array(lm['coord']) - centroid
```

这样保存后的单牙样本以自身几何中心为原点。该设计使模型更关注单牙几何形态、局部曲率和方向结构，而不是病人在扫描坐标系中的位置差异。

### 4.4 地标类别

本阶段标准化使用 5 类方向地标：

- Mesial
- Distal
- InnerPoint
- OuterPoint
- FacialPoint

其中任务书核心要求是前 4 类方向表面，即近中、远中、舌侧 / 内侧、颊侧 / 外侧。FacialPoint 在当前实现中作为第 5 个可选预测目标保留，有助于扩展表面方向表达。

### 4.5 距离图特征

在分割脚本中还实现了 `compute_distance_maps()`，为每颗牙计算形状为 `(5, num_faces)` 的距离图特征 `_f.npy`。其含义是：

- 对每个面片计算几何中心。
- 分别计算该面片中心到 5 类地标的欧氏距离。
- 如果某颗牙缺少某一类地标，则该通道填充为 `20.0`。

该距离图可以作为未来基于面片级预测或 Point Transformer v3 风格模型的监督信号，也为处理缺失标注提供了明确策略。

## 5. 数据统计与完整度分析

数据统计相关脚本包括：

- `data preparation部分/single_teeth_num_calculation.py`
- `data preparation部分/single_teeth_satisified_num_calculation.py`

根据已有阶段总结，本阶段数据准备结果如下：

- 成功处理原始牙弓数量：155 个。
- 成功分割出的独立单牙数量：3,180 颗。
- 四个核心方向标注完整的单牙数量：2,025 颗。
- 存在部分地标缺失的单牙数量：1,155 颗。

完整样本定义为同时包含：

- Mesial
- Distal
- InnerPoint
- OuterPoint

对于不完整样本，本阶段没有直接丢弃，而是采用保留策略：缺失的距离通道填充为 `20.0`，训练阶段则通过 mask 忽略缺失地标对应的损失。这种做法的优点是保留了更多单牙几何样本，有助于模型学习不同牙位和形态的分布；缺点是训练逻辑必须严格处理缺失标签，否则会把伪标签误当成真实监督。

## 6. 数据集划分

数据集划分脚本位于：

- `data preparation部分/single_teeth_split_script.py`

划分原则是 **病人 / 牙弓级别隔离**，而不是按单颗牙随机划分。该设计非常重要，因为同一病人的不同牙齿在形态、扫描方式和标注风格上可能高度相关。如果随机按单牙划分，训练集和测试集可能包含来自同一病人的不同牙齿，导致评估结果偏乐观。

当前划分设置为：

- 训练集比例：85%
- 测试集比例：15%
- 随机种子：42
- 中期报告记录的实际划分规模：训练集 131 个牙弓，测试集 24 个牙弓。

目标目录结构为：

```text
Single_Teeth_Y/
  train/
    <patient_id>_single/
      <patient_id>_<jaw>_tooth_<tooth_id>.off
      <patient_id>_<jaw>_tooth_<tooth_id>.json
      <patient_id>_<jaw>_tooth_<tooth_id>_f.npy
  test/
    <patient_id>_single/
      ...
```

该结构保留了病人 / 牙弓级目录，便于后续追溯数据来源和检查数据泄露风险。

## 7. 点云预采样与 6D 输入构建

点云预处理脚本位于：

- `pre_sample_pc.py`

早期脚本中曾保留一个仅采样 XYZ 点云的版本，即从单牙 `.off` 均匀采样点并保存为 `_pc2048.npy`。当前实际启用的是 `pre_sample_6d()`，它将单牙网格转换为带法向量的 6D 点云：

```text
[x, y, z, nx, ny, nz]
```

具体步骤为：

1. 递归查找单牙数据目录中的所有 `.off` 文件。
2. 使用 `trimesh.load()` 加载单牙 mesh。
3. 使用 `trimesh.sample.sample_surface(mesh, 4096)` 在表面采样 4096 个点。
4. 根据采样返回的 `face_indices` 获取对应面片法向量。
5. 拼接点坐标和法向量，得到 `(4096, 6)` 数组。
6. 保存为 `_pc4096_normals.npy`。
7. 同步复制对应 `.json` 地标文件到新目录。

最终 6D 点云目录为：

```text
Single_Teeth_PC_6D/
  train/
  test/
```

6D 输入相比纯 XYZ 输入的主要优势是引入了局部表面朝向信息。对于牙齿表面方向判断任务，法向量可以帮助模型感知曲面外向、凹凸变化和牙冠形态。

## 8. PyTorch Dataset 设计

数据加载逻辑位于：

- `dataset.py`

核心类为 `SingleToothPCDataset`。它递归查找数据目录下所有 `*_normals.npy` 文件，并为每个样本加载：

- 6D 点云：`points`
- 5 个地标坐标：`landmarks`
- 地标有效性 mask：`lm_mask`
- FDI 牙位编号：`tooth_id`
- 样本文件名：`sid`

### 8.1 地标 mask

Dataset 中定义了固定类别顺序：

```python
STANDARD_CLASSES = ["Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint"]
```

加载 JSON 后，如果某个类别存在，则填入其坐标并将对应 `lm_mask` 设为 1；如果不存在，则坐标保持为 0，mask 设为 0。该 mask 后续用于 loss 和 metric，确保缺失地标不会参与训练损失。

### 8.2 FDI 牙位编号

牙位编号通过文件名解析：

```python
tooth_id_str = filename.split("tooth_")[1].split("_")[0]
tooth_id = int(tooth_id_str)
```

该编号作为模型中的 tooth embedding 输入，用于提供牙位先验。因为不同牙位的解剖形态和方向关系并不完全相同，引入 FDI 编号可以帮助模型区分门牙、犬牙、前磨牙和磨牙等形态差异。

### 8.3 旋转数据增强

Dataset 支持训练时随机 3D 旋转增强：

```python
rot_matrix = R.random().as_matrix().astype(np.float32)
points = points @ rot_matrix.T
normals = normals @ rot_matrix.T
landmarks[valid_idx] = landmarks[valid_idx] @ rot_matrix.T
```

这里对点坐标、法向量和有效地标执行完全一致的旋转，保持几何关系正确。该增强有助于模型学习旋转等变或旋转鲁棒的方向预测能力，减少对固定扫描姿态的过拟合。

## 9. 模型结构

模型定义位于：

- `point_transformer_model.py`

核心类为 `LandmarkPointTransformer`，目标是从单牙 6D 点云回归 5 个方向地标的 3D 坐标，输出形状为：

```text
(batch_size, 5, 3)
```

### 9.1 输入

模型输入包括：

- `pts`：形状为 `(B, N, 6)` 的点云，其中 N=4096。
- `tooth_id`：形状为 `(B,)` 的 FDI 编号。

模型将：

- 前 3 维 XYZ 作为空间参考。
- 全部 6 维 XYZ + normal 作为初始点特征。

### 9.2 TransitionDown

模型使用三层 `TransitionDown` 对点数量进行随机下采样，并通过 1x1 Conv、BatchNorm 和 ReLU 提升特征维度：

- 输入 6 维特征到 `embed_dim=96`
- 96 到 192
- 192 到 384

该模块的作用是逐步减少点数量，增加特征表达能力，降低后续 attention 的计算成本。

### 9.3 PointTransformerLayer

模型包含两个 `PointTransformerLayer`，内部实现为多头 self-attention：

- 使用 Linear 生成 Q、K、V。
- 执行 multi-head attention。
- 加 residual connection、LayerNorm 和 ReLU。

该实现更接近通用 Transformer attention 层，而不是完整包含局部邻域位置编码的 Point Transformer v3。但它已经具备点集全局关系建模能力，适合作为本阶段 AI 原型。

### 9.4 FDI 牙位 embedding

模型定义：

```python
self.tooth_emb = nn.Embedding(50, 64)
```

全局点云特征经过 pooling 后为 384 维，牙位 embedding 为 64 维，两者拼接成 448 维特征，再输入全连接回归头：

```text
384 point cloud feature + 64 tooth embedding = 448
```

### 9.5 回归头

回归头结构为：

- Linear 448 -> 512
- BatchNorm1d
- ReLU
- Dropout 0.3
- Linear 512 -> 256
- ReLU
- Linear 256 -> 15

最后 reshape 为 `(B, 5, 3)`，表示 5 个地标的三维坐标。

## 10. 训练流程

训练脚本位于：

- `train_point_transformer.py`

主要超参数如下：

```text
NUM_LANDMARKS = 5
BATCH_SIZE = 32
EPOCHS = 200
LR = 3e-4
optimizer = AdamW
weight_decay = 1e-4
scheduler = ReduceLROnPlateau
```

训练目录设置为：

```text
F:\NDCS_3DS_data\Single_Teeth_PC_6D\train
F:\NDCS_3DS_data\Single_Teeth_PC_6D\test
```

训练集启用随机旋转增强，验证集关闭增强。

### 10.1 Masked Smooth L1 Loss

训练使用自定义 `landmark_loss()`：

```python
loss = F.smooth_l1_loss(pred, gt, reduction='none')
loss = loss.sum(dim=-1)
return (loss * mask).sum() / valid_sum
```

其特点是：

- 对每个地标的 XYZ 坐标计算 Smooth L1。
- 对 XYZ 三个维度求和，得到每个地标一个 loss。
- 用 `lm_mask` 屏蔽缺失地标。
- 只对有效地标求平均。

这与前面保留部分缺失样本的数据策略相匹配。

### 10.2 评估指标

验证阶段使用 `mean_euclidean_error()` 计算平均欧氏距离误差，单位为 mm：

```python
diff = torch.norm(pred - gt, dim=-1)
metric = (diff * mask).sum() / mask.sum()
```

该指标直观表示预测地标与真实地标之间的平均空间距离，比单纯 loss 更容易解释。

### 10.3 稳定训练机制

训练脚本中还加入了几项稳定机制：

- 使用 AMP autocast。
- CUDA 下使用 `bfloat16`，降低 NaN 风险。
- 使用 `GradScaler`。
- 使用 gradient clipping，最大范数为 1.0。
- 使用 ReduceLROnPlateau，当验证误差长期不改善时降低学习率。
- 保存验证误差最好的权重到 `best_single_tooth_model_6d.pt`。

这些设计使原型训练更稳定，也方便后续迭代比较不同模型结构。

### 10.4 中期训练结果

根据 `Midterm讲稿.docx` 和 `YIN JUNYAO_midterm report.docx` 中记录的模型训练结果，当前最佳模型的定量表现如下：

| 指标 | 结果 |
|---|---:|
| 训练 epoch 总数 | 200 |
| 最佳 epoch | 173 |
| 最佳 epoch 学习率 | 4.7e-6 |
| Train Loss | 1.3624 |
| Validation Loss | 0.7967 |
| Mean Euclidean Error (MEE) | 1.1636 mm |
| 临床参考阈值 | < 2.0 mm |

MEE 表示预测地标与真实地标之间的平均三维欧氏距离。中期报告中将 2.0 mm 以内的误差作为数字牙科 landmark detection 和后续 CAD/CAM 工作流中较常见的临床可接受范围。当前模型达到 1.1636 mm，说明基于 6D 点云、FDI 牙位 embedding 和 masked Smooth L1 loss 的方案已经具备较好的空间定位精度。

从训练过程描述来看，模型在 200 个 epoch 内表现出较稳定的收敛趋势，最佳结果出现在训练后期，即 Epoch 173。学习率已经由初始 `3e-4` 通过 `ReduceLROnPlateau` 降至 `4.7e-6`，这说明 scheduler 在验证误差趋缓后逐步降低学习率，有助于模型进一步细化 landmark 坐标预测。

该结果也修正了第一版总结中的一个空白：虽然项目目录内没有单独保存完整训练日志或曲线文件，但中期汇报材料已经记录了关键训练结果，可作为 final report 中模型性能分析的主要依据。

## 11. 可视化验证原型

可视化脚本位于：

- `visualizer.py`

该脚本实现了一个 Streamlit 应用，用于加载训练好的模型权重，对单牙 `.off` 或 `.obj` 模型进行推理，并在 3D 网格上显示方向着色。

### 11.1 推理流程

可视化应用的推理流程为：

1. 加载 `best_single_tooth_model_6d.pt`。
2. 从数据目录递归查找 `.off` 或 `.obj` 单牙模型。
3. 用户在侧边栏选择一个牙齿模型。
4. 使用 trimesh 加载 mesh。
5. 从 mesh 表面采样 4096 个点。
6. 拼接点坐标和面法向量，构造 `(4096, 6)` 输入。
7. 从文件名中通过正则提取 `tooth_id`。
8. 调用模型预测 5 个方向地标。
9. 将预测地标映射回 mesh 顶点附近区域并着色。

### 11.2 表面着色逻辑

可视化并不是直接输出面片分类，而是使用预测地标进行基于距离的顶点着色：

- 初始化所有顶点为牙齿本色。
- 对每个被选中的方向地标，计算所有顶点到该地标的距离。
- 如果顶点距离某个地标小于用户设置的 `color_radius`，并且该地标是目前最近的方向点，则将该顶点染成该方向颜色。

这类似一个局部 Voronoi / 最近地标着色策略。它的优点是实现简单、交互直观；局限是颜色区域大小依赖 `color_radius`，并不能完全等价于真实解剖表面分割。

### 11.3 UI 功能

当前 Streamlit 原型支持：

- 单牙模型选择。
- FDI 牙位显示。
- 顶点数、面片数显示。
- 颜色扩散半径调节。
- 选择显示哪些方向表面。
- 可选显示 AI 预测地标点。
- 使用 Plotly 交互式旋转、缩放、查看 3D 牙齿。
- 自定义模型权重路径，不要求 `best_single_tooth_model_6d.pt` 必须放在项目当前目录下。
- 自定义数据根目录，方便切换 train/test 或其他验证数据集。
- 自动读取与 mesh 同名的 ground-truth `.json` 文件，并在存在真实地标时计算误差指标。
- 单牙级别显示 MEE、Success Rate@1mm、Success Rate@2mm、Success Rate@3mm。
- 展开查看每个方向地标的预测坐标、真实坐标和误差。
- 批量评估当前 data root 下的前 N 颗牙，输出 dataset-level MEE、median error、max error 和各阈值 success rate。

默认显示前 4 个任务核心方向：

- Mesial
- Distal
- Lingual / Inner
- Buccal / Outer

Facial / Occlusal 作为第 5 类可选显示。

### 11.4 中期可视化样例

中期报告中记录了一个具体的系统可视化样例：`0140W3ND_lower_tooth_31.off`。该样本包含：

- FDI Tooth ID：31
- 顶点数：2,764
- 面片数：5,255
- 可视化颜色扩散半径：6.00 mm

系统在该样本上能够显示 AI 预测地标，并基于 Voronoi / 最近地标距离规则渲染四个核心方向区域：

- Mesial：粉红 / 红色系
- Distal：蓝色
- Lingual / Inner：绿色
- Buccal / Outer：黄色

该例子说明可视化原型不仅能运行模型推理，也能把预测结果转换为可被临床人员直观检查的 3D 表面颜色区域。

### 11.5 可视化指标扩展与 Batch Evaluation

在中期可视化原型基础上，后续进一步扩展了 `visualizer.py` 的评估能力。原始可视化主要用于展示预测地标和表面着色，定量指标主要来自训练脚本中的 MEE。修改后，Streamlit 页面可以在可视化阶段直接计算和展示地标定位质量。

新增逻辑包括：

- 侧边栏增加 `Model weights path` 输入框，可直接填写 best 权重文件的绝对路径。
- 侧边栏增加 `Data root` 输入框，可切换待评估的单牙数据目录。
- 选择单颗牙后，程序会自动查找同路径、同文件名的 `.json` ground truth。
- 如果存在 ground truth，则计算每个有效地标的三维欧氏距离误差。
- 单牙页面显示 `MEE` 和 `Success Rate@1mm / 2mm / 3mm`。
- 页面提供 per-landmark error table，用于查看每个方向的预测点、真实点和误差。
- 侧边栏增加 batch evaluation，可指定最多评估多少颗牙，并汇总所有有效 landmark 的统计指标。

这里使用 **Success Rate** 而不是 Precision / Recall，是因为当前模型对每个解剖类别固定输出一个预测地标，同一类别下最多对应一个 ground-truth 地标。在这种一对一 landmark localization 设定中，Precision 和 Recall 会退化为相同的阈值命中率。因此，用 Success Rate@kmm 表达“有多少有效地标预测落在 k mm 以内”更加准确。

### 11.6 Batch Evaluation 结果记录

根据当前文件夹中的 `阶段1_batch_eva数据.csv`，使用修改后的 Streamlit batch evaluation 对 50 颗牙进行了批量评估。结果如下：

| 指标 | 结果 |
|---|---:|
| Evaluated Teeth | 50 |
| Skipped Teeth | 0 |
| Valid Landmark Samples | 237 |
| MEE | 1.1944 mm |
| Median Error | 0.6833 mm |
| Max Error | 8.5445 mm |
| Success Rate@1mm | 74.68% |
| Hits@1mm | 177 / 237 |
| Success Rate@2mm | 87.34% |
| Hits@2mm | 207 / 237 |
| Success Rate@3mm | 91.14% |
| Hits@3mm | 216 / 237 |

这个 batch evaluation 结果与中期报告中的整体 MEE=1.1636 mm 接近，说明在 50 颗牙的可视化抽样评估中，模型仍保持约 1.2 mm 的平均 landmark 定位误差。Success Rate@2mm 达到 87.34%，说明大多数有效地标已经落在中期报告采用的 2.0 mm 临床参考阈值内。Success Rate@3mm 达到 91.14%，进一步说明模型对多数地标具有稳定的空间定位能力。

同时，Max Error=8.5445 mm 表明仍存在少量失败或偏差较大的 landmark。后续分析可以进一步查看 per-landmark error table，定位这些 outlier 是否集中在某些牙位、某些方向类别、缺失标注附近样本，或由几何异常和扫描质量导致。

## 12. 当前代码文件职责总结

| 文件 | 主要职责 |
|---|---|
| `NDCS Project 1 - AI single Tooth Orientation Prediction.pdf` | 项目任务书，定义项目背景、技能要求和目标交付 |
| `data preparation部分工作总结.docx` | 已有数据准备阶段总结，记录数据来源、分割数量、完整度和划分策略 |
| `Midterm讲稿.docx` | 中期汇报讲稿，记录项目背景、数据集规模、模型结构、MEE=1.16 mm 的阶段性结果和后续计划 |
| `YIN JUNYAO_midterm report.docx` | 中期报告正文，记录训练配置、最佳 epoch、loss、MEE、临床阈值和可视化样例 |
| `阶段1_batch_eva数据.csv` | 修改 Streamlit 后进行 batch evaluation 的结果，记录 50 颗牙上的 MEE、median/max error 和 Success Rate@1/2/3mm |
| `data preparation部分/seg_for_single_teeth.py` | 从全颌牙弓分割单牙，归一化坐标，保存 OFF/JSON/NPY |
| `data preparation部分/single_teeth_num_calculation.py` | 统计分割出的单牙总数 |
| `data preparation部分/single_teeth_satisified_num_calculation.py` | 统计四个核心方向标注完整的单牙数量 |
| `data preparation部分/single_teeth_split_script.py` | 按病人 / 牙弓级别划分 train/test |
| `pre_sample_pc.py` | 将单牙 mesh 采样为 4096 点 6D 点云 |
| `dataset.py` | PyTorch Dataset 和 DataLoader，加载 6D 点云、地标、mask、FDI 编号 |
| `point_transformer_model.py` | Point Transformer 风格地标回归模型 |
| `train_point_transformer.py` | 训练、验证、保存最佳模型权重 |
| `visualizer.py` | Streamlit + Plotly 可视化推理、表面着色、单牙指标和 batch evaluation 原型 |

## 13. 技术路线总结

第一阶段形成的技术路线可以概括为：

```text
原始全颌 OBJ + 顶点标签 + 地标 JSON
        ↓
基于 FDI 顶点标签分割单牙
        ↓
单牙中心化归一化
        ↓
保存 OFF 几何、JSON 地标、NPY 距离图
        ↓
按病人 / 牙弓级别划分 train/test
        ↓
从单牙 OFF 采样 4096 个表面点
        ↓
拼接 XYZ + 法向量，构造 6D 点云
        ↓
PyTorch Dataset 加载点云、地标、mask、FDI 编号
        ↓
Point Transformer 风格模型回归 5 个方向地标
        ↓
Masked Smooth L1 Loss 训练
        ↓
平均欧氏距离误差验证
        ↓
Streamlit / Plotly 可视化预测地标和方向着色
        ↓
可视化阶段计算 MEE 与 Success Rate@1/2/3mm
```

## 14. 第一阶段的关键工程决策

### 14.1 先预测地标，再映射表面

任务最终要求是识别牙齿表面方向。本阶段选择先预测方向地标，再通过地标附近区域进行可视化着色。这是一种合理的原型策略，因为原始数据中方向标注主要以关键点形式存在，而不是完整面片级表面分割标签。

该策略的优点：

- 与现有标注格式匹配。
- 训练目标明确，监督信号容易构造。
- 预测误差可以用 mm 级距离直接评估。
- 可视化实现简单，便于和导师 / 业务方沟通。

该策略的局限：

- 地标点不等于完整表面区域。
- 着色范围依赖半径阈值。
- 对牙齿复杂曲面或多峰区域，最近地标规则可能不够准确。

后续可以在地标回归基础上进一步构建面片级或点级方向分类模型。

### 14.2 保留部分缺失标注样本

本阶段没有只使用 2,025 个完整样本，而是保留了 1,155 个部分缺失标注样本。该策略提升了可用于学习牙齿几何分布的数据量，但要求训练阶段必须使用 mask 忽略缺失标签。

当前 `dataset.py` 和 `train_point_transformer.py` 已经实现了这一点：缺失地标 `lm_mask=0`，loss 和 metric 中不参与计算。

### 14.3 引入法向量作为输入

从纯 XYZ 点云升级为 XYZ + normal 的 6D 点云，是本阶段模型输入设计的重要改进。牙齿方向识别不仅依赖点的位置，也依赖局部表面朝向和形态。法向量有助于模型区分牙齿不同侧面的几何差异。

### 14.4 引入 FDI 牙位先验

不同牙位的形态差异明显，同一个方向在不同牙齿上的几何表现并不完全相同。因此模型加入 FDI tooth embedding，将牙位编号作为额外条件输入。这使模型能够学习“同一牙位类型下方向地标通常出现在哪里”的先验。

## 15. 当前阶段局限

第一阶段已经完成了可运行原型，但仍有以下限制需要在 final report 或后续阶段中说明：

1. 模型尚不是真正完整的 Point Transformer v3 实现。
   当前 attention 层未显式建模局部邻域和相对位置编码，更接近全局 self-attention 点云模型。

2. 表面识别当前通过地标距离间接实现。
   可视化中的方向着色是基于预测地标和半径阈值，并非模型直接输出每个点或面片的方向类别。

3. 部分文件中的中文注释出现编码乱码。
   代码逻辑可读，但注释在当前环境下存在 mojibake，后续正式整理代码时建议统一保存为 UTF-8。

4. 数据路径为本地绝对路径。
   多个脚本中使用 `F:\NDCS_3DS_data\...`，后续需要改为配置文件、命令行参数或环境变量，方便复现实验。

5. 当前缺少可复现实验日志和训练曲线文件。
   中期报告已经记录了最佳模型结果，包括 Epoch 173、Train Loss 1.3624、Validation Loss 0.7967 和 MEE 1.1636 mm。但项目目录内仍未看到独立保存的逐 epoch 日志、训练曲线图或实验配置表，后续建议补齐，方便复现实验和对比不同模型版本。

6. 当前没有自动化单元测试。
   数据处理和训练流程依赖真实数据路径，建议后续加入小型 synthetic mesh 测试或 smoke test。

7. 评估指标目前集中在地标坐标误差。
   对最终表面分类目标，还需要补充方向分类准确率、表面 IoU、点级 / 面片级 accuracy 等指标。

## 16. 后续阶段建议

基于第一阶段成果，第二阶段可以优先推进以下方向：

1. 系统化记录并管理模型训练结果。
   当前中期汇报已记录最佳结果 MEE 1.1636 mm。后续应继续保存每次实验的配置、逐 epoch loss / MEE、训练曲线、模型版本和权重路径，形成可比较的实验表。

2. 将路径和超参数参数化。
   使用 argparse、YAML 或 JSON 配置文件管理数据路径、batch size、learning rate、epoch、模型权重路径等。

3. 改进模型结构。
   可尝试真正的 Point Transformer / PointNet++ / DGCNN / PointMLP 等点云网络，并比较 3D 输入与 6D 输入效果。

4. 从地标回归扩展到表面分类。
   可以利用距离图 `_f.npy` 或人工规则生成弱标签，将任务转为点级 / 面片级方向分类。

5. 优化可视化映射策略。
   当前半径着色可以升级为基于 mesh geodesic distance、nearest surface region、softmax confidence 或 point-wise classification 的更稳定着色。

6. 完善数据质量分析。
   进一步统计不同牙位的样本数量、完整标注比例、上下颌分布、缺失类别分布，识别数据偏差。

7. 增加推理接口。
   将模型推理封装为独立函数或 API，方便后续集成到 Unity、Web 或现有 dental design system。

## 17. 可用于 final report 的阶段性结论

第一阶段完成了 AI single Tooth Orientation Prediction 项目的基础设施建设。工作从原始全颌 3D 扫描数据出发，构建了独立单牙数据集，并形成了包含几何模型、方向地标、距离图特征和 6D 点云的训练数据格式。通过病人 / 牙弓级别的数据划分，阶段工作降低了训练和测试之间的数据泄露风险。

在模型方面，本阶段实现了一个基于 Point Transformer 思路的单牙地标回归网络。模型输入为 4096 个带法向量的 6D 采样点，并结合 FDI 牙位 embedding 作为先验信息，输出 Mesial、Distal、InnerPoint、OuterPoint 和 FacialPoint 五个方向地标。训练过程采用 masked Smooth L1 loss，能够兼容部分缺失标注的样本，并使用平均欧氏距离误差作为验证指标。

根据中期汇报材料，当前最佳模型在 Epoch 173 达到 Train Loss 1.3624、Validation Loss 0.7967，并在测试 / 验证集上取得 MEE 1.1636 mm。该结果低于中期报告中引用的 2.0 mm 临床可接受阈值，说明本阶段模型已经具备较好的单牙方向 landmark 定位能力，可作为后续表面级自动着色和 SMART RPD 设计流程集成的基础。

在应用验证方面，本阶段实现了 Streamlit + Plotly 可视化原型。该工具能够加载单牙模型，调用训练好的模型进行方向地标预测，并以不同颜色在 3D 牙齿表面上显示方向区域，为后续业务验证和系统集成提供了直观基础。

在最新的可视化版本中，系统还支持直接在 Streamlit 中读取 ground truth JSON，计算单牙和 batch-level 的 MEE 与 Success Rate@1/2/3mm。基于 50 颗牙、237 个有效 landmark 的 batch evaluation，模型取得 MEE 1.1944 mm、Success Rate@1mm 74.68%、Success Rate@2mm 87.34%、Success Rate@3mm 91.14%。这些结果为 final report 中展示可视化系统的量化验证提供了额外依据。

总体而言，第一阶段已经完成了从数据准备、特征构建、模型训练到可视化验证的端到端原型，为后续提高模型性能、完善表面级预测和集成到实际 dental design workflow 奠定了基础。
