# AlphaFold 3 SI Algorithm Implementation Guide

本文档将 AlphaFold 3 论文补充材料(Supplementary Information)中的 31 个算法映射到代码实现文件。

## 算法映射表

| # | SI Algorithm | 页码 | 实现位置 | 类/函数名 |
|---|--------------|------|----------|-----------|
| 1 | MainInferenceLoop | 9 | `alphafold3_pytorch/alphafold3.py` | `Alphafold3.forward()` (6217行) |
| 2 | InputFeatureEmbedder | 10 | `alphafold3_pytorch/alphafold3.py` | `InputFeatureEmbedder` (4293行) |
| 3 | RelativePositionEncoding | 10 | `alphafold3_pytorch/alphafold3.py` | `RelativePositionEncoding` (1692行) |
| 4 | one_hot | 11 | `alphafold3_pytorch/alphafold3.py` | 内置函数 `F.one_hot()` |
| 5 | AtomAttentionEncoder | 12 | `alphafold3_pytorch/alphafold3.py` | `InputFeatureEmbedder` 中的 atom attention 部分 (4293行) |
| 6 | AtomAttentionDecoder | 13 | `alphafold3_pytorch/alphafold3.py` | `DiffusionModule` 中的 atom decoder 部分 (2368行) |
| 7 | AtomTransformer | 13 | `alphafold3_pytorch/alphafold3.py` | `DiffusionTransformer` (2049行) |
| 8 | MsaModule | 15 | `alphafold3_pytorch/alphafold3.py` | `MSAModule` (1191行) |
| 9 | OuterProductMean | 15 | `alphafold3_pytorch/alphafold3.py` | `OuterProductMean` (1057行) |
| 10 | MSAPairWeightedAveraging | 15 | `alphafold3_pytorch/alphafold3.py` | `MSAPairWeightedAveraging` (1123行) |
| 11 | Transition | 16 | `alphafold3_pytorch/alphafold3.py` | `Transition` (567行) |
| 12 | TriangleMultiplicationOutgoing | 16 | `alphafold3_pytorch/alphafold3.py` | `TriangleMultiplication` (748行, outgoing=True) |
| 13 | TriangleMultiplicationIncoming | 16 | `alphafold3_pytorch/alphafold3.py` | `TriangleMultiplication` (748行, outgoing=False) |
| 14 | TriangleAttentionStartingNode | 17 | `alphafold3_pytorch/alphafold3.py` | `TriangleAttention` (922行, node='starting') |
| 15 | TriangleAttentionEndingNode | 17 | `alphafold3_pytorch/alphafold3.py` | `TriangleAttention` (922行, node='ending') |
| 16 | TemplateEmbedder | 18 | `alphafold3_pytorch/alphafold3.py` | `TemplateEmbedder` (1769行) |
| 17 | PairformerStack | 19 | `alphafold3_pytorch/alphafold3.py` | `PairformerStack` (1450行) |
| 18 | SampleDiffusion | 20 | `alphafold3_pytorch/alphafold3.py` | `ElucidatedAtomDiffusion.sample()` (2710行) |
| 19 | CentreRandomAugmentation | 20 | `alphafold3_pytorch/alphafold3.py` | `CentreRandomAugmentation` (4204行) |
| 20 | DiffusionModule | 21 | `alphafold3_pytorch/alphafold3.py` | `DiffusionModule` (2368行) |
| 21 | DiffusionConditioning | 21 | `alphafold3_pytorch/alphafold3.py` | `PairwiseConditioning` + `SingleConditioning` (1946行, 1989行) |
| 22 | FourierEmbedding | 22 | `alphafold3_pytorch/alphafold3.py` | `FourierEmbedding` (1928行) |
| 23 | DiffusionTransformer | 22 | `alphafold3_pytorch/alphafold3.py` | `DiffusionTransformer` (2049行) |
| 24 | AttentionPairBias | 22 | `alphafold3_pytorch/alphafold3.py` | `AttentionPairBias` (812行) |
| 25 | ConditionedTransitionBlock | 23 | `alphafold3_pytorch/alphafold3.py` | `ConditionWrapper` (692行) + `Transition` (567行) |
| 26 | AdaLN | 23 | `alphafold3_pytorch/alphafold3.py` | `AdaptiveLayerNorm` (658行) |
| 27 | SmoothLDDTLoss | 24 | `alphafold3_pytorch/alphafold3.py` | `SmoothLDDTLoss` (3173行) |
| 28 | weighted_rigid_align | 25 | `alphafold3_pytorch/alphafold3.py` | `WeightedRigidAlign` (3239行) |
| 29 | expressCoordinatesInFrame | 27 | `alphafold3_pytorch/alphafold3.py` | `ComputeAlignmentError` 中的坐标变换 (4147行) |
| 30 | computeAlignmentError | 27 | `alphafold3_pytorch/alphafold3.py` | `ComputeAlignmentError` (4147行) |
| 31 | ConfidenceHead | 28 | `alphafold3_pytorch/alphafold3.py` | `ConfidenceHead` (4597行) |

## 详细说明

### 1. 主推理循环 (MainInferenceLoop)
- **实现**: `Alphafold3.forward()` 方法
- **功能**: 协调整个模型的推理流程，包括特征嵌入、循环优化、diffusion采样和置信度预测

### 2. 输入特征嵌入器 (InputFeatureEmbedder)
- **实现**: `InputFeatureEmbedder` 类
- **功能**: 将原子、残基和分子特征嵌入到向量空间，处理参考构象信息

### 3. 相对位置编码 (RelativePositionEncoding)
- **实现**: `RelativePositionEncoding` 类
- **功能**: 编码token之间的相对位置、链和实体关系

### 4-7. Atom Attention 模块
- **Encoder**: 在 `InputFeatureEmbedder` 中实现序列局部原子注意力
- **Decoder**: 在 `DiffusionModule` 中将token级激活广播回原子
- **Transformer**: `DiffusionTransformer` 实现原子级的transformer

### 8-10. MSA 模块
- **MSAModule**: 处理多序列比对信息
- **OuterProductMean**: MSA和pair表示之间的信息传递
- **MSAPairWeightedAveraging**: MSA行的加权平均注意力

### 11. Transition Layer
- **实现**: `Transition` 类使用 SwiGLU 激活函数
- **功能**: MLP层，用于增加网络容量

### 12-15. Triangle Update 模块
- **Multiplication**: `TriangleMultiplication` 处理outgoing和incoming边
- **Attention**: `TriangleAttention` 处理starting和ending节点
- **功能**: 更新pair表示的三角形一致性

### 16. 模板嵌入器 (TemplateEmbedder)
- **实现**: `TemplateEmbedder` 类
- **功能**: 将结构模板信息嵌入到pair表示中

### 17. Pairformer Stack
- **实现**: `PairformerStack` 类 (48个block)
- **功能**: 主干网络，交替更新single和pair表示

### 18-26. Diffusion 模块
- **SampleDiffusion**: EDM采样循环
- **DiffusionModule**: 主扩散模块，生成3D坐标
- **DiffusionTransformer**: 处理噪声坐标的transformer
- **Conditioning**: 从trunk嵌入条件信息

### 27-31. 辅助Head
- **SmoothLDDTLoss**: 可微分的LDDT损失
- **WeightedRigidAlign**: 加权刚体对齐
- **ComputeAlignmentError**: 计算PAE (Predicted Alignment Error)
- **ConfidenceHead**: 预测pLDDT、PAE、PDE等置信度指标

## 注意事项

1. 本实现是基于PyTorch的AlphaFold3重新实现，不是官方代码
2. 某些算法细节可能与论文有所不同，但总体架构保持一致
3. 代码包含大量中文注释，方便理解
4. 主要实现集中在 `alphafold3_pytorch/alphafold3.py` 单个文件中

## 参考
- SI论文: "Accurate structure prediction of biomolecular interactions with AlphaFold 3" Supplementary Information
- 代码仓库: alphafold3-pytorch
