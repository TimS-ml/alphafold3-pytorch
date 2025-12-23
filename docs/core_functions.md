# AlphaFold 3 核心函数逻辑说明

本文档详细说明 AlphaFold 3 中各个核心模块的功能逻辑和实现细节。

## 目录
- [1. Input Feature Embedding](#1-input-feature-embedding)
- [2. MSA Processing](#2-msa-processing)
- [3. Template Embedding](#3-template-embedding)
- [4. Pairformer Stack](#4-pairformer-stack)
- [5. Diffusion Module](#5-diffusion-module)
- [6. Confidence Prediction](#6-confidence-prediction)
- [7. Loss Functions](#7-loss-functions)

---

## 1. Input Feature Embedding

### 1.1 InputFeatureEmbedder
**位置**: `alphafold3_pytorch/alphafold3.py:4293`

**功能**: 将原子和token级别的输入特征嵌入到高维向量空间

**核心逻辑**:
1. **原子特征嵌入**:
   - 嵌入原子的参考位置 (`ref_pos`)
   - 嵌入原子元素类型 (`ref_element`)
   - 嵌入原子电荷 (`ref_charge`)
   - 嵌入原子名称 (`ref_atom_name_chars`)

2. **Token特征嵌入**:
   - 嵌入残基/核苷酸类型 (`restype`)
   - 嵌入MSA profile
   - 嵌入deletion信息

3. **序列局部原子注意力**:
   - 在局部序列窗口内对原子进行attention
   - 聚合原子级信息到token级

**输入**:
- `atom_inputs`: 原子级特征字典
- `atompair_inputs`: 原子对特征
- `additional_token_feats`: 额外的token特征

**输出**:
- `single`: Token级单表示 `[batch, n_tokens, dim_single]`
- `pairwise`: Token对表示 `[batch, n_tokens, n_tokens, dim_pairwise]`
- `atom_single_repr`: 原子级表示(可选)

### 1.2 RelativePositionEncoding
**位置**: `alphafold3_pytorch/alphafold3.py:1692`

**功能**: 编码token之间的相对位置关系

**核心逻辑**:
1. **相对残基位置**: 同链内残基的相对距离
2. **相对token位置**: 同残基内token的相对距离
3. **链标识**: 是否属于同一条链
4. **实体标识**: 是否属于同一实体(序列)

**数学表达**:
```
d_residue = clip(i - j + r_max, 0, 2*r_max) if same_chain else 2*r_max + 1
d_token = clip(token_i - token_j + r_max, 0, 2*r_max) if same_residue else 2*r_max + 1
```

---

## 2. MSA Processing

### 2.1 MSAModule
**位置**: `alphafold3_pytorch/alphafold3.py:1191`

**功能**: 从多序列比对(MSA)中提取进化信息并更新pair表示

**核心逻辑**:
1. **MSA嵌入**: 将MSA序列、deletion嵌入到向量空间
2. **外积均值** (`OuterProductMean`): MSA → Pair信息流
3. **MSA行注意力** (`MSAPairWeightedAveraging`): 使用pair bias的注意力
4. **Pair stack**:
   - Triangle Multiplication (outgoing/incoming)
   - Triangle Attention (starting/ending)
   - Transition

**架构**:
```
MSA Embedding
    ↓
[循环 N_block 次]
    MSA → OuterProductMean → Pair +=
    MSA += MSAPairWeightedAveraging(Pair bias)
    MSA += Transition
    Pair += Triangle Multiplication (out)
    Pair += Triangle Multiplication (in)
    Pair += Triangle Attention (start)
    Pair += Triangle Attention (end)
    Pair += Transition
```

### 2.2 OuterProductMean
**位置**: `alphafold3_pytorch/alphafold3.py:1057`

**功能**: 通过外积将MSA信息传递到pair表示

**数学表达**:
```python
# MSA行投影到两个向量
a_si, b_si = Linear(LayerNorm(msa))  # [n_seq, n_token, c]

# 外积 + 跨序列平均
o_ij = mean_over_sequences(a_si ⊗ b_sj)  # [n_token, n_token, c²]

# 线性投影到pair维度
z_ij = Linear(o_ij)  # [n_token, n_token, dim_pair]
```

### 2.3 MSAPairWeightedAveraging
**位置**: `alphafold3_pytorch/alphafold3.py:1123`

**功能**: MSA行的加权注意力,使用pair表示作为bias

**核心逻辑**:
- 每行MSA独立进行attention
- Attention weights完全由pair表示决定(无query-key点积)
- 使用gating机制

**数学表达**:
```python
v = Linear(LayerNorm(msa))  # value
b = Linear(LayerNorm(pair))  # bias from pair
g = sigmoid(Linear(msa))  # gate

w_ij = softmax(b_ij)  # attention weights
o_si = g_si ⊙ Σ_j w_ij * v_sj  # output

msa_out = Linear(concat_over_heads(o))
```

---

## 3. Template Embedding

### TemplateEmbedder
**位置**: `alphafold3_pytorch/alphafold3.py:1769`

**功能**: 将结构模板信息嵌入到pair表示

**核心逻辑**:
1. **模板特征嵌入**:
   - Template distogram (Cβ距离分布)
   - Template unit vectors (局部坐标系中的单位向量)
   - Template restype
   - Template masks

2. **模板处理**:
   - 每个模板独立通过mini-PairformerStack处理
   - 所有模板的输出取平均

3. **信息整合**:
   - 模板信息与trunk的pair表示融合

**架构**:
```
对每个模板:
    template_features → Embed → pair_t
    pair_t += Linear(trunk_pair)
    pair_t → PairformerStack(N_block=2)

average_over_templates(pair_t) → output
```

---

## 4. Pairformer Stack

### PairformerStack
**位置**: `alphafold3_pytorch/alphafold3.py:1450`

**功能**: AlphaFold 3的主干网络,迭代更新single和pair表示

**核心逻辑**:
- **总层数**: 48 blocks
- **每个block**:
  1. Pair → Triangle Multiplication (outgoing)
  2. Pair → Triangle Multiplication (incoming)
  3. Pair → Triangle Attention (starting node)
  4. Pair → Triangle Attention (ending node)
  5. Pair → Transition
  6. Single → Attention with Pair Bias
  7. Single → Transition

**与AlphaFold 2的区别**:
- 移除了MSA representation(只保留single)
- 移除了outer product mean(single不影响pair)
- 移除了column attention
- 使用SwiGLU代替ReLU

**架构**:
```
[循环 48 次]
    # Pair Stack
    pair += DropoutRow(TriangleMultiplication_out(pair))
    pair += DropoutRow(TriangleMultiplication_in(pair))
    pair += DropoutRow(TriangleAttention_start(pair))
    pair += DropoutCol(TriangleAttention_end(pair))
    pair += Transition(pair)

    # Single Stack
    single += AttentionPairBias(single, pair_bias=pair)
    single += Transition(single)
```

### 4.1 TriangleMultiplication
**位置**: `alphafold3_pytorch/alphafold3.py:748`

**功能**: 通过三角形更新强制pair表示的几何一致性

**数学表达** (Outgoing):
```python
pair_norm = LayerNorm(pair)
a_ij, b_ij = sigmoid(Linear(pair_norm)) ⊙ Linear(pair_norm)
g_ij = sigmoid(Linear(pair_norm))

# 三角形更新: z_ij 通过中间节点k更新
z_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ik ⊙ b_jk))
```

**Incoming**: 改变求和方向 `Σ_k a_ki ⊙ b_kj`

### 4.2 TriangleAttention
**位置**: `alphafold3_pytorch/alphafold3.py:922`

**功能**: 沿着triangle的边进行self-attention

**核心逻辑** (Starting Node):
- 对每个起始节点i,在所有终止节点k上做attention
- 使用pair(j,k)作为bias

**数学表达**:
```python
q_ij, k_ij, v_ij = Linear(LayerNorm(pair))
b_ij = Linear(pair)
g_ij = sigmoid(Linear(pair))

# Attention along k dimension
a_ijk = softmax_k(q_ij^T k_ik / √c + b_jk)
o_ij = g_ij ⊙ Σ_k a_ijk * v_ik
```

**Ending Node**: 沿着不同维度 `softmax_k(q_ij^T k_kj + b_ki)`

---

## 5. Diffusion Module

### 5.1 ElucidatedAtomDiffusion
**位置**: `alphafold3_pytorch/alphafold3.py:2710`

**功能**: 使用扩散模型生成原子3D坐标

**核心逻辑**:
1. **训练时**:
   - 对ground truth坐标添加高斯噪声
   - 训练denoiser预测干净坐标
   - 使用EDM (Elucidated Diffusion Model)参数化

2. **推理时**:
   - 从纯噪声开始
   - 迭代去噪200步
   - 每步应用random augmentation

**EDM采样循环**:
```python
x = sample_noise()  # 初始化为噪声

for t in noise_schedule:
    # 添加噪声
    noise = λ * √(t² - t_prev²) * N(0,I)
    x_noisy = x + noise

    # 去噪
    x_denoised = DiffusionModule(x_noisy, t, conditioning)

    # 更新
    δ = (x - x_denoised) / t
    dt = t_next - t
    x = x_noisy + η * dt * δ
```

### 5.2 DiffusionModule
**位置**: `alphafold3_pytorch/alphafold3.py:2368`

**功能**: 扩散denoiser网络

**架构**:
```
噪声坐标 r_noisy
    ↓
[Conditioning]
    single, pair = DiffusionConditioning(trunk_s, trunk_z, t)
    ↓
[Atom Attention Encoder]
    r_noisy → AtomAttentionEncoder → token_repr
    (序列局部attention,聚合原子→token)
    ↓
[Token Transformer]
    token_repr += Linear(single)
    token_repr → DiffusionTransformer(24 blocks)
    (全局token-level attention with pair bias)
    ↓
[Atom Attention Decoder]
    token_repr → broadcast → atom_repr
    atom_repr → AtomTransformer → r_update
    (序列局部attention,token→原子)
    ↓
[输出]
    x_out = σ²/(σ²+t²) * x_noisy + σt/√(σ²+t²) * r_update
```

**关键特性**:
- **两层架构**: 先atom→token,再token→atom
- **序列局部attention**: 每32个atom attend to 128个邻近atom
- **条件信息**: 来自trunk的single/pair + 时间嵌入
- **无空间归纳偏置**: 仅用单个linear layer嵌入坐标

### 5.3 DiffusionTransformer
**位置**: `alphafold3_pytorch/alphafold3.py:2049`

**功能**: Token级transformer,处理噪声表示

**架构** (每个block):
```
token_repr += AttentionPairBias(token, single_cond, pair_cond)
token_repr += ConditionedTransitionBlock(token, single_cond)
```

**与标准Transformer的区别**:
- 使用Adaptive LayerNorm (AdaLN)调制
- Attention使用pair representation作为bias
- Transition使用single conditioning

---

## 6. Confidence Prediction

### ConfidenceHead
**位置**: `alphafold3_pytorch/alphafold3.py:4597`

**功能**: 预测结构质量的多个置信度指标

**输出指标**:
1. **pLDDT** (per-atom): 局部距离差异测试
2. **PAE** (pairwise): 对齐误差
3. **PDE** (pairwise): 距离误差
4. **resolved** (per-atom): 原子是否被实验解析

**架构**:
```
预测坐标 + trunk(single, pair)
    ↓
Embed pair distances
    pair += Linear(one_hot(||x_i - x_j||))
    ↓
Mini PairformerStack (4 blocks)
    single, pair → update
    ↓
Projection heads:
    pLDDT = softmax(Linear(single))  # [n_atoms, 50 bins]
    PAE = softmax(Linear(pair))       # [n_tokens, n_tokens, 64 bins]
    PDE = softmax(Linear(pair+pair^T)) # [n_tokens, n_tokens, 64 bins]
    resolved = softmax(Linear(single)) # [n_atoms, 2]
```

**pLDDT计算**:
- 对每个atom,计算其到所有polymer token的距离
- 统计<0.5Å, <1Å, <2Å, <4Å的比例
- 仅考虑ground truth中<15Å(protein)或<30Å(NA)的pairs

**PAE计算**:
- 构建每个token的局部坐标系
- 计算在不同坐标系下的对齐误差
- 预测误差的分布

---

## 7. Loss Functions

### 7.1 Diffusion Loss
**位置**: `alphafold3_pytorch/alphafold3.py:2710` (ElucidatedAtomDiffusion)

**组成**:
1. **MSE Loss**: 预测坐标与对齐后ground truth的加权MSE
2. **Smooth LDDT Loss**: 可微分的LDDT
3. **Bond Loss**: 确保共价键长度正确

**数学表达**:
```python
# 1. 加权刚体对齐
gt_aligned = weighted_rigid_align(gt, pred, weights)

# 2. 加权MSE
w = 1 + 5*is_NA + 10*is_ligand
L_mse = mean(w * ||pred - gt_aligned||²)

# 3. Bond loss (仅fine-tuning)
L_bond = mean(| ||pred_i - pred_j|| - ||gt_i - gt_j|| |²)
         for (i,j) in covalent_bonds

# 4. Smooth LDDT
ε = 1/4 * Σ sigmoid(threshold - |d_pred - d_gt|)
    for threshold in [0.5, 1, 2, 4]Å
L_lddt = 1 - mean(ε)

# 总loss
L = (t²+σ²)/(t+σ)² * (L_mse + L_bond) + L_lddt
```

**权重设计**:
- 上调核酸和配体的权重,因为它们样本少
- 时间依赖权重: 高噪声时权重小,低噪声时权重大

### 7.2 Confidence Loss
**位置**: `alphafold3_pytorch/alphafold3.py:4597`

**组成**:
```python
L_confidence = L_plddt + L_pae + L_pde + L_resolved

# pLDDT: 交叉熵
L_plddt = -mean(lddt_true * log(p_plddt))

# PAE: 交叉熵
L_pae = -mean(pae_true * log(p_pae))

# PDE: 交叉熵
L_pde = -mean(pde_true * log(p_pde))

# Resolved: 二分类交叉熵
L_resolved = -mean(resolved_true * log(p_resolved))
```

### 7.3 Distogram Loss
**位置**: `alphafold3_pytorch/alphafold3.py:4441`

**功能**: 预测token对之间的距离分布

**数学表达**:
```python
# 距离离散化到64个bins
d_ij = ||x_rep(i) - x_rep(j)||
d_bin = discretize(d_ij, bins=[3.25, 3.75, ..., 50.75]Å)

# 交叉熵
L_distogram = -mean(one_hot(d_bin) * log(p_distogram))
```

### 7.4 总Loss
**位置**: `alphafold3_pytorch/alphafold3.py:6204`

```python
L_total = α_diffusion * L_diffusion +
          α_confidence * (L_plddt + L_pae + L_pde + L_resolved) +
          α_distogram * L_distogram

# 权重
α_diffusion = 4
α_confidence = 1e-4
α_distogram = 3e-2
α_pae = 0 (除最后fine-tuning阶段为1)
```

---

## 关键设计思想

1. **统一表示**: 所有分子类型(蛋白/NA/配体)用相同架构处理
2. **Token化**: 标准残基1 token,配体per-atom token
3. **无几何偏置**: Diffusion模块不使用SE(3)等变性
4. **两级架构**: Atom ↔ Token双向转换
5. **序列局部attention**: 控制计算复杂度
6. **EDM扩散**: 使用改进的扩散模型参数化
7. **多尺度监督**: 同时优化坐标、距离、置信度

---

## 参考资料
- AlphaFold 3 论文: Nature 2024
- 代码实现: alphafold3-pytorch
- SI文档: 算法伪代码详解
