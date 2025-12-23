# AlphaFold 3 推理逻辑与函数调用顺序

本文档详细说明 AlphaFold 3 在推理阶段的完整函数调用流程。

## 整体流程概览

```
输入准备
    ↓
特征嵌入 (Input Embedding)
    ↓
主干网络循环 (Trunk Recycling)
    ↓
扩散采样 (Diffusion Sampling)
    ↓
置信度预测 (Confidence Prediction)
    ↓
输出结构
```

---

## 详细调用序列

### 阶段 1: 模型初始化
**入口**: `Alphafold3.__init__()`

```
1. 创建子模块
   ├─ InputFeatureEmbedder (特征嵌入器)
   ├─ RelativePositionEncoding (相对位置编码)
   ├─ TemplateEmbedder (模板嵌入器, 可选)
   ├─ MSAModule (MSA模块, 可选)
   ├─ PairformerStack (主干网络, 48层)
   ├─ ElucidatedAtomDiffusion (扩散模型)
   │   └─ DiffusionModule (去噪网络)
   ├─ ConfidenceHead (置信度预测头)
   └─ DistogramHead (距离图预测头)
```

---

### 阶段 2: 推理前向传播
**入口**: `Alphafold3.forward(return_loss=False)`

#### 2.1 输入特征嵌入
**函数调用**: `InputFeatureEmbedder.forward()`

```python
# 步骤 1: 嵌入原子级特征
atom_feats = embed_atom_features(
    ref_pos,          # 参考构象坐标
    ref_element,      # 元素类型
    ref_charge,       # 电荷
    ref_atom_names    # 原子名称
)

# 步骤 2: 序列局部原子注意力 (Atom Attention Encoder)
# 对应 SI Algorithm 5: AtomAttentionEncoder
for block in range(3):
    # 2.1 创建atom pair特征
    pair_feats = embed_atom_pair_offsets(ref_pos)

    # 2.2 序列局部attention (窗口大小: 32 query atoms, 128 key atoms)
    atom_repr = local_atom_transformer(
        queries=atom_feats,
        keys=atom_feats,
        values=atom_feats,
        pair_bias=pair_feats,
        window_size=(32, 128)
    )

# 步骤 3: 聚合到token级
single = aggregate_atoms_to_tokens(atom_repr)  # [n_tokens, dim_single]

# 步骤 4: 嵌入额外的token特征
single = concat(single, restype, profile, deletion_mean)

返回: single, pairwise, atom_single_repr
```

**对应算法**: SI Algorithm 2 (InputFeatureEmbedder), Algorithm 5 (AtomAttentionEncoder)

#### 2.2 相对位置编码
**函数调用**: `RelativePositionEncoding.forward()`

```python
# 计算相对位置编码
rel_pos_encoding = RelativePositionEncoding(
    residue_index,    # 残基索引
    token_index,      # Token索引
    asym_id,          # 链ID
    entity_id,        # 实体ID
    sym_id            # 对称ID
)

# 添加到pair表示
pairwise += rel_pos_encoding  # [n_tokens, n_tokens, dim_pair]
```

**对应算法**: SI Algorithm 3 (RelativePositionEncoding)

#### 2.3 主干网络循环 (Trunk Recycling)
**循环次数**: `num_recycles` (默认3次,包括初始前向)

```python
# 初始化循环状态
single_prev = 0
pairwise_prev = 0

for recycle_iter in range(num_recycles):
    # 2.3.1 添加循环连接
    single_init = single + Linear(LayerNorm(single_prev))
    pairwise_init = pairwise + Linear(LayerNorm(pairwise_prev))

    # 2.3.2 模板嵌入 (可选)
    if use_templates:
        template_embed = TemplateEmbedder(
            template_feats,
            pairwise_init
        )
        pairwise_init += template_embed

    # 2.3.3 MSA模块 (可选)
    if use_msa:
        pairwise_init += MSAModule(
            msa_feats,
            pairwise_init,
            single_init
        )

    # 2.3.4 Pairformer主干 (48层)
    single_trunk, pairwise_trunk = PairformerStack(
        single_init,
        pairwise_init
    )

    # 2.3.5 保存用于下次循环
    single_prev = single_trunk
    pairwise_prev = pairwise_trunk

# 最终trunk输出
single_final = single_trunk
pairwise_final = pairwise_trunk
```

**对应算法**: SI Algorithm 1 (MainInferenceLoop), Algorithm 16 (TemplateEmbedder), Algorithm 8 (MsaModule), Algorithm 17 (PairformerStack)

##### 2.3.2 详解: TemplateEmbedder
**函数调用**: `TemplateEmbedder.forward()`

```python
def TemplateEmbedder(template_feats, pairwise):
    # 对每个模板独立处理
    template_embeds = []

    for t in range(num_templates):
        # 拼接模板特征
        template_pair = concat(
            template_distogram[t],      # Cβ距离直方图
            template_unit_vectors[t],   # 局部坐标系单位向量
            template_backbone_mask[t],
            template_pseudo_beta_mask[t]
        )

        # 嵌入 + 添加trunk pair
        pair_t = Linear(template_pair)
        pair_t += Linear(LayerNorm(pairwise))

        # Mini Pairformer (2层)
        for block in range(2):
            pair_t += TriangleMultiplication(pair_t, outgoing=True)
            pair_t += TriangleMultiplication(pair_t, outgoing=False)
            pair_t += TriangleAttention(pair_t, node='starting')
            pair_t += TriangleAttention(pair_t, node='ending')
            pair_t += Transition(pair_t)

        template_embeds.append(LayerNorm(pair_t))

    # 平均所有模板
    return mean(template_embeds)
```

**对应算法**: SI Algorithm 16 (TemplateEmbedder)

##### 2.3.3 详解: MSAModule
**函数调用**: `MSAModule.forward()`

```python
def MSAModule(msa_feats, pairwise, single_inputs):
    # 嵌入MSA特征
    msa = Linear(concat(msa_seq, has_deletion, deletion_value))
    msa += Linear(single_inputs)  # 广播single到每行MSA

    # MSA处理循环 (4个blocks)
    for block in range(4):
        # MSA → Pair: Outer Product Mean
        pairwise += OuterProductMean(msa)

        # Pair → MSA: Pair-biased attention
        msa += MSAPairWeightedAveraging(msa, pairwise)
        msa += Transition(msa)

        # Pair stack
        pairwise += TriangleMultiplication(pairwise, outgoing=True)
        pairwise += TriangleMultiplication(pairwise, outgoing=False)
        pairwise += TriangleAttention(pairwise, node='starting')
        pairwise += TriangleAttention(pairwise, node='ending')
        pairwise += Transition(pairwise)

    return pairwise
```

**对应算法**: SI Algorithm 8 (MsaModule), Algorithm 9 (OuterProductMean), Algorithm 10 (MSAPairWeightedAveraging)

##### 2.3.4 详解: PairformerStack
**函数调用**: `PairformerStack.forward()`

```python
def PairformerStack(single, pairwise):
    # 48层循环
    for block in range(48):
        # === Pair Stack ===
        # Triangle updates
        pairwise += Dropout(TriangleMultiplication(pairwise, outgoing=True))
        pairwise += Dropout(TriangleMultiplication(pairwise, outgoing=False))
        pairwise += Dropout(TriangleAttention(pairwise, node='starting'))
        pairwise += Dropout(TriangleAttention(pairwise, node='ending'))
        pairwise += Transition(pairwise)

        # === Single Stack ===
        # Attention with pair bias
        single += AttentionPairBias(
            queries=single,
            keys=single,
            values=single,
            pair_bias=pairwise
        )
        single += Transition(single)

    return single, pairwise
```

**对应算法**: SI Algorithm 17 (PairformerStack), Algorithm 12-15 (Triangle操作), Algorithm 24 (AttentionPairBias), Algorithm 11 (Transition)

#### 2.4 扩散采样生成坐标
**函数调用**: `ElucidatedAtomDiffusion.sample()`

```python
# 扩散采样 (200步)
def sample_diffusion():
    # 步骤 1: 初始化为纯噪声
    atom_pos = sample_gaussian_noise()  # [n_atoms, 3]

    # 步骤 2: 定义噪声时间表
    noise_schedule = get_edm_noise_schedule(
        num_steps=200,
        sigma_min=0.0004,
        sigma_max=160,
        rho=7
    )

    # 步骤 3: 迭代去噪
    for step, (t_curr, t_next) in enumerate(noise_schedule):
        # 3.1 中心化 + 随机旋转/平移
        atom_pos = CentreRandomAugmentation(atom_pos)

        # 3.2 添加额外噪声 (高阶求解器)
        t_hat = t_curr * (gamma + 1)
        noise = lambda_noise * sqrt(t_hat^2 - t_curr^2) * N(0,I)
        atom_pos_noisy = atom_pos + noise

        # 3.3 去噪预测
        atom_pos_denoised = DiffusionModule(
            noisy_coords=atom_pos_noisy,
            timestep=t_hat,
            single_trunk_repr=single_final,
            pair_trunk_repr=pairwise_final,
            atom_feats=atom_feats
        )

        # 3.4 更新坐标 (Euler step)
        direction = (atom_pos - atom_pos_denoised) / t_hat
        dt = t_next - t_hat
        atom_pos = atom_pos_noisy + eta * dt * direction

    return atom_pos  # 最终预测坐标
```

**对应算法**: SI Algorithm 18 (SampleDiffusion), Algorithm 19 (CentreRandomAugmentation), Algorithm 20 (DiffusionModule)

##### 2.4.1 详解: DiffusionModule (去噪网络)
**函数调用**: `DiffusionModule.forward()`

```python
def DiffusionModule(noisy_coords, timestep, single_trunk, pair_trunk, atom_feats):
    # === 步骤 1: 条件信息处理 ===
    # Pair conditioning
    pair_cond = concat(pair_trunk, RelativePositionEncoding())
    pair_cond = Linear(LayerNorm(pair_cond))
    for _ in range(2):
        pair_cond += Transition(pair_cond, n_hidden=2)

    # Single conditioning
    single_cond = concat(single_trunk, single_inputs)
    single_cond = Linear(LayerNorm(single_cond))

    # 时间嵌入
    time_embed = FourierEmbedding(log(timestep / sigma_data))
    single_cond += Linear(LayerNorm(time_embed))

    for _ in range(2):
        single_cond += Transition(single_cond, n_hidden=2)

    # === 步骤 2: 坐标归一化 ===
    coords_normed = noisy_coords / sqrt(timestep^2 + sigma_data^2)

    # === 步骤 3: Atom Attention Encoder ===
    # 序列局部atom→token聚合
    token_repr, atom_skip_repr, atom_skip_cond, atompair_skip = \
        AtomAttentionEncoder(
            atom_coords=coords_normed,
            atom_feats=atom_feats,
            single_trunk=single_trunk,
            pair_trunk=pair_cond
        )

    # === 步骤 4: Token-level Transformer ===
    token_repr += Linear(LayerNorm(single_cond))

    # 24层 Diffusion Transformer
    for block in range(24):
        # Attention with pair bias + adaptive conditioning
        token_repr += AttentionPairBias(
            token_repr,
            single_cond=single_cond,
            pair_cond=pair_cond
        )

        # Conditioned transition with AdaLN
        token_repr += ConditionedTransitionBlock(
            token_repr,
            single_cond=single_cond
        )

    token_repr = LayerNorm(token_repr)

    # === 步骤 5: Atom Attention Decoder ===
    # Token→atom广播
    coords_update = AtomAttentionDecoder(
        token_repr,
        atom_skip_repr,
        atom_skip_cond,
        atompair_skip
    )

    # === 步骤 6: 坐标输出 ===
    # EDM预调节器公式
    c_skip = sigma_data^2 / (sigma_data^2 + timestep^2)
    c_out = sigma_data * timestep / sqrt(sigma_data^2 + timestep^2)

    coords_out = c_skip * noisy_coords + c_out * coords_update

    return coords_out
```

**对应算法**: SI Algorithm 20 (DiffusionModule), Algorithm 21 (DiffusionConditioning), Algorithm 22 (FourierEmbedding), Algorithm 23 (DiffusionTransformer)

#### 2.5 置信度预测
**函数调用**: `ConfidenceHead.forward()`

```python
def ConfidenceHead(single_inputs, single_trunk, pair_trunk, pred_coords):
    # 步骤 1: 构建pair特征
    pair = pair_trunk.clone()
    pair += Linear(single_inputs) + Linear(single_inputs.T)

    # 步骤 2: 嵌入预测的原子间距离
    rep_atom_dists = compute_representative_atom_distances(pred_coords)
    dist_one_hot = one_hot(rep_atom_dists, bins=[3.375, 5.125, ..., 21.375])
    pair += Linear(dist_one_hot)

    # 步骤 3: Mini Pairformer (4层)
    single = single_trunk
    for block in range(4):
        # Pair stack
        pair += TriangleMultiplication(pair, outgoing=True)
        pair += TriangleMultiplication(pair, outgoing=False)
        pair += TriangleAttention(pair, node='starting')
        pair += TriangleAttention(pair, node='ending')
        pair += Transition(pair)

        # Single stack
        single += AttentionPairBias(single, pair_bias=pair)
        single += Transition(single)

    # 步骤 4: 预测各类置信度
    # pLDDT: per-atom local confidence
    plddt_logits = Linear(single)  # [n_atoms, 50 bins]
    plddt = softmax(plddt_logits)

    # PAE: pairwise alignment error
    pae_logits = Linear(pair)  # [n_tokens, n_tokens, 64 bins]
    pae = softmax(pae_logits)

    # PDE: pairwise distance error
    pde_logits = Linear(pair + pair.T)  # 对称化
    pde = softmax(pde_logits)

    # Resolved: experimentally resolved prediction
    resolved_logits = Linear(single)  # [n_atoms, 2]
    resolved = softmax(resolved_logits)

    return {
        'plddt': plddt,
        'pae': pae,
        'pde': pde,
        'resolved': resolved
    }
```

**对应算法**: SI Algorithm 31 (ConfidenceHead)

#### 2.6 距离图预测 (可选)
**函数调用**: `DistogramHead.forward()`

```python
def DistogramHead(pair_repr):
    # 预测token对之间的距离分布
    distogram_logits = Linear(pair_repr)  # [n_tokens, n_tokens, 64 bins]
    distogram = softmax(distogram_logits)
    return distogram
```

---

## 完整推理调用栈

```
Alphafold3.forward(return_loss=False)
│
├─ [1] InputFeatureEmbedder()
│   ├─ embed_atom_features()
│   ├─ AtomAttentionEncoder() [Algorithm 5]
│   │   └─ local_atom_transformer() × 3 blocks
│   └─ aggregate_atoms_to_tokens()
│
├─ [2] RelativePositionEncoding() [Algorithm 3]
│
├─ [3] Trunk Recycling Loop (3次)
│   │
│   ├─ [3.1] TemplateEmbedder() [Algorithm 16]
│   │   └─ mini_pairformer(2 blocks) × num_templates
│   │
│   ├─ [3.2] MSAModule() [Algorithm 8]
│   │   └─ [4 blocks]
│   │       ├─ OuterProductMean() [Algorithm 9]
│   │       ├─ MSAPairWeightedAveraging() [Algorithm 10]
│   │       ├─ Transition() [Algorithm 11]
│   │       ├─ TriangleMultiplication() × 2 [Algorithm 12-13]
│   │       ├─ TriangleAttention() × 2 [Algorithm 14-15]
│   │       └─ Transition()
│   │
│   └─ [3.3] PairformerStack() [Algorithm 17]
│       └─ [48 blocks]
│           ├─ TriangleMultiplication(outgoing)
│           ├─ TriangleMultiplication(incoming)
│           ├─ TriangleAttention(starting)
│           ├─ TriangleAttention(ending)
│           ├─ Transition(pair)
│           ├─ AttentionPairBias(single) [Algorithm 24]
│           └─ Transition(single)
│
├─ [4] ElucidatedAtomDiffusion.sample() [Algorithm 18]
│   └─ [200 denoising steps]
│       ├─ CentreRandomAugmentation() [Algorithm 19]
│       └─ DiffusionModule() [Algorithm 20]
│           ├─ DiffusionConditioning() [Algorithm 21]
│           │   ├─ PairwiseConditioning()
│           │   ├─ SingleConditioning()
│           │   └─ FourierEmbedding() [Algorithm 22]
│           ├─ AtomAttentionEncoder()
│           ├─ DiffusionTransformer() [Algorithm 23]
│           │   └─ [24 blocks]
│           │       ├─ AttentionPairBias()
│           │       └─ ConditionedTransitionBlock() [Algorithm 25]
│           │           └─ AdaLN() [Algorithm 26]
│           └─ AtomAttentionDecoder()
│
├─ [5] ConfidenceHead() [Algorithm 31]
│   └─ mini_pairformer(4 blocks)
│       └─ predict(pLDDT, PAE, PDE, Resolved)
│
└─ [6] DistogramHead() (可选)
    └─ predict(distance_distribution)
```

---

## 关键时间复杂度

| 模块 | 复杂度 | 说明 |
|------|--------|------|
| InputFeatureEmbedder | O(N_atoms²) | 序列局部attention |
| PairformerStack | O(N_tokens² × N_layers) | 48层,每层O(N²) |
| DiffusionModule (per step) | O(N_atoms² + N_tokens²) | Atom attention + Token transformer |
| Diffusion Sampling | O(N_steps × N_tokens²) | 200步 × 每步 |
| ConfidenceHead | O(N_tokens²) | 4层mini-pairformer |

**总复杂度**: O(N_recycles × N_tokens² × 48 + N_diffusion_steps × N_tokens²)

---

## 内存优化策略

1. **Gradient Checkpointing**: 在Pairformer和Diffusion中重计算中间激活
2. **序列局部Attention**: 限制atom attention在局部窗口(32×128)
3. **两级架构**: Atom→Token聚合,降低Transformer的输入大小
4. **混合精度**: 使用FP16/BF16加速计算

---

## 总结

AlphaFold 3的推理流程可以概括为:
1. **特征嵌入**: 原子→token级表示
2. **主干处理**: Recycling + Pairformer深度更新
3. **结构生成**: 扩散模型迭代去噪
4. **质量评估**: 置信度head预测pLDDT/PAE等

整个流程遵循"粗粒化→细粒度→评估"的架构设计哲学。
