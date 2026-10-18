# MiniGPT

用 PyTorch 从零实现的 Decoder-only GPT，按三个阶段递进：**基础 Transformer → KV Cache 增量推理 → MHA/GQA/MQA 统一注意力**。每个阶段配套 pytest 用例，`main` 分支保持全绿（当前 25 passed）。

## 演进路线

### 阶段 1：基础 Decoder-only GPT

- `model/mini_gpt.py`：Token / Position Embedding → N × DecoderBlock → LayerNorm → LM Head
- `model/attention.py`：CausalSelfAttention，softmax **前**屏蔽未来位置；`model/decoder_block.py`：Pre-LN + 残差；`model/feed_forward.py`
- `tests/test_model.py`：前向 `[B, T, V]` 形状、反向传播、未来位置 attention 权重 ≈ 0、超长输入抛 `ValueError`

### 阶段 2：KV Cache 增量推理 + 现代 LLM 结构

- `model/cached_gpt.py` / `cached_attention.py` / `cached_block.py`：Prefill 建立各层缓存，Decode 只算新 token；RoPE 位置偏移随缓存长度推进
- `model/rope.py`（作用于 Q/K 的旋转位置编码）、`model/rms_norm.py`（不减均值的均方根归一化）、`model/swiglu.py`（门控 FFN）
- `tests/test_kv_cache.py`：缓存形状、单步 / 多步 decode、分块 prefill 与全序列推理同位置 logits 一致性、矩形 attention mask、越界与非法 token 校验

### 阶段 3：MHA / GQA / MQA 统一实现

- `model/grouped_*.py`：以 `num_q_heads` / `num_kv_heads` 参数化——`H_kv = H_q` 即 MHA，`1 < H_kv < H_q` 即 GQA，`H_kv = 1` 即 MQA（约束 `H_q % H_kv == 0`）
- `tests/test_gqa.py`：三种头配置下 cached vs full logits 一致性、分块 prefill、多步 decode、KV Cache 显存缩减验证、非法头数校验，以及 **MHA 配置与阶段 1 实现的输出对齐**

关键一致性结论：`model.eval()` 下，KV Cache 逐步推理与全序列推理在同位置 logits 的最大绝对误差约 `2e-7`（float32，RoPE offset 正确处理）。演示见 `python -m demos.demo_cached`、`python -m demos.demo_gqa`。

## 快速开始

```bash
pip install -r requirements.txt   # torch>=2.2, pytest>=8
python -m pytest tests/ -v        # 25 个用例
python -m demos.demo_cached       # cache vs full 一致性演示
python -m demos.demo_gqa          # MQA 配置下的 KV 显存对比演示
```

## 目录结构

```text
MiniGPT/
├─ model/          # 全部模型模块（三阶段 + 共享的 rope/rms_norm/swiglu）
├─ tests/          # 三个阶段的 pytest 套件
├─ demos/          # 可运行演示脚本
├─ docs/           # 各阶段历史运行输出存档 + 最新测试报告
├─ conftest.py     # 使仓库根目录可被测试导入
└─ requirements.txt
```

## 局限与后续

- 教学规模实现（d_model 64–128、2–3 层），未做推理性能优化（如缓存 `torch.cat` 的拷贝开销、GQA 分组广播可进一步向量化）
- 后续计划：基于 nanoGPT 的字符级从零预训练、TRL + LoRA 的 DPO 偏好对齐实验（独立仓库）

## 参考

- [nanoGPT](https://github.com/karpathy/nanoGPT)（karpathy）
- RoPE: Su et al., *RoFormer*, 2021；RMSNorm: Zhang & Sennrich, 2019；SwiGLU: Shazeer, 2020；MQA: Shazeer, 2019；GQA: Ainslie et al., 2023
