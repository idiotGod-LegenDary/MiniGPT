"""Week 04 · Day 06：将 LoRA 真正注入 MiniGPT Attention。

本文件是【仅新增文件】。不得覆盖 model/attention.py、model/mini_gpt.py、
training/adapter_io.py、training/loop.py 等之前已经完成的文件。

本课只为基础版 MiniGPT 的 CausalSelfAttention 中 W_q / W_v 线性层注入 LoRA；
保留原有前向实现，注入方式是替换模块对象而不是重新实现 Attention。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA 包装层：基础 Linear + 低秩增量，输入最后一维为 in_features。

    预期计算：base(x) + (alpha/rank) * B(A(dropout(x)))。
    这里 A、B 使用 nn.Parameter 存储，不单独建立 nn.Linear 模块。
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        """TODO 1：初始化可训练的 LoRA A/B 与冻结基础权重。

        参数：
        - base：现有 nn.Linear 层，必须【保留原对象】（不复制数值/不重置权重）。
        - rank：正整数（bool 非法），且不超过 min(in_features, out_features)。
        - alpha：有限的正实数（bool 非法）；缩放比例为 alpha/rank。
        - dropout：有限实数，范围 [0,1)，作用于 LoRA 分支输入。

        属性契约：
        - self.base 与传入 base 指向同一个 nn.Linear 对象；base 参数全部冻结。
        - self.rank、self.alpha、self.scale 为相应的 int/float 值。
        - self.lora_dropout 是 nn.Dropout(dropout)。
        - self.lora_A：形状 [rank, in_features] 的 nn.Parameter；Kaiming 初始化。
        - self.lora_B：形状 [out_features, rank] 的 nn.Parameter；全零初始化。
        - A/B 的 requires_grad 均为 True，保持 base 原有 dtype/device。

        错误：base 类型非法时 TypeError；rank/alpha/dropout 不合法时 ValueError。
        提示：可使用 torch.empty(..., device=base.weight.device, dtype=base.weight.dtype)、
              nn.init.kaiming_uniform_()、nn.init.zeros_()、param.requires_grad_(False)。
        """
        super().__init__()
        # TODO 1：完整参数校验、注册基础层与 LoRA 参数、正确初始化。
        if not isinstance(base,nn.Linear):
            raise TypeError("base type should be nn.Linear")       
        
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("rank type should be int")
        if rank <= 0 or rank > min(base.in_features,base.out_features):
            raise ValueError("rank error")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive real number")

        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be a finite real in [0, 1)")
        
        self.base = base 
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha/rank
        self.dropout = dropout
        
        base.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))

        self.lora_dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO 2：计算基础投影和 LoRA 增量之和。

        输入：任意前导维度的浮点 Tensor，最后一维 = base.in_features，
             如 [B,T,D] 或 [B,D]。
        输出：形状 [..., base.out_features]，保持基础层的 dtype/device。
        必须保留 Autograd 计算图，让 LoRA 参数能接收梯度。
        注意：不要原地改动 base.weight、不要在 forward 中 merge 或 detach。
        提示：F.linear(self.lora_dropout(x), self.lora_A)，
              再 F.linear(中间值, self.lora_B) * self.scale。
        """
        base_out = self.base(x)
        return base_out + F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B
        ) * self.scale


def inject_lora_attention(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("W_q", "W_v"),
) -> list[str]:
    """TODO 3：对真实 MiniGPT Attention 的指定投影层进行 LoRA 注入。

    参数：
    - model：现有 MiniGPT(nn.Module)，不要求更改其类定义。
    - rank/alpha/dropout：传给每个 LoRALinear。
    - target_modules：Attention 上要替换的投影名称，默认 ("W_q","W_v")。
      允许的名称为 W_q、W_k、W_v、out_proj；不得重复、不得为空。

    返回：排序后的全部 Adapter 参数完整名称，例如：
      ['net.0.attn.W_q.lora_A', 'net.0.attn.W_q.lora_B',
       'net.0.attn.W_v.lora_A', 'net.0.attn.W_v.lora_B', ...]

    必须：
    1. 只处理 model.attention.CausalSelfAttention 的目标子模块；
       可通过 model.named_modules() 找到它们。
    2. 注入前完成检查：模型中有 Attention、目标存在、均为未包装的 nn.Linear，
       配置合法，rank 不超过每个目标投影的输入/输出维度；
       任一项检查失败，不得出现【部分注入】或改变原模型冻结状态。
    3. 校验后用 LoRALinear(base=old_linear, ...) 包装，通过 setattr(attn, 名称, wrapper)
       安装到原 Attention，不修改模型 forward 实现。
    4. 收集所有 '.lora_A'、'.lora_B' 的完整参数名。
    5. 直接复用 Day 04 的 training.adapter_io.set_adapter_trainable()：
       仅 LoRA 参数可训练，模型其他所有参数都冻结。
    6. 二次注入同一个目标应拒绝（ValueError），不能叠加两层 LoRA。

    错误：无目标/未知或重复名称/已经注入/非法超参 -> ValueError；
          不正确的 model/target_modules 类型 -> TypeError。
    """
    # TODO 3：先完整校验，再包装投影，并复用 Day 04 冻结工具。
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    if isinstance(target_modules, str) or not isinstance(target_modules, Sequence):
        raise TypeError("target_modules must be a sequence of str")

    allowed = {"W_q", "W_k", "W_v", "out_proj"}
    targets = list(target_modules)
    if len(targets) == 0:
        raise ValueError("target_modules must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("target_modules contains duplicates")
    
    for t in targets:
        if not isinstance(t, str):
            raise TypeError("target_modules must contain str")
        if t not in allowed:
            raise ValueError(f"unknown target module: {t}")
        
    # ---- 收集 Attention 及其目标投影（先完整校验）----
    from model.attention import CausalSelfAttention  # 按实际路径导入

    attn_modules = []          # [(attn_qualname, attn_module)]
    to_wrap = []               # [(attn_module, attr_name, old_linear, qualname)]

    for name, module in model.named_modules(): # 取出CausalSelfAttention的所有子模块
        if isinstance(module, CausalSelfAttention):
            attn_modules.append((name, module))

    if not attn_modules:
        raise ValueError("no CausalSelfAttention found in model")

    for attn_name, attn in attn_modules:# 再从所有模块中找出需要的层

        for attr in targets: # 按照targets里的模块名寻找匹配
            if not hasattr(attn, attr):
                raise ValueError(f"{attn_name} has no target {attr}")
            
            old = getattr(attn, attr)
            if isinstance(old, LoRALinear):
                raise ValueError(f"{attn_name}.{attr} already injected")
            if not isinstance(old, nn.Linear):
                raise ValueError(f"{attn_name}.{attr} is not nn.Linear")
            if rank > min(old.in_features, old.out_features):
                raise ValueError(f"rank too large for {attn_name}.{attr}")
            
            to_wrap.append((attn, attr, old, f"{attn_name}.{attr}" if attn_name else attr))

    # ---- 校验通过后统一注入 ----
    for attn, attr, old, _qual in to_wrap:
        wrapper = LoRALinear(old, rank=rank, alpha=alpha, dropout=dropout)
        setattr(attn, attr, wrapper)       # nn.Module 属性 → 覆盖 _modules 同名条目

    # ---- 收集 LoRA 参数完整名 ----
    lora_names: list[str] = []
    for name, _ in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            lora_names.append(name)
    lora_names.sort()

    # ---- 复用 Day 04 冻结工具 ----
    from training.adapter_io import set_adapter_trainable
    set_adapter_trainable(model, lora_names)

    return lora_names

