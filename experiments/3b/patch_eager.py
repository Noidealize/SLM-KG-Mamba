# -*- coding: utf-8 -*-
"""
修复：transformers 5.15.0 的 sdpa 注意力在 RTX 5070 Ti (Blackwell) 上
批量前向非确定性产生 NaN（debug_attn.py 已证实：eager 3 次全干净，
sdpa 3 次中 2 次出现 2916~3483 个 NaN 值）。

做法：monkey-patch LlamaForCausalLM.from_pretrained 注入
attn_implementation="eager"，零侵入 SMETimes 源码。
在 import models.* 之前 import 本模块即可生效。
"""
import transformers
from transformers import LlamaForCausalLM

_orig = LlamaForCausalLM.from_pretrained.__func__


def _patched(cls, *args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    return _orig(cls, *args, **kwargs)


LlamaForCausalLM.from_pretrained = classmethod(_patched)
print("[patch_eager] LlamaForCausalLM.from_pretrained patched -> attn_implementation=eager")
