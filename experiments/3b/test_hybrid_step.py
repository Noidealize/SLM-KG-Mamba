# -*- coding: utf-8 -*-
"""Hybrid-Adam + CPU 裁判 + 冻结权重校验的逻辑验证（纯 CPU，秒级，不碰 GPU、不启动训练）。

从 train_smetimes.py 提取 _sanitize/_make_hybrid_step/_hash_frozen/_integrity_check
源码（避免 import 触发 patch_eager 与模型加载），验证：
  1. 小参数 GPU 分支（CPU 张量上跑同代码路径）+ 大参数 CPU 分支的更新结果
     与 torch 原生 Adam/AdamW（foreach=False）逐位一致；
  2. 爆炸梯度（>10）整步跳过；
  3. grad NaN/Inf 净化；
  4. 小参数 CPU 裁判：GPU 侧被写入有限坏值/NaN 后，裁判检出并用 CPU 权威值
     修复，修复后与无损坏参照一致（2026-08-30 epoch 3 崩毁教训的加固）；
  5. 冻结权重哈希校验：正常路径通过；坏值检出后立即中止（SystemExit(1)）。
用法：  D:/ANACONDA/envs/pytorch/python.exe test_hybrid_step.py
"""
import ast
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).parent.joinpath("train_smetimes.py").read_text(encoding="utf-8")
compile(SRC, "train_smetimes.py", "exec")  # 语法检查

# _make_hybrid_step 引用的模块级全局：步数计数器、校验间隔（置超大值使
# 定期校验不触发）、校验函数桩（本测试单独验证 _integrity_check）。
ns = {"torch": torch, "_STEP_COUNT": 0, "_INTEGRITY_INTERVAL": 2 ** 62,
      "_integrity_check": lambda *a, **k: None}
for node in ast.parse(SRC).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CPU_BIG_NUMEL"
                                             for t in node.targets):
        exec(compile(ast.Module([node], type_ignores=[]), "<t>", "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name in ("_sanitize", "_make_hybrid_step"):
        exec(compile(ast.Module([node], type_ignores=[]), "<t>", "exec"), ns)
_sanitize, _make_hybrid_step = ns["_sanitize"], ns["_make_hybrid_step"]
CPU_BIG_NUMEL = ns["CPU_BIG_NUMEL"]


def build():
    return [torch.nn.Parameter(torch.randn(5, 3) * 0.1),          # 小参数
            torch.nn.Parameter(torch.randn(1) * 0.1),             # 标量小参数
            torch.nn.Parameter(torch.randn(CPU_BIG_NUMEL + 7) * 0.1)]  # 触发大参数 CPU 路径


def check_vs_native(opt_cls, hybrid_decoupled, **kw):
    torch.manual_seed(0)
    p_ref, p_hyb = build(), build()
    for a, b in zip(p_ref, p_hyb):
        b.data.copy_(a.data)
    ref = opt_cls(p_ref, lr=0.01, weight_decay=0.01, foreach=False, **kw)
    hyb = opt_cls(p_hyb, lr=0.01, weight_decay=0.01, foreach=False, **kw)
    orig_step = opt_cls.step  # 保存原生实现
    patched = _make_hybrid_step(decoupled_wd=hybrid_decoupled)
    for _ in range(3):
        grads = [torch.randn_like(p) for p in p_ref]
        for p, g in zip(p_ref, grads):
            p.grad = g
        for p, g in zip(p_hyb, grads):
            p.grad = g.clone()
        orig_step(ref)
        patched(hyb)
    worst = 0.0
    for a, b in zip(p_ref, p_hyb):
        worst = max(worst, (a - b).abs().max().item())
    print(f"{opt_cls.__name__} 3 步更新后与原生实现最大偏差: {worst:.3e}")
    assert worst < 1e-6, f"偏差过大: {worst}"


def check_skip_and_sanitize():
    torch.manual_seed(0)
    p = build()
    hyb = torch.optim.Adam(p, lr=0.01)
    patched = _make_hybrid_step(decoupled_wd=False)
    # 小参数爆炸（100 > 10）：该参数本步应保持不变（首步 m/v 为零 → 更新量为 0）
    p[0].grad = torch.full((5, 3), 100.0)
    # 小参数 grad 带 NaN：净化后正常更新
    g1 = torch.randn(1)
    g1[0] = float("nan")
    p[1].grad = g1
    # 大参数正常
    p[2].grad = torch.randn(CPU_BIG_NUMEL + 7)
    before0 = p[0].detach().clone()
    patched(hyb)
    assert torch.equal(p[0], before0), "爆炸参数应整步跳过"
    assert not torch.isnan(p[1]).any(), "NaN grad 净化后参数不应出现 NaN"
    # CPU 裁判状态应已建立
    assert "cpu_w" in hyb.state[p[1]] and "cpu_m" in hyb.state[p[1]], "裁判状态未建立"
    print("爆炸跳过 + NaN 净化 + 裁判状态建立: OK")


def check_referee_heal():
    torch.manual_seed(0)

    def build_small():
        return [torch.nn.Parameter(torch.randn(5, 3) * 0.1),
                torch.nn.Parameter(torch.randn(1) * 0.1)]

    p_a, p_b = build_small(), build_small()
    for a, b in zip(p_a, p_b):
        b.data.copy_(a.data)
    hyb_a, hyb_b = torch.optim.Adam(p_a, lr=0.01), torch.optim.Adam(p_b, lr=0.01)
    patched = _make_hybrid_step(decoupled_wd=False)
    for step in range(3):
        grads = [torch.randn_like(q) for q in p_a]
        for q, g in zip(p_a, grads):
            q.grad = g
        for q, g in zip(p_b, grads):
            q.grad = g.clone()
        if step == 2:
            # 模拟 GPU 静默损坏：有限坏值 + NaN（epoch 3 崩毁的形态，isnan 拦不住）
            with torch.no_grad():
                p_b[0][0, 0] += 1e5
                p_b[1][0] = float("nan")
        patched(hyb_a)
        patched(hyb_b)
    for a, b in zip(p_a, p_b):
        assert not torch.isnan(b).any(), f"修复后仍有 NaN: {b.shape}"
        assert torch.allclose(a, b, atol=1e-6), "裁判修复后应与无损坏参照一致"
    print("小参数裁判：有限坏值/NaN 检出 + CPU 权威值修复 OK")


def check_referee_state_continuity():
    # 步骤 1 带 NaN 梯度、步骤 2-3 干净：CPU 裁判状态必须全程同步。
    # 回归测试 2026-08-30 修复的缺陷：NaN 梯度曾被 torch.equal 误判成
    # "D2H 坏读"而跳过裁判更新，导致 CPU 副本状态掉队、后续误修复。
    torch.manual_seed(0)
    p_hyb = [torch.nn.Parameter(torch.randn(4, 2) * 0.1)]
    p_ref = [torch.nn.Parameter(p_hyb[0].detach().clone())]
    hyb = torch.optim.Adam(p_hyb, lr=0.01)
    ref = torch.optim.Adam(p_ref, lr=0.01, foreach=False)
    patched = _make_hybrid_step(decoupled_wd=False)
    for step in range(3):
        g = torch.randn(4, 2)
        if step == 0:
            g[1, 0] = float("nan")      # 应被净化（=0）而非触发"坏读"分支
        p_hyb[0].grad = g.clone()
        p_ref[0].grad = g.clone()
        if step == 0:
            p_ref[0].grad[1, 0] = 0.0   # 原生参照：仅第 0 步人工净化同一位置
        patched(hyb)
        ref.step()
    worst = (p_hyb[0] - p_ref[0]).abs().max().item()
    assert worst < 1e-6, f"NaN 净化后应等价于原生 Adam（人工净化参照），偏差 {worst}"
    # 裁判状态与 GPU 参数应逐位一致（无掉队）
    track = (p_hyb[0].detach().to("cpu") - hyb.state[p_hyb[0]]["cpu_w"]).abs().max().item()
    assert track < 1e-6, f"裁判状态掉队: {track}"
    print("NaN 梯度后裁判状态连续同步: OK")


def check_integrity():
    ns2 = {"torch": torch, "sys": sys, "_INTEGRITY": {}, "_STEP_COUNT": 0,
           "_LAST_CHECK_STEP": -10 ** 9, "_CHUNK": 4}
    for node in ast.parse(SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_hash_frozen", "_integrity_check"):
            exec(compile(ast.Module([node], type_ignores=[]), "<t>", "exec"), ns2)
    m = torch.nn.Module()
    m.a = torch.nn.Parameter(torch.randn(100))
    m.b = torch.nn.Parameter(torch.randn(50))
    for p in m.parameters():
        p.requires_grad = False
    ns2["_INTEGRITY"]["model"] = m
    n, s, sq = ns2["_hash_frozen"](m)
    assert n == 150, f"哈希元素数错误: {n}"
    ns2["_INTEGRITY"]["ref"] = (n, s, sq)
    ns2["_integrity_check"]("测试")  # 正常路径：打印 OK，不中止
    with torch.no_grad():
        m.a[3] += 1e6  # 有限坏值
    try:
        ns2["_integrity_check"]("测试")
        raise AssertionError("冻结权重被改坏后应中止（SystemExit）")
    except SystemExit as e:
        assert e.code == 1
    print("冻结权重哈希校验：OK 路径 + 坏值中止路径 OK")


if __name__ == "__main__":
    check_vs_native(torch.optim.Adam, hybrid_decoupled=False)      # 非 decoupled wd
    check_vs_native(torch.optim.AdamW, hybrid_decoupled=True)      # decoupled wd
    check_skip_and_sanitize()
    check_referee_state_continuity()
    check_referee_heal()
    check_integrity()
    print("ALL PASSED")
