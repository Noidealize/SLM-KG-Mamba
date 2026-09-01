# -*- coding: utf-8 -*-
"""
训练包装器（论文基准数据集复现用，LLM 换 Llama-3.2-3B-Instruct）：
1. 加载 patch_eager（sdpa→eager，修复 RTX 5070 Ti 上 sdpa 内核的非确定性 NaN）
2. monkey-patch SMETimes_Llama.Model.forecast 加 NaN 检测重试
   （eager 下批量前向仍偶发瞬态 GPU NaN，重试一次即恢复，预处理已验证）
3. monkey-patch torch.optim.Adam/AdamW.step -> Hybrid-Adam + CPU 裁判
   （≥1M 元素≈4MB 的参数走 CPU 权威路径——RTX 5070 Ti 上大张量 elementwise
   更新内核系统性出错；小参数 GPU 原地更新 + CPU 副本每步重算比对，坏值即修复。
   同步点 ~10/步，CPU 侧逐元素运算限 2 线程，避免 CPU 满载 + WDDM 整机假死）
4. 冻结权重完整性校验：模型构造时记录参考哈希，每 2000 步与每次验证前重算，
   静默坏值立即中止（2026-08-30 epoch 3 崩毁审计的教训，详见下方注释）

用法（与直接调用 train.py 完全一致）:
    python train_smetimes.py --task_name long_term_forecast --is_training 1 ...
"""
import os
import sys
import runpy
from pathlib import Path

SMETIMES_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SMETIMES_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# CPU 侧 Adam 数学（37.8MB 逐元素运算）是内存带宽瓶颈，2 线程即可饱和；
# 放开线程数会瞬时打满全部核心，WDDM 下整机界面假死（2026-08-30 实测冻机）。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import patch_eager  # noqa: E402,F401  # 必须在 models.* import 之前生效

# 触发 models.SMETimes_Llama 加载后再 patch forecast
import models.SMETimes_Llama as SML  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)  # 与 OMP_NUM_THREADS 双保险：CPU 满载是冻机主因

_orig_forecast = SML.Model.forecast
MAX_RETRY = 10


def _slice1(t, i):
    return None if t is None else t[i:i + 1]


def _has_bad(t):
    # NaN/Inf 或幅值异常（标准化空间输出 >1e7 必为坏值；正常值 ±5 内）
    if (torch.isnan(t) | torch.isinf(t)).any():
        return True
    return bool((t.abs() > 1e7).any())


def _fix_input(t, name):
    """修复输入张量中的坏值（H2D 拷贝 kernel 概率性写坏）：
    x_enc 归一化值域 ±5、x_mark 嵌入值域 ±50，>1e6 必为坏值。
    torch.where 保持梯度流（坏位置梯度为 0，好位置正常）。"""
    if t is None:
        return t
    bad = (torch.isnan(t) | torch.isinf(t)) | (t.abs() > 1e6)
    if bad.any():
        t = torch.where(bad, torch.zeros_like(t), t)
        print(f"[NaN-guard] {name} had {bad.sum().item()} bad elems -> zeroed")
    return t


def _safe_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    """NaN/Inf 兜底：整批 forward → 若含 NaN/Inf，逐样本定位坏样本 → 用好样本替换后重跑。
    与预处理重试一致：坏样本单独重跑通常变干净（瞬态 GPU 内核问题），
    但训练需要保持 batch 梯度流，所以用替换策略。"""
    for attempt in range(MAX_RETRY):
        x_enc = _fix_input(x_enc, "x_enc")
        x_mark_enc = _fix_input(x_mark_enc, "x_mark_enc")
        out = _orig_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not _has_bad(out):
            return out

        bs = x_enc.shape[0]
        # 1) 逐样本定位（no_grad 仅检测）
        bad = []
        with torch.no_grad():
            for i in range(bs):
                oi = _orig_forecast(self, _slice1(x_enc, i), _slice1(x_mark_enc, i),
                                    _slice1(x_dec, i), _slice1(x_mark_dec, i))
                if _has_bad(oi):
                    bad.append(i)
        if not bad:
            print(f"[NaN-guard] batch bad but no single-sample bad (attempt {attempt + 1}), retrying whole batch...")
            continue  # 纯 batch 组合问题，重跑整批

        # 2) 坏样本替换为 batch 内好样本（保持 requires_grad 以维持梯度流）
        good = [i for i in range(bs) if i not in bad]
        src = good[0] if good else None
        x_enc_f = x_enc.clone()
        x_mark_enc_f = x_mark_enc.clone() if x_mark_enc is not None else None
        for i in bad:
            if src is not None:
                x_enc_f[i:i + 1] = x_enc_f[src:src + 1]
                if x_mark_enc_f is not None:
                    x_mark_enc_f[i:i + 1] = x_mark_enc_f[src:src + 1]
            else:
                x_enc_f[i:i + 1] = 0.0
                if x_mark_enc_f is not None:
                    x_mark_enc_f[i:i + 1] = 0.0
            print(f"[NaN-guard] attempt {attempt + 1}: replaced bad sample {i} "
                  f"(x_enc min/max = {x_enc[i].min().item():.4f}/{x_enc[i].max().item():.4f})")
        x_enc, x_mark_enc = x_enc_f, x_mark_enc_f
        x_enc = _fix_input(x_enc, "x_enc(repl)")
        x_mark_enc = _fix_input(x_mark_enc, "x_mark_enc(repl)")
        out = _orig_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not _has_bad(out):
            return out
        print(f"[NaN-guard] attempt {attempt + 1}: still bad after replacement, retrying...")
    # 诊断：重试耗尽时 dump 可训练参数状态，定位坏权重
    print("=== guard exhausted, dumping trainable params ===")
    for n, pp in self.named_parameters():
        if pp.requires_grad:
            v = pp.detach()
            print(f"  {n} {list(v.shape)}: NaN={torch.isnan(v).sum().item()} "
                  f"Inf={torch.isinf(v).sum().item()} max_abs={v.abs().max().item():.3e}")
    raise RuntimeError(f"LLM 前向 NaN/Inf 重试 {MAX_RETRY} 次仍失败")


SML.Model.forecast = _safe_forecast
print("[train_smetimes] SMETimes_Llama.Model.forecast patched with NaN-guard (locate+replace)")

# ---------------------------------------------------------------------------
# 完整性校验（2026-08-30 审计结论驱动的加固）：
# 首次完整训练在 epoch 3 末静默崩毁：全程零 NaN、所有防护（NaN-guard、
# 爆炸跳过、CPU 读回校验）都没触发，但 loss 从 0.65 骤升 2 万+，之后每步
# ~5/8 参数梯度爆炸被跳过、模型冻结在坏状态。说明权重被"有限坏值"写坏
# （本机 GPU 已知故障：1e22~5e25 级有限坏值，isnan 拦不住）。两个盲区：
#   1) 小参数 GPU 原地更新没有权威校验（Hybrid-Adam 为治冻机的取舍）；
#   2) 冻结 Llama 权重（12GB）加载后从未再校验。
# 本块补齐盲区 2：
#   - 模型构造完成（llama 已冻结）时记录冻结权重参考哈希（fp64 分块求和，
#     16M 元素/块，防止 fp64 转换把 16GB 显存撑爆）；
#   - 每 2000 步 + 每次 model.eval()（验证/测试前）全量重算比对；
#   - 不一致先重算一次（排除 GPU 瞬态坏读），仍不一致立即中止训练——
#     坏权重继续训练只会浪费 GPU 时间，且可能把坏检查点存成"最优"。
# 盲区 1 由 Hybrid-Adam 的小参数 CPU 裁判补齐（见 _make_hybrid_step）。
# ---------------------------------------------------------------------------

_INTEGRITY = {}          # "model" -> 冻结的 llama 模型；"ref" -> (n, sum, sumsq)
_STEP_COUNT = 0          # 全局训练步数（Adam/AdamW 两个 patch 实例共享）
_LAST_CHECK_STEP = -10 ** 9
_INTEGRITY_INTERVAL = 2000   # 每 2000 步（约 4 分钟）全量校验一次
_CHUNK = 16 * 1024 * 1024    # fp64 分块尺寸：16M 元素 = 128MB 暂存


def _hash_frozen(model):
    """冻结权重哈希：n 个元素、fp64 分块 sum 与 sumsq。
    fp64 保证 CPU 参考与 GPU 校验间误差 ~1e-13，比对阈值可取 1e-9。"""
    n, s, sq = 0, 0.0, 0.0
    for p in model.parameters():
        if p.requires_grad:
            continue
        flat = p.detach().reshape(-1)
        for i in range(0, flat.numel(), _CHUNK):
            c = flat[i:i + _CHUNK].to(torch.float64)
            n += c.numel()
            s += c.sum().item()
            sq += (c * c).sum().item()
    return n, s, sq


def _integrity_check(reason=""):
    """冻结 Llama 权重与参考哈希比对；不一致重算一次，仍不一致立即中止。"""
    global _LAST_CHECK_STEP
    _LAST_CHECK_STEP = _STEP_COUNT
    model = _INTEGRITY.get("model")
    ref = _INTEGRITY.get("ref")
    if model is None or ref is None:
        return
    n, s, sq = _hash_frozen(model)
    if not (n == ref[0] and abs(s - ref[1]) <= 1e-9 * max(1.0, abs(ref[1]))
            and abs(sq - ref[2]) <= 1e-9 * max(1.0, abs(ref[2]))):
        n2, s2, sq2 = _hash_frozen(model)  # GPU 坏读可能是瞬时的：重算一次
        if not (n2 == ref[0] and abs(s2 - ref[1]) <= 1e-9 * max(1.0, abs(ref[1]))
                and abs(sq2 - ref[2]) <= 1e-9 * max(1.0, abs(ref[2]))):
            print("=" * 72)
            print(f"[integrity] INTEGRITY FAILURE at step {_STEP_COUNT} ({reason})")
            print(f"  参考: n={ref[0]}, sum={ref[1]:.6e}, sumsq={ref[2]:.6e}")
            print(f"  实际: n={n}, sum={s:.6e}, sumsq={sq:.6e}")
            print("  冻结 Llama 权重被静默改坏（有限坏值，NaN 检测拦不住），训练立即中止。")
            print("  请保留本日志；本次运行的检查点与 result 文件均不可用，需重新训练。")
            print("=" * 72)
            sys.exit(1)
        print(f"[integrity] step {_STEP_COUNT}: 首次校验坏读（瞬态），重算通过 ({reason})")
        return
    print(f"[integrity] frozen-weights hash OK at step {_STEP_COUNT} ({reason})")


def _find_llama(module):
    for m in module.modules():
        if type(m).__name__ == "LlamaForCausalLM":
            return m
    return None


_orig_model_init = SML.Model.__init__


def _model_init_with_ref(self, *args, **kwargs):
    r = _orig_model_init(self, *args, **kwargs)  # 返回后 llama 已冻结（requires_grad=False）
    try:
        llama = _find_llama(self)
        if llama is None:
            print("[integrity] WARNING: 未找到 LlamaForCausalLM 子模型，冻结权重校验停用")
        else:
            ref = _hash_frozen(llama)
            if ref[0] == 0:
                print("[integrity] WARNING: llama 没有冻结参数（requires_grad=False），校验停用")
            else:
                _INTEGRITY["model"] = llama
                _INTEGRITY["ref"] = ref
                print(f"[integrity] 冻结权重参考已记录: {ref[0]} 个元素, "
                      f"sum={ref[1]:.6e}, sumsq={ref[2]:.6e}")
    except Exception as e:
        print(f"[integrity] WARNING: 参考记录失败（{e}），校验停用")
    return r


SML.Model.__init__ = _model_init_with_ref
print("[train_smetimes] SMETimes_Llama.Model.__init__ patched -> record frozen-weights reference")

_orig_module_eval = torch.nn.Module.eval


def _module_eval_with_check(self):
    if _STEP_COUNT - _LAST_CHECK_STEP > 500:
        _integrity_check("model.eval()（验证/测试前）")
    return _orig_module_eval(self)


torch.nn.Module.eval = _module_eval_with_check
print("[train_smetimes] nn.Module.eval patched -> integrity check before validation/testing")

# ---------------------------------------------------------------------------
# Hybrid-Adam：小参数 GPU 原地更新，大参数（≥CPU_BIG_NUMEL 元素≈4MB）CPU 权威更新。
# 背景（本机实测）：
#   - RTX 5070 Ti 上 ≥16MB 张量的 GPU addcmul/addcdiv（Adam 更新路径）以 ~5% 概率
#     产出 NaN/Inf，且非 NaN 时数值与 CPU 参考相差 ~1e2；CPU 更新 20/20 干净。
#     → experts.0.weight [3072,3072]=37.8MB 必须走 CPU 权威路径（防护与旧版一致）。
#   - 全量 CPU-Adam 时，每步 8 参数 × (D2H + 双 H2D + 校验) ≈ 40+ 次 GPU 同步 +
#     多线程 CPU 逐元素运算：CPU 满载、GPU 利用率低（观察 0.137s/iter 大部分
#     耗在同步与拷贝），整机满载还导致界面假死/冻机（2026-08-30 冻在 epoch 3）。
#   - 小参数（encoder/decoder/gate/fusion_gate/add_scale 共 ~2.4MB）GPU 原地更新
#     + CPU 裁判：每步把净化后的梯度双重 D2H 到 CPU，用与 GPU 完全相同的
#     决策/公式重算，再与 GPU 结果比对（阈值 1e-3，正常偏差 ~1e-7）；不一致
#     即 GPU 静默损坏，用 CPU 权威值写回修复。来源：2026-08-30 首次完整训练
#     在 epoch 3 末崩毁——全程零 NaN、零防护触发，权重被"有限坏值"写坏，
#     小参数无权威校验正是当时的盲区。
# ---------------------------------------------------------------------------
import torch.optim  # noqa: E402

CPU_BIG_NUMEL = 1_000_000  # ≥1M 元素（fp32≈4MB）走 CPU；实测坏值仅出现在 ≥16MB，留 4 倍余量


def _sanitize(t, label):
    """把 NaN/Inf 位置清零（backward GPU kernel 偶发瞬态 NaN 的兜底）。
    m/v 状态不被污染，其余有限梯度照常贡献更新。"""
    bad = torch.isnan(t) | torch.isinf(t)
    n = bad.sum().item()
    if n:
        t = t.clone()
        t[bad] = 0.0
        print(f"[Hybrid-Adam] WARNING: {label} had {n} NaN/Inf -> zeroed")
    return t


def _make_hybrid_step(decoupled_wd=False, max_grad=10.0):
    @torch.no_grad()
    def _hybrid_step(self, closure=None):
        global _STEP_COUNT
        _STEP_COUNT += 1
        if _STEP_COUNT % _INTEGRITY_INTERVAL == 0:
            _integrity_check(f"每 {_INTEGRITY_INTERVAL} 步定期校验")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # 逐参数更新。grad 先净化 NaN/Inf（清零），再检测爆炸：
        # 若某参数 grad 出现过大的值（>max_grad，正常值 0.05~0.8，爆炸可达 1e36），
        # 该参数整步跳过（清零）。爆炸梯度的方向不可信（llama backward 数值不稳定），
        # 且 Adam 对巨大梯度天然饱和（更新≈±lr），跳过几乎无损失。
        # 不能 clamp：实测 clamp 让 encoder 沿垃圾方向持续游走，训练发散至 inf；
        # 也不能用全局 norm clip：encoder 的爆炸会把 coef 拖到 0，连累 experts/decoder。
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group.get("weight_decay", 0.0)
            big, small = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                (big if p.numel() >= CPU_BIG_NUMEL else small).append(p)

            # ---- 小参数：CPU 决定 + GPU 执行 + CPU 裁判复算（坏值即修复） ----
            # 流程（2026-08-30 epoch 3 崩毁审计的加固）：
            #   1) 全部小参数梯度合并成一个缓冲区，双重 D2H 读回（NaN 感知比对，
            #      坏读重读一次；仍不一致则本步全部小参数跳过更新——绝不冒险，
            #      且 GPU/CPU 双方状态仍同步前进，不会出现裁判掉队）；
            #   2) CPU 侧做净化/爆炸决策，把清洗后的梯度写回 p.grad（H2D）；
            #   3) GPU 用清洗后梯度原地 Adam——与 CPU 裁判输入完全一致；
            #   4) CPU 用同样输入复算 Adam（~2.4MB，毫秒级），与 GPU 结果比对
            #      （阈值 1e-3，正常偏差 ~1e-7）；不一致（含 NaN）即 GPU 静默
            #      损坏，用 CPU 权威值双拷贝投票写回修复。
            # 同步开销正常路径 ~3 次/步。
            if small:
                grad_shapes = [tuple(p.grad.shape) for p in small]
                grad_numels = [p.grad.numel() for p in small]
                g_flat = torch.cat([p.grad.detach().reshape(-1) for p in small])
                g1 = g_flat.to("cpu")
                g2 = g_flat.to("cpu")
                same = lambda a, b: bool(
                    torch.eq(a, b).logical_or(torch.isnan(a) & torch.isnan(b)).all())
                if not same(g1, g2):  # 坏读：重读一次
                    g3 = g_flat.to("cpu")
                    if not same(g2, g3):
                        print("[Hybrid-Adam] WARNING: grad D2H 坏读持续 -> 本步全部小参数跳过更新")
                        g1 = torch.zeros_like(g2)
                    else:
                        g1 = g3
                # CPU 侧净化/爆炸决策（GPU 使用写回的同一份梯度，双端输入一致）
                clean_slices = []
                off = 0
                for shp, n in zip(grad_shapes, grad_numels):
                    sl = g1[off:off + n]
                    off += n
                    if (torch.isnan(sl) | torch.isinf(sl)).any():
                        sl = sl.clone()
                        sl[torch.isnan(sl) | torch.isinf(sl)] = 0.0
                        print(f"[Hybrid-Adam] WARNING: grad{list(shp)} had NaN/Inf -> zeroed")
                    if max_grad and bool(sl.abs().max() > max_grad):
                        sl = torch.zeros_like(sl)
                        print(f"[Hybrid-Adam] grad exploded -> skip update on {list(shp)}")
                    clean_slices.append(sl)
                for p, sl in zip(small, clean_slices):
                    p.grad.copy_(sl.reshape_as(p.grad))  # H2D 写回清洗后梯度
                for p in small:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        # CPU 裁判副本（初始值 = 当前参数 + 零动量）
                        state["cpu_w"] = p.detach().to("cpu").clone()
                        state["cpu_m"] = torch.zeros_like(p, device="cpu")
                        state["cpu_v"] = torch.zeros_like(p, device="cpu")
                    state["step"] += 1
                    t = state["step"]
                    g = p.grad
                    if decoupled_wd:
                        if wd != 0:
                            p.mul_(1 - lr * wd)
                    elif wd != 0:
                        g = g.add(p, alpha=wd)
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    m_hat = m / (1 - b1 ** t)
                    v_hat = v / (1 - b2 ** t)
                    p.addcdiv_(m_hat, v_hat.sqrt_().add_(eps), value=-lr)
                    if t % 50 == 0 and (torch.isnan(p).any() or torch.isinf(p).any()):
                        print(f"[Hybrid-Adam] WARNING: GPU param {list(p.shape)} NaN/Inf at step {t}")
                # CPU 裁判复算（输入 = clean_slices，与 GPU 完全一致）
                cpu_w_new = []
                for p, sl in zip(small, clean_slices):
                    st = self.state[p]
                    t = st["step"]
                    g_c = sl.reshape(p.shape)
                    pw = st["cpu_w"]
                    if decoupled_wd:
                        if wd != 0:
                            pw = pw * (1 - lr * wd)
                    elif wd != 0:
                        g_c = g_c + pw * wd
                    m_c, v_c = st["cpu_m"], st["cpu_v"]
                    m_c.mul_(b1).add_(g_c, alpha=1 - b1)
                    v_c.mul_(b2).addcmul_(g_c, g_c, value=1 - b2)
                    m_hat = m_c / (1 - b1 ** t)
                    v_hat = v_c / (1 - b2 ** t)
                    pw = pw.addcdiv(m_hat, v_hat.sqrt().add(eps), value=-lr)
                    st["cpu_w"] = pw
                    st["cpu_m"], st["cpu_v"] = m_c, v_c
                    cpu_w_new.append(pw)
                # 与 GPU 结果比对（合并一次 D2H），不一致用 CPU 权威值修复
                p_flat = torch.cat([p.detach().reshape(-1) for p in small]).to("cpu")
                off = 0
                for p, pw in zip(small, cpu_w_new):
                    n = pw.numel()
                    p_cpu = p_flat[off:off + n]
                    off += n
                    diff = (p_cpu - pw.reshape(-1)).abs().max().item()
                    if not (diff <= 1e-3):  # 正常 GPU/CPU 偏差 ~1e-7；NaN/Inf 也进此分支
                        print(f"[Hybrid-Adam] WARNING: 小参数 {list(p.shape)} 与 CPU 裁判不一致 "
                              f"(diff={diff:.3e}) -> 用 CPU 权威值修复")
                        for _ in range(3):
                            p1 = pw.to(p.device)
                            p2 = pw.to(p.device)
                            if torch.equal(p1, p2):
                                p.data.copy_(p1)
                                if torch.equal(p.data, p1):
                                    break
                        else:
                            print(f"[Hybrid-Adam] 修复失败 on {list(p.shape)} -> 下一步重试")

            # ---- 大参数：CPU 权威路径（与旧版完全一致的防护链） ----
            for p in big:
                g_cpu = _sanitize(p.grad.detach().to("cpu"), f"grad{list(p.shape)}")
                if max_grad and g_cpu.abs().max().item() > max_grad:
                    g_cpu.zero_()
                    print(f"[Hybrid-Adam] grad exploded -> skip update on {list(p.shape)}")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, device="cpu")
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu")
                state["step"] += 1
                t = state["step"]
                # GPU->CPU 读回验证：读回 kernel 也会概率性写坏（实测读回 5e25 级
                # 有限坏值，sanitize 拦不住）。与 CPU 权威副本对比，不一致则用权威
                # 副本（GPU 存储坏由写回时的投票/重分配修复）。
                p_cpu = _sanitize(p.detach().to("cpu"), f"param{list(p.shape)}")
                if "cpu_w" in state:
                    diff = float((p_cpu - state["cpu_w"]).abs().max())
                    if diff > 0.05:  # 正常每步更新幅度 <= lr*10 = 0.01
                        print(f"[Hybrid-Adam] GPU readback corrupted on {list(p.shape)} "
                              f"(diff={diff:.2e}) -> use authoritative CPU copy")
                        p_cpu = state["cpu_w"].clone()
                if decoupled_wd:
                    if wd != 0:
                        p_cpu.mul_(1 - lr * wd)
                elif wd != 0:
                    g_cpu.add_(p_cpu, alpha=wd)
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(b1).add_(g_cpu, alpha=1 - b1)
                v.mul_(b2).addcmul_(g_cpu, g_cpu, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                p_cpu.addcdiv_(m_hat, v_hat.sqrt_().add_(eps), value=-lr)
                assert not torch.isnan(p_cpu).any(), f"CPU update NaN on {p.shape}"
                state["cpu_w"] = p_cpu.detach().clone()
                # 写回 GPU：坏值定位 + 精准重写。GPU copy/读路径会概率性坏
                # （flaky 显存，实测 1e22~5e25 级有限坏值），坏位置通常少数。
                # 策略：整张拷贝 → 定位坏位置 → 只重写坏位置，最多 12 轮；
                # 仍坏则重分配存储（绕开坏页）；再坏则跳过本参数本次更新
                # （GPU 上保留旧值，下一步会再次尝试修复）——绝不崩溃训练。
                for retry in range(12):
                    # 双拷贝投票：两次独立 H2D 一致才认为源拷贝可信
                    p1 = p_cpu.to(p.device)
                    p2 = p_cpu.to(p.device)
                    if not torch.equal(p1, p2):
                        continue  # H2D 源拷贝坏，重来
                    p.data.copy_(p1)
                    mask = p.data != p1
                    n_bad = int(mask.sum())
                    if n_bad == 0:
                        break
                    if retry >= 2:
                        print(f"[Hybrid-Adam] patch {n_bad} bad elems on {list(p.shape)} "
                              f"(round {retry + 1})")
                    p.data[mask] = p1[mask]
                else:
                    print(f"[Hybrid-Adam] reallocating storage for {list(p.shape)}")
                    p.data = p_cpu.clone().to(p.device)
                    p1 = p_cpu.to(p.device)
                    if not torch.equal(p.data, p1):
                        print(f"[Hybrid-Adam] storage still broken on {list(p.shape)} "
                              f"-> skip update this step")
        return loss

    return _hybrid_step


torch.optim.Adam.step = _make_hybrid_step(decoupled_wd=False)
torch.optim.AdamW.step = _make_hybrid_step(decoupled_wd=True)
print("[train_smetimes] torch.optim.Adam/AdamW.step patched -> hybrid + CPU referee "
      "(big: CPU authoritative; small: GPU update + per-step CPU verify/heal)")

if __name__ == "__main__":
    sys.argv = [os.path.join(SMETIMES_DIR, "train.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(SMETIMES_DIR, "train.py"), run_name="__main__")
