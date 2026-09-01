# -*- coding: utf-8 -*-
"""训练日志审计：按 epoch 统计梯度爆炸/NaN 事件与每 100-iter 窗口的平均 loss，
定位模型崩溃的准确时间点。
用法：  D:/ANACONDA/envs/pytorch/python.exe audit_train_log.py
"""
import pathlib
import re

LOG = pathlib.Path(__file__).parent / "logs" / "train_ETTh1_3b.log"

IT = re.compile(r"iters: (\d+), epoch: (\d+) \| loss: ([\d.e+\-]+)")
EXPL = re.compile(r"\[Hybrid-Adam\] grad exploded")
NANZ = re.compile(r"\[Hybrid-Adam\] WARNING")
NANF = re.compile(r"\[NaN-guard\]")

wins = {}          # (epoch, iter) -> 累计 loss（每 epoch 内单调递增）
expl_by_epoch = {} # epoch -> 爆炸次数
nan_by_epoch = {}  # epoch -> WARNING 次数
guard_by_epoch = {}
epoch = 0
first_expl_line = None
guard_lines = []

for line in LOG.open(encoding="utf-16"):
    m = IT.search(line)
    if m:
        epoch, i, v = int(m.group(2)), int(m.group(1)), float(m.group(3))
        wins[(epoch, i)] = v
    elif EXPL.search(line):
        expl_by_epoch[epoch] = expl_by_epoch.get(epoch, 0) + 1
        if first_expl_line is None:
            first_expl_line = line.strip()
    elif NANZ.search(line):
        nan_by_epoch[epoch] = nan_by_epoch.get(epoch, 0) + 1
    elif NANF.search(line):
        guard_by_epoch[epoch] = guard_by_epoch.get(epoch, 0) + 1
        if len(guard_lines) < 3:
            guard_lines.append(line.strip())

print(f"第一条爆炸记录: {first_expl_line}")
print(f"NaN-guard 行样本: {guard_lines if guard_lines else '无'}")
print()

print(f"{'epoch':>5} {'爆炸次数':>8} {'爆炸率':>8} {'WARNING':>8} {'NaN-guard':>9}")
for e in sorted(set(expl_by_epoch) | set(nan_by_epoch) | set(guard_by_epoch)):
    n = expl_by_epoch.get(e, 0)
    rate = n / 6889 * 100
    print(f"{e:>5} {n:>8} {rate:>7.1f}% {nan_by_epoch.get(e, 0):>8} {guard_by_epoch.get(e, 0):>9}")

print()
print("每 100-iter 窗口平均 loss（仅显示每 epoch 的关键窗口）:")
prev_by_epoch = {}
for (e, i) in sorted(wins):
    v = wins[(e, i)]
    prev = prev_by_epoch.get(e)
    mean = v / i if i else v  # 累计均值到该窗口（近似）
    if prev is not None:
        window_mean = (v - prev) / 100.0
    else:
        window_mean = v / i
    prev_by_epoch[e] = v
    # 只打印每 epoch 首个窗口、loss 首次超过 1/100/1000 的窗口
    flag = ""
    if i == 100:
        flag = "epoch 首个窗口"
    if window_mean > 1000:
        flag = "!!! 窗口均值 > 1000"
    elif window_mean > 100:
        flag = "!! 窗口均值 > 100"
    elif window_mean > 1:
        flag = "! 窗口均值 > 1"
    if flag:
        print(f"  epoch {e:>2} iter {i:>5}: 窗口均值 {window_mean:>12.3f}  {flag}")
