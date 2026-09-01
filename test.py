# -*- coding: utf-8 -*-
"""测试入口：等价于 python train.py --is_training 0 <args>。

用法:
    python test.py --task_name long_term_forecast --model_id ETTh1_672_96 ...

测试阶段会读取 results/checkpoints/ 下的检查点，预测图与指标写入
results/test_results/ 和 results/result_long_term_forecast.txt。
"""
import os
import runpy
import sys

_TRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

if __name__ == "__main__":
    sys.argv = [_TRAIN_PATH, "--is_training", "0"] + sys.argv[1:]
    runpy.run_path(_TRAIN_PATH, run_name="__main__")
