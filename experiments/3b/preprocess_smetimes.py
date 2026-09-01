# -*- coding: utf-8 -*-
"""
预处理包装器（论文基准数据集，LLM 换 3B）：
import patch_eager 后再运行官方 preprocess.py。
强制 --no_use_amp（fp32）：fp16 批量 LLM 前向在这块卡上产生 NaN（数值溢出），
与 1B 时代结论一致。3B fp32 权重 ~12.9GB，batch_size 建议 8~16。

用法:
    python preprocess_smetimes.py --dataset ETTh1 --batch_size 16 --llm_ckp_dir <3B目录>
注意: --mix_embeds 模式下 .pt 嵌入维度 = LLM hidden，换 3B 必须全部重新预处理。
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

import patch_eager  # noqa: E402,F401  必须在 models.* import 之前生效

if __name__ == "__main__":
    # 从独立实验目录运行时，让官方 preprocess.py 的 ./data 指向 dataset_3b，
    # 从而不覆盖主工程中 1B 使用的 ETTh1.pt 等嵌入。
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    isolated_dataset = os.path.join(exp_dir, "dataset_3b")
    expected_dataset = os.path.join(exp_dir, "data")
    if not os.path.exists(isolated_dataset):
        raise FileNotFoundError("请先运行 prepare_3b_data.ps1 创建独立的 3B 数据目录")
    if not os.path.exists(expected_dataset):
        raise FileNotFoundError("缺少 data 目录联接，请重新运行 prepare_3b_data.ps1")
    os.chdir(exp_dir)
    sys.argv = [os.path.join(SMETIMES_DIR, "scripts", "preprocess.py")] + sys.argv[1:]
    sys.argv += ["--no_use_amp"]  # 强制 fp32（放最后，argparse 后者覆盖前者）
    runpy.run_path(os.path.join(SMETIMES_DIR, "scripts", "preprocess.py"), run_name="__main__")
