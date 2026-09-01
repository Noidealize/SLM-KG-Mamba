# RUL semantic feasibility audit

## Scope

Static/code-level and synthetic-input feasibility audit only. No formal C-MAPSS
test result or predictive-effectiveness evidence was produced.

## Audited files

- Local: `experiments/rul_semantic/*`, `experiments/effectiveness/*`,
  `models/SMETimes_Llama.py`, `models/Preprocess_Llama.py`.
- External read-only: `config.py`, `data/data_preprocessing.py`,
  `knowledge_graph/backbone.py`, `models/ma_rdg_mamba.py`, and training entry points
  in `D:/CODEXPROJECT/MA_RDG_Mamba_RUL_complete_v1/ma_rdg_mamba_rul`.

## Findings

1. SLM caches are offline embeddings of manually written sensor cards, not knowledge extraction.
2. `transe`, `slm`, and `slm_shuffled` replace only the 20% similarity component.
   `none` previously retained a unit diagonal; it now supplies an exact-zero component.
3. KG construction restores Python/NumPy/Torch RNG state; DataLoaders use explicit seeds.
4. External preprocessing splits by engine and fits normalization on training engines only.
5. The old local trainer called `prepare_subset()`, which always opened train/test/RUL
   and evaluated test. The new loader opens only `train_<subset>.txt`.
6. External `MARDGMamba` registers `a_k` and passes it into `ResidualDynamicGraph`;
   graph context is fused into every Mamba token, showing a real static dependency.
7. External `disable_knowledge=True` substitutes identity but still calls/fuses the graph;
   it is not No-Graph. The local adapter defines true F0, F1, F2, and F3 paths.
8. Synthetic tests verify bypass and KG sensitivity. WSL2/CUDA smoke then verified
   F0-F3 forward/backward, checkpoint save, and reload with official `mamba_ssm`.
9. Output paths formerly omitted KG seed/config hash. The safe path includes both.
10. `s20=W21` was wrong and is corrected to `s20=W31`.
11. External input order is `s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21`.
    Exact constants differ: FD001 has s1/s5/s10/s16/s18/s19; FD003 has
    s1/s5/s16/s18/s19; FD002/FD004 have none. The fixed 14 is code policy, not a
    verified common subset-specific constant filter.
12. The SMETimes Llama files belong to the ETT branch and are not called by RUL code.

## Classification

- 已审计: call chain, data split, sensor order, seed roles.
- 已实现并通过单元测试: KG validation/projection, cache invariants, F0-F3 router.
- 已通过可行性 smoke: FD001 train/validation-only loading/window generation and
  F0-F3 official-Mamba forward/backward/save/reload in WSL2/CUDA.
- 尚未验证: real SLM cache generation because no SLM model/cache was supplied.
- 需要人工审核: every Reference KG edge and its evidence.
- 留待正式实验: metrics, multiple seeds, statistics, official test access.

No current result establishes effectiveness, robustness, formal Go/No-Go, or
readiness for Dual Trust.
