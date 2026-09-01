# RUL semantic and graph feasibility layer

Status: diagnostic feasibility only. Outputs are not paper evidence and the safe
loader never opens official C-MAPSS test or RUL files.

The refactor separates two independent random processes:

- `model_seed`: model initialization and data-loader order;
- `kg_seed`: TransE knowledge-backbone construction, fixed across model seeds.

The semantic names are:

- `transe`: original TransE cosine similarity;
- `slm`: cached pretrained-SLM sensor embeddings;
- `slm_shuffled`: cached SLM embeddings with a deterministic sensor permutation;
- `none` / `no_semantic`: exact-zero semantic component, with other weights renormalized.
- explicit aliases: `semantic_similarity_slm` and `semantic_similarity_shuffled`.

The SLM path embeds manually authored sensor cards. It is not full SLM
entity/relation/evidence extraction. Cache loading fails closed on metadata,
sensor order, dimensions, NaN/Inf, zero norms, and semantic-matrix SHA256.

## Layout

- `semantic_backbone.py`: audited backbone construction and cache validation.
- `prepare_slm_semantics.py`: offline frozen-SLM embedding generator.
- `graph_paths.py`: explicit F0 bypass, F1 data graph, F2 Reference KG, and F3 fixed fusion.
- `predictor_adapter.py`: local external-model adapter without modifying the dependency.
- `data_protocol.py`: engine-level train/validation loader that never opens official test.
- `tests/`: CPU-only invariant tests.

Output paths include subset, semantic mode, graph path, model seed, KG seed, and
configuration hash. Protocols declare `diagnostic_feasibility_only`,
`paper_evidence=false`, and `official_test_used=false`.

## Frozen experiment inputs

- `fixed14_sensor_policy.json` freezes one ordered 14-sensor input contract for
  FD001-FD004; runtime or subset-specific sensor removal is forbidden.
- `formal_experiment_protocol.json` freezes F0-F3 comparisons, matched seeds,
  stage gates, metrics, and reviewed-KG artifact hashes.
- `validate_formal_protocol.py` checks the frozen artifacts and all four training
  files without resolving or opening official test/RUL files.
- `run_protocol.py` executes the frozen matched-seed pilot/confirmatory matrix,
  skips only hash-compatible completed runs, stops on the first failure, and writes
  one UTF-8 log per subset/path/seed under `<output-root>/logs/`.
- `summarize_pilot.py` audits Pilot completeness and exports validation results;
  missing jobs remain explicit and are not silently dropped.
