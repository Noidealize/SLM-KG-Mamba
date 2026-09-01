# MA-RDG-Mamba Semantic Refactor

This directory is an additive refactor of `MA_RDG_Mamba_RUL_complete_v1`.
It does not overwrite the original training scripts or results.

The refactor separates two independent random processes:

- `model_seed`: model initialization and data-loader order;
- `kg_seed`: TransE knowledge-backbone construction, fixed across model seeds.

It also makes the 20% semantic component of the mechanism backbone replaceable:

- `transe`: original TransE cosine similarity;
- `slm`: cached pretrained-SLM sensor embeddings;
- `slm_shuffled`: cached SLM embeddings with a deterministic sensor permutation;
- `none`: no semantic component, with direct/path weights renormalized.

The direct topology and path components remain unchanged. SLM inference is
offline, so the RUL model has no language-model cost during training or
deployment.

## Layout

- `semantic_backbone.py`: audited backbone construction and cache validation.
- `prepare_slm_semantics.py`: offline frozen-SLM embedding generator.
- `experiment.py`: protocol/configuration helpers for the original trainer.
- `tests/`: CPU-only invariant tests.

This first refactor intentionally does not duplicate the CUDA Mamba model. The
existing `MARDGMamba` remains the numerical predictor; the refactor produces a
drop-in `a_k` matrix for it.

