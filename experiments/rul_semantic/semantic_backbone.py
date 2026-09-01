"""Deterministic, auditable semantic backbones for MA-RDG-Mamba."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Literal

import numpy as np
import torch


SemanticMode = Literal[
    "transe", "slm", "slm_shuffled", "none",
    "semantic_similarity_slm", "semantic_similarity_shuffled", "no_semantic",
    "reference_kg_projected", "no_kg",
]

MODE_ALIASES = {
    "semantic_similarity_slm": "slm",
    "semantic_similarity_shuffled": "slm_shuffled",
    "no_semantic": "none",
}


@dataclass(frozen=True)
class BackboneSpec:
    mode: SemanticMode = "transe"
    kg_seed: int = 42
    direct_weight: float = 0.50
    path_weight: float = 0.30
    semantic_weight: float = 0.20
    slm_cache: Path | None = None
    shuffle_seed: int = 1042
    reference_kg_matrix: Path | None = None


@dataclass(frozen=True)
class BackboneResult:
    matrix: np.ndarray
    components: dict[str, np.ndarray]
    audit: dict[str, object]


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("semantic embeddings contain a zero-norm row")
    normalized = embeddings / norms
    similarity = normalized @ normalized.T
    # Match the original TransE component's [0, 1] range.
    similarity = (similarity + 1.0) / 2.0
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def load_slm_cache(path: Path, expected_sensors: list[str]) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SLM cache not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"embeddings", "sensor_columns", "metadata_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"SLM cache missing keys: {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        sensors = [str(x) for x in payload["sensor_columns"].tolist()]
        metadata = json.loads(str(payload["metadata_json"].item()))
    if sensors != expected_sensors:
        raise ValueError(
            "SLM cache sensor order does not match model input order: "
            f"cache={sensors}, expected={expected_sensors}"
        )
    if embeddings.shape[0] != len(expected_sensors):
        raise ValueError("SLM cache row count does not match sensor count")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("SLM cache embeddings contain NaN or Inf")
    if np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError("SLM cache embeddings contain a zero-norm row")
    required_metadata = {
        "model_name", "model_config_sha256", "weights_id", "tokenizer_id",
        "transformers_version", "pooling", "sensor_cards_sha256",
        "generated_at_utc", "embedding_dim", "semantic_matrix_sha256",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(f"SLM cache metadata missing keys: {sorted(missing_metadata)}")
    if int(metadata["embedding_dim"]) != embeddings.shape[1]:
        raise ValueError("SLM cache embedding_dim metadata does not match embeddings")
    if metadata["pooling"] not in {"mean", "last_valid_token"}:
        raise ValueError("SLM cache pooling must be mean or last_valid_token")
    expected_sha = _sha256_array(_cosine_similarity(embeddings))
    if metadata["semantic_matrix_sha256"] != expected_sha:
        raise ValueError("SLM cache semantic matrix hash mismatch")
    return embeddings, metadata


def _import_original(original_root: Path):
    original_root = Path(original_root).resolve()
    if not (original_root / "knowledge_graph" / "backbone.py").is_file():
        raise FileNotFoundError(f"Not an MA-RDG-Mamba source root: {original_root}")
    root_text = str(original_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import config  # type: ignore
    from knowledge_graph.backbone import build_mechanism_backbone  # type: ignore
    return config, build_mechanism_backbone


def _call_builder_preserving_rng(builder, sensor_columns: list[str], kg_seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(kg_seed); np.random.seed(kg_seed); torch.manual_seed(kg_seed)
        return builder(sensor_columns, kg_seed)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def load_projected_kg(path: Path, sensor_count: int) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"projected KG not found: {path}")
    if path.suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as payload:
            if "matrix" not in payload.files:
                raise ValueError("projected KG archive must contain 'matrix'")
            matrix = payload["matrix"]
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (sensor_count, sensor_count):
        raise ValueError(f"projected KG shape mismatch: {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("projected KG must be finite and within [0,1]")
    return matrix


def build_backbone(
    original_root: Path,
    sensor_columns: list[str],
    spec: BackboneSpec,
) -> BackboneResult:
    """Build one fixed knowledge backbone for all matched model seeds.

    The original builder is called exactly once with ``kg_seed``. Python,
    NumPy and Torch RNG states are restored afterwards, so knowledge creation
    cannot perturb model initialization or data-loader randomness.
    """
    mode = MODE_ALIASES.get(spec.mode, spec.mode)
    config, original_builder = _import_original(original_root)
    _, original_parts = _call_builder_preserving_rng(
        original_builder, sensor_columns, spec.kg_seed
    )

    direct = np.asarray(original_parts["direct"], dtype=np.float32)
    path = np.asarray(original_parts["path"], dtype=np.float32)
    transe = np.asarray(original_parts["semantic"], dtype=np.float32)

    cache_metadata: dict[str, object] = {}
    permutation: list[int] | None = None
    if mode == "transe":
        semantic = transe
    elif mode in {"slm", "slm_shuffled"}:
        if spec.slm_cache is None:
            raise ValueError(f"semantic mode {spec.mode!r} requires slm_cache")
        embeddings, cache_metadata = load_slm_cache(spec.slm_cache, sensor_columns)
        if mode == "slm_shuffled":
            rng = np.random.default_rng(spec.shuffle_seed)
            permutation = rng.permutation(len(sensor_columns)).tolist()
            embeddings = embeddings[permutation]
        semantic = _cosine_similarity(embeddings)
        if mode == "slm_shuffled":
            original_semantic = _cosine_similarity(load_slm_cache(spec.slm_cache, sensor_columns)[0])
            off_diagonal = ~np.eye(len(sensor_columns), dtype=bool)
            if np.allclose(semantic[off_diagonal], original_semantic[off_diagonal]):
                raise ValueError("shuffled SLM semantics did not change non-diagonal structure")
    elif mode == "none":
        semantic = np.zeros_like(direct)
    elif mode == "reference_kg_projected":
        if spec.reference_kg_matrix is None:
            raise ValueError("reference_kg_projected requires reference_kg_matrix")
        projected = load_projected_kg(spec.reference_kg_matrix, len(sensor_columns))
        direct = np.zeros_like(projected); path = np.zeros_like(projected)
        transe = np.zeros_like(projected); semantic = projected
    elif mode == "no_kg":
        direct = np.zeros_like(direct); path = np.zeros_like(path)
        transe = np.zeros_like(transe); semantic = np.zeros_like(direct)
    else:
        raise ValueError(f"unsupported semantic mode: {spec.mode}")

    if mode == "no_kg":
        wd = wp = ws = 0.0
    elif mode == "reference_kg_projected":
        wd, wp, ws = 0.0, 0.0, 1.0
    elif mode == "none":
        total = spec.direct_weight + spec.path_weight
        if total <= 0:
            raise ValueError("direct_weight + path_weight must be positive")
        wd, wp, ws = spec.direct_weight / total, spec.path_weight / total, 0.0
    else:
        total = spec.direct_weight + spec.path_weight + spec.semantic_weight
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"backbone weights must sum to 1, got {total}")
        wd, wp, ws = spec.direct_weight, spec.path_weight, spec.semantic_weight

    matrix = wd * direct + wp * path + ws * semantic
    threshold = float(config.KG_SPARSITY_THRESHOLD)
    matrix[matrix < threshold] = 0.0
    matrix = np.clip(matrix, 0.0, 1.0).astype(np.float32)
    if mode != "no_kg":
        np.fill_diagonal(matrix, 1.0)
    components = {"direct": direct, "path": path, "transe": transe, "semantic": semantic}
    audit: dict[str, object] = {
        "semantic_mode": spec.mode,
        "canonical_semantic_mode": mode,
        "kg_seed": spec.kg_seed,
        "shuffle_seed": spec.shuffle_seed if permutation is not None else None,
        "permutation": permutation,
        "sensor_columns": list(sensor_columns),
        "weights": {"direct": wd, "path": wp, "semantic": ws},
        "matrix_sha256": _sha256_array(matrix),
        "component_sha256": {name: _sha256_array(value) for name, value in components.items()},
        "slm_cache": str(spec.slm_cache.resolve()) if spec.slm_cache else None,
        "slm_metadata": cache_metadata,
    }
    return BackboneResult(matrix=matrix, components=components, audit=audit)
