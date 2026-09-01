# Reference KG v1 draft

This directory is a feasibility prototype, not a Gold KG. Human review (reviewer:
Noidealize) accepted the six direct-definition edges draft-001..006. draft-007 was
accepted under the dataset-scope policy with `conflict_status=unresolved` and is
excluded from projection. draft-004b/005b remain `needs_review` and are
projection-eligible for diagnostics only. Nf/NRf are decomposed into measured
quantities plus separately scoped structural edges.

- `schema.json`: allowed entity/relation types and edge fields.
- `sensor_ontology.json`: all 21 raw sensors, distinct from the current fixed 14-sensor input.
- `reference_kg_v1_draft.jsonl`: unreviewed candidates for validation/projection only.
- `reference_kg_v1_reviewed_feasibility.jsonl`: signed, immutable-by-convention
  feasibility snapshot; it remains non-Gold while pending/collided edges exist.
- `reference_kg_v1_reviewed_feasibility.manifest.json`: hashes, review counts,
  unresolved items, and projection metadata for the frozen snapshot.
- `evidence_registry.jsonl`: document-level records with separately addressable spans.
- `projection_rules.json`: deterministic bounded multi-hop sensor projection.
- `validators/validate_kg.py`: validation and projection CLI.
- `freeze_reviewed_kg.py`: fail-closed snapshot and manifest generator.

Missing evidence on an unreviewed edge is a warning. Missing evidence on an
accepted edge, a reference to a nonexistent/unsuitable document or span, a span
that does not declare support for its edge, or a projected unresolved conflict is
an error.
