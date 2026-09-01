# Current context

## Current goal

Establish the first engineering feasibility basis for EC-DTGM / TrustKG-Mamba:
auditable sensor-card semantics, a reviewable Reference KG draft, explicit F0-F3
graph paths, and a train/validation-only C-MAPSS protocol.

## Key decisions and constraints

- This phase is diagnostic feasibility validation, not a formal experiment.
- SLM sensor-card cosine similarity and projected KG adjacency are separate artifacts.
- The Reference KG is `reference_kg_v1_draft`; unverified content is never Gold KG.
- Evidence, model scope, conflicts, projection eligibility, and review recommendations
  are explicit machine-validated fields.
- Official C-MAPSS test and RUL files remain unopened in this phase.
- The external project is read-only; graph controls are implemented locally.
- No performance, robustness, statistical, or Dual Trust claims are permitted.

## Current progress

- Audited local RUL/effectiveness code and the external MA-RDG-Mamba call chain.
- Corrected sensor-card `s20` from W21 to W31.
- Implemented strict SLM cache validation and explicit pooling.
- Added Reference KG schema, ontology, draft, projection, validator, and tests.
- Added CPU-tested F0-F3 routing and a local predictor adapter; all four paths
  completed a one-batch Linux/CUDA feasibility smoke with checkpoint reload.
- Verified a train/validation-only FD001 80/20 engine split.

## Unresolved questions

- T24, T30, P30, Nf, NRf, and W31 definitions have evidence sufficient for human
  acceptance review. The two fan-speed structural bridges use related-model evidence
  and require a scope decision. W32 retains an exact-dataset versus model-guide conflict.
- Sensor selection is not frozen; the external predictor uses one fixed 14-sensor
  set although FD001-FD004 constant-sensor statistics differ.
- The working runtime is WSL2 `Ubuntu-D` with `/home/administrator/mamba_rul`;
  native Windows environments still do not provide official `mamba_ssm`.
- No real SLM cache/model path was supplied; cache generation remains unexecuted.

## Next action

Complete the bounded decisions in `knowledge/human_review_checklist.md`, then freeze
the subset-specific sensor policy and formal experiment protocol. No broad retrieval
round is currently justified.

## Rejected approaches

- Treating SLM cosine similarity as a KG.
- Calling identity adjacency a No-Graph ablation.
- Using external `prepare_subset()` during training because it opens test/RUL files.
- Promoting unsourced candidate edges or proceeding to Dual Trust now.
