# Human review checklist

Only three bounded decisions remain. No broad literature or web retrieval is
needed. Two gaps are closed by policy and must not be reopened during this review:

- The PHM08 generator code was searched through GitHub, NTRS, and DASHlink and was
  not found. Do not infer W31/W32 semantics from unavailable generator code.
- NASA/TM-2010-216831 was not opened. Its contents, including any purported
  “Nf = low-pressure-shaft speed” wording, must not be cited.

Allowed values from `schema.json`:

- `review_status`: `unverified | accepted | rejected | needs_review`
- `conflict_status`: `none | unresolved | resolved`
- `reference_scope`: `PHM08_FD001_FD004_dataset_column_semantics |
  C_MAPSS_90K_model_semantics | cross_model_support_only`

## A. Direct definitions (recommended: accept)

Review `draft-001`, `draft-002`, `draft-003`, `draft-004a`, `draft-005a`, and
`draft-006` against their registered spans.

`draft-001` through `draft-005a` cite NASA/TM-2007-215026 printed p.4,
Table 1.2 (the output-variable table). `draft-006` instead cites printed p.5,
Table 1.3, a non-output workspace-variable table. W31 is not part of the guide's
27-element output vector; this distinction is relevant to the W32 conflict below.

If accepted, record the reviewer's name or initials and set `review_status=accepted`.
If the cited wording or encoded meaning is wrong, set `review_status=rejected` and
record the reason in `notes`. Do not change a relation tail without a new evidence
review.

## B. Fan-speed structural transfer (human engineering decision)

Review `draft-004b` and `draft-005b`. Both cite NASA/TM-2008-215303, a related
C-MAPSS model rather than the PHM08 generator:

- `draft-004b`: Table 1, “N1 Fan, LPC, LPT”.
- `draft-005b`: model description, “the corrected versions of the two spool speeds
  are designated RN1 and RN2”.

Decide whether this related-model evidence is sufficient for feasibility-only graph
projection. The conservative recommendation is to keep both `needs_review` and
`projection_eligible=true` for prototype diagnostics, but not promote them to Gold
KG until exact PHM08 generator/model evidence becomes available.

## C. W32 scope conflict (required before publication use)

The registered evidence currently establishes a genuine scope conflict:

1. Saxena et al. (2008), Table 2, defines PHM08 dataset sensor 21/W32 as LPT
   coolant bleed.
2. NASA/TM-2007-215026, Table 1.3, labels W32 as HPT coolant bleed.

The latest audit also reports two additional model-side sources: NASA/TM-2008-215303
Table 5/Fig. 6 and Zinnecker (2014). They strengthen the model-side interpretation,
but are not yet registered as addressable evidence spans in `evidence_registry.jsonl`.
They must not be counted as formal KG evidence until their exact excerpts, locations,
metadata, and support declarations are registered and validated. This is a targeted
registration task, not a new broad retrieval round.

No generator code was found, so the conflict is not resolved. Choose one policy:

1. Dataset-scope policy (recommended for this predictor): set
   `review_status=accepted`, keep
   `reference_scope=PHM08_FD001_FD004_dataset_column_semantics`, keep
   `conflict_status=unresolved`, and keep `projection_eligible=false`. Sensor s21
   remains an input; only its KG-projection edge is withheld.
2. Strict unresolved policy: retain `review_status=needs_review`,
   `conflict_status=unresolved`, and `projection_eligible=false`.

With the current schema, `conflict_status=unresolved` always requires
`projection_eligible=false`. Do not set `conflict_status=resolved`: the evidence does
not resolve the conflict. Enabling W32 projection later requires an explicit scoped
conflict design plus coordinated changes to `schema.json`, `README.md`, and
`tests/test_knowledge.py`.

Never generalize a PHM08 dataset-scope decision to all C-MAPSS model variables.

## Review record

- Date: 2026-09-02, reviewer: Noidealize.
- Section A: draft-001..006 accepted (`review_status=accepted`).
- Section B: draft-004b/005b kept `needs_review`, `projection_eligible=true`
  (feasibility diagnostics only).
- Section C: policy 1 — draft-007 accepted under PHM08 dataset scope,
  `conflict_status=unresolved`, `projection_eligible=false`.
- Post-review validation: 0 errors, 0 warnings; unit tests 8/8.
