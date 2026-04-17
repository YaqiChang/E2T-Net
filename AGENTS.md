# AGENTS.md

## Purpose
This repository uses this file as the persistent instruction source for Codex and other coding agents.

The current repository goal is to evolve E2T-Net from a PTINet-style baseline toward an evidence-based intention prediction framework. At the current stage, the active task is only pose data integration unless a task file explicitly says otherwise.

Read this file before making any code change.

---

## Repository Principles

1. Preserve the baseline path.
2. Make the smallest safe change.
3. Keep changes modular and reviewable.
4. Do not rewrite unrelated modules.
5. Do not silently change training semantics.
6. Do not introduce broad abstractions before feasibility is verified.
7. Prefer explicit tensor interfaces and comments for all new code.

---

## Baseline Compatibility Rules

1. Existing baseline behavior must remain unchanged when a new feature is disabled.
2. Existing configs should remain usable whenever possible.
3. New features must be guarded by config flags.
4. The old train and eval path must remain runnable unless a task explicitly authorizes a migration.

---

## Required Engineering Workflow

For every non-trivial task, follow this order:

1. Inspect relevant files first.
2. State assumptions clearly in code comments or docs.
3. Implement the smallest viable version first.
4. Add a smoke test or sanity check.
5. Run the smoke test.
6. Report exact files modified and exact commands executed.
7. Document unresolved assumptions and risks.

Do not skip validation.

---

## Validation Rules

Every non-trivial change should satisfy as many of the following as applicable.

### Level 1
Static sanity
- imports succeed
- no syntax errors
- no unresolved references

### Level 2
Smoke test
- dataset can load one sample
- dataloader can produce one batch
- model can run one forward pass
- output shapes are correct
- no NaN or Inf appears

### Level 3
Integration feasibility
- feature can be enabled by config
- baseline still runs when feature is disabled
- training can start at least one iteration when feature is enabled

A task is not done until at least a smoke test has been executed and reported.

---

## Minimal Change Policy

1. Edit only files needed for the current task.
2. Avoid cosmetic formatting changes in unrelated code.
3. Avoid renaming stable interfaces unless necessary.
4. Reuse existing utilities when safe.
5. Prefer adapters over invasive rewrites when schemas are uncertain.

---

## Data and Shape Explicitness

For every new module, document expected tensor shapes in code comments.

For pose-related features, use these normalized conventions unless the task explicitly changes them:

- pose: B x T x J x 2
- pose_conf: B x T x J
- pose_feat_seq: B x T x D

If raw files use a different schema, convert them into this standard form at the adapter layer.

---

## Missing Data Policy

Missing or unreadable optional files must not crash the pipeline by default.

For offline pose files:
- missing pose should fall back to zero pose
- missing confidence should fall back to zero confidence
- count and report missing cases when possible

---

## Documentation Requirements

For every meaningful feature addition, update or create a short document under docs.

At minimum, report:
- purpose
- files changed
- assumptions
- validation commands
- known risks

---

## Out of Scope by Default

Unless the current task file explicitly requests them, do not implement:
- evidence accumulation
- SSM accumulator
- belief-conditioned trajectory decoding
- loss redesign
- pose graph network
- large training refactor
- performance optimization pass
- style-only cleanup

---

## Task File Policy

Check the current task file before coding.

Use the task file to determine:
- current stage objective
- allowed scope
- required deliverables
- acceptance criteria

If the task file conflicts with this file, follow both by using the narrower scope and the safer implementation path.

---

## Reporting Format

At the end of each task, report:

1. Files modified
2. Main logic added or changed
3. Smoke tests executed
4. Exact commands run
5. Observed outputs
6. Remaining risks or assumptions

Be concrete and concise.

## Python Environment Rule

Use the repository's designated Python environment for all validation and smoke tests.
Do not add fallback patches for missing libraries until you have first verified that the correct environment is active.
Before any workaround for missing imports, report:
- exact python path
- python version
- torch import result
- torchvision import result
- cv2 import result
Use this interpreter for every Python command:
/home/meta/anaconda3/envs/3dhuman/bin/python