# Repository guidance

## Scope

- This repo supports the first theoretical-limitations paper, primarily its simulations.
- `research_proposal.md` is broader context; do not imply all proposal extensions are implemented.
- Keep simulations, empirical evidence, and other paper visuals as separate source/output groups.

## Stable contracts

- Use uv plus `pyproject.toml` and `uv.lock`; do not add requirements files.
- A generator at `src/wkde_cad/<group>/<stem>.py` writes to `outputs/<group>/<stem>/`.
- Figure 1 is the unweighted atlas; Figure 2 is the weighted atlas.
- Preserve the main atlas PNGs byte-for-byte unless the user explicitly requests a visual change:
  - Figure 1 SHA256: `61cd39da00961c75c48ba9030d4e294065cf525c3ce3e98850301cd8c9f8b071`
  - Figure 2 SHA256: `c276c9f47ac170facd29b08e522af24cef274a50066d6a838966a59f22059bc7`

## Scientific interpretation

- Atlas columns are controlled one-factor slices embedded in a multidimensional envelope.
- For unweighted perfect separation, use the exact condition `N_cal + 1 >= m/(alpha*s) = 1/(alpha*pi1)` instead of another phase plot.
- The red zero line is a rank-feasibility boundary; for weighted WCS it is only a resolution diagnostic.
- Figure 1 uses nested calibration/test repeats; Figure 2 uses accepted trials and mixes `(N_cal, rho)` within effective-size bins. Keep this distinction explicit.
- Figure 3 is a descriptive atlas collapse plus a matched-effective-size sufficiency stress test, not a causal decomposition.
- Randomization and clipping belong to supplements unless the manuscript develops those extensions further.
- Current empirical CSVs are exploratory inputs, not a finished empirical figure.

## Workflow

- Prefer validated caches, vectorized/batched calculations, and existing process-level parallelism.
- Keep cache keys tied to all design-changing parameters; smoke runs must not poison canonical caches.
- Avoid adding an optimization framework for marginal runtime gains.
- Run `uv run python -m pytest -q`, `uv run ruff check .`, and the atlas hash test after structural changes.
