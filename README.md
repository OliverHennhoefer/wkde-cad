# WKDE-CAD

This repository creates the simulation and empirical evidence for the first
paper on theoretical limits of conformal anomaly detection (CAD). The broader
research proposal is context, not the implemented scope of this repository.

## Paper evidence map

| Role | Generator | Main question |
| --- | --- | --- |
| Figure 1 | `figure1_unweighted_power_atlas.py` | Where does standard split-conformal CAD retain or lose power? |
| Figure 2 | `figure2_weighted_power_atlas.py` | How do covariate-shift weights change the operational envelope? |
| Figure 3 | `figure3_operational_sufficiency.py` | Does rank resolution organize the transition, and is matched effective calibration size sufficient? |
| Supplement | `supplement_randomized_pvalue_instability.py` | How unstable are randomized decisions near a discrete boundary? |
| Supplement | `supplement_weight_clipping_frontier.py` | What validity-power trade-off does weight clipping induce? |

Figures 1 and 2 are the former unweighted and weighted operational atlases,
promoted without changing their image bytes. Their columns are controlled
one-factor slices: each varies one of test size, anomaly prevalence/count, or
nominal FDR while holding the displayed baseline factors fixed. This is the
intended ceteris-paribus view inside the broader multidimensional atlas.

Figure 3 adds two complementary summaries. Its top row collapses equal-weighted
atlas cells against the rank-resolution diagnostic. Its bottom row compares
weighted runs at matched realized effective calibration size across shift
regimes. Residual differences show that effective size is useful but does not
fully characterize the normalized weights or WCS decision mechanism.

For standard CAD with perfect score separation, the useful content of the old
standalone phase-transition figures is the exact necessary rank-feasibility
relation

\[
N_{\mathrm{cal}} + 1 \geq \frac{m}{\alpha s}
= \frac{1}{\alpha\pi_1}.
\]

The zero contour in the atlases visualizes the corresponding rank-resolution
boundary. In weighted WCS it is a diagnostic marker, not a complete
no-discovery certificate.

The empirical benchmark is retained separately under `wkde_cad.empirical`.
Its current CSVs cover the configured nine-dataset, 25-seed grid and remain
exploratory raw results, not a promoted paper figure. A future empirical figure
should be designed from an explicitly fixed grid rather than reviving the
removed projection.

## Layout

```text
src/wkde_cad/
  simulations/   main and supplementary simulation generators
  empirical/     benchmark experiments and configuration
  visuals/       reserved for non-simulation paper visuals
outputs/
  simulations/<generator-stem>/
  empirical/benchmark/
tests/
  simulations/
  empirical/
```

Every plot generator writes to `outputs/<group>/<generator-stem>/`. A leaf is
therefore traceable directly to the Python file that created it.

## Environment

Use uv and the committed lockfile:

```powershell
uv sync --frozen
```

There is intentionally no requirements file.

## Generate results

The default runs use validated CSV caches when their simulation design matches.
Pass `--force` only when a full recomputation is intended.

```powershell
uv run python -m wkde_cad.simulations.figure1_unweighted_power_atlas
uv run python -m wkde_cad.simulations.figure2_weighted_power_atlas
uv run python -m wkde_cad.simulations.figure3_operational_sufficiency
uv run python -m wkde_cad.simulations.supplement_randomized_pvalue_instability
uv run python -m wkde_cad.simulations.supplement_weight_clipping_frontier
```

Parallel cell execution and batched WCS evaluation are already built into the
long simulations. For smoke checks, use small trial/repeat counts with a
temporary output patch in tests; do not overwrite a canonical cache with a
smoke run. Figure 3 stores accepted matched-ESS trials and normally rerenders
from that cache.

Run the empirical benchmark with:

```powershell
uv run python -m wkde_cad.empirical.benchmark --config src/wkde_cad/empirical/benchmark/empirical_benchmark.toml
```

## Verify

```powershell
uv run python -m pytest -q
uv run ruff check .
```

Tests cover the output-path contract, cache compatibility, core statistical
logic, and byte hashes of the two protected atlas PNGs.
