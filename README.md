# RFNBO price pathways and renewable hydrogen plant design

This repository accompanies the manuscript **RFNBO Price Pathways and Renewable Hydrogen Plant Design: An Origin Aware Engineering Study of Selected European Market Areas**, prepared for submission to the *International Journal of Green Energy*.

The repository contains the final manuscript source and PDF, vector figures, editable result tables, and the Python code used for the audited engineering comparisons. It is a selected European public data screen. It is not a legal RFNBO certification tool, a project level congestion model, or an EU 27 prevalence estimate.

## Contents

- `manuscript/`: LaTeX source, bibliography, Taylor and Francis template files, figures, and the compiled manuscript PDF.
- `code/`: model, public data preparation, experiment, audit, and literature verification scripts.
- `tables/`: the editable tables reported in the manuscript.
- `figures/`: a convenient copy of the final artwork.
- `data/DATA_SOURCES.md`: public source records and redistribution boundary.
- `REPRODUCIBILITY.md`: reproduction notes and interpretation boundary.
- `CITATION.cff`: citation metadata.

## Reproduction

Raw provider files are not redistributed. Obtain them from the public sources listed in `data/DATA_SOURCES.md`, place them in a local raw data directory, and set the paths described by the preparation scripts. All derived data and results should be written outside the raw directory.

The three audited experiment entry points are:

```text
python code/run_esb_phase_shift.py --workers 2
python code/run_esb_cross_period_hourly.py --workers 2
python code/run_esb_parameter_lhs.py --draws 12 --seed 20260904 --workers 2
```

Use `code/audit_esb_upgrade.py` after the runs. The reported numerical tables are deterministic summaries of the archived run. E1 circular shifts are counterfactual timing tests; E2 published forecast profiles are profile sensitivities and do not establish forecast vintage compliance; E3 is a bounded engineering stress box and not a probability model.

## Citation and versioning

The initial public package is tagged `v1.0.0`; the manuscript link update is
released as `v1.0.1`. The versioned package is available at
<https://github.com/williamjay1/rfnbo-hydrogen-europe-ijge/releases/tag/v1.0.1>.
GitHub does not mint DOIs. If an archival DOI is later created through a
recognized repository, it should be added to this README, `CITATION.cff`, and
the manuscript data statement. No DOI is invented in this repository.

## Data and license note

The raw inputs remain subject to the terms of their original providers and are not redistributed here. The repository owner should select and add an explicit license for the code, manuscript source, figures, and derived tables before or during archival deposit.
