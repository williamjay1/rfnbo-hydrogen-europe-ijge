# Reproducibility notes

This source package contains the editable LaTeX manuscript, the vector and
high-resolution companion artwork, the result tables used in the paper, and
the Python scripts that generated the canonical summaries. Raw downloads are
not redistributed here because their original providers apply source-specific
terms and because the project keeps the immutable raw repository separately.
The public URLs, identifiers, processing notes, and result hashes are recorded
in the manuscript references, run manifests, and `package_manifest.json`.
The package includes `CITATION.cff`. The archived v1.0.2 package has the
Zenodo DOI <https://doi.org/10.5281/zenodo.22648097>.

## Canonical project layout

The audited run separated a read-only raw repository from a writable project
work area. The scripts retain a `PROJECT` root so that a rerun can be
compared directly with the archived result tables. When the package is
moved to another machine, replace the `PROJECT` constants in the copied
scripts with that machine's local project root; keep raw inputs read only.
No machine-specific drive path is required by this documentation.

## Reproduction sequence

From a local clone of this repository, after the public source data have been
prepared, the three additional experiment entry points are:

```text
python <PROJECT>\code\run_esb_phase_shift.py --workers 2
python <PROJECT>\code\run_esb_cross_period_hourly.py --workers 2
python <PROJECT>\code\run_esb_parameter_lhs.py --draws 12 --seed 20260904 --workers 2
```

The final numerical audit is run with `code/audit_esb_upgrade.py` against the
experiment result root. The manuscript source in `manuscript/` can be compiled
with the official Taylor and Francis Interact files included in that folder.
The canonical evidence does not require an ENTSO-E credential: authenticated
API data were not used as a hidden input. Historical public snapshots and
openly documented market and generation series are listed in the manuscript
references and `data/DATA_SOURCES.md`.

## Interpretation boundary

The manuscript reports a selected-European public-data engineering screen.
It does not identify EU-27 project-level congestion, certify legal RFNBO
compliance, or validate authenticated forecast vintages. The package is
intended to make those boundaries and the reported numerical evidence easy
to inspect.
