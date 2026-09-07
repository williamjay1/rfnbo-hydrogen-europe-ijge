# Public data sources

The analysis uses public generation profiles, public day ahead price records, and a public or openly documented EUA proxy. The repository contains processing code and derived result tables, but does not redistribute provider raw files.

The exact source URLs, identifiers, timestamps, coverage checks, processing rules, and file hashes are recorded in the manuscript references, the project evidence ledger, and the run manifests retained by the author. The public package is intentionally limited to materials that can be redistributed under the source terms.

## Reproduction boundary

Users must independently download the source files from the original provider pages and verify their terms before running the preparation scripts. The scripts do not infer missing hours, authenticated forecast vintage, legal RFNBO certification, project interconnection feasibility, or physical network congestion.

## Files to retain locally but not upload here

- provider generation and price downloads;
- EUA source extracts where redistribution is restricted;
- ENTSO E credentials or tokens, if a user performs a separate authenticated validation;
- local caches and solver checkpoints.

The manuscript reports a selected European public data engineering screen. The public package is therefore a reproducibility aid for the reported estimand, not a redistribution of all upstream data.
