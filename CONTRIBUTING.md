# Contributing

This is a research-alpha repository. Keep changes narrowly scoped and describe
the physical problem, variables, gauge, boundary conditions and numerical method
affected. A passing software test does not establish a new scientific claim.

Install `requirements-dev.txt` with Python 3.12 and run `python -B -m pytest -q`.
For changes that affect the portable constraint bridge, also run its complete
example into a new directory and retain unsuccessful reports. Do not edit saved
evidence, loosen tolerances or replace a failed experiment with selected results.

Exported files are bound by `provenance/export-manifest.json`. When proposing
changes to them, preserve the original baseline and explicitly identify the new
candidate; never silently replace a historical source hash. Keep third-party
copyright notices intact.

Reports should distinguish source verification, local reproduction, hosted CI,
independent external reproduction and physical validation. Include the commit,
Python/dependency versions, command and output report with a reproducible issue.

Do not commit environment files, credentials, generated output directories,
large binary data or local machine configuration.


The binary64 candidate manifest also binds root README.md and .gitignore as
new-candidate packaging metadata. After an intentional edit to either, update
only its byte count and SHA256 in experiments/case12_binary64_v1/manifest.json
from the LF Git blob and record that metadata-only change. Preserve all frozen
export identities, mathematical sources, parent inputs and registered limits.
Before pushing, run the full portable suite, not only the edited test files;
then inspect both hosted Python jobs and the C++ job for that pushed commit.
