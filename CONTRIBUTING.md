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
