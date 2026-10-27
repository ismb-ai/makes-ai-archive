# Preparation changes

Original download archives are preserved separately and were not modified.

- Preserved the recovered project layouts and historical source versions.
- Removed notebook outputs, execution counts, cell metadata, embedded attachments and nonessential notebook metadata from the repository copy.
- Relocated `.ipynb_checkpoints` to visible `notebook-checkpoints` directories so preserved checkpoint source can be included in a repository.
- Represented byte-identical standalone downloads through existing project files and their provenance entries.
- Excluded datasets, generated reports, compiled Python files, trained models, archive ZIPs and operating-system metadata from this repository copy.
- Completed `import seaborn as sns` in `projects/mkrecsysX/src/abc.py`, matching the file’s existing `sns` calls.
- Commented the notebook-only `%matplotlib inline` directive in `projects/Forecasting_v1/init.py`; corrected its two references to `training_data` to use the already-loaded `data` variable.

The machine-readable change log and source manifest record original and prepared hashes. Algorithms and historical model designs were otherwise preserved. Absolute data paths and historical dependencies still need configuration for execution.
