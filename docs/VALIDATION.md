# Validation

- 814 Python files parsed, including the validation tool: 3 syntax errors.
- 129 notebooks parsed as JSON; saved outputs and attachments removed.
- All 320 distinct original recovered code-file hashes are represented in the provenance manifest.
- Original archives were read with ZIP integrity checking during recovery; the large derived similarity CSV remains inside its original ZIP and was not extracted for source inspection.
- The targeted credential-pattern scan of recovered sources reported 0 matches. This is a limited static check, not a comprehensive security audit.

No application runtime, dependency compatibility, data-dependent functionality or prediction accuracy was tested. The notebook check validates the document structure and output removal; it does not execute notebook cells. The validation command returns a non-zero status while preserved incomplete Python prototypes remain.

## Preserved incomplete prototypes

- `versions/Hume-v1-c927162d/Hume-v1/src/Bayesian-RecSys.py`, line 113: invalid syntax. The recovered Hume example leaves vocabulary sizes unspecified. No model dimensions were invented.
- `projects/Hume/src/Bayesian-RecSys.py`, line 113: invalid syntax. The recovered Hume example leaves vocabulary sizes unspecified. No model dimensions were invented.
- `projects/Hume-v1/src/Bayesian-RecSys.py`, line 113: invalid syntax. The recovered Hume example leaves vocabulary sizes unspecified. No model dimensions were invented.
