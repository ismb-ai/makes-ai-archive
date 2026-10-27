# Dependencies and data

Original dependency files are preserved in their original project/version paths. No package installation or environment migration was performed.

## Existing dependency declarations

- `projects/RecSysDNA/src/requirements.txt`
- `projects/analytics/requirements(1).txt`
- `projects/analytics/requirements.txt`
- `projects/forecast_makes-main/src/pyproject.toml`
- `projects/forecast_makes-main 2/src/pyproject.toml`
- `projects/mkrecsysX/src/analytics/requirements.txt`
- `projects/mkrecsysX/src/autoencoder/requirements.txt`
- `projects/recsys/src/main/requirements.txt`
- `projects/rfm-cluster/src/requirements.txt`

## Data requirements

The original archives contain sales/customer datasets, derived tables and model inputs. These are excluded from this repository copy. Several scripts refer to fixed local CSV paths, which need to be configured before running. Notebook outputs were cleared to avoid carrying their embedded tables and charts into a repository.

Analytics variants commonly use columns such as `Client_ID`, `Product_ID`, `Sales_Date` and `Monetary_Value`; specific modules may also require names, quantities, costs or other fields. Verify the selected application’s input schema in its source. No synthetic dataset has been claimed as a substitute for its original inputs.

## Import inventory

The machine-readable [import inventory](imports.json) is extracted from the recovered Python files. It includes standard-library and local modules as well as external packages; it is not an installation requirements file.
