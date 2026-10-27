# Makes AI — complete source archive

Recovered Python applications and experiments for business analytics, recommendation systems and forecasting. **21 project folders** and historical source versions are preserved here, representing **320 distinct original Python/notebook file contents** across 1,001 recovered occurrences. These are file-version counts, not counts of unique algorithms.

## Explore the portfolio

| Repository | Focus |
| --- | --- |
| [Customer analytics & CRM intelligence](https://github.com/ismb-ai/customer-analytics-crm) | RFM, retention, ABC analysis and business dashboards |
| [Recommendation systems](https://github.com/ismb-ai/recommendation-systems) | Classical, neural and hybrid recommendation experiments |
| [Demand forecasting](https://github.com/ismb-ai/demand-forecasting) | Temporal windows, statistical models and neural forecasting |

This repository is the complete historical record. Start with the thematic repositories for an introduction, a small executable example and a code tour.

## Archive structure

- `projects/`: all 21 recovered project folders.
- `versions/`: source extracted from historical ZIP versions.
- `standalone/`: additional non-duplicate source files.
- [Project catalogue](docs/PROJECTS.md): project descriptions and candidate entry points.
- [Source manifest](docs/source-manifest.json): original/prepared SHA-256 hashes and source mappings.
- [Preparation notes](docs/PREPARATION.md): notebook sanitisation and two small syntax/name fixes.
- [Validation](docs/VALIDATION.md): the scope and limits of source checks.

Private datasets, generated business reports and trained models are excluded. Notebook outputs and attachments have been cleared. Original downloads remain outside the public portfolio.

## Status and attribution

Historical applications are preserved rather than presented as a uniformly supported package. Three copies of an unfinished Hume Bayesian example contain empty vocabulary-size assignments; the source validator reports them. Full application execution, training and predictive performance have not been validated.

Original copyright and author comments remain in place. The `forecast_makes-main` copies credit Ricardo Costa in their Poetry metadata and are related team-project source. No new licence or claim of sole authorship is applied to the collection.

```sh
python tools/check_sources.py
```

The checker exits non-zero for the three documented unfinished files. See the thematic examples for checks that can run independently of the historical environments.
