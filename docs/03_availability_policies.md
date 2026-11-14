# Availability policies

Every configuration declares `data.availability_policy`. The value states which
information a forecast made at an origin may read. Runs, input snapshots, feature
specifications, predictions and summaries record it. Results from different
policies are never pooled into one comparison.

| Policy | Inputs | Use |
| --- | --- | --- |
| `competition_v1` | Sales and transactions through the origin, the promotion plan, store metadata, the full holiday/event table and Kaggle-supplied oil for target dates | Reproduces the historical benchmark |
| `asof_v1` | The same history, plan and store metadata; no oil; holiday rows except type `Event` | Simulates information available at the origin |

`asof_v1` rests on assumptions the dataset cannot verify. Historical `onpromotion`
values stand in for a promotion plan made before the origin. Non-Event calendar
rows are treated as scheduled in advance. Store metadata is treated as unchanged.
Event rows are all excluded, including scheduled ones. Treat results under this
policy as a stricter simulation, not as evidence of a deployed forecast.

## Configurations

| Config | Mode | Policy |
| --- | --- | --- |
| `configs/smoke.yaml` | Synthetic smoke | `competition_v1` |
| `configs/smoke-asof.yaml` | Synthetic smoke | `asof_v1` |
| `configs/full.yaml` | Kaggle data | `competition_v1` |
| `configs/asof.yaml` | Kaggle data | `asof_v1` |

```bash
make baseline CONFIG=configs/smoke-asof.yaml
make baseline CONFIG=configs/asof.yaml
```

The notebooks choose their configuration from `RETAIL_FORECAST_MODE` (`full` or
`smoke`) and `RETAIL_FORECAST_POLICY` (`competition_v1`, the default, or `asof_v1`).
A missing or unknown policy fails before any data is read.

## Request boundary

`retail_forecast.data_contracts.ForecastRequest` holds the inputs for one origin:
sales and transaction history, a complete promotion plan for every requested series
and horizon date, store metadata, the calendar and, only under `competition_v1`,
oil. It never contains target-period sales. Validation runs before fitting.
It rejects these cases with a named category:

| Category | Example |
| --- | --- |
| `HISTORY_AFTER_ORIGIN` | A history row dated after the origin |
| `FUTURE_PLAN_INCOMPLETE` | A requested series/date missing from the promotion plan |
| `UNSUPPORTED_SERIES` | A store/family outside the snapshot |
| `NONFINITE_VALUE` | `NaN` or infinity in sales, counts or promotions |
| `POLICY_MISMATCH` | An unknown policy, or oil supplied under `asof_v1` |
| `SCHEMA_INVALID` | Unknown columns, duplicate keys, invalid dates or negative values |

Missing history stays missing and is flagged; it is never filled with zero sales.
A valid `asof_v1` request needs no `test.csv`, `sample_submission.csv`, oil or
future labels. The benchmark projection builds requests from the seven competition
files; labels are joined afterward by store, family and date for scoring.
Requests carry the 56 days of history the features need. The seasonal comparators
read the same window, so a series with gaps in those 56 days has fewer same-weekday
observations. The Kaggle panel has a row for every series on every date except
December 25, so seasonal predictions there match the full-history implementation.

## Outputs

Baseline runs write `predictions/research-oof.parquet` with each row's policy,
input contract version, snapshot ID, run ID and `lineage_status=research_partial`.
That status marks fold predictions made before verified model bundles exist.
Runs also write `inputs/data-card.json`, which describes the dataset's coverage,
meaning, missingness, policy assumptions and rights status. Each of its numbers
references the input manifest or EDA summary it came from.
