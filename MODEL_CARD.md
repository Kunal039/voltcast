# Model Card: VoltCast forecaster and peak-hour classifier

## Model details
| | Forecaster | Peak-hour classifier |
|---|---|---|
| Task | Regression: household active power (kW) for each of the next 1–48 hours | Binary classification: will the next hour exceed the peak threshold (75th percentile of training consumption, 1.59 kW)? |
| Algorithm | `RandomForestRegressor` (scikit-learn), 200 trees, max depth 16, min 5 samples per leaf | `RandomForestClassifier` (200 trees, min 5 samples per leaf, balanced class weights) |
| Inputs | 31 features computed only from past readings: lags (1–168 h), rolling mean/std, calendar, previous-hour electrical readings | same 31 features |
| Selection | Best validation RMSE among the syllabus algorithms (linear, ridge, decision tree, KNN, SVM, random forest), default and GridSearchCV-tuned, compared with 2 naive baselines | Best validation F1 among default decision tree, GridSearchCV-tuned decision tree and random forest |
| Versioning | MLflow model registry (`voltcast-forecaster`), DVC pipeline (`dvc.lock`), Git | MLflow experiment `household-electricity-forecasting-peak-classifier` |

## Intended use
- Short-term forecasts and peak-hour alerts for **one household's** smart-meter data, to help residents shift flexible loads and to illustrate a complete MLOps workflow.
- **Not** intended for grid dispatch, billing, or decisions about individuals (credit, tariffs, eligibility).

## Training data
UCI *Individual Household Electric Power Consumption* (Hebrail & Berard, 2012, CC BY 4.0): 2,075,259 one-minute readings from one house in Sceaux, France, Dec 2006 – Nov 2010; 1.25% missing, resampled to hourly. Chronological split: train to Dec 2009, validation Jan–May 2010, test Jun–Nov 2010.

## Performance (held-out test set, Jun–Nov 2010)
| Forecaster | MAE | RMSE | R² |
|---|---|---|---|
| Random forest (deployed) | 0.295 kW | 0.434 kW | 0.620 |
| Persistence baseline | 0.377 kW | 0.574 kW | 0.335 |

| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random forest classifier | 0.857 | 0.562 | 0.718 | 0.631 | 0.910 |

## Fairness and bias
The system makes no decisions about people, so fairness is assessed as **equal quality of service across conditions** (`reports/fairness.json`, `fig_fairness.png`). Test MAE by slice:

- **Time of day:** night 0.16 kW, evening 0.39 kW. Forecasts are least reliable exactly when demand peaks.
- **Day type:** weekdays 0.28 kW, weekends 0.33 kW. Weekend routines are less regular.
- **Season:** summer 0.25 kW, autumn 0.33 kW. The training data is winter-heavy; monitoring flags this drift (PSI 0.73 on the water-heater/AC sub-meter).
- **Usage level:** low-use hours 0.15 kW, high-use hours 0.44 kW.

**Sampling bias:** the data comes from a single French household in 2006–2010. The model must not be assumed to work for other homes, climates (e.g. Indian summers with AC load), tariffs or appliance mixes without retraining and re-evaluation.

## Transparency
- Permutation feature importance is published (`fig_feature_importance.png`). The top inputs are previous-hour current, previous-hour power and the water-heater/AC sub-meter, which is physically sensible.
- Every metric, figure and report number is produced by code in this repository and can be reproduced with `dvc repro`.
- Forecasts include a 95% interval, so users see the uncertainty, not just a single number.

## Privacy
High-resolution meter data can reveal when people are home, asleep or away, which is sensitive personal data.
- Only a public, anonymised research dataset is used; no personal identifiers are stored.
- The prediction log (`logs/predictions.jsonl`) stores forecasts and timestamps only, never the submitted meter history.
- For real deployments: process data on-premise or in the user's own account, aggregate to hourly before storage, obtain consent, and follow India's Digital Personal Data Protection Act, 2023.

## Security
- Input validation with Pydantic: types, non-negative readings, positive voltage, a 1–48 h horizon, minimum history length, duplicate timestamps rejected.
- Request bodies limited to 2 MB (`MAX_CONTENT_LENGTH`).
- Models are loaded only from files produced by this pipeline; MLflow logging marks the single trusted tree type explicitly rather than trusting arbitrary pickles.
- Rejected requests are counted in Prometheus (`voltcast_prediction_errors_total`) so abuse is visible in Grafana.

## Monitoring and retraining
Prometheus scrapes request rate, latency, forecast volume, forecast distribution and peak alerts from `/api/metrics`; Grafana shows them. A pipeline stage computes PSI drift and rolling MAE and recommends retraining when PSI > 0.2 or MAE degrades by more than 20%. GitHub Actions retrains weekly.
