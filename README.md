# Nepal River Flood Prediction

Machine-learning models for river-discharge monitoring across ten Nepali river stations, using daily weather and discharge data from January 2023 to August 2026.

Two tasks are tackled on the same dataset:

| Task | Question | Target |
|---|---|---|
| **Classification** | Is the river in a high-flow state? | `flow_class` = 1 when `river_discharge_m3s` ≥ 90th percentile (27.84 m³/s) |
| **Regression** | How much water will flow tomorrow? | `next_discharge` = next day's `river_discharge_m3s` per station |

## Repository contents

| File | Description |
|---|---|
| `CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv` | Dataset: 13,390 daily records, 19 columns |
| `classification.ipynb` | High-flow classification (Logistic Regression, Decision Tree, Random Forest) |
| `regression.ipynb` | Next-day discharge regression (Linear Regression, Random Forest, Decision Tree) |

## Dataset

- **Rows:** 13,390 (1,339 days × 10 stations), 2023-01-01 to 2026-08-31
- **No duplicates and no missing values** in the raw file
- **Stations:** Bahrabise, Belsot, Bhada Bridge, Chameliya/Nayalbadi, Chatara, Chisapani, Devghat, Khokana, Kusum, Rasuwagadhi

| Station | River | Basin |
|---|---|---|
| Bahrabise | Bhote Koshi | Koshi |
| Belsot | Kamala | Kamala |
| Bhada Bridge | Babai | Babai |
| Chameliya/Nayalbadi | Chamelia | Mahakali |
| Chatara | Saptakoshi | Koshi |
| Chisapani | Karnali | Karnali |
| Devghat | Narayani | Narayani/Gandak |
| Khokana | Bagmati | Bagmati |
| Kusum | West Rapti | Rapti |
| Rasuwagadhi | Bhote Koshi | Narayani |

**Columns:** `date`, `location`, `river`, `basin`, `dhm_station`, `latitude`, `longitude`, `elevation_m`, `precipitation_mm`, `soil_moisture_0_100cm_m3m3`, `rain_mm`, `temperature_mean_c`, `dew_point_mean_c`, `precipitation_hours`, `relative_humidity_mean_pct`, `wind_speed_max_kmh`, `wind_gusts_max_kmh`, `wind_direction_dominant_deg`, `river_discharge_m3s`.

## Method

1. Load the CSV, check duplicates and nulls.
2. Parse `date`, derive `month`, drop rows with unparseable dates.
3. Build the target (high-flow flag, or next-day discharge via a per-station `shift(-1)`).
4. Features: 11 weather/terrain variables, `river_discharge_m3s`, `month`, plus one-hot encoded `location`, `river`, `basin`.
5. 80/20 train/test split (`random_state=42`).
6. Train and compare three models per task; inspect errors and feature importance.

## Results

### Classification (test set, 1,056 rows)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9991 | 0.9895 | 1.000 | 0.9947 |
| Decision Tree | 0.9991 | 0.9895 | 1.000 | 0.9947 |
| Random Forest | 0.9991 | 0.9895 | 1.000 | 0.9947 |

Confusion matrix (Logistic Regression): 961 true negatives, 1 false positive, 0 false negatives, 94 true positives.
**Read the caveats below before trusting these scores.**

### Regression

| Model | MSE | MAE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 504.08 | 6.69 | 22.45 | **0.920** |
| Random Forest | 761.01 | **3.21** | 27.59 | 0.879 |
| Decision Tree | 1684.70 | 4.10 | 41.05 | 0.731 |

- Linear Regression has the best R² and RMSE. Random Forest has the lowest MAE, so it is closer on typical days but misses large peaks by more.
- Most important feature (Decision Tree): `river_discharge_m3s` (83%), then soil moisture (6.8%) and precipitation (6.4%).
- Error distribution (Linear Regression): mean −0.14, max +448, min −76. Nine of the ten largest errors are at Kusum (West Rapti), almost all under-predictions of monsoon peaks.
- Adding lag features (`discharge_lag1`, `rain_lag1`) to a depth-6 Decision Tree lowered R² from 0.731 to 0.655 in the notebook.

## Known issues to fix

These affect how far the results above can be trusted.

1. **Date parsing drops 61% of the data.** The CSV dates are ISO (`YYYY-MM-DD`), but the notebooks use `pd.to_datetime(..., dayfirst=True)`. Days 1–12 are parsed with day and month swapped; days 13–31 become `NaT` and are dropped (8,110 of 13,390 rows). Only 5,280 rows are used, and the `month` feature actually holds the day of the month. Fix: `pd.to_datetime(df['date'])`.
2. **Target leakage in classification.** `flow_class` is defined from `river_discharge_m3s`, which is also an input feature, so near-perfect scores are expected. Remove it, or predict *tomorrow's* high-flow class instead.
3. **Decision Tree and Random Forest classifiers are scored with the wrong predictions.** Both cells call `model.predict(x_test)` (the logistic model), so all three rows are identical. Use `d_model.predict` and `rf_model.predict`.
4. **Global, not per-station, threshold.** The comment says "per station" but the 90th percentile is computed over all stations, so the high-flow class is dominated by the largest rivers.
5. **Random split on time-series data.** Neighbouring days land in both train and test. Use a chronological split.
6. **Unscaled features** cause the logistic-regression convergence warning; scale them or raise `max_iter`.
7. **Hard-coded Windows paths.** Change the `pd.read_csv(...)` path to wherever you keep the CSV.

## Suggested next steps

- Fix the date parsing and re-run both notebooks.
- Use a time-based train/test split and per-station thresholds.
- Drop current discharge from the classifier inputs (or predict the next day's class).
- Add rolling rainfall and soil-moisture features, and tune the tree models.
- Evaluate separately on monsoon (June–September) extremes.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook
```

Open `classification.ipynb` or `regression.ipynb`, update the CSV path in the first code cell, and run all cells.
