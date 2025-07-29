# COVID-19 Case Prediction using Facebook Prophet

This project aims to forecast global COVID-19 confirmed cases using Facebook Prophet, a time series forecasting library developed by Meta. It includes data loading, preprocessing, visualization, forecasting, and evaluation to demonstrate the power of Prophet in modeling pandemic-related time series data.

---

## 📊 Project Overview

- **Objective:** Predict future COVID-19 confirmed cases using historical data with daily granularity.
- **Model Used:** Facebook Prophet (fbprophet) with support for daily and yearly seasonality.
- **Tech Stack:** Python, Pandas, Matplotlib, Seaborn, fbprophet

---

## 🗂️ Dataset

- **Source File:** `covid_19_clean_complete.csv`
- **Total Rows:** 49,068
- **Columns:**
  - `Province/State`, `Country/Region`, `Lat`, `Long`, `Date`
  - `Confirmed`, `Deaths`, `Recovered`, `Active`, `WHO Region`

Missing values were primarily observed in the `Province/State` field, which were handled as part of preprocessing.

---

## 📦 Installation

Use the following commands to install the required libraries:

```bash
pip install pystan==2.19.1.1
pip install fbprophet
```

---

## 🛠️ How to Run

1. **Load the Dataset**
   ```python
   import pandas as pd
   df = pd.read_csv('/content/covid_19_clean_complete.csv')
   ```

2. **Preprocess the Data**
   - Convert `Date` column to datetime format
   - Group by `Date` and aggregate confirmed cases globally
   - Rename columns to `ds` and `y` as required by Prophet

3. **Train the Model**
   ```python
   from fbprophet import Prophet
   m = Prophet(daily_seasonality=True, yearly_seasonality=True)
   m.fit(df_prophet)
   ```

4. **Forecast Future Cases**
   ```python
   future = m.make_future_dataframe(periods=30)
   forecast = m.predict(future)
   ```

---

## 📈 Visualizations

- Top 5 countries by total deaths and recoveries using Seaborn bar plots.
- Forecast plot of COVID-19 confirmed cases using Prophet.
- Prophet's component plots for trends, weekly and yearly seasonality.
- Change points plotted using `add_changepoints_to_plot`.

---

## 📉 Model Evaluation

Evaluation is done using Prophet’s diagnostics module:

- Cross-validation:
   ```python
   from fbprophet.diagnostics import cross_validation
   df_cv = cross_validation(m, horizon='30 days', period='15 days', initial='90 days')
   ```

- Performance Metrics:
   ```python
   from fbprophet.diagnostics import performance_metrics
   df_performance = performance_metrics(df_cv)
   ```

- Plot RMSE/MSE:
   ```python
   from fbprophet.plot import plot_cross_validation_metric
   plot_cross_validation_metric(df_cv, metric='rmse')
   ```

---

## 🔍 Key Results

- Accurate 30-day ahead predictions of COVID-19 cases
- Visual component decomposition of seasonality
- Model performance tracked via RMSE, MSE, MAE, MAPE

---

## 📚 Libraries Used

- `pandas`
- `matplotlib`
- `seaborn`
- `fbprophet`
- `pystan`

---

## 📌 Conclusion

The project successfully demonstrates a practical application of Facebook Prophet for forecasting pandemic data. It highlights how historical trends can guide future planning and health response.

> **Disclaimer:** This project is for educational purposes and should not be used for public health decision-making without proper validation.

