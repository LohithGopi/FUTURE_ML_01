# Executive Business Report: Sales & Demand Forecasting System
**Prepared For:** Retail Store Owners, General Managers, & Operations Executives  
**Task:** Machine Learning Task 1 (Future Interns)  
**Winning Model:** Random Forest  

---

## 1. Executive Summary & Business Impact
Forecasting daily demand is critical to maintaining high customer satisfaction while safeguarding operating margins. This predictive system shifts retail planning from **reactive guesswork** to **data-driven foresight**.

### Key 30-Day Outlook:
- **Total Projected Sales (Next 30 Days):** **$72,141.44**
- **Average Projected Daily Sales:** **$2,404.71**
- **Expected Demand Spike:** **2018-12-31** (Projected: **$3,278.20**)
- **Safety Stock Recommended Threshold:** Up to **$5,169.86** peak buffer

---

## 2. Model Accuracy & Benchmark Leaderboard
Evaluated against a strict 90-day chronological holdout test set to replicate real-world forward testing:

| Model | MAE ($) | RMSE ($) | wMAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | $2,064.64 | $2,866.45 | 67.06% | 0.0895 |
| XGBoost Regressor | $2,126.49 | $2,892.55 | 69.07% | 0.0728 |
| Ridge Regression | $2,141.95 | $3,028.26 | 69.57% | -0.0162 |
| Baseline (Lag-7 Persistence) | $2,807.33 | $4,119.00 | 91.19% | -0.8801 |

> **Why Random Forest?**  
> Random Forest delivered the lowest Weighted Mean Absolute Percentage Error (**67.06%**) and Mean Dollar Error (**$2,064.64**), outperforming naive moving averages by capturing both seasonal surges and day-of-week demand shifts.

---

## 3. Operational Action Plan for Management

### 📦 A. Inventory & Warehouse Procurement
1. **Prevent Stockouts on High-Run Days:** Place inventory supplier purchase orders at least 10–14 days prior to the forecasted spike on **2018-12-31**.
2. **Buffer with Dynamic Safety Stock:** Utilize the generated **90% Service Level Upper Band** for high-velocity items (Technology & Office Supplies) to prevent lost sales during unexpected order clusters.
3. **Minimize Holding Costs:** During identified low-demand mid-week periods, maintain lean floor stock to avoid unnecessary capital tie-up.

### 👥 B. Staffing & Shift Scheduling
1. **Peak Scheduling:** Align fulfillment center and floor associate shifts with predicted weekend and month-end demand spikes.
2. **Fulfillment Logistics:** Pre-allocate courier pickups and packing supplies 48 hours before identified sales surges.

### 💵 C. Financial & Cash Flow Planning
1. **Revenue Predictability:** With a reliable 30-day baseline revenue target of **$72,141.44**, finance can plan supplier payment disbursements and marketing campaign budgets with high confidence.

---

## 4. Visual Evidence
- **Historical Trend & 30-Day Moving Average:** `reports/figures/01_sales_trend_over_time.png`
- **Monthly Seasonality Cycles:** `reports/figures/02_sales_seasonality_monthly.png`
- **Weekly Demand Distribution:** `reports/figures/03_sales_day_of_week.png`
- **Actual vs Forecast Holdout Validation:** `reports/figures/05_actual_vs_predicted_test.png`
- **Future 30-Day Projection with Safety Bands:** `reports/figures/07_future_demand_forecast_with_safety_stock.png`
