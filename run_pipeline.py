import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader import load_raw_superstore, clean_sales_data, aggregate_daily_sales
from src.feature_engineering import prepare_modeling_dataset
from src.model import get_models, train_and_evaluate_all, generate_future_forecast


def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12


def generate_eda_visualizations(raw_df: pd.DataFrame, daily_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting_style()

    # 1. Sales Trend Over Time (Daily + 30-Day Rolling Mean)
    plt.figure(figsize=(14, 6))
    rolling_30 = daily_df["total_sales"].rolling(30, min_periods=7).mean()
    plt.plot(daily_df["order_date"], daily_df["total_sales"], color="#90CAF9", alpha=0.5, label="Daily Actual Sales")
    plt.plot(daily_df["order_date"], rolling_30, color="#1565C0", linewidth=2.5, label="30-Day Moving Average Trend")
    plt.title("Historical Daily Sales & 30-Day Moving Average Trend")
    plt.xlabel("Order Date")
    plt.ylabel("Sales Volume ($)")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_sales_trend_over_time.png"), dpi=300)
    plt.close()

    # 2. Monthly Seasonality (Aggregated Across Years)
    raw_df_copy = raw_df.copy()
    raw_df_copy["month_name"] = raw_df_copy["order_date"].dt.strftime("%b")
    raw_df_copy["month_num"] = raw_df_copy["order_date"].dt.month
    monthly_sales = raw_df_copy.groupby(["month_num", "month_name"])["sales"].sum().reset_index().sort_values("month_num")

    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Blues_d", n_colors=12)
    ax = sns.barplot(data=monthly_sales, x="month_name", y="sales", hue="month_name", palette=palette, legend=False)
    plt.title("Seasonality: Total Historical Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue ($)")
    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_sales_seasonality_monthly.png"), dpi=300)
    plt.close()

    # 3. Day of Week Demand Distribution
    raw_df_copy["day_name"] = raw_df_copy["order_date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_sales = raw_df_copy.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=day_sales, x="day_name", y="sales", hue="day_name", palette="crest", legend=False)
    plt.title("Average Order Value by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Mean Order Value ($)")
    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_sales_day_of_week.png"), dpi=300)
    plt.close()

    # 4. Sales Distribution by Product Category
    if "category" in raw_df_copy.columns:
        cat_summary = raw_df_copy.groupby("category").agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum") if "profit" in raw_df_copy.columns else ("sales", "sum")
        ).reset_index()

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=cat_summary, x="category", y="total_sales", hue="category", palette="viridis", legend=False)
        plt.title("Revenue by Product Category")
        plt.xlabel("Category")
        plt.ylabel("Total Sales ($)")
        for p in ax.patches:
            ax.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_category_performance.png"), dpi=300)
        plt.close()


def generate_evaluation_plots(test_df: pd.DataFrame, predictions: dict, summary_df: pd.DataFrame, best_model_name: str, output_dir: str):
    setup_plotting_style()

    # 5. Out-of-Time Actual vs Predicted on Holdout 90-day Test Set
    plt.figure(figsize=(14, 6))
    plt.plot(test_df["order_date"], test_df["total_sales"], label="Actual Daily Sales", color="#263238", linewidth=1.8, marker="o", markersize=3)
    plt.plot(test_df["order_date"], predictions[best_model_name], label=f"Predicted ({best_model_name})", color="#E65100", linewidth=2.2, linestyle="--")
    plt.title(f"Holdout Test Performance: Actual vs. Predicted Demand ({best_model_name})")
    plt.xlabel("Date (Last 90 Days of History)")
    plt.ylabel("Sales Volume ($)")
    plt.legend(frameon=True, facecolor="white", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_actual_vs_predicted_test.png"), dpi=300)
    plt.close()

    # 6. Leaderboard Metrics Comparison (MAE and wMAPE)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    color1 = "#1976D2"
    color2 = "#D32F2F"

    bars1 = ax1.bar(x - width/2, summary_df["MAE ($)"], width, label="MAE ($)", color=color1)
    ax1.set_ylabel("Mean Absolute Error ($)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, summary_df["wMAPE (%)"], width, label="wMAPE (%)", color=color2, alpha=0.85)
    ax2.set_ylabel("Weighted MAPE (%)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["Model"], rotation=15, ha="right")
    plt.title("Model Performance Leaderboard Comparison")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_model_comparison_metrics.png"), dpi=300)
    plt.close()


def generate_forecast_plot(historical_tail_df: pd.DataFrame, forecast_df: pd.DataFrame, output_dir: str):
    setup_plotting_style()

    plt.figure(figsize=(14, 6))
    plt.plot(historical_tail_df["order_date"], historical_tail_df["total_sales"], label="Recent Actual Sales", color="#37474F", linewidth=1.8)
    plt.plot(forecast_df["date"], forecast_df["predicted_sales"], label="Forecasted Sales (Next 30 Days)", color="#00897B", linewidth=2.5, marker="s", markersize=4)

    # Shaded safety stock buffer
    plt.fill_between(
        forecast_df["date"],
        forecast_df["conservative_lower"],
        forecast_df["safety_stock_upper"],
        color="#80CBC4",
        alpha=0.35,
        label="Safety Stock Buffer (90% Service Level)"
    )

    plt.title("Future 30-Day Demand Projection with Recommended Safety Stock Bands")
    plt.xlabel("Timeline")
    plt.ylabel("Projected Daily Sales ($)")
    plt.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_future_demand_forecast_with_safety_stock.png"), dpi=300)
    plt.close()


def generate_business_report(summary_df: pd.DataFrame, forecast_df: pd.DataFrame, best_model_name: str, output_path: str):
    best_mae = summary_df.loc[summary_df["Model"] == best_model_name, "MAE ($)"].values[0]
    best_wmape = summary_df.loc[summary_df["Model"] == best_model_name, "wMAPE (%)"].values[0]
    total_proj_sales = forecast_df["predicted_sales"].sum()
    avg_daily_sales = forecast_df["predicted_sales"].mean()
    peak_day = forecast_df.loc[forecast_df["predicted_sales"].idxmax()]

    report = f"""# Executive Business Report: Sales & Demand Forecasting System
**Prepared For:** Retail Store Owners, General Managers, & Operations Executives  
**Task:** Machine Learning Task 1 (Future Interns)  
**Winning Model:** {best_model_name}  

---

## 1. Executive Summary & Business Impact
Forecasting daily demand is critical to maintaining high customer satisfaction while safeguarding operating margins. This predictive system shifts retail planning from **reactive guesswork** to **data-driven foresight**.

### Key 30-Day Outlook:
- **Total Projected Sales (Next 30 Days):** **${total_proj_sales:,.2f}**
- **Average Projected Daily Sales:** **${avg_daily_sales:,.2f}**
- **Expected Demand Spike:** **{peak_day['date'].strftime('%Y-%m-%d')}** (Projected: **${peak_day['predicted_sales']:,.2f}**)
- **Safety Stock Recommended Threshold:** Up to **${forecast_df['safety_stock_upper'].max():,.2f}** peak buffer

---

## 2. Model Accuracy & Benchmark Leaderboard
Evaluated against a strict 90-day chronological holdout test set to replicate real-world forward testing:

| Model | MAE ($) | RMSE ($) | wMAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in summary_df.iterrows():
        report += f"| {row['Model']} | ${row['MAE ($)']:,.2f} | ${row['RMSE ($)']:,.2f} | {row['wMAPE (%)']}% | {row['R2 Score']} |\n"

    report += f"""
> **Why {best_model_name}?**  
> {best_model_name} delivered the lowest Weighted Mean Absolute Percentage Error (**{best_wmape}%**) and Mean Dollar Error (**${best_mae:,.2f}**), outperforming naive moving averages by capturing both seasonal surges and day-of-week demand shifts.

---

## 3. Operational Action Plan for Management

### 📦 A. Inventory & Warehouse Procurement
1. **Prevent Stockouts on High-Run Days:** Place inventory supplier purchase orders at least 10–14 days prior to the forecasted spike on **{peak_day['date'].strftime('%Y-%m-%d')}**.
2. **Buffer with Dynamic Safety Stock:** Utilize the generated **90% Service Level Upper Band** for high-velocity items (Technology & Office Supplies) to prevent lost sales during unexpected order clusters.
3. **Minimize Holding Costs:** During identified low-demand mid-week periods, maintain lean floor stock to avoid unnecessary capital tie-up.

### 👥 B. Staffing & Shift Scheduling
1. **Peak Scheduling:** Align fulfillment center and floor associate shifts with predicted weekend and month-end demand spikes.
2. **Fulfillment Logistics:** Pre-allocate courier pickups and packing supplies 48 hours before identified sales surges.

### 💵 C. Financial & Cash Flow Planning
1. **Revenue Predictability:** With a reliable 30-day baseline revenue target of **${total_proj_sales:,.2f}**, finance can plan supplier payment disbursements and marketing campaign budgets with high confidence.

---

## 4. Visual Evidence
- **Historical Trend & 30-Day Moving Average:** `reports/figures/01_sales_trend_over_time.png`
- **Monthly Seasonality Cycles:** `reports/figures/02_sales_seasonality_monthly.png`
- **Weekly Demand Distribution:** `reports/figures/03_sales_day_of_week.png`
- **Actual vs Forecast Holdout Validation:** `reports/figures/05_actual_vs_predicted_test.png`
- **Future 30-Day Projection with Safety Bands:** `reports/figures/07_future_demand_forecast_with_safety_stock.png`
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    print("==========================================================")
    print("Sales & Demand Forecasting System - Future Interns ML 01")
    print("==========================================================")

    data_path = os.path.join("data", "raw", "superstore.csv")
    figures_dir = os.path.join("reports", "figures")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load & Clean Data
    print("\n[Step 1/5] Loading and cleaning raw transactional data...")
    raw_df = load_raw_superstore(data_path)
    clean_df = clean_sales_data(raw_df)
    clean_df.to_csv(os.path.join(processed_dir, "cleaned_transactions.csv"), index=False)
    print(f" -> Processed {len(clean_df):,} verified transactions.")

    # 2. Resample & Daily Aggregation
    print("\n[Step 2/5] Aggregating continuous daily sales time series...")
    daily_df = aggregate_daily_sales(clean_df)
    daily_df.to_csv(os.path.join(processed_dir, "daily_sales.csv"), index=False)
    print(f" -> Constructed continuous timeline of {len(daily_df)} days ({daily_df['order_date'].min().strftime('%Y-%m-%d')} to {daily_df['order_date'].max().strftime('%Y-%m-%d')}).")

    # 3. Exploratory Data Analysis & Visualizations
    print("\n[Step 3/5] Generating exploratory analysis charts...")
    generate_eda_visualizations(clean_df, daily_df, figures_dir)
    print(f" -> Saved 4 EDA figures to {figures_dir}")

    # 4. Feature Engineering & Chronological Train-Test Split
    print("\n[Step 4/5] Engineering time features, lags, and rolling metrics...")
    clean_modeling_df, feature_cols, train_df, test_df, X_train, y_train, X_test, y_test = prepare_modeling_dataset(
        daily_df, target_col="total_sales", test_days=90
    )
    clean_modeling_df.to_csv(os.path.join(processed_dir, "features_sales.csv"), index=False)
    print(f" -> Generated {len(feature_cols)} features (calendar, lags [1,7,14,21,28,30], rolling averages).")
    print(f" -> Chronological Split: Train Set = {len(train_df)} days | Test Holdout = {len(test_df)} days.")

    # 5. Train Models & Evaluate
    print("\n[Step 5/5] Training models and generating forecasts...")
    models = get_models(random_state=42)
    summary_df, fitted_models, predictions = train_and_evaluate_all(models, X_train, y_train, X_test, y_test)
    summary_df.to_csv(os.path.join("reports", "model_comparison_leaderboard.csv"), index=False)

    print("\n--- MODEL LEADERBOARD ---")
    print(summary_df.to_string(index=False))

    best_model_name = summary_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    print(f"\nWinning Model: {best_model_name}")

    # Evaluation plots
    generate_evaluation_plots(test_df, predictions, summary_df, best_model_name, figures_dir)

    # Future 30-Day Forecast
    print("\nGenerating 30-day future demand forecast with safety stock intervals...")
    forecast_df = generate_future_forecast(best_model, daily_df, feature_cols=feature_cols, forecast_days=30, service_level_z=1.65)
    forecast_df.to_csv(os.path.join("reports", "future_30d_forecast.csv"), index=False)

    recent_tail_df = daily_df.tail(60)
    generate_forecast_plot(recent_tail_df, forecast_df, figures_dir)

    # Executive Business Summary
    business_report_path = os.path.join("reports", "business_executive_summary.md")
    generate_business_report(summary_df, forecast_df, best_model_name, business_report_path)
    print(f" -> Saved executive business report to {business_report_path}")

    print("\n Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
