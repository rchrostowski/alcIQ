import math
import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Pavlish Inventory Optimizer",
    layout="wide"
)

st.title("🍾 Pavlish Inventory & Order Optimizer")

st.markdown(
    """
This is a **prototype for Pavlish Beverage Company**.

- Upload POS exports (sales + inventory + products) or use the sample data.
- The app will **forecast demand** and **recommend what to order** by SKU.
- You can filter by vendor and export a clean order file.
"""
)

# ---------- Helper functions ----------

@st.cache_data
def load_sample_data():
    sales = pd.read_csv("data/sales_sample.csv", parse_dates=["date"])
    inventory = pd.read_csv("data/inventory_sample.csv")
    products = pd.read_csv("data/products_sample.csv")
    return sales, inventory, products


def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file, parse_dates=parse_dates)


def compute_demand_stats(sales_df, lookback_days=30, today=None):
    """Compute average daily demand and std dev for each SKU."""
    if today is None:
        today = sales_df["date"].max()

    cutoff = today - pd.Timedelta(days=lookback_days - 1)
    recent = sales_df[sales_df["date"].between(cutoff, today)]

    # If no data after filter, fall back to all
    if recent.empty:
        recent = sales_df.copy()

    # Aggregate qty by sku + day
    daily = (
        recent.groupby(["sku", "date"], as_index=False)["qty_sold"]
        .sum()
    )

    # Compute stats by sku
    stats = (
        daily.groupby("sku")["qty_sold"]
        .agg(
            avg_daily_demand="mean",
            std_daily_demand="std",
            days_sold="count"
        )
        .reset_index()
    )

    # Replace NaN std with 0 if only 1 day of data
    stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0.0)
    return stats


def compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=30,
    safety_z=1.65,
    review_period_days=7
):
    if sales_df is None or inventory_df is None or products_df is None:
        return None

    # Ensure SKU is the key
    for df in [sales_df, inventory_df, products_df]:
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    # Merge everything
    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    # Fill missing inventory with 0
    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)

    # If a SKU has no sales history, set demand to 0 for now
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    # Core inventory math
    d_bar = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)  # default 5 days if missing

    # Reorder Point: ROP = d_bar * L + z * sigma * sqrt(L)
    merged["reorder_point"] = d_bar * L + safety_z * sigma * np.sqrt(L)

    # Target stock = demand over (L + review period) + safety stock (approx)
    target_days = L + review_period_days
    merged["target_stock"] = d_bar * target_days + safety_z * sigma * np.sqrt(L)

    # If avg demand is basically zero, don't recommend anything
    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    # Compute raw recommended order
    merged["raw_order_qty"] = merged["target_stock"] - merged["on_hand_qty"]

    # Don't order negative
    merged["raw_order_qty"] = merged["raw_order_qty"].clip(lower=0)

    # Round to case sizes
    case_size = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / case_size)
    merged["recommended_order_qty"] = merged["order_cases"] * case_size

    # If recommended is tiny (< 1 bottle), zero it out
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    # Financials
    merged["unit_cost"] = merged["cost"]
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    # Simple margin estimate if we have unit_price
    latest_prices = (
        sales_df.sort_values("date")
        .groupby("sku")["unit_price"]
        .last()
        .reset_index()
        .rename(columns={"unit_price": "last_unit_price"})
    )
    merged = merged.merge(latest_prices, on="sku", how="left")
    merged["estimated_margin_per_unit"] = merged["last_unit_price"] - merged["unit_cost"]
    merged["estimated_profit_on_order"] = (
        merged["recommended_order_qty"] * merged["estimated_margin_per_unit"]
    )

    # Stockout risk: if on_hand < ROP
    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    # Clean up columns for display
    display_cols = [
        "sku",
        "brand",
        "product_name",
        "category",
        "size",
        "vendor",
        "on_hand_qty",
        "avg_daily_demand",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
        "order_cases",
        "unit_cost",
        "extended_cost",
        "last_unit_price",
        "estimated_margin_per_unit",
        "estimated_profit_on_order",
        "stockout_risk",
        "lead_time_days",
        "case_size",
    ]

    merged = merged[display_cols].sort_values(
        by=["stockout_risk", "vendor", "estimated_profit_on_order"],
        ascending=[False, True, False]
    )

    return merged


# ---------- Sidebar: data + parameters ----------

st.sidebar.header("Data & Settings")

use_sample = st.sidebar.toggle("Use sample data (demo Pavlish-style)", value=True)

sales_file = None
inventory_file = None
products_file = None

if not use_sample:
    sales_file = st.sidebar.file_uploader(
        "Sales CSV (date, sku, product_name, qty_sold, unit_price)",
        type=["csv"],
        key="sales"
    )
    inventory_file = st.sidebar.file_uploader(
        "Inventory CSV (sku, on_hand_qty)",
        type=["csv"],
        key="inventory"
    )
    products_file = st.sidebar.file_uploader(
        "Products CSV (sku, brand, product_name, category, size, vendor, cost, case_size, lead_time_days)",
        type=["csv"],
        key="products"
    )

lookback_days = st.sidebar.slider(
    "Lookback window for demand (days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

safety_z = st.sidebar.slider(
    "Safety factor (Z-score, higher = fewer stockouts, more inventory)",
    min_value=0.0,
    max_value=3.0,
    value=1.65,
    step=0.05
)

review_period_days = st.sidebar.slider(
    "Review period (days between orders)",
    min_value=3,
    max_value=21,
    value=7,
    step=1
)

# ---------- Load data ----------

if use_sample:
    sales_df, inventory_df, products_df = load_sample_data()
else:
    sales_df = load_csv(sales_file, parse_dates=["date"])
    inventory_df = load_csv(inventory_file)
    products_df = load_csv(products_file)

if sales_df is None or inventory_df is None or products_df is None:
    st.warning("⬅️ Upload all three CSVs or toggle sample data in the sidebar.")
    st.stop()

# ---------- Compute recommendations ----------

recs = compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=lookback_days,
    safety_z=safety_z,
    review_period_days=review_period_days
)

if recs is None or recs.empty:
    st.warning("No recommendations could be computed. Check your data.")
    st.stop()

# ---------- Filters ----------

vendors = ["(All Vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
selected_vendor = st.selectbox("Filter by vendor", vendors)

filtered = recs.copy()
if selected_vendor != "(All Vendors)":
    filtered = filtered[filtered["vendor"] == selected_vendor]

# Only show SKUs with non-zero recommendation unless user wants full view
show_all = st.checkbox("Show SKUs with zero recommended order", value=False)
if not show_all:
    filtered = filtered[filtered["recommended_order_qty"] > 0]

# ---------- KPIs ----------

total_order_cost = filtered["extended_cost"].sum()
total_est_profit = filtered["estimated_profit_on_order"].sum()
num_skus_to_order = (filtered["recommended_order_qty"] > 0).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Recommended Order Cost", f"${total_order_cost:,.2f}")
col2.metric("Estimated Profit From This Order", f"${total_est_profit:,.2f}")
col3.metric("Number of SKUs to Order", int(num_skus_to_order))

st.markdown("### 📋 Recommended Order Detail")

st.dataframe(
    filtered.style.format(
        {
            "on_hand_qty": "{:.0f}",
            "avg_daily_demand": "{:.2f}",
            "reorder_point": "{:.1f}",
            "target_stock": "{:.1f}",
            "recommended_order_qty": "{:.0f}",
            "order_cases": "{:.0f}",
            "unit_cost": "${:.2f}",
            "extended_cost": "${:.2f}",
            "last_unit_price": "${:.2f}",
            "estimated_margin_per_unit": "${:.2f}",
            "estimated_profit_on_order": "${:.2f}",
        }
    ),
    use_container_width=True,
    height=500
)

# ---------- Download ----------

st.markdown("#### 📦 Export Order as CSV")

download_cols = [
    "vendor",
    "sku",
    "brand",
    "product_name",
    "size",
    "category",
    "recommended_order_qty",
    "order_cases",
    "unit_cost",
    "extended_cost",
]

out_df = filtered[download_cols].copy()
csv_buffer = io.StringIO()
out_df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="Download Order CSV",
    data=csv_data,
    file_name="pavlish_recommended_order.csv",
    mime="text/csv"
)

# ---------- Extra views ----------

st.markdown("---")
st.markdown("### 🔍 Inventory Health Snapshot")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Top SKUs by stockout risk (high demand, low inventory)**")
    risky = recs[recs["stockout_risk"] == "HIGH"].copy()
    risky = risky.sort_values("avg_daily_demand", ascending=False).head(10)
    st.dataframe(
        risky[["sku", "brand", "product_name", "on_hand_qty", "avg_daily_demand", "reorder_point"]],
        use_container_width=True
    )

with col_b:
    st.markdown("**Overstocked SKUs (very slow movers)**")
    slow = recs[recs["avg_daily_demand"] < 0.2].copy()
    slow = slow.sort_values("on_hand_qty", ascending=False).head(10)
    st.dataframe(
        slow[["sku", "brand", "product_name", "on_hand_qty", "avg_daily_demand"]],
        use_container_width=True
    )

st.markdown(
    """
This is a **free pilot build for Pavlish Beverage**.  
Once you plug in real POS exports, this will produce *store-specific, data-driven order sheets*.
"""
)
