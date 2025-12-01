from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="alcIQ – Liquor Inventory Optimizer",
    page_icon="🍾",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------

st.title("🍾 alcIQ – Liquor Inventory & Order Optimizer")

st.markdown(
    """
**alcIQ** turns your recent sales and inventory into **clear, easy-to-use order recommendations**.

For liquor & package stores, alcIQ helps you:

- See **which items are at risk of running out**
- Avoid **over-ordering slow movers**
- Build a **clean order file by vendor**
- Understand **how each product is selling**

Use the sample data or upload your own.
"""
)

st.divider()

# ------------------------------------------------------------------------------
# Data loading (no caching to avoid stale sample data)
# ------------------------------------------------------------------------------

def load_sample_data():
    """Always load latest CSVs from /data (no caching)."""
    base_dir = Path(__file__).parent
    sales = pd.read_csv(base_dir / "data" / "sales_sample.csv", parse_dates=["date"])
    inventory = pd.read_csv(base_dir / "data" / "inventory_sample.csv")
    products = pd.read_csv(base_dir / "data" / "products_sample.csv")
    return sales, inventory, products

def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file, parse_dates=parse_dates)

# ------------------------------------------------------------------------------
# Demand forecast (blended average: long window + recent window)
# ------------------------------------------------------------------------------

def compute_demand_stats(
    sales_df,
    lookback_days=30,
    today=None,
    short_window_days=7,
    alpha=0.7,
):
    """Blended avg daily demand: reacts fast to trends but stays stable."""
    if today is None:
        today = sales_df["date"].max()

    short_window_days = min(short_window_days, lookback_days)

    # Long window
    long_cutoff = today - pd.Timedelta(days=lookback_days - 1)
    sales_long = sales_df[sales_df["date"].between(long_cutoff, today)]
    if sales_long.empty:
        sales_long = sales_df.copy()

    daily_long = (
        sales_long.groupby(["sku", "date"], as_index=False)["qty_sold"].sum()
    )
    long_stats = (
        daily_long.groupby("sku")["qty_sold"]
        .agg(
            long_avg_daily="mean",
            long_std_daily="std",
            long_days_sold="count",
        )
        .reset_index()
    )

    # Short window
    short_cutoff = today - pd.Timedelta(days=short_window_days - 1)
    sales_short = sales_df[sales_df["date"].between(short_cutoff, today)]
    if sales_short.empty:
        sales_short = sales_long.copy()

    daily_short = (
        sales_short.groupby(["sku", "date"], as_index=False)["qty_sold"].sum()
    )
    short_stats = (
        daily_short.groupby("sku")["qty_sold"]
        .agg(short_avg_daily="mean")
        .reset_index()
    )

    # Merge & blend
    stats = long_stats.merge(short_stats, on="sku", how="left")
    stats["short_avg_daily"] = stats["short_avg_daily"].fillna(stats["long_avg_daily"])

    stats["avg_daily_demand"] = (
        alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    )

    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]

# ------------------------------------------------------------------------------
# Inventory + reorder calculation
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=30,
    safety_z=1.65,
    review_period_days=7,
):
    if sales_df is None or inventory_df is None or products_df is None:
        return None

    # Normalize SKU type
    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    # Demand stats
    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    # Merge everything
    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    d_bar = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)

    # Reorder point
    merged["reorder_point"] = d_bar * L + safety_z * sigma * np.sqrt(L)

    # Target stock
    target_days = L + review_period_days
    merged["target_stock"] = d_bar * target_days + safety_z * sigma * np.sqrt(L)

    # Zero demand → no stock
    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    # Raw order
    merged["raw_order_qty"] = merged["target_stock"] - merged["on_hand_qty"]
    merged["raw_order_qty"] = merged["raw_order_qty"].clip(lower=0)

    # Round to case sizes
    cases = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / cases)
    merged["recommended_order_qty"] = merged["order_cases"] * cases
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    # Financials
    merged["unit_cost"] = merged["cost"]
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    # Last selling price
    last_prices = (
        sales_df.sort_values("date")
        .groupby("sku")["unit_price"]
        .last()
        .reset_index()
        .rename(columns={"unit_price": "last_unit_price"})
    )
    merged = merged.merge(last_prices, on="sku", how="left")

    merged["estimated_margin_per_unit"] = (
        merged["last_unit_price"] - merged["unit_cost"]
    )
    merged["estimated_profit_on_order"] = (
        merged["recommended_order_qty"] * merged["estimated_margin_per_unit"]
    )

    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    display_cols = [
        "sku","brand","product_name","category","size","vendor",
        "on_hand_qty","avg_daily_demand","reorder_point","target_stock",
        "recommended_order_qty","order_cases",
        "unit_cost","extended_cost","last_unit_price",
        "estimated_margin_per_unit","estimated_profit_on_order",
        "stockout_risk","lead_time_days","case_size",
    ]

    merged = merged[display_cols].sort_values(
        by=["stockout_risk","vendor","estimated_profit_on_order"],
        ascending=[False, True, False]
    )
    return merged

# ------------------------------------------------------------------------------
# Sidebar (uploads + settings)
# ------------------------------------------------------------------------------

st.sidebar.header("Data & Configuration")

use_sample = st.sidebar.toggle("Use sample data built into alcIQ", value=True)

if not use_sample:
    st.sidebar.markdown("**Upload your POS CSV files**")
    sales_file = st.sidebar.file_uploader("Sales CSV", type=["csv"])
    inventory_file = st.sidebar.file_uploader("Inventory CSV", type=["csv"])
    products_file = st.sidebar.file_uploader("Products CSV", type=["csv"])
else:
    sales_file = inventory_file = products_file = None

st.sidebar.markdown("---")

lookback_days = st.sidebar.slider(
    "Days of sales history to analyze",
    7, 120, 30
)
safety_z = st.sidebar.slider(
    "Safety level (higher = fewer stockouts)",
    0.0, 3.0, 1.65, step=0.05
)
review_period_days = st.sidebar.slider(
    "Days between vendor orders",
    3, 28, 7
)

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

if use_sample:
    sales_df, inventory_df, products_df = load_sample_data()
else:
    sales_df = load_csv(sales_file, parse_dates=["date"])
    inventory_df = load_csv(inventory_file)
    products_df = load_csv(products_file)

if sales_df is None or inventory_df is None or products_df is None:
    st.warning("Upload all 3 CSV files, or enable sample data.")
    st.stop()

# ------------------------------------------------------------------------------
# Compute recommendations
# ------------------------------------------------------------------------------

recs = compute_reorder_recommendations(
    sales_df, inventory_df, products_df,
    lookback_days=lookback_days,
    safety_z=safety_z,
    review_period_days=review_period_days
)

if recs is None or recs.empty:
    st.warning("Could not generate recommendations.")
    st.stop()

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_order, tab_health, tab_sku, tab_vendor = st.tabs([
    "📦 Recommended Order",
    "📊 Inventory Health",
    "🔍 SKU Explorer",
    "🏷️ Vendor Summary",
])

# ------------------------------------------------------------------------------
# TAB 1: Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("📦 Recommended Order")
    st.markdown(
        """
This shows **what to order**, based on:
- how fast each item sells,
- how much you have left,
- vendor lead times,
- how often you order.

Filter by vendor or download the clean order file.
"""
    )

    vendors = ["(All)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected_vendor != "(All)":
        df = df[df["vendor"] == selected_vendor]

    if not st.checkbox("Show zero-quantity recommendations", value=False):
        df = df[df["recommended_order_qty"] > 0]

    # KPIs
    total_cost = df["extended_cost"].sum()
    total_profit = df["estimated_profit_on_order"].sum()
    count_items = (df["recommended_order_qty"] > 0).sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Cost", f"${total_cost:,.2f}")
    k2.metric("Estimated Profit", f"${total_profit:,.2f}")
    k3.metric("Items to Order", int(count_items))

    # Friendly display names
    display_df = df.rename(columns={
        "sku":"SKU","brand":"Brand","product_name":"Product",
        "category":"Category","size":"Size","vendor":"Vendor",
        "on_hand_qty":"On Hand","avg_daily_demand":"Avg Daily Sales",
        "reorder_point":"Reorder Point","target_stock":"Target Stock",
        "recommended_order_qty":"Order Units","order_cases":"Order Cases",
        "unit_cost":"Cost/Unit","extended_cost":"Total Cost",
        "last_unit_price":"Last Price","estimated_margin_per_unit":"Margin/Unit",
        "estimated_profit_on_order":"Profit From Order",
        "stockout_risk":"Risk","lead_time_days":"Lead Time",
        "case_size":"Case Size"
    })

    st.dataframe(
        display_df.style.format({
            "On Hand":"{:.0f}", "Avg Daily Sales":"{:.2f}",
            "Reorder Point":"{:.1f}", "Target Stock":"{:.1f}",
            "Order Units":"{:.0f}", "Order Cases":"{:.0f}",
            "Cost/Unit":"${:.2f}", "Total Cost":"${:.2f}",
            "Last Price":"${:.2f}", "Margin/Unit":"${:.2f}",
            "Profit From Order":"${:.2f}",
        }),
        use_container_width=True,
        height=500
    )

    # Export
    out = display_df[[
        "Vendor","SKU","Brand","Product","Size",
        "Order Units","Order Cases","Cost/Unit","Total Cost"
    ]]

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "Download Order CSV",
        buf.getvalue(),
        "alciq_order.csv",
        "text/csv"
    )

# ------------------------------------------------------------------------------
# TAB 2: Inventory Health
# ------------------------------------------------------------------------------

with tab_health:
    st.subheader("📊 Inventory Health")
    st.markdown(
        """
Left = **fast movers that might run out**  
Right = **slow movers holding too much money**
"""
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### ⚠️ High Stockout Risk")
        risky = recs[recs["stockout_risk"] == "HIGH"].copy()
        risky = risky.sort_values("avg_daily_demand", ascending=False).head(15)
        risky_display = risky.rename(columns={
            "sku":"SKU","brand":"Brand","product_name":"Product",
            "on_hand_qty":"On Hand","avg_daily_demand":"Avg Daily Sales",
            "reorder_point":"Reorder Point"
        })
        st.dataframe(risky_display[
            ["SKU","Brand","Product","On Hand","Avg Daily Sales","Reorder Point"]
        ], use_container_width=True)

    with right:
        st.markdown("### 🐌 Slow Movers (Overstock Risk)")
        slow = recs[recs["avg_daily_demand"] < 0.25].copy()
        slow = slow.sort_values("on_hand_qty", ascending=False).head(15)
        slow_display = slow.rename(columns={
            "sku":"SKU","brand":"Brand","product_name":"Product",
            "on_hand_qty":"On Hand","avg_daily_demand":"Avg Daily Sales"
        })
        st.dataframe(slow_display[
            ["SKU","Brand","Product","On Hand","Avg Daily Sales"]
        ], use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: SKU Explorer
# ------------------------------------------------------------------------------

with tab_sku:
    st.subheader("🔍 SKU Explorer")

    options = (
        recs[["sku","product_name","vendor","category"]]
        .drop_duplicates()
        .copy()
    )
    options["label"] = options.apply(
        lambda r: f"{r['sku']} – {r['product_name']} ({r['vendor']})",
        axis=1
    )

    selected = st.selectbox("Pick a product", sorted(options["label"]))
    selected_sku = options[options["label"] == selected]["sku"].iloc[0]

    row = recs[recs["sku"] == selected_sku].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On Hand", int(row["on_hand_qty"]))
    c2.metric("Avg Daily Sales", f"{row['avg_daily_demand']:.2f}")
    c3.metric("Reorder Point", f"{row['reorder_point']:.1f}")
    c4.metric("Target Stock", f"{row['target_stock']:.1f}")

    st.markdown("#### Daily Sales Trend")
    sku_sales = (
        sales_df[sales_df["sku"].astype(str) == selected_sku]
        .groupby("date")["qty_sold"]
        .sum()
        .reset_index()
        .set_index("date")
    )

    if sku_sales.empty:
        st.info("No sales history for this product.")
    else:
        st.line_chart(sku_sales["qty_sold"])

# ------------------------------------------------------------------------------
# TAB 4: Vendor Summary
# ------------------------------------------------------------------------------

with tab_vendor:
    st.subheader("🏷️ Vendor Summary")

    vendor_df = recs.copy()
    vendor_df["vendor"] = vendor_df["vendor"].fillna("(Unknown)")

    summary = (
        vendor_df.groupby("vendor")
        .agg(
            cost=("extended_cost","sum"),
            profit=("estimated_profit_on_order","sum"),
            catalog=("sku","nunique"),
            ordering=("recommended_order_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    pretty = summary.rename(columns={
        "vendor":"Vendor",
        "cost":"Order Cost",
        "profit":"Est Profit",
        "catalog":"# Products",
        "ordering":"# Ordering",
    })

    st.dataframe(
        pretty.style.format({
            "Order Cost":"${:,.2f}",
            "Est Profit":"${:,.2f}",
            "# Products":"{:.0f}",
            "# Ordering":"{:.0f}",
        }),
        use_container_width=True,
    )

    st.markdown("#### Spend by Vendor")
    st.bar_chart(pretty.set_index("Vendor")["Order Cost"])

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------

st.divider()
st.caption(
    "alcIQ – prototype inventory optimization tool for liquor & package stores."
)
 
