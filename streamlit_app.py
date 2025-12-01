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
# UI Styling (no custom fonts, clean dark theme)
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Use Streamlit's native font */
    html, body, [class*="css"] {
        font-family: inherit !important;
    }

    /* Layout spacing */
    .main {
        padding: 2rem 3rem;
    }

    /* Metric card styling */
    .metric-card {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        background: #0f172a;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .metric-card p {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        color: #f8fafc;
    }

    /* Subtitle */
    .alciq-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------

st.title("🍾 alcIQ – Liquor Inventory & Order Optimizer")

st.markdown(
    """
    <p class="alciq-subtitle">
    Turn your sales + inventory into clear, accurate order recommendations.
    Reduce stockouts. Avoid overstock. Order smarter.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to use (recommended for first-time users)", expanded=True):
    st.markdown(
        """
        1. Start with **sample data** in the sidebar.
        2. When ready, turn off sample mode and upload your:
           • Sales CSV  
           • Inventory CSV  
           • Products CSV  
        3. Adjust assumptions in the sidebar.  
        4. Review **📦 Recommended Order** and export your order CSV.

        > alcIQ is a prototype. Double-check before placing real orders.
        """
    )

st.divider()

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_sample_data():
    base_dir = Path(__file__).parent
    sales_path = base_dir / "data" / "sales_sample.csv"
    inv_path = base_dir / "data" / "inventory_sample.csv"
    prod_path = base_dir / "data" / "products_sample.csv"
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inventory = pd.read_csv(inv_path)
    products = pd.read_csv(prod_path)
    return sales, inventory, products


def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, parse_dates=parse_dates)
    except Exception as e:
        st.error(f"Error reading **{uploaded_file.name}**: {e}")
        return None

# ------------------------------------------------------------------------------
# Demand stats
# ------------------------------------------------------------------------------

def compute_demand_stats(
    sales_df: pd.DataFrame,
    lookback_days: int = 30,
    today=None,
    short_window_days: int = 7,
    alpha: float = 0.7,
):
    if today is None:
        today = sales_df["date"].max()

    short_window_days = min(short_window_days, lookback_days)

    # Long window
    long_cutoff = today - pd.Timedelta(days=lookback_days - 1)
    sales_long = sales_df[sales_df["date"].between(long_cutoff, today)]
    if sales_long.empty:
        sales_long = sales_df.copy()
    daily_long = sales_long.groupby(["sku", "date"], as_index=False)["qty_sold"].sum()
    long_stats = (
        daily_long.groupby("sku")["qty_sold"]
        .agg(
            long_avg_daily="mean",
            long_std_daily="std",
        )
        .reset_index()
    )

    # Short window
    short_cutoff = today - pd.Timedelta(days=short_window_days - 1)
    sales_short = sales_df[sales_df["date"].between(short_cutoff, today)]
    if sales_short.empty:
        sales_short = sales_long.copy()
    daily_short = sales_short.groupby(["sku", "date"], as_index=False)["qty_sold"].sum()
    short_stats = (
        daily_short.groupby("sku")["qty_sold"]
        .agg(short_avg_daily="mean")
        .reset_index()
    )

    stats = long_stats.merge(short_stats, on="sku", how="left")
    stats["short_avg_daily"] = stats["short_avg_daily"].fillna(stats["long_avg_daily"])

    stats["avg_daily_demand"] = (
        alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    )
    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]

# ------------------------------------------------------------------------------
# Reorder recommendations
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=30,
    safety_z=1.65,
    review_period_days=7,
):

    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0)

    d = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)

    merged["reorder_point"] = d * L + safety_z * sigma * np.sqrt(L)
    merged["target_stock"] = d * (L + review_period_days) + safety_z * sigma * np.sqrt(L)

    merged.loc[d < 0.1, ["reorder_point", "target_stock"]] = 0

    merged["raw_order_qty"] = (merged["target_stock"] - merged["on_hand_qty"]).clip(lower=0)

    case_sizes = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / case_sizes)

    merged["recommended_order_qty"] = merged["order_cases"] * case_sizes
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    merged["unit_cost"] = merged["cost"]
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    last_prices = (
        sales_df.sort_values("date")
        .groupby("sku")["unit_price"]
        .last()
        .reset_index()
        .rename(columns={"unit_price": "last_unit_price"})
    )
    merged = merged.merge(last_prices, on="sku", how="left")

    merged["estimated_margin_per_unit"] = merged["last_unit_price"] - merged["unit_cost"]
    merged["estimated_profit_on_order"] = (
        merged["recommended_order_qty"] * merged["estimated_margin_per_unit"]
    )

    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    # Column order
    cols = [
        "sku", "brand", "product_name", "category", "size", "vendor",
        "on_hand_qty", "avg_daily_demand", "reorder_point", "target_stock",
        "recommended_order_qty", "order_cases",
        "unit_cost", "extended_cost", "last_unit_price",
        "estimated_margin_per_unit", "estimated_profit_on_order",
        "stockout_risk", "lead_time_days", "case_size",
    ]

    merged = merged[cols].sort_values(
        ["stockout_risk", "vendor", "estimated_profit_on_order"],
        ascending=[False, True, False],
    )

    return merged

# ------------------------------------------------------------------------------
# Sidebar – uploads & settings
# ------------------------------------------------------------------------------

st.sidebar.header("Data & Configuration")

use_sample = st.sidebar.toggle("Use sample data built into alcIQ", value=True)

if use_sample:
    sales_file = inventory_file = products_file = None
else:
    st.sidebar.markdown("**Upload your POS CSV files**")
    sales_file = st.sidebar.file_uploader(
        "Sales CSV (date, sku, product_name, qty_sold, unit_price)",
        type=["csv"],
    )
    inventory_file = st.sidebar.file_uploader(
        "Inventory CSV (sku, on_hand_qty)",
        type=["csv"],
    )
    products_file = st.sidebar.file_uploader(
        "Products CSV (sku, brand, product_name, category, size, vendor, cost, case_size, lead_time_days)",
        type=["csv"],
    )

    with st.sidebar.expander("Download blank templates"):
        st.sidebar.download_button(
            "Sales template",
            "date,sku,product_name,qty_sold,unit_price\n",
            "alciq_sales_template.csv",
        )
        st.sidebar.download_button(
            "Inventory template",
            "sku,on_hand_qty\n",
            "alciq_inventory_template.csv",
        )
        st.sidebar.download_button(
            "Products template",
            "sku,brand,product_name,category,size,vendor,cost,case_size,lead_time_days\n",
            "alciq_products_template.csv",
        )

st.sidebar.markdown("---")
lookback_days = st.sidebar.slider("Days of sales history", 7, 120, 30)
safety_z = st.sidebar.slider("Safety level", 0.0, 3.0, 1.65, 0.05)
review_period_days = st.sidebar.slider("Days between orders", 3, 28, 7)

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
    st.warning("Upload all 3 files or enable sample data.")
    st.stop()

# Schema checks
required_sales = {"date", "sku", "product_name", "qty_sold", "unit_price"}
required_inv = {"sku", "on_hand_qty"}
required_prod = {
    "sku", "brand", "product_name", "category", "size",
    "vendor", "cost", "case_size", "lead_time_days"
}

def check_cols(df, req, name):
    missing = req - set(df.columns)
    if missing:
        st.error(f"{name} CSV missing columns: {missing}")
        return False
    return True

ok = True
ok &= check_cols(sales_df, required_sales, "Sales")
ok &= check_cols(inventory_df, required_inv, "Inventory")
ok &= check_cols(products_df, required_prod, "Products")

if not ok:
    st.stop()

# ------------------------------------------------------------------------------
# Compute recommendations
# ------------------------------------------------------------------------------

recs = compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days,
    safety_z,
    review_period_days,
)

if recs.empty:
    st.warning("No recommendations generated.")
    st.stop()

# Overview cards
total_cost = recs["extended_cost"].sum()
total_profit = recs["estimated_profit_on_order"].sum()
high_risk = (recs["stockout_risk"] == "HIGH").sum()

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-card"><h3>Total order cost</h3><p>${total_cost:,.0f}</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><h3>Estimated profit</h3><p>${total_profit:,.0f}</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><h3>High-risk SKUs</h3><p>{int(high_risk)}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_order, tab_health, tab_sku, tab_vendor = st.tabs(
    ["📦 Recommended Order", "📊 Inventory Health", "🔍 SKU Explorer", "🏷️ Vendor Summary"]
)

# ------------------------------------------------------------------------------
# TAB: Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("📦 Recommended Order")

    vendors = ["All vendors"] + sorted(recs["vendor"].unique())
    selected = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected != "All vendors":
        df = df[df["vendor"] == selected]

    df_disp = df[df["recommended_order_qty"] > 0]

    st.dataframe(df_disp, use_container_width=True, height=500)

    buf = io.StringIO()
    df_disp.to_csv(buf, index=False)
    st.download_button("Download order (CSV)", buf.getvalue(), "alciq_order.csv")

# ------------------------------------------------------------------------------
# TAB: Inventory Health
# ------------------------------------------------------------------------------

with tab_health:
    st.subheader("📊 Inventory Health")

    fast = recs[recs["stockout_risk"] == "HIGH"].sort_values("avg_daily_demand", ascending=False)
    slow = recs[recs["avg_daily_demand"] < 0.25].sort_values("on_hand_qty", ascending=False)

    col1, col2 = st.columns(2)
    col1.markdown("### ⚠️ Risk of running out")
    col1.dataframe(fast.head(15), use_container_width=True)

    col2.markdown("### 🐌 Overstock risk")
    col2.dataframe(slow.head(15), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB: SKU Explorer
# ------------------------------------------------------------------------------

with tab_sku:
    st.subheader("🔍 SKU Explorer")

    options = recs["sku"].unique()
    chosen = st.selectbox("Select a SKU", options)

    row = recs[recs["sku"] == chosen].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("On hand", row["on_hand_qty"])
    m2.metric("Avg daily sales", f"{row['avg_daily_demand']:.2f}")
    m3.metric("Reorder point", f"{row['reorder_point']:.1f}")
    m4.metric("Target stock", f"{row['target_stock']:.1f}")

    st.markdown("#### Sales history")
    sku_sales = (
        sales_df[sales_df["sku"].astype(str) == chosen]
        .groupby("date")["qty_sold"]
        .sum()
    )
    st.line_chart(sku_sales)

# ------------------------------------------------------------------------------
# TAB: Vendor Summary
# ------------------------------------------------------------------------------

with tab_vendor:
    st.subheader("🏷️ Vendor Summary")

    summary = (
        recs.groupby("vendor")
        .agg(
            total_cost=("extended_cost", "sum"),
            total_profit=("estimated_profit_on_order", "sum"),
            skus=("sku", "nunique"),
            items=("recommended_order_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("vendor")["total_cost"])

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------

st.divider()
st.caption("alcIQ – prototype tool for liquor & beverage retailers.")


