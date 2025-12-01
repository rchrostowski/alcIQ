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
**alcIQ** turns raw POS exports into **data-driven purchase orders** for liquor & beverage retailers.

- Ingest recent **sales**, **inventory**, and **product master** data.
- Estimate **demand** and calculate **reorder points** and **target stock levels**.
- Generate a **recommended order** by SKU and vendor, exportable as CSV.

Use the bundled sample data to explore the workflow, or upload your own CSVs in the sidebar.
"""
)

st.divider()

# ------------------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------------------

@st.cache_data
def load_sample_data():
    """Load bundled sample data (kept in the repo under data/)."""
    base_dir = Path(__file__).parent
    sales_path = base_dir / "data" / "sales_sample.csv"
    inventory_path = base_dir / "data" / "inventory_sample.csv"
    products_path = base_dir / "data" / "products_sample.csv"

    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inventory = pd.read_csv(inventory_path)
    products = pd.read_csv(products_path)
    return sales, inventory, products


def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file, parse_dates=parse_dates)


# ------------------------------------------------------------------------------
# Core demand + inventory logic
# ------------------------------------------------------------------------------

def compute_demand_stats(sales_df, lookback_days=30, today=None):
    """
    Compute average daily demand and volatility for each SKU
    over a configurable lookback window.
    """
    if today is None:
        today = sales_df["date"].max()

    cutoff = today - pd.Timedelta(days=lookback_days - 1)
    recent = sales_df[sales_df["date"].between(cutoff, today)]

    # Fallback: if the filtered window is empty, use all available data
    if recent.empty:
        recent = sales_df.copy()

    # Aggregate quantity by sku + day
    daily = (
        recent.groupby(["sku", "date"], as_index=False)["qty_sold"]
        .sum()
    )

    # Compute demand statistics per SKU
    stats = (
        daily.groupby("sku")["qty_sold"]
        .agg(
            avg_daily_demand="mean",
            std_daily_demand="std",
            days_sold="count",
        )
        .reset_index()
    )

    # If only 1 day of data, std is NaN → treat as 0
    stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0.0)
    return stats


def compute_reorder_recommendations(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    lookback_days: int = 30,
    safety_z: float = 1.65,
    review_period_days: int = 7,
) -> pd.DataFrame | None:
    """
    Combine sales, inventory, and product master data to generate
    reorder recommendations per SKU.

    - Uses a simple continuous review model:
      ROP = d_bar * L + z * sigma * sqrt(L)
    - Target stock covers lead time + review period.
    """
    if sales_df is None or inventory_df is None or products_df is None:
        return None

    # Ensure SKU is a consistent key type
    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    # Merge product master, current inventory, and demand stats
    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    # Fill missing inventory with 0
    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)

    # If a SKU has no sales history, treat demand and volatility as 0 for now
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    # Core inventory math
    d_bar = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)  # default lead time if not provided

    # Reorder Point (ROP): expected demand over lead time + safety stock
    merged["reorder_point"] = d_bar * L + safety_z * sigma * np.sqrt(L)

    # Target stock: demand over lead time + next review period + safety
    target_days = L + review_period_days
    merged["target_stock"] = d_bar * target_days + safety_z * sigma * np.sqrt(L)

    # Do not recommend stock for SKUs with essentially zero demand
    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    # Raw recommended order = what we "should" hold minus what we have
    merged["raw_order_qty"] = merged["target_stock"] - merged["on_hand_qty"]

    # No negative orders
    merged["raw_order_qty"] = merged["raw_order_qty"].clip(lower=0)

    # Round up to case sizes
    case_size = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / case_size)
    merged["recommended_order_qty"] = merged["order_cases"] * case_size

    # Very small orders (<1 unit) → treat as zero
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    # Financials
    merged["unit_cost"] = merged["cost"]
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    # Approximate margin using most recent unit price from sales
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

    # Simple stockout risk flag
    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    # Select and order columns for presentation
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
        ascending=[False, True, False],
    )

    return merged


# ------------------------------------------------------------------------------
# Sidebar – data & configuration
# ------------------------------------------------------------------------------

st.sidebar.header("Data & Configuration")

use_sample = st.sidebar.toggle(
    "Use bundled sample data",
    value=True,
    help="Turn this off to upload your own CSV exports.",
)

sales_file = None
inventory_file = None
products_file = None

if not use_sample:
    st.sidebar.markdown("**Upload store data**")

    sales_file = st.sidebar.file_uploader(
        "Sales CSV",
        type=["csv"],
        help="Expected columns: date, sku, product_name, qty_sold, unit_price",
        key="sales",
    )
    inventory_file = st.sidebar.file_uploader(
        "Inventory CSV",
        type=["csv"],
        help="Expected columns: sku, on_hand_qty",
        key="inventory",
    )
    products_file = st.sidebar.file_uploader(
        "Products CSV",
        type=["csv"],
        help="Expected columns: sku, brand, product_name, category, size, vendor, cost, case_size, lead_time_days",
        key="products",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Forecast & policy assumptions**")

lookback_days = st.sidebar.slider(
    "Demand lookback window (days)",
    min_value=7,
    max_value=120,
    value=30,
    step=1,
    help="Number of days of sales history used to estimate average daily demand.",
)

safety_z = st.sidebar.slider(
    "Service level (safety factor, Z-score)",
    min_value=0.0,
    max_value=3.0,
    value=1.65,
    step=0.05,
    help="Higher values reduce stockouts but increase inventory (1.65 ≈ 95% service).",
)

review_period_days = st.sidebar.slider(
    "Order review period (days)",
    min_value=3,
    max_value=28,
    value=7,
    step=1,
    help="Typical number of days between orders for the same vendor.",
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
    st.warning("Upload all three CSVs in the sidebar, or enable **Use bundled sample data**.")
    st.stop()

# ------------------------------------------------------------------------------
# Compute recommendations
# ------------------------------------------------------------------------------

recs = compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=lookback_days,
    safety_z=safety_z,
    review_period_days=review_period_days,
)

if recs is None or recs.empty:
    st.warning("No recommendations could be generated. Please verify your data.")
    st.stop()

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_order, tab_health, tab_sku, tab_vendor = st.tabs(
    ["📦 Recommended Order", "📊 Inventory Health", "🔍 SKU Explorer", "🏷️ Vendor Summary"]
)

# ------------------------------------------------------------------------------
# Tab 1 – Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("Recommended Purchase Order")

    # Filter by vendor
    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    filtered = recs.copy()
    if selected_vendor != "(All vendors)":
        filtered = filtered[filtered["vendor"] == selected_vendor]

    show_all = st.checkbox(
        "Include SKUs with zero recommended order",
        value=False,
        help="Turn this on if you want to see all SKUs, not just those to reorder.",
    )
    if not show_all:
        filtered = filtered[filtered["recommended_order_qty"] > 0]

    # KPIs
    total_order_cost = filtered["extended_cost"].sum()
    total_est_profit = filtered["estimated_profit_on_order"].sum()
    num_skus_to_order = (filtered["recommended_order_qty"] > 0).sum()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total recommended order cost", f"${total_order_cost:,.2f}")
    kpi2.metric("Estimated profit on this order", f"${total_est_profit:,.2f}")
    kpi3.metric("SKUs with positive order", int(num_skus_to_order))

    st.markdown("#### Line-item detail")

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
        height=480,
    )

    st.markdown("#### Export order")

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
        label="Download recommended order as CSV",
        data=csv_data,
        file_name="alciq_recommended_order.csv",
        mime="text/csv",
        help="Attach this file to an email or upload it to your distributor portal.",
    )

# ------------------------------------------------------------------------------
# Tab 2 – Inventory Health
# ------------------------------------------------------------------------------

with tab_health:
    st.subheader("Inventory Health Overview")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**High stockout risk (fast movers, low inventory)**")
        risky = recs[recs["stockout_risk"] == "HIGH"].copy()
        risky = risky.sort_values("avg_daily_demand", ascending=False).head(10)
        if risky.empty:
            st.info("No SKUs currently flagged as high stockout risk at this service level.")
        else:
            st.dataframe(
                risky[
                    [
                        "sku",
                        "brand",
                        "product_name",
                        "on_hand_qty",
                        "avg_daily_demand",
                        "reorder_point",
                    ]
                ],
                use_container_width=True,
            )

    with col_right:
        st.markdown("**Overstocked / very slow movers**")
        slow = recs[recs["avg_daily_demand"] < 0.2].copy()
        slow = slow.sort_values("on_hand_qty", ascending=False).head(10)
        if slow.empty:
            st.info("No SKUs meet the current overstock / slow-mover criteria.")
        else:
            st.dataframe(
                slow[
                    [
                        "sku",
                        "brand",
                        "product_name",
                        "on_hand_qty",
                        "avg_daily_demand",
                    ]
                ],
                use_container_width=True,
            )

    st.markdown(
        """
Use this view to identify:

- **Critical items** that are likely to stock out without an order.
- **Dead or slow inventory** that may be candidates for discounting, promotion, or delisting.
        """
    )

# ------------------------------------------------------------------------------
# Tab 3 – SKU Explorer
# ------------------------------------------------------------------------------

with tab_sku:
    st.subheader("SKU Explorer")

    sku_options = (
        recs[["sku", "product_name", "vendor", "category"]]
        .drop_duplicates()
        .copy()
    )

    if sku_options.empty:
        st.info("No SKUs available to explore.")
    else:
        sku_options["label"] = sku_options.apply(
            lambda r: f"{r['sku']} – {r['product_name']} ({r['vendor']})", axis=1
        )
        label_to_sku = dict(zip(sku_options["label"], sku_options["sku"]))

        selected_label = st.selectbox(
            "Select a SKU",
            sorted(label_to_sku.keys()),
        )
        selected_sku = label_to_sku[selected_label]

        sku_row = recs[recs["sku"] == selected_sku].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("On hand", f"{sku_row['on_hand_qty']:.0f}")
        c2.metric("Avg daily demand", f"{sku_row['avg_daily_demand']:.2f}")
        c3.metric("Reorder point", f"{sku_row['reorder_point']:.1f}")
        c4.metric("Target stock", f"{sku_row['target_stock']:.1f}")

        st.markdown("#### Recent daily sales")

        sku_sales = (
            sales_df[sales_df["sku"].astype(str) == selected_sku]
            .groupby("date")["qty_sold"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        if sku_sales.empty:
            st.info("No sales history available for this SKU in the selected window.")
        else:
            sku_sales = sku_sales.set_index("date")
            st.line_chart(sku_sales["qty_sold"])

# ------------------------------------------------------------------------------
# Tab 4 – Vendor Summary
# ------------------------------------------------------------------------------

with tab_vendor:
    st.subheader("Vendor Summary")

    vendor_df = recs.copy()
    vendor_df["vendor"] = vendor_df["vendor"].fillna("(Unspecified)")

    summary = (
        vendor_df.groupby("vendor")
        .agg(
            total_order_cost=("extended_cost", "sum"),
            total_estimated_profit=("estimated_profit_on_order", "sum"),
            skus_in_catalog=("sku", "nunique"),
            skus_with_order=("recommended_order_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("total_order_cost", ascending=False)
    )

    st.dataframe(
        summary.style.format(
            {
                "total_order_cost": "${:,.2f}",
                "total_estimated_profit": "${:,.2f}",
                "skus_in_catalog": "{:.0f}",
                "skus_with_order": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Spend by vendor")

    if not summary.empty:
        chart_data = summary.set_index("vendor")[["total_order_cost"]]
        st.bar_chart(chart_data)

st.divider()
st.caption(
    "alcIQ – prototype decision support tool for liquor and beverage retailers. "
    "For demonstration purposes only; configuration and assumptions should be calibrated to each store."
)


