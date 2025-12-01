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
# Minimal styling tweaks (keeps it professional & clean)
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        padding: 2rem 3rem;
    }
    h1, h2, h3, h4 {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .alciq-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    .metric-card {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        background: #020617;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
        color: #9ca3af;
    }
    .metric-card p {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0;
        color: #f9fafb;
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
    A decision-support tool that turns your recent sales and inventory into
    clear, vendor-ready order recommendations – so you avoid stockouts without
    drowning in slow movers.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to use this pilot (recommended for first-time use)", expanded=True):
    st.markdown(
        """
        1. **Start with the sample data** in the left sidebar to see how alcIQ works.
        2. When you're ready, turn off **“Use sample data built into alcIQ”**.
        3. Upload 3 CSVs exported from your POS:
           - **Sales** – at least the last 30–60 days
           - **Inventory** – current on-hand by SKU
           - **Products** – SKUs with vendor, costs, case sizes, and lead times
        4. Adjust assumptions in the sidebar:
           - Sales lookback window
           - Safety level (how cautious you want to be about stockouts)
           - How often you typically order from vendors
        5. Review the **📦 Recommended Order** tab, filter by vendor, and export CSVs
           you can send to suppliers or paste into your ordering system.

        > alcIQ is a prototype. Always double-check quantities before placing a live order.
        """
    )

st.divider()

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_sample_data():
    """
    Always load the latest CSVs from data/ (no caching).
    """
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
        st.error(f"Problem reading **{uploaded_file.name}**: {e}")
        return None


# ------------------------------------------------------------------------------
# Demand forecast (blended: long-window + short-window)
# ------------------------------------------------------------------------------

def compute_demand_stats(
    sales_df: pd.DataFrame,
    lookback_days: int = 30,
    today=None,
    short_window_days: int = 7,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Compute blended avg daily demand and volatility per SKU.

    - Long window: average over `lookback_days`
    - Short window: average over `short_window_days`
    - Blended avg = alpha * short + (1 - alpha) * long
    """
    if today is None:
        today = sales_df["date"].max()

    # Ensure short window <= long window
    short_window_days = min(short_window_days, lookback_days)

    # ----- Long window -----
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

    # ----- Short window -----
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

    # ----- Merge & blend -----
    stats = long_stats.merge(short_stats, on="sku", how="left")
    stats["short_avg_daily"] = stats["short_avg_daily"].fillna(
        stats["long_avg_daily"]
    )

    stats["avg_daily_demand"] = (
        alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    )
    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]


# ------------------------------------------------------------------------------
# Inventory + reorder recommendations
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    lookback_days: int = 30,
    safety_z: float = 1.65,
    review_period_days: int = 7,
) -> pd.DataFrame | None:
    """
    Combine sales, inventory, and products into reorder recommendations per SKU.
    """
    if sales_df is None or inventory_df is None or products_df is None:
        return None

    # Make SKU type consistent
    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    # Fill missing basics
    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    d_bar = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)

    # Reorder point
    merged["reorder_point"] = d_bar * L + safety_z * sigma * np.sqrt(L)

    # Target stock (lead time + review period)
    target_days = L + review_period_days
    merged["target_stock"] = d_bar * target_days + safety_z * sigma * np.sqrt(L)

    # No demand → don't hold stock
    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    # Raw order
    merged["raw_order_qty"] = merged["target_stock"] - merged["on_hand_qty"]
    merged["raw_order_qty"] = merged["raw_order_qty"].clip(lower=0)

    # Case rounding
    case_sizes = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / case_sizes)
    merged["recommended_order_qty"] = merged["order_cases"] * case_sizes
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    # Financials
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

    merged["estimated_margin_per_unit"] = (
        merged["last_unit_price"] - merged["unit_cost"]
    )
    merged["estimated_profit_on_order"] = (
        merged["recommended_order_qty"] * merged["estimated_margin_per_unit"]
    )

    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    cols = [
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

    merged = merged[cols].sort_values(
        by=["stockout_risk", "vendor", "estimated_profit_on_order"],
        ascending=[False, True, False],
    )
    return merged


# ------------------------------------------------------------------------------
# Sidebar – uploads & settings
# ------------------------------------------------------------------------------

st.sidebar.header("Data & Configuration")

use_sample = st.sidebar.toggle(
    "Use sample data built into alcIQ", value=True
)

if not use_sample:
    st.sidebar.markdown("**Upload your POS CSV files**")
    sales_file = st.sidebar.file_uploader(
        "Sales CSV (date, sku, product_name, qty_sold, unit_price)",
        type=["csv"],
        key="sales",
    )
    inventory_file = st.sidebar.file_uploader(
        "Inventory CSV (sku, on_hand_qty)",
        type=["csv"],
        key="inventory",
    )
    products_file = st.sidebar.file_uploader(
        "Products CSV (sku, brand, product_name, category, size, vendor, cost, case_size, lead_time_days)",
        type=["csv"],
        key="products",
    )

    with st.sidebar.expander("Need templates? Download blank CSV files"):
        st.sidebar.download_button(
            "Download sales template",
            data="date,sku,product_name,qty_sold,unit_price\n",
            file_name="alciq_sales_template.csv",
            mime="text/csv",
        )
        st.sidebar.download_button(
            "Download inventory template",
            data="sku,on_hand_qty\n",
            file_name="alciq_inventory_template.csv",
            mime="text/csv",
        )
        st.sidebar.download_button(
            "Download products template",
            data=(
                "sku,brand,product_name,category,size,vendor,cost,case_size,lead_time_days\n"
            ),
            file_name="alciq_products_template.csv",
            mime="text/csv",
        )
else:
    sales_file = inventory_file = products_file = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Assumptions**")

lookback_days = st.sidebar.slider(
    "Days of sales history to analyze",
    min_value=7,
    max_value=120,
    value=30,
)
safety_z = st.sidebar.slider(
    "Safety level (higher = fewer stockouts, more inventory)",
    min_value=0.0,
    max_value=3.0,
    value=1.65,
    step=0.05,
)
review_period_days = st.sidebar.slider(
    "Days between orders to the same vendor",
    min_value=3,
    max_value=28,
    value=7,
)

# ------------------------------------------------------------------------------
# Load data (and show a quick summary so you can confirm it's updated)
# ------------------------------------------------------------------------------

if use_sample:
    sales_df, inventory_df, products_df = load_sample_data()
else:
    sales_df = load_csv(sales_file, parse_dates=["date"])
    inventory_df = load_csv(inventory_file)
    products_df = load_csv(products_file)

if sales_df is None or inventory_df is None or products_df is None:
    st.warning("Upload all three CSV files, or turn on sample data.")
    st.stop()

# Basic schema validation
required_sales_cols = {"date", "sku", "product_name", "qty_sold", "unit_price"}
required_inventory_cols = {"sku", "on_hand_qty"}
required_products_cols = {
    "sku",
    "brand",
    "product_name",
    "category",
    "size",
    "vendor",
    "cost",
    "case_size",
    "lead_time_days",
}

def check_columns(df, required, name: str):
    missing = required - set(df.columns)
    if missing:
        st.error(
            f"{name} file is missing these required columns: {', '.join(sorted(missing))}"
        )
        return False
    return True

ok = True
ok &= check_columns(sales_df, required_sales_cols, "Sales")
ok &= check_columns(inventory_df, required_inventory_cols, "Inventory")
ok &= check_columns(products_df, required_products_cols, "Products")

if not ok:
    st.stop()

# Quick sanity panel so you can *see* the data is new
with st.expander("Data summary (for sanity check)", expanded=False):
    st.write(
        f"Sales rows: **{len(sales_df):,}**, "
        f"distinct SKUs in sales: **{sales_df['sku'].nunique():,}**"
    )
    st.write(
        f"Sales date range: **{sales_df['date'].min().date()} → {sales_df['date'].max().date()}**"
    )
    st.write(
        f"Inventory rows: **{len(inventory_df):,}**, "
        f"Products rows: **{len(products_df):,}**"
    )

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
    st.warning("Could not generate any recommendations. Check your data.")
    st.stop()

# High-level overview cards
total_cost_all = recs["extended_cost"].sum()
total_profit_all = recs["estimated_profit_on_order"].sum()
high_risk_count = (recs["stockout_risk"] == "HIGH").sum()

oc1, oc2, oc3 = st.columns(3)
with oc1:
    st.markdown(
        f'<div class="metric-card"><h3>Order value (all vendors)</h3><p>${total_cost_all:,.0f}</p></div>',
        unsafe_allow_html=True,
    )
with oc2:
    st.markdown(
        f'<div class="metric-card"><h3>Estimated profit on order</h3><p>${total_profit_all:,.0f}</p></div>',
        unsafe_allow_html=True,
    )
with oc3:
    st.markdown(
        f'<div class="metric-card"><h3>SKUs at high stockout risk</h3><p>{int(high_risk_count)}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_order, tab_health, tab_sku, tab_vendor = st.tabs(
    [
        "📦 Recommended Order",
        "📊 Inventory Health",
        "🔍 SKU Explorer",
        "🏷️ Vendor Summary",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1 – Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("📦 Recommended Order")

    st.markdown(
        """
        This table shows **what alcIQ suggests you order next**, based on:

        - How fast each item sells (recent days weighted more)
        - How much you have on hand right now
        - Lead time from vendor
        - How often you typically order
        """
    )

    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected_vendor != "(All vendors)":
        df = df[df["vendor"] == selected_vendor]

    if not st.checkbox(
        "Show products with zero recommended order", value=False
    ):
        df = df[df["recommended_order_qty"] > 0]

    total_cost = df["extended_cost"].sum()
    total_profit = df["estimated_profit_on_order"].sum()
    items_to_order = (df["recommended_order_qty"] > 0).sum()

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Total cost of this order", f"${total_cost:,.2f}")
    kc2.metric("Estimated profit from this order", f"${total_profit:,.2f}")
    kc3.metric("Number of products to order", int(items_to_order))

    display_df = df.rename(
        columns={
            "sku": "SKU",
            "brand": "Brand",
            "product_name": "Product",
            "category": "Category",
            "size": "Size",
            "vendor": "Vendor",
            "on_hand_qty": "On hand (units)",
            "avg_daily_demand": "Avg daily sales (units)",
            "reorder_point": "Reorder point (units)",
            "target_stock": "Target stock (units)",
            "recommended_order_qty": "Recommended order (units)",
            "order_cases": "Recommended cases",
            "unit_cost": "Cost per unit ($)",
            "extended_cost": "Total cost ($)",
            "last_unit_price": "Recent selling price ($)",
            "estimated_margin_per_unit": "Margin per unit ($)",
            "estimated_profit_on_order": "Est. profit on this item ($)",
            "stockout_risk": "Stockout risk",
            "lead_time_days": "Lead time (days)",
            "case_size": "Units per case",
        }
    )

    st.dataframe(
        display_df.style.format(
            {
                "On hand (units)": "{:.0f}",
                "Avg daily sales (units)": "{:.2f}",
                "Reorder point (units)": "{:.1f}",
                "Target stock (units)": "{:.1f}",
                "Recommended order (units)": "{:.0f}",
                "Recommended cases": "{:.0f}",
                "Cost per unit ($)": "${:.2f}",
                "Total cost ($)": "${:.2f}",
                "Recent selling price ($)": "${:.2f}",
                "Margin per unit ($)": "${:.2f}",
                "Est. profit on this item ($)": "${:.2f}",
                "Lead time (days)": "{:.0f}",
                "Units per case": "{:.0f}",
            }
        ),
        use_container_width=True,
        height=500,
    )

    st.markdown("#### Export order as CSV")

    order_export = display_df[
        [
            "Vendor",
            "SKU",
            "Brand",
            "Product",
            "Size",
            "Recommended order (units)",
            "Recommended cases",
            "Cost per unit ($)",
            "Total cost ($)",
        ]
    ].copy()

    buf = io.StringIO()
    order_export.to_csv(buf, index=False)
    st.download_button(
        label="Download recommended order (CSV)",
        data=buf.getvalue(),
        file_name="alciq_recommended_order.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------------------
# TAB 2 – Inventory Health
# ------------------------------------------------------------------------------

with tab_health:
    st.subheader("📊 Inventory Health")

    st.markdown(
        """
        **Left:** items selling quickly with low inventory (**risk of running out**)  
        **Right:** items selling very slowly but with a lot on hand (**overstock**)
        """
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### ⚠️ Fast movers with low inventory")
        risky = recs[recs["stockout_risk"] == "HIGH"].copy()
        risky = risky.sort_values("avg_daily_demand", ascending=False).head(15)
        risky_display = risky.rename(
            columns={
                "sku": "SKU",
                "brand": "Brand",
                "product_name": "Product",
                "on_hand_qty": "On hand (units)",
                "avg_daily_demand": "Avg daily sales (units)",
                "reorder_point": "Reorder point (units)",
            }
        )
        if risky_display.empty:
            st.info("No products are currently flagged as high stockout risk.")
        else:
            st.dataframe(
                risky_display[
                    [
                        "SKU",
                        "Brand",
                        "Product",
                        "On hand (units)",
                        "Avg daily sales (units)",
                        "Reorder point (units)",
                    ]
                ],
                use_container_width=True,
            )

    with col_right:
        st.markdown("### 🐌 Very slow movers with high stock (overstock risk)")
        slow = recs[recs["avg_daily_demand"] < 0.25].copy()
        slow = slow.sort_values("on_hand_qty", ascending=False).head(15)
        slow_display = slow.rename(
            columns={
                "sku": "SKU",
                "brand": "Brand",
                "product_name": "Product",
                "on_hand_qty": "On hand (units)",
                "avg_daily_demand": "Avg daily sales (units)",
            }
        )
        if slow_display.empty:
            st.info("No products meet the slow-mover criteria in this dataset.")
        else:
            st.dataframe(
                slow_display[
                    [
                        "SKU",
                        "Brand",
                        "Product",
                        "On hand (units)",
                        "Avg daily sales (units)",
                    ]
                ],
                use_container_width=True,
            )

# ------------------------------------------------------------------------------
# TAB 3 – SKU Explorer
# ------------------------------------------------------------------------------

with tab_sku:
    st.subheader("🔍 SKU Explorer")

    sku_options = (
        recs[["sku", "product_name", "vendor", "category"]]
        .drop_duplicates()
        .copy()
    )
    sku_options["label"] = sku_options.apply(
        lambda r: f"{r['sku']} – {r['product_name']} ({r['vendor']})", axis=1
    )

    selected_label = st.selectbox(
        "Choose a product to inspect", sorted(sku_options["label"])
    )
    selected_sku = sku_options.loc[
        sku_options["label"] == selected_label, "sku"
    ].iloc[0]

    sku_row = recs[recs["sku"] == selected_sku].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On hand (units)", f"{sku_row['on_hand_qty']:.0f}")
    c2.metric("Avg daily sales (units)", f"{sku_row['avg_daily_demand']:.2f}")
    c3.metric("Reorder point (units)", f"{sku_row['reorder_point']:.1f}")
    c4.metric("Target stock (units)", f"{sku_row['target_stock']:.1f}")

    st.markdown("#### Daily sales history")

    sku_sales = (
        sales_df[sales_df["sku"].astype(str) == selected_sku]
        .groupby("date")["qty_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    if sku_sales.empty:
        st.info("No sales history for this product in the current data.")
    else:
        sku_sales = sku_sales.set_index("date")
        st.line_chart(sku_sales["qty_sold"])

# ------------------------------------------------------------------------------
# TAB 4 – Vendor Summary
# ------------------------------------------------------------------------------

with tab_vendor:
    st.subheader("🏷️ Vendor Summary")

    vendor_df = recs.copy()
    vendor_df["vendor"] = vendor_df["vendor"].fillna("(Unspecified)")

    summary = (
        vendor_df.groupby("vendor")
        .agg(
            total_cost=("extended_cost", "sum"),
            total_profit=("estimated_profit_on_order", "sum"),
            catalog_skus=("sku", "nunique"),
            ordering_skus=(
                "recommended_order_qty",
                lambda x: (x > 0).sum(),
            ),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )

    summary_display = summary.rename(
        columns={
            "vendor": "Vendor",
            "total_cost": "Total order cost ($)",
            "total_profit": "Est. profit on this order ($)",
            "catalog_skus": "Products in catalog",
            "ordering_skus": "Products on this order",
        }
    )

    st.dataframe(
        summary_display.style.format(
            {
                "Total order cost ($)": "${:,.2f}",
                "Est. profit on this order ($)": "${:,.2f}",
                "Products in catalog": "{:.0f}",
                "Products on this order": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Spend by vendor")
    if not summary_display.empty:
        chart_data = summary_display.set_index("Vendor")[["Total order cost ($)"]]
        st.bar_chart(chart_data)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------

st.divider()
st.caption(
    "All numbers are estimates based on recent data and simple inventory logic."
)

