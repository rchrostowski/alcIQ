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
**alcIQ** turns your recent sales and inventory into **clear, easy-to-use order recommendations.**

If you’re a store owner or manager, alcIQ helps you:

- See **what you’re likely to run out of** soon.
- Avoid **over-ordering slow movers** that tie up cash.
- Build a **clean order file by vendor** that you can send to your reps or upload to portals.

Use the sample data to see how it works, or upload your own CSV exports in the sidebar.
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

    # Select and order columns for presentation (internal names)
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
    "Use sample data built into alcIQ",
    value=True,
    help="Turn this off to upload your own CSV exports.",
)

sales_file = None
inventory_file = None
products_file = None

if not use_sample:
    st.sidebar.markdown("**Step 1 – Upload your data**")
    st.sidebar.caption(
        "Export these files from your POS/back office system and upload them here."
    )

    sales_file = st.sidebar.file_uploader(
        "Sales file (CSV)",
        type=["csv"],
        help="One row per day and product. Columns: date, sku, product_name, qty_sold, unit_price.",
        key="sales",
    )
    inventory_file = st.sidebar.file_uploader(
        "Current inventory file (CSV)",
        type=["csv"],
        help="Current on-hand counts. Columns: sku, on_hand_qty.",
        key="inventory",
    )
    products_file = st.sidebar.file_uploader(
        "Product / vendor file (CSV)",
        type=["csv"],
        help="Product details. Columns: sku, brand, product_name, category, size, vendor, cost, case_size, lead_time_days.",
        key="products",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 2 – Assumptions**")
st.sidebar.caption("These control how aggressively alcIQ protects against stockouts.")

lookback_days = st.sidebar.slider(
    "How many past days of sales to look at?",
    min_value=7,
    max_value=120,
    value=30,
    step=1,
    help="alcIQ uses this many days of sales to estimate how fast each product sells.",
)

safety_z = st.sidebar.slider(
    "How much do you want to avoid stockouts?",
    min_value=0.0,
    max_value=3.0,
    value=1.65,
    step=0.05,
    help="Higher = fewer stockouts but more inventory. Around 1.6 is a good starting point.",
)

review_period_days = st.sidebar.slider(
    "How many days between orders (for the same vendor)?",
    min_value=3,
    max_value=28,
    value=7,
    step=1,
    help="If you normally order from a vendor once per week, set this to 7.",
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
    st.warning("Upload all three CSVs in the sidebar, or turn on **Use sample data built into alcIQ**.")
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
    st.subheader("📦 Recommended Order")

    st.markdown(
        """
**What this section shows**

Each row is a product alcIQ thinks you **should consider ordering** based on:

- How fast it’s been selling recently.
- How long it takes to arrive from the vendor.
- How often you place orders.

**How to use it**

1. Filter by vendor.
2. Review the recommended quantities.
3. Download the CSV and send it to your reps or upload to their portal.
"""
    )

    # Filter by vendor
    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Show products from which vendor?", vendors)

    filtered = recs.copy()
    if selected_vendor != "(All vendors)":
        filtered = filtered[filtered["vendor"] == selected_vendor]

    show_all = st.checkbox(
        "Include products with a recommended order of 0",
        value=False,
        help="Turn this on if you want to see everything, not just items alcIQ suggests reordering.",
    )
    if not show_all:
        filtered = filtered[filtered["recommended_order_qty"] > 0]

    # KPIs
    total_order_cost = filtered["extended_cost"].sum()
    total_est_profit = filtered["estimated_profit_on_order"].sum()
    num_skus_to_order = (filtered["recommended_order_qty"] > 0).sum()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total cost of this order", f"${total_order_cost:,.2f}")
    kpi2.metric("Estimated profit on this order", f"${total_est_profit:,.2f}")
    kpi3.metric("Number of products to order", int(num_skus_to_order))

    st.markdown("#### Line-item detail")

    # Friendlier column names for display
    display_df = filtered.rename(
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
        height=480,
    )

    st.markdown(
        """
**Helpful columns to focus on**

- **On hand (units)** – what you currently have in the store.
- **Avg daily sales (units)** – roughly how many you sell per day.
- **Reorder point (units)** – when you get down to this level, it’s time to reorder.
- **Recommended order (units)** – what alcIQ suggests adding on this order.
"""
    )

    st.markdown("#### Export order")

    download_cols = [
        "Vendor",
        "SKU",

