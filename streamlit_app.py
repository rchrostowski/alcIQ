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
- Avoid **over-ordering slow movers that tie up cash**
- Build a **clean order file by vendor**
- Understand **how each product is selling**

Upload a **single Excel file** with three sheets:
`Sales`, `Inventory`, and `Products`.
"""
)

st.divider()

# ------------------------------------------------------------------------------
# Load Excel File
# ------------------------------------------------------------------------------

def load_from_excel(uploaded_file):
    if uploaded_file is None:
        return None, None, None

    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return None, None, None

    required_sheets = {"Sales", "Inventory", "Products"}
    missing = required_sheets - set(xls.sheet_names)
    if missing:
        st.error("Excel file is missing sheets: " + ", ".join(missing))
        return None, None, None

    try:
        sales = pd.read_excel(xls, sheet_name="Sales", parse_dates=["date"])
        inventory = pd.read_excel(xls, sheet_name="Inventory")
        products = pd.read_excel(xls, sheet_name="Products")
    except Exception as e:
        st.error(f"Error reading sheets: {e}")
        return None, None, None

    return sales, inventory, products

# ------------------------------------------------------------------------------
# Demand forecast
# ------------------------------------------------------------------------------

def compute_demand_stats(sales_df, lookback_days=30, today=None, short_window_days=7, alpha=0.7):
    if today is None:
        today = sales_df["date"].max()

    short_window_days = min(short_window_days, lookback_days)

    long_cutoff = today - pd.Timedelta(days=lookback_days - 1)
    sales_long = sales_df[sales_df["date"].between(long_cutoff, today)]
    if sales_long.empty:
        sales_long = sales_df.copy()

    daily_long = (
        sales_long.groupby(["sku", "date"], as_index=False)["qty_sold"].sum()
    )
    long_stats = (
        daily_long.groupby("sku")["qty_sold"]
        .agg(long_avg_daily="mean", long_std_daily="std")
        .reset_index()
    )

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

    stats = long_stats.merge(short_stats, on="sku", how="left")

    stats["short_avg_daily"] = stats["short_avg_daily"].fillna(stats["long_avg_daily"])
    stats["avg_daily_demand"] = alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]

# ------------------------------------------------------------------------------
# Reorder recommendation engine
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    lookback_days=30,
    safety_z=1.65,
    review_period_days=7,
):
    if any(df is None for df in [sales_df, inventory_df, products_df]):
        return None

    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days)
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

    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    merged["raw_order_qty"] = (merged["target_stock"] - merged["on_hand_qty"]).clip(lower=0)

    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / merged["case_size"].replace(0, 1))
    merged["recommended_order_qty"] = merged["order_cases"] * merged["case_size"].replace(0, 1)

    merged["unit_cost"] = merged["cost"]
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    last_prices = (
        sales_df.sort_values("date")
        .groupby("sku")["unit_price"]
        .last()
        .rename("last_unit_price")
        .reset_index()
    )

    merged = merged.merge(last_prices, on="sku", how="left")

    merged["estimated_margin_per_unit"] = merged["last_unit_price"] - merged["unit_cost"]
    merged["estimated_profit_on_order"] = merged["recommended_order_qty"] * merged["estimated_margin_per_unit"]

    merged["stockout_risk"] = np.where(
        merged["on_hand_qty"] < merged["reorder_point"], "HIGH", "LOW"
    )

    return merged.sort_values(["stockout_risk", "vendor", "estimated_profit_on_order"], ascending=[False, True, False])

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------

st.sidebar.header("Upload Data")

excel_file = st.sidebar.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"],
)

with st.sidebar.expander("Download Excel template"):
    from io import BytesIO

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # Instructions
        instructions = pd.DataFrame([
            ["Sales", "date", "Date of sale (YYYY-MM-DD)."],
            ["Sales", "sku", "Product ID used in your POS."],
            ["Sales", "product_name", "Name of item sold."],
            ["Sales", "qty_sold", "Units sold that day."],
            ["Sales", "unit_price", "Selling price per unit."],
            ["Inventory", "sku", "Product ID."],
            ["Inventory", "on_hand_qty", "Units currently in stock."],
            ["Products", "sku", "Product ID (master key)."],
            ["Products", "brand", "Brand name."],
            ["Products", "product_name", "Full product name."],
            ["Products", "category", "Vodka, Tequila, etc."],
            ["Products", "size", "750ml, 12pk, etc."],
            ["Products", "vendor", "Distributor name."],
            ["Products", "cost", "Your unit cost."],
            ["Products", "case_size", "Units per case."],
            ["Products", "lead_time_days", "Vendor delivery lead time."],
        ], columns=["Sheet", "Column", "Description"])

        instructions.to_excel(writer, sheet_name="Instructions", index=False)

        pd.DataFrame(columns=["date", "sku", "product_name", "qty_sold", "unit_price"])\
            .to_excel(writer, sheet_name="Sales", index=False)

        pd.DataFrame(columns=["sku", "on_hand_qty"])\
            .to_excel(writer, sheet_name="Inventory", index=False)

        pd.DataFrame(columns=[
            "sku", "brand", "product_name", "category", "size",
            "vendor", "cost", "case_size", "lead_time_days"
        ]).to_excel(writer, sheet_name="Products", index=False)

    st.sidebar.download_button(
        "Download Excel template",
        data=buf.getvalue(),
        file_name="alcIQ_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.sidebar.markdown("---")
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Days of sales history", 7, 120, 30)
safety_z = st.sidebar.slider("Safety level", 0.0, 3.0, 1.65, 0.05)
review_period_days = st.sidebar.slider("Days between orders", 3, 28, 7)

# ------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------

sales_df, inventory_df, products_df = load_from_excel(excel_file)

if sales_df is None:
    st.warning("Upload an Excel file to continue.")
    st.stop()

# Data sanity
with st.expander("Data Summary"):
    st.write(f"Sales rows: {len(sales_df):,}")
    st.write(f"Inventory rows: {len(inventory_df):,}")
    st.write(f"Products rows: {len(products_df):,}")

# ------------------------------------------------------------------------------
# Compute recs
# ------------------------------------------------------------------------------

recs = compute_reorder_recommendations(
    sales_df, inventory_df, products_df,
    lookback_days, safety_z, review_period_days
)

if recs is None or recs.empty:
    st.warning("Unable to generate recommendations — check your data.")
    st.stop()

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Recommended Order",
    "📊 Inventory Health",
    "🔍 SKU Explorer",
    "🏷️ Vendor Summary",
    "🧾 Clean Order Report",
])

# ------------------------------------------------------------------------------
# TAB 1 — Recommended Order
# ------------------------------------------------------------------------------

with tab1:
    st.subheader("📦 Recommended Order")

    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected_vendor != "(All vendors)":
        df = df[df["vendor"] == selected_vendor]

    df = df[df["recommended_order_qty"] > 0]

    total_cost = df["extended_cost"].sum()
    items_to_order = len(df)

    c1, c2 = st.columns(2)
    c1.metric("Total Cost", f"${total_cost:,.2f}")
    c2.metric("Items to Order", items_to_order)

    st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5 — Clean Order REPORT
# ------------------------------------------------------------------------------

with tab5:
    st.subheader("🧾 Clean Order Report")

    base = recs[recs["recommended_order_qty"] > 0].copy()

    vendors_clean = ["(All vendors combined)"] + sorted(base["vendor"].dropna().unique())
    sel_vendor = st.selectbox("Vendor for clean report", vendors_clean)

    report = base if sel_vendor == "(All vendors combined)" else base[base["vendor"] == sel_vendor]

    clean = report.rename(columns={
        "vendor": "Vendor",
        "sku": "SKU",
        "brand": "Brand",
        "product_name": "Product",
        "size": "Size",
        "order_cases": "Cases to order",
        "recommended_order_qty": "Units to order",
        "unit_cost": "Cost per unit ($)",
        "extended_cost": "Line total ($)"
    })[
        ["Vendor", "SKU", "Brand", "Product", "Size",
         "Cases to order", "Units to order", "Cost per unit ($)", "Line total ($)"]
    ]

    clean["Cost per unit ($)"] = clean["Cost per unit ($)"].round(2)
    clean["Line total ($)"] = clean["Line total ($)"].round(2)

    st.dataframe(clean, use_container_width=True)

    buf = io.StringIO()
    clean.to_csv(buf, index=False)

    fname = (
        "alciq_clean_order_all_vendors.csv" if sel_vendor == "(All vendors combined)"
        else f"alciq_clean_order_{sel_vendor}.csv"
    ).replace(" ", "_")

    st.download_button(
        "Download Clean Order CSV",
        data=buf.getvalue(),
        file_name=fname,
        mime="text/csv",
    )

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------

st.divider()
st.caption("alcIQ – Liquor inventory intelligence for modern retailers.")





