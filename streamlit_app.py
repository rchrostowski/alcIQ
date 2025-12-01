import io
from io import BytesIO

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
# Access Code Gate
# ------------------------------------------------------------------------------

st.markdown("### 🔐 Secure Access")

# You control these codes manually (assign one per store)
VALID_CODES = {
    "LCQ-TEST",  # your internal testing code
    "LCQ-001",
    "LCQ-002",
    "LCQ-003",
    # Add more codes as you onboard stores
}

code = st.text_input("Enter your alcIQ Access Code", type="password")

if code not in VALID_CODES:
    st.warning("Please enter a valid access code to continue.")
    st.stop()

st.success("Access granted.")
st.divider()

# ------------------------------------------------------------------------------
# Header & onboarding
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

Upload a **single Excel file** with three sheets: `Sales`, `Inventory`, and `Products`.
"""
)

st.markdown(
    """
### 🧭 How to use alcIQ

1. **Download the Excel template** from the left sidebar  
2. Export reports from your POS and paste the data into:
   - **Sales** – daily sales per SKU  
   - **Inventory** – current on-hand quantities  
   - **Products** – your item catalog (one row per SKU)
3. Upload your completed Excel file using the uploader on the left  
4. alcIQ will analyze:
   - Recent sales velocity  
   - Vendor lead times  
   - Case sizes  
   - Stockout risk  
   - Profitability per item  
5. Review your **Recommended Order** and download a clean **vendor-ready order file**
"""
)

st.divider()

# ------------------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------------------

def friendly_error(message: str):
    st.error(message)


def load_from_excel(uploaded_file):
    """Load Sales, Inventory, Products from a single Excel workbook with friendly errors."""
    if uploaded_file is None:
        return None, None, None

    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception:
        friendly_error(
            "I couldn't read that Excel file.\n\n"
            "Make sure you're uploading a valid `.xlsx` file exported from Excel."
        )
        return None, None, None

    required_sheets = {"Sales", "Inventory", "Products"}
    missing = required_sheets - set(xls.sheet_names)
    if missing:
        friendly_error(
            "Your Excel file is missing one or more required sheets.\n\n"
            "Please make sure your file includes **Sales**, **Inventory**, and **Products**."
        )
        return None, None, None

    try:
        sales = pd.read_excel(xls, sheet_name="Sales")
        inventory = pd.read_excel(xls, sheet_name="Inventory")
        products = pd.read_excel(xls, sheet_name="Products")
    except Exception:
        friendly_error(
            "There was a problem reading one of the sheets.\n\n"
            "Please double-check that the sheet names are exactly **Sales**, **Inventory**, and **Products**."
        )
        return None, None, None

    # Parse date column nicely
    if "date" not in sales.columns:
        friendly_error(
            "Your **Sales** sheet is missing the `date` column.\n\n"
            "Please add a `date` column with values like `2024-01-15`."
        )
        return None, None, None

    try:
        sales["date"] = pd.to_datetime(sales["date"])
    except Exception:
        friendly_error(
            "Some dates in your **Sales** sheet could not be understood.\n\n"
            "Make sure dates look like `2024-01-15` (YYYY-MM-DD)."
        )
        return None, None, None

    required_sales_cols = ["sku", "product_name", "qty_sold", "unit_price"]
    for col in required_sales_cols:
        if col not in sales.columns:
            friendly_error(
                f"Your **Sales** sheet is missing the `{col}` column.\n\n"
                f"Please add `{col}` to your Sales sheet and re-upload."
            )
            return None, None, None

    required_inv_cols = ["sku", "on_hand_qty"]
    for col in required_inv_cols:
        if col not in inventory.columns:
            friendly_error(
                f"Your **Inventory** sheet is missing the `{col}` column.\n\n"
                "Please make sure Inventory has `sku` and `on_hand_qty`."
            )
            return None, None, None

    required_prod_cols = [
        "sku",
        "brand",
        "product_name",
        "category",
        "size",
        "vendor",
        "cost",
        "case_size",
        "lead_time_days",
    ]
    for col in required_prod_cols:
        if col not in products.columns:
            friendly_error(
                f"Your **Products** sheet is missing the `{col}` column.\n\n"
                "Please redownload the template and copy your product data into it."
            )
            return None, None, None

    return sales, inventory, products

# ------------------------------------------------------------------------------
# Demand forecast
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
    """
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

    stats["avg_daily_demand"] = (
        alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    )
    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]

# ------------------------------------------------------------------------------
# Reorder engine
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    lookback_days: int = 30,
    safety_z: float = 1.65,
    review_period_days: int = 7,
) -> pd.DataFrame | None:
    if any(df is None for df in [sales_df, inventory_df, products_df]):
        return None

    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    d = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)

    merged["reorder_point"] = d * L + safety_z * sigma * np.sqrt(L)
    merged["target_stock"] = d * (L + review_period_days) + safety_z * np.sqrt(L)

    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

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

    return merged.sort_values(
        ["stockout_risk", "vendor", "estimated_profit_on_order"],
        ascending=[False, True, False],
    )

# ------------------------------------------------------------------------------
# Sidebar – upload, template, settings (with session_state)
# ------------------------------------------------------------------------------

st.sidebar.header("Upload Data")

excel_file = st.sidebar.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"],
    help="Use the template below for the required structure.",
)

with st.sidebar.expander("Download Excel template"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        instructions = pd.DataFrame(
            [
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
                ["Products", "lead_time_days", "Vendor delivery lead time in days."],
            ],
            columns=["Sheet", "Column", "Description"],
        )
        instructions.to_excel(writer, sheet_name="Instructions", index=False)

        pd.DataFrame(columns=["date", "sku", "product_name", "qty_sold", "unit_price"]).to_excel(
            writer, sheet_name="Sales", index=False
        )
        pd.DataFrame(columns=["sku", "on_hand_qty"]).to_excel(
            writer, sheet_name="Inventory", index=False
        )
        pd.DataFrame(
            columns=[
                "sku",
                "brand",
                "product_name",
                "category",
                "size",
                "vendor",
                "cost",
                "case_size",
                "lead_time_days",
            ]
        ).to_excel(writer, sheet_name="Products", index=False)

    st.sidebar.download_button(
        "Download Excel template",
        data=buf.getvalue(),
        file_name="alcIQ_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.sidebar.markdown("---")
st.sidebar.header("Settings")

# Default settings in session_state
if "lookback_days" not in st.session_state:
    st.session_state["lookback_days"] = 30
if "safety_z" not in st.session_state:
    st.session_state["safety_z"] = 1.65
if "review_period_days" not in st.session_state:
    st.session_state["review_period_days"] = 7

lookback_days = st.sidebar.slider(
    "Days of sales history to analyze",
    min_value=7,
    max_value=120,
    value=st.session_state["lookback_days"],
)
safety_z = st.sidebar.slider(
    "Safety level (higher = fewer stockouts, more inventory)",
    min_value=0.0,
    max_value=3.0,
    value=float(st.session_state["safety_z"]),
    step=0.05,
)
review_period_days = st.sidebar.slider(
    "Days between orders to the same vendor",
    min_value=3,
    max_value=28,
    value=st.session_state["review_period_days"],
)

# Persist into session_state
st.session_state["lookback_days"] = lookback_days
st.session_state["safety_z"] = safety_z
st.session_state["review_period_days"] = review_period_days

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

sales_df, inventory_df, products_df = load_from_excel(excel_file)

if sales_df is None or inventory_df is None or products_df is None:
    st.warning("Upload the Excel file with all three sheets to continue.")
    st.stop()

with st.expander("Data summary (for sanity check)", expanded=False):
    st.write(
        f"Sales rows: **{len(sales_df):,}**, distinct SKUs in sales: **{sales_df['sku'].nunique():,}**"
    )
    st.write(
        f"Sales date range: **{sales_df['date'].min().date()} → {sales_df['date'].max().date()}**"
    )
    st.write(
        f"Inventory rows: **{len(inventory_df):,}**, Products rows: **{len(products_df):,}**"
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
    st.warning("Could not generate any recommendations. Please check your data.")
    st.stop()

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_order, tab_health, tab_sku, tab_vendor, tab_report = st.tabs(
    [
        "📦 Recommended Order",
        "📊 Inventory Health",
        "🔍 SKU Explorer",
        "🏷️ Vendor Summary",
        "🧾 Clean Order Report",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1 – Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("📦 Recommended Order")

    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected_vendor != "(All vendors)":
        df = df[df["vendor"] == selected_vendor]

    df = df[df["recommended_order_qty"] > 0]

    total_cost = df["extended_cost"].sum()
    total_profit = df["estimated_profit_on_order"].sum()
    items_to_order = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total cost of this order", f"${total_cost:,.2f}")
    c2.metric("Estimated profit from this order", f"${total_profit:,.2f}")
    c3.metric("Number of products to order", int(items_to_order))

    st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2 – Inventory Health
# ------------------------------------------------------------------------------

with tab_health:
    st.subheader("📊 Inventory Health")

    st.markdown(
        """
**Left:** items selling quickly with low inventory (risk of running out)  
**Right:** items selling very slowly but with a lot on hand (overstock)
"""
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### ⚠️ Fast movers with low inventory")
        risky = recs[recs["stockout_risk"] == "HIGH"].copy()
        risky = risky.sort_values("avg_daily_demand", ascending=False).head(15)
        if risky.empty:
            st.info("No products are currently flagged as high stockout risk.")
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
        st.markdown("### 🐌 Very slow movers with high stock (overstock risk)")
        slow = recs[recs["avg_daily_demand"] < 0.25].copy()
        slow = slow.sort_values("on_hand_qty", ascending=False).head(15)
        if slow.empty:
            st.info("No products meet the slow-mover criteria in this data.")
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
            ordering_skus=("recommended_order_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )

    st.dataframe(summary, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5 – Clean Order Report
# ------------------------------------------------------------------------------

with tab_report:
    st.subheader("🧾 Clean Order Report")

    st.markdown(
        """
This view is designed to be a **simple, vendor-ready order file** – minimal columns,
no extra analytics. You can export this and send it directly to suppliers or paste
into their ordering portals.
"""
    )

    base = recs[recs["recommended_order_qty"] > 0].copy()

    if base.empty:
        st.info("No products currently have a positive recommended order.")
    else:
        vendors_clean = ["(All vendors combined)"] + sorted(
            base["vendor"].dropna().unique().tolist()
        )
        sel_vendor = st.selectbox(
            "Choose which vendor to generate an order for",
            vendors_clean,
        )

        report_df = base.copy()
        if sel_vendor != "(All vendors combined)":
            report_df = report_df[report_df["vendor"] == sel_vendor]

        clean = report_df.rename(
            columns={
                "vendor": "Vendor",
                "sku": "SKU",
                "brand": "Brand",
                "product_name": "Product",
                "size": "Size",
                "order_cases": "Cases to order",
                "recommended_order_qty": "Units to order",
                "unit_cost": "Cost per unit ($)",
                "extended_cost": "Line total ($)",
            }
        )[
            [
                "Vendor",
                "SKU",
                "Brand",
                "Product",
                "Size",
                "Cases to order",
                "Units to order",
                "Cost per unit ($)",
                "Line total ($)",
            ]
        ].sort_values(["Vendor", "Brand", "Product"])

        clean["Cost per unit ($)"] = clean["Cost per unit ($)"].round(2)
        clean["Line total ($)"] = clean["Line total ($)"].round(2)

        st.dataframe(
            clean.style.format(
                {
                    "Cases to order": "{:.0f}",
                    "Units to order": "{:.0f}",
                    "Cost per unit ($)": "${:.2f}",
                    "Line total ($)": "${:.2f}",
                }
            ),
            use_container_width=True,
            height=500,
        )

        # CSV export
        csv_buf = io.StringIO()
        clean.to_csv(csv_buf, index=False)

        if sel_vendor == "(All vendors combined)":
            base_name = "alciq_clean_order_all_vendors"
        else:
            safe_vendor = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "_" for c in sel_vendor
            ).strip()
            base_name = f"alciq_clean_order_{safe_vendor}"

        st.download_button(
            "Download Clean Order CSV",
            data=csv_buf.getvalue(),
            file_name=f"{base_name}.csv",
            mime="text/csv",
        )

        # Excel export (openpyxl)
        xls_buf = BytesIO()
        with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
            clean.to_excel(writer, index=False, sheet_name="Order")

        st.download_button(
            "Download Clean Order Excel (.xlsx)",
            data=xls_buf.getvalue(),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ------------------------------------------------------------------------------
# Footer & privacy
# ------------------------------------------------------------------------------

st.divider()
st.markdown(
    """
### 🔐 Privacy & Data

### 🔐 Privacy & Data

Your data is **100% private**.

alcIQ does **not** store, save, log, transmit, share, or sell any of your data — ever.

Your Excel file is processed **only inside this session**, directly in your browser and temporary app memory.  
The moment you close the app or refresh the page, all data disappears permanently.

We cannot see your sales numbers.  
We cannot see your inventory.  
We cannot see your product file.  

Your information belongs entirely to **you**, and alcIQ simply analyzes it to generate your order recommendations.
---
"""
)
st.caption("alcIQ – Liquor inventory intelligence for modern retailers.")








