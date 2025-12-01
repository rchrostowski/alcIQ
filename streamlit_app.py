import io
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# For PDF generation (Phase 2 – PO PDF)
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="alcIQ – Liquor Inventory Optimizer",
    page_icon="🍾",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Access Code Gate (Phase 1 – Control access)
# ------------------------------------------------------------------------------

st.markdown("### 🔐 Secure Access")

VALID_CODES = {
    "ALCIQ-TEST",
    "ALCIQ-001",
    "ALCIQ-002",
    "ALCIQ-003",
    "LCQ-TEST",
    # add more per customer
}

code = st.text_input("Enter your alcIQ Access Code", type="password")

if code not in VALID_CODES:
    st.warning("Please enter a valid access code to continue.")
    st.stop()

st.success("Access granted.")
st.info(
    "Your data is processed only for this session. We do **not** store, save, or "
    "sell any of your information. You can also request a signed NDA."
)
st.divider()

# ------------------------------------------------------------------------------
# Header & onboarding
# ------------------------------------------------------------------------------

st.title("🍾 alcIQ – Liquor Inventory & Order Optimizer")

st.markdown(
    """
**alcIQ** turns your recent sales and inventory into **clear, vendor-ready order recommendations**.

For liquor & package stores, alcIQ helps you:

- See **which items are at risk of running out**
- Avoid **over-ordering slow movers that tie up cash**
- Build a **clean order file by vendor**
- Understand **how each product is actually selling**

Upload a **single Excel file** with these sheets:

- `Sales`
- `Inventory`
- `Products`
- (Optional) `Discounts`
"""
)

st.markdown(
    """
### 🧭 How to use alcIQ

1. Download the Excel template from the sidebar  
2. Export / copy data from your POS into:
   - **Sales** – daily sales per SKU  
   - **Inventory** – current on-hand quantities  
   - **Products** – your product catalog (one row per SKU)  
   - **Discounts** (optional) – quantity discount levels per SKU  
3. (Optional) Upload an extra CSV of **online orders** (DoorDash, CityHive, etc.)  
4. Upload the Excel file into alcIQ  
5. Review the **Recommended Order** and **Clean Order / PO PDFs** per vendor
"""
)

st.divider()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def friendly_error(message: str):
    st.error(message)

def load_from_excel(uploaded_file):
    """
    Load Sales, Inventory, Products, and optional Discounts from a single Excel workbook,
    with friendly error messages. This is the main data load for alcIQ.
    """
    if uploaded_file is None:
        return None, None, None, None

    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception:
        friendly_error(
            "I couldn't read that Excel file.\n\n"
            "Make sure you're uploading a valid `.xlsx` file exported from Excel."
        )
        return None, None, None, None

    required_sheets = {"Sales", "Inventory", "Products"}
    missing = required_sheets - set(xls.sheet_names)
    if missing:
        friendly_error(
            "Your Excel file is missing one or more required sheets.\n\n"
            "Please make sure your file includes **Sales**, **Inventory**, and **Products**."
        )
        return None, None, None, None

    # Optional Discounts sheet
    has_discounts = "Discounts" in xls.sheet_names

    try:
        sales = pd.read_excel(xls, sheet_name="Sales")
        inventory = pd.read_excel(xls, sheet_name="Inventory")
        products = pd.read_excel(xls, sheet_name="Products")
        discounts = pd.read_excel(xls, sheet_name="Discounts") if has_discounts else None
    except Exception:
        friendly_error(
            "There was a problem reading one of the sheets.\n\n"
            "Double-check that the sheet names are exactly **Sales**, **Inventory**, **Products**, "
            "and (optionally) **Discounts**."
        )
        return None, None, None, None

    # Parse date in Sales
    if "date" not in sales.columns:
        friendly_error(
            "Your **Sales** sheet is missing the `date` column.\n\n"
            "Please add a `date` column with values like `2024-01-15`."
        )
        return None, None, None, None

    try:
        sales["date"] = pd.to_datetime(sales["date"])
    except Exception:
        friendly_error(
            "Some dates in your **Sales** sheet could not be understood.\n\n"
            "Make sure dates look like `2024-01-15` (YYYY-MM-DD)."
        )
        return None, None, None, None

    required_sales_cols = ["sku", "product_name", "qty_sold", "unit_price"]
    for col in required_sales_cols:
        if col not in sales.columns:
            friendly_error(
                f"Your **Sales** sheet is missing the `{col}` column.\n\n"
                f"Please add `{col}` to your Sales sheet and re-upload."
            )
            return None, None, None, None

    required_inv_cols = ["sku", "on_hand_qty"]
    for col in required_inv_cols:
        if col not in inventory.columns:
            friendly_error(
                f"Your **Inventory** sheet is missing the `{col}` column.\n\n"
                "Make sure Inventory has at least `sku` and `on_hand_qty`."
            )
            return None, None, None, None

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
                "Redownload the template and copy your product data into it."
            )
            return None, None, None, None

    # Discounts optional: validate minimally if present
    if discounts is not None:
        if "sku" not in discounts.columns:
            friendly_error(
                "Your **Discounts** sheet must have at least a `sku` column.\n\n"
                "Other columns can be `break_qty_1`, `price_1`, `break_qty_2`, `price_2`, etc."
            )
            return None, None, None, None

    return sales, inventory, products, discounts

def load_online_orders_csv(uploaded_file):
    """
    Optional: load online orders (DoorDash, CityHive, web store, etc.) as extra sales.
    Must have date, sku, product_name, qty_sold, unit_price columns.
    """
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        friendly_error(
            "I couldn't read the online orders CSV.\n\n"
            "Make sure it's a standard `.csv` file with a header row."
        )
        return None

    required_cols = ["date", "sku", "product_name", "qty_sold", "unit_price"]
    for col in required_cols:
        if col not in df.columns:
            friendly_error(
                f"Your online orders CSV is missing `{col}`.\n\n"
                "Expected columns: date, sku, product_name, qty_sold, unit_price."
            )
            return None

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        friendly_error(
            "Some dates in your online orders CSV could not be understood.\n\n"
            "Make sure dates look like `2024-01-15`."
        )
        return None

    return df

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
    Long window + short window, blended with weight alpha toward short.
    """
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
        .agg(long_avg_daily="mean", long_std_daily="std")
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

    stats = long_stats.merge(short_stats, on="sku", how="left")
    stats["short_avg_daily"] = stats["short_avg_daily"].fillna(stats["long_avg_daily"])

    stats["avg_daily_demand"] = (
        alpha * stats["short_avg_daily"] + (1 - alpha) * stats["long_avg_daily"]
    )
    stats["std_daily_demand"] = stats["long_std_daily"].fillna(0.0)

    return stats[["sku", "avg_daily_demand", "std_daily_demand"]]

# ------------------------------------------------------------------------------
# Discount logic (Phase 2 – quantity discount levels)
# ------------------------------------------------------------------------------

def apply_discounts(merged: pd.DataFrame, discounts_df: pd.DataFrame | None):
    """
    Given merged SKU-level data with `recommended_order_qty` and base `unit_cost`,
    and an optional Discounts sheet, compute:

    - effective_unit_cost
    - discount_applied (True/False)
    - next_break_qty (if any)
    - savings_vs_base

    Discounts sheet format:
    sku, break_qty_1, price_1, break_qty_2, price_2, ...
    """
    merged["effective_unit_cost"] = merged["unit_cost"]
    merged["discount_applied"] = False
    merged["savings_vs_base"] = 0.0
    merged["next_break_qty"] = np.nan

    if discounts_df is None or discounts_df.empty:
        return merged

    # Ensure sku as string
    discounts_df = discounts_df.copy()
    discounts_df["sku"] = discounts_df["sku"].astype(str)

    for idx, row in merged.iterrows():
        sku = str(row["sku"])
        qty = row["recommended_order_qty"]
        base_cost = row["unit_cost"]

        if qty <= 0:
            continue

        disc_rows = discounts_df[discounts_df["sku"] == sku]
        if disc_rows.empty:
            continue

        d_row = disc_rows.iloc[0]

        # Collect all (break_qty, price) pairs
        breaks = []
        for i in range(1, 6):  # allow up to 5 discount tiers
            bq_col = f"break_qty_{i}"
            pr_col = f"price_{i}"
            if bq_col in d_row and pr_col in d_row:
                bq = d_row[bq_col]
                pr = d_row[pr_col]
                if pd.notna(bq) and pd.notna(pr):
                    breaks.append((float(bq), float(pr)))

        if not breaks:
            continue

        # Sort by break quantity
        breaks.sort(key=lambda x: x[0])

        # Determine applicable price tier
        applicable_price = base_cost
        applied = False
        for bq, price in breaks:
            if qty >= bq:
                applicable_price = price
                applied = True

        # Compute next break if exists
        next_break = np.nan
        for bq, price in breaks:
            if qty < bq:
                next_break = bq
                break

        merged.at[idx, "effective_unit_cost"] = applicable_price
        merged.at[idx, "discount_applied"] = applied

        if applied:
            base_total = qty * base_cost
            disc_total = qty * applicable_price
            merged.at[idx, "savings_vs_base"] = base_total - disc_total
        if not np.isnan(next_break):
            merged.at[idx, "next_break_qty"] = next_break

    return merged

# ------------------------------------------------------------------------------
# Reorder engine
# ------------------------------------------------------------------------------

def compute_reorder_recommendations(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    discounts_df: pd.DataFrame | None,
    lookback_days: int = 30,
    safety_z: float = 1.65,
    review_period_days: int = 7,
) -> pd.DataFrame | None:
    """Combine sales, inventory, products, and discounts into reorder recommendations."""
    if any(df is None for df in [sales_df, inventory_df, products_df]):
        return None

    # Ensure sku strings
    for df in (sales_df, inventory_df, products_df):
        df["sku"] = df["sku"].astype(str)

    stats = compute_demand_stats(sales_df, lookback_days=lookback_days)
    stats["sku"] = stats["sku"].astype(str)

    merged = (
        products_df.merge(inventory_df, on="sku", how="left")
        .merge(stats, on="sku", how="left")
    )

    # Optional last purchase info columns from Products sheet
    for col in ["last_purchase_qty", "last_purchase_cost", "last_purchase_date"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    d = merged["avg_daily_demand"]
    sigma = merged["std_daily_demand"]
    L = merged["lead_time_days"].fillna(5)

    # Reorder point & target stock
    merged["reorder_point"] = d * L + safety_z * sigma * np.sqrt(L)
    merged["target_stock"] = d * (L + review_period_days) + safety_z * sigma * np.sqrt(L)

    # If almost no demand, don't hold stock
    merged.loc[merged["avg_daily_demand"] < 0.1, ["reorder_point", "target_stock"]] = 0

    # Raw order
    merged["raw_order_qty"] = (merged["target_stock"] - merged["on_hand_qty"]).clip(lower=0)

    case_sizes = merged["case_size"].replace(0, np.nan).fillna(1)
    merged["order_cases"] = np.ceil(merged["raw_order_qty"] / case_sizes)
    merged["recommended_order_qty"] = merged["order_cases"] * case_sizes
    merged.loc[merged["recommended_order_qty"] < 1, "recommended_order_qty"] = 0

    # Unit cost
    merged["unit_cost"] = merged["cost"]

    # Apply discounts (if any)
    merged = apply_discounts(merged, discounts_df)

    # Extended cost using effective unit cost
    merged["extended_cost"] = merged["recommended_order_qty"] * merged["effective_unit_cost"]

    # Last selling price
    last_prices = (
        sales_df.sort_values("date")
        .groupby("sku")["unit_price"]
        .last()
        .reset_index()
        .rename(columns={"unit_price": "last_unit_price"})
    )
    merged = merged.merge(last_prices, on="sku", how="left")

    merged["estimated_margin_per_unit"] = merged["last_unit_price"] - merged["effective_unit_cost"]
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
# Sidebar – upload, template, settings
# ------------------------------------------------------------------------------

st.sidebar.header("Upload Data")

excel_file = st.sidebar.file_uploader(
    "Upload alcIQ Excel file (.xlsx)",
    type=["xlsx"],
    help="Use the template below for the required structure.",
)

online_orders_file = st.sidebar.file_uploader(
    "Optional online orders CSV (DoorDash, CityHive, web store)",
    type=["csv"],
    help="Columns: date, sku, product_name, qty_sold, unit_price",
)

with st.sidebar.expander("Download Excel template"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Instructions
        instructions = pd.DataFrame(
            [
                ["Sales", "date", "Date of sale (YYYY-MM-DD)."],
                ["Sales", "sku", "Product code / UPC used at the register."],
                ["Sales", "product_name", "Product name (e.g., 'Tito's 1.75L')."],
                ["Sales", "qty_sold", "Units sold that day."],
                ["Sales", "unit_price", "Retail price per unit (what customer pays)."],
                ["Inventory", "sku", "Product code (must match Sales and Products)."],
                ["Inventory", "on_hand_qty", "Units currently in stock at the store."],
                ["Products", "sku", "Product code (one row per unique item)."],
                ["Products", "brand", "Brand (e.g., Tito's, Jameson, Truly)."],
                ["Products", "product_name", "Full product name as sold."],
                ["Products", "category", "Vodka, Whiskey, RTD, Beer, Wine, etc."],
                ["Products", "size", "Size (1.75L, 750mL, 12pk 12oz, etc.)."],
                ["Products", "vendor", "Distributor / wholesaler you order from."],
                ["Products", "cost", "Your cost per unit from that vendor."],
                ["Products", "case_size", "Units per case (if ordering by case)."],
                ["Products", "lead_time_days", "Typical days from order to delivery."],
                ["Products", "last_purchase_qty", "Last quantity ordered (optional)."],
                ["Products", "last_purchase_cost", "Cost per unit on last order (optional)."],
                ["Products", "last_purchase_date", "Date of last order (optional)."],
                ["Discounts", "sku", "Product code this discount applies to."],
                ["Discounts", "break_qty_1", "First quantity break (e.g., 10 cases)."],
                ["Discounts", "price_1", "Unit cost at that break (e.g., 12.50)."],
                ["Discounts", "break_qty_2", "Second quantity break (optional)."],
                ["Discounts", "price_2", "Unit cost at second break (optional)."],
            ],
            columns=["Sheet", "Column", "Description"],
        )
        instructions.to_excel(writer, sheet_name="Instructions", index=False)

        # Empty structured sheets
        pd.DataFrame(
            columns=["date", "sku", "product_name", "qty_sold", "unit_price"]
        ).to_excel(writer, sheet_name="Sales", index=False)

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
                "last_purchase_qty",
                "last_purchase_cost",
                "last_purchase_date",
            ]
        ).to_excel(writer, sheet_name="Products", index=False)

        pd.DataFrame(
            columns=[
                "sku",
                "break_qty_1",
                "price_1",
                "break_qty_2",
                "price_2",
                "break_qty_3",
                "price_3",
            ]
        ).to_excel(writer, sheet_name="Discounts", index=False)

    st.sidebar.download_button(
        "Download alcIQ Excel template",
        data=buf.getvalue(),
        file_name="alcIQ_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.sidebar.markdown("---")
st.sidebar.header("Settings")

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
    "Days between vendor orders",
    min_value=3,
    max_value=28,
    value=st.session_state["review_period_days"],
)

st.session_state["lookback_days"] = lookback_days
st.session_state["safety_z"] = safety_z
st.session_state["review_period_days"] = review_period_days

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

sales_df, inventory_df, products_df, discounts_df = load_from_excel(excel_file)

if sales_df is None or inventory_df is None or products_df is None:
    st.warning("Upload the Excel file with Sales, Inventory, Products (and optional Discounts) to continue.")
    st.stop()

# Optional online orders integration (Phase 3 – simple connector)
online_orders_df = load_online_orders_csv(online_orders_file)
if online_orders_df is not None and not online_orders_df.empty:
    sales_df = pd.concat([sales_df, online_orders_df], ignore_index=True)

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
    if discounts_df is not None:
        st.write(f"Discount rows: **{len(discounts_df):,}**")

# ------------------------------------------------------------------------------
# Compute recommendations
# ------------------------------------------------------------------------------

recs = compute_reorder_recommendations(
    sales_df,
    inventory_df,
    products_df,
    discounts_df,
    lookback_days=lookback_days,
    safety_z=safety_z,
    review_period_days=review_period_days,
)

if recs is None or recs.empty:
    st.warning("Could not generate any recommendations. Please check that your data has some recent sales.")
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
        "🧾 Clean Order / POs",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1 – Recommended Order
# ------------------------------------------------------------------------------

with tab_order:
    st.subheader("📦 Recommended Order")

    st.markdown(
        """
This table shows **what alcIQ suggests you order**, based on:

- How fast each item sells (recent days weighted more)
- How much you have left on the shelf
- How long it takes to get a delivery
- How often you place orders with that vendor
- Any **discount tiers** you’ve defined
"""
    )

    vendors = ["(All vendors)"] + sorted(recs["vendor"].dropna().unique().tolist())
    selected_vendor = st.selectbox("Filter by vendor", vendors)

    df = recs.copy()
    if selected_vendor != "(All vendors)":
        df = df[df["vendor"] == selected_vendor]

    df = df[df["recommended_order_qty"] > 0]

    total_cost = df["extended_cost"].sum()
    total_profit = df["estimated_profit_on_order"].sum()
    items_to_order = len(df)
    total_savings = df["savings_vs_base"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost of this order", f"${total_cost:,.2f}")
    c2.metric("Est. profit from this order", f"${total_profit:,.2f}")
    c3.metric("Products on this order", int(items_to_order))
    c4.metric("Savings from discounts (vs base cost)", f"${total_savings:,.2f}")

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
        "effective_unit_cost",
        "savings_vs_base",
        "extended_cost",
        "last_unit_price",
        "estimated_margin_per_unit",
        "estimated_profit_on_order",
        "stockout_risk",
        "last_purchase_qty",
        "last_purchase_cost",
        "last_purchase_date",
        "next_break_qty",
        "discount_applied",
    ]

    display_df = df[display_cols].rename(
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
            "order_cases": "Cases to order",
            "unit_cost": "Base cost per unit ($)",
            "effective_unit_cost": "Effective cost per unit ($)",
            "savings_vs_base": "Savings vs base cost ($)",
            "extended_cost": "Total line cost ($)",
            "last_unit_price": "Recent selling price ($)",
            "estimated_margin_per_unit": "Margin per unit ($)",
            "estimated_profit_on_order": "Est. profit on this item ($)",
            "stockout_risk": "Stockout risk",
            "last_purchase_qty": "Last purchase qty",
            "last_purchase_cost": "Last purchase cost per unit ($)",
            "last_purchase_date": "Last purchase date",
            "next_break_qty": "Next discount break qty",
            "discount_applied": "Discount applied?",
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
                "Cases to order": "{:.0f}",
                "Base cost per unit ($)": "${:.2f}",
                "Effective cost per unit ($)": "${:.2f}",
                "Savings vs base cost ($)": "${:.2f}",
                "Total line cost ($)": "${:.2f}",
                "Recent selling price ($)": "${:.2f}",
                "Margin per unit ($)": "${:.2f}",
                "Est. profit on this item ($)": "${:.2f}",
                "Last purchase qty": "{:.0f}",
                "Last purchase cost per unit ($)": "${:.2f}",
                "Next discount break qty": "{:.0f}",
            }
        ),
        use_container_width=True,
        height=550,
    )

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
                ].rename(
                    columns={
                        "sku": "SKU",
                        "brand": "Brand",
                        "product_name": "Product",
                        "on_hand_qty": "On hand (units)",
                        "avg_daily_demand": "Avg daily sales (units)",
                        "reorder_point": "Reorder point (units)",
                    }
                ),
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
                ].rename(
                    columns={
                        "sku": "SKU",
                        "brand": "Brand",
                        "product_name": "Product",
                        "on_hand_qty": "On hand (units)",
                        "avg_daily_demand": "Avg daily sales (units)",
                    }
                ),
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
            total_savings=("savings_vs_base", "sum"),
            catalog_skus=("sku", "nunique"),
            ordering_skus=("recommended_order_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )

    summary_display = summary.rename(
        columns={
            "vendor": "Vendor",
            "total_cost": "Total order cost ($)",
            "total_profit": "Est. profit on this order ($)",
            "total_savings": "Savings from discounts ($)",
            "catalog_skus": "Products in catalog",
            "ordering_skus": "Products on this order",
        }
    )

    st.dataframe(
        summary_display.style.format(
            {
                "Total order cost ($)": "${:,.2f}",
                "Est. profit on this order ($)": "${:,.2f}",
                "Savings from discounts ($)": "${:,.2f}",
                "Products in catalog": "{:.0f}",
                "Products on this order": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# TAB 5 – Clean Order / POs (with PDF)
# ------------------------------------------------------------------------------

def generate_po_pdf(vendor_name: str, order_df: pd.DataFrame) -> bytes:
    """
    Generate a simple Purchase Order PDF in memory and return bytes.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Purchase Order – {vendor_name}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated by alcIQ on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 30

    # Table header
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "SKU")
    c.drawString(margin + 80, y, "Product")
    c.drawString(margin + 250, y, "Size")
    c.drawRightString(margin + 350, y, "Cases")
    c.drawRightString(margin + 420, y, "Units")
    c.drawRightString(margin + 500, y, "Unit Cost")
    c.drawRightString(margin + 570, y, "Line Total")
    y -= 15
    c.line(margin, y, width - margin, y)
    y -= 10

    c.setFont("Helvetica", 9)
    total = 0.0

    for _, row in order_df.iterrows():
        if y < 80:
            c.showPage()
            y = height - margin
        sku = str(row["SKU"])
        product = str(row["Product"])
        size = str(row["Size"])
        cases = int(row["Cases to order"])
        units = int(row["Units to order"])
        unit_cost = float(row["Cost per unit ($)"])
        line_total = float(row["Line total ($)"])
        total += line_total

        c.drawString(margin, y, sku[:10])
        c.drawString(margin + 80, y, product[:28])
        c.drawString(margin + 250, y, size[:10])
        c.drawRightString(margin + 350, y, str(cases))
        c.drawRightString(margin + 420, y, str(units))
        c.drawRightString(margin + 500, y, f"{unit_cost:,.2f}")
        c.drawRightString(margin + 570, y, f"{line_total:,.2f}")
        y -= 14

    # Total
    y -= 10
    c.line(margin, y, width - margin, y)
    y -= 20
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(margin + 570, y, f"Total: ${total:,.2f}")

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

with tab_report:
    st.subheader("🧾 Clean Order Report & Purchase Orders")

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
                "effective_unit_cost": "Cost per unit ($)",
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

        # Excel export
        xls_buf = BytesIO()
        with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
            clean.to_excel(writer, index=False, sheet_name="Order")

        st.download_button(
            "Download Clean Order Excel (.xlsx)",
            data=xls_buf.getvalue(),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # PDF PO export (per vendor only)
        if sel_vendor != "(All vendors combined)":
            po_df = clean[clean["Vendor"] == sel_vendor].copy()
            po_pdf_bytes = generate_po_pdf(sel_vendor, po_df)
            st.download_button(
                f"Download Purchase Order PDF for {sel_vendor}",
                data=po_pdf_bytes,
                file_name=f"PO_{sel_vendor.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

# ------------------------------------------------------------------------------
# Footer – Privacy
# ------------------------------------------------------------------------------

st.divider()
st.markdown(
    """
### 🔐 Privacy & Data

Your data is **100% private**.

alcIQ does **not** store, save, log, transmit, share, or sell any of your data — ever.

Your Excel file (and any online orders CSV) are processed only while this app is open.
Once you close or refresh, the data is gone.

We cannot see your sales numbers.  
We cannot see your inventory.  
We cannot see your product file.  

Your information belongs entirely to **you** — alcIQ simply analyzes it to generate smarter order recommendations.
"""
)
st.caption("alcIQ – Inventory intelligence for modern liquor & beverage retailers.")









