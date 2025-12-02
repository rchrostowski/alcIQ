import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AlcIQ – Inventory Intelligence for Beverage Retailers",
    layout="wide",
)

# ---- Access code gate ----
ACCESS_CODE = "LCQ-TEST"  # <- give this to Pavlish

# ---- Data file path (auto-generated if missing) ----
DATA_PATH = Path("data/alcIQ_sample.csv")

# ---- Column mappings (adjust to your real data later) ----
DATE_COL = "date"                  # transaction date
SKU_COL = "sku"                    # unique SKU code
NAME_COL = "product_name"          # human-readable name
CAT_COL = "category"               # e.g., Beer, Seltzer, Wine
SUPPLIER_COL = "supplier"          # supplier/distributor
UNITS_COL = "units_sold"           # units sold on that date
REV_COL = "revenue"                # sales revenue on that date
INV_COL = "inventory_on_hand"      # units on hand end-of-day
COST_COL = "unit_cost"             # unit cost
LEADTIME_COL = "lead_time_days"    # lead time in days

# ---- Global defaults (can be overridden in Settings page) ----
DEFAULT_HISTORY_DAYS = 90
DEFAULT_FORECAST_DAYS = 14
DEFAULT_SAFETY_FACTOR = 0.5
DEFAULT_TARGET_SERVICE_DAYS = 21   # how many days of stock we aim to cover
DEFAULT_MIN_SLOW_DAILY_UNITS = 0.02
DEFAULT_FAST_TOP_N = 15
DEFAULT_SLOW_TOP_N = 15

# Initialize Streamlit session state for settings
if "history_days" not in st.session_state:
    st.session_state["history_days"] = DEFAULT_HISTORY_DAYS
if "forecast_days" not in st.session_state:
    st.session_state["forecast_days"] = DEFAULT_FORECAST_DAYS
if "safety_factor" not in st.session_state:
    st.session_state["safety_factor"] = DEFAULT_SAFETY_FACTOR
if "target_service_days" not in st.session_state:
    st.session_state["target_service_days"] = DEFAULT_TARGET_SERVICE_DAYS
if "min_slow_daily_units" not in st.session_state:
    st.session_state["min_slow_daily_units"] = DEFAULT_MIN_SLOW_DAILY_UNITS
if "fast_top_n" not in st.session_state:
    st.session_state["fast_top_n"] = DEFAULT_FAST_TOP_N
if "slow_top_n" not in st.session_state:
    st.session_state["slow_top_n"] = DEFAULT_SLOW_TOP_N


# ============================================================
# SAMPLE DATA GENERATOR (PAVLISH-STYLE DRIVE-THRU)
# ============================================================

def generate_sample_dataset(
    path: Path,
    n_skus: int = 250,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
) -> None:
    """
    Generate a realistic Pavlish-style beverage drive-thru dataset:
    - Beer-heavy mix of SKUs
    - Seasonal seltzers + wine/spirit spikes
    - Multiple suppliers
    - Daily sales, revenue, inventory, cost, lead times
    """
    rng = np.random.default_rng(42)
    random.seed(42)

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # Category mix
    categories = [
        "Domestic Beer",
        "Import & Craft Beer",
        "Hard Seltzer",
        "Cider",
        "Spirits",
        "Wine",
        "NA Beer",
        "Energy Drinks",
        "Mixers",
    ]
    cat_weights = np.array([0.35, 0.22, 0.18, 0.05, 0.08, 0.06, 0.02, 0.02, 0.02])
    cat_weights = cat_weights / cat_weights.sum()

    suppliers_by_cat = {
        "Domestic Beer": ["MillerCo", "ABInBev", "Coors", "Pavlish Local"],
        "Import & Craft Beer": ["Heineken", "LocalCraft", "GuinnessCo"],
        "Hard Seltzer": ["WhiteClaw", "Truly", "VizzyCo"],
        "Cider": ["AngryOrchard", "LocalCider"],
        "Spirits": ["BeamSuntory", "Diageo", "LocalDistilling"],
        "Wine": ["WineHouse", "NapaValleyCo", "EuroVine"],
        "NA Beer": ["Heineken0", "AthleticBrewing"],
        "Energy Drinks": ["RedBull", "Monster", "Celsius"],
        "Mixers": ["CocaCola", "PepsiCo", "GenericMixers"],
    }

    sku_rows = []
    for i in range(n_skus):
        sku_id = f"SKU-{i+1:04d}"
        cat = rng.choice(categories, p=cat_weights)
        supplier = random.choice(suppliers_by_cat[cat])

        # Base daily demand by category (cases or units per day)
        base_demand_map = {
            "Domestic Beer": (2.0, 6.0),
            "Import & Craft Beer": (1.0, 3.5),
            "Hard Seltzer": (0.8, 4.5),
            "Cider": (0.3, 1.0),
            "Spirits": (0.2, 0.8),
            "Wine": (0.2, 0.9),
            "NA Beer": (0.1, 0.5),
            "Energy Drinks": (0.5, 2.0),
            "Mixers": (0.5, 1.5),
        }
        lam_low, lam_high = base_demand_map.get(cat, (0.2, 1.0))
        base_lambda = rng.uniform(lam_low, lam_high)

        # Cost & price
        cost_map = {
            "Domestic Beer": (14, 24),
            "Import & Craft Beer": (16, 28),
            "Hard Seltzer": (15, 26),
            "Cider": (12, 20),
            "Spirits": (12, 40),
            "Wine": (9, 25),
            "NA Beer": (9, 16),
            "Energy Drinks": (1.0, 1.4),
            "Mixers": (0.7, 1.2),
        }
        c_low, c_high = cost_map.get(cat, (8, 20))
        unit_cost = rng.uniform(c_low, c_high)

        margin_pct = rng.uniform(0.18, 0.32)
        unit_price = unit_cost * (1 + margin_pct)

        # Lead time in days
        lead_time_map = {
            "Domestic Beer": (3, 7),
            "Import & Craft Beer": (4, 10),
            "Hard Seltzer": (4, 10),
            "Cider": (7, 14),
            "Spirits": (7, 21),
            "Wine": (7, 21),
            "NA Beer": (5, 10),
            "Energy Drinks": (3, 7),
            "Mixers": (3, 7),
        }
        lt_low, lt_high = lead_time_map.get(cat, (5, 10))
        lead_time_days = int(rng.integers(lt_low, lt_high + 1))

        sku_rows.append(
            {
                "sku": sku_id,
                "category": cat,
                "supplier": supplier,
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "lead_time_days": lead_time_days,
            }
        )

    sku_df = pd.DataFrame(sku_rows)

    # Product names
    name_by_cat = {
        "Domestic Beer": ["Light Lager 30pk", "Pilsner 24pk", "American Lager 12pk"],
        "Import & Craft Beer": ["IPA 6pk", "Stout 4pk", "Belgian Ale 6pk"],
        "Hard Seltzer": ["Lemon Lime 12pk", "Variety Pack 12pk", "Peach 12pk"],
        "Cider": ["Crisp Apple 6pk", "Dry Cider 6pk"],
        "Spirits": ["Vodka 750ml", "Whiskey 750ml", "Rum 1L", "Tequila 750ml"],
        "Wine": ["Cabernet 750ml", "Pinot Grigio 750ml", "Rose 750ml"],
        "NA Beer": ["NA Lager 12pk", "IPA 6pk Non-Alc"],
        "Energy Drinks": ["Energy 16oz", "Sugar-free Energy 16oz"],
        "Mixers": ["Tonic 1L", "Cola 2L", "Club Soda 1L"],
    }

    def make_name(cat: str) -> str:
        base = random.choice(name_by_cat.get(cat, ["Beverage"]))
        return f"{base}"

    sku_df["product_name"] = sku_df["category"].apply(make_name)

    # Build daily rows with seasonality + weekend effects + simple reordering
    rows = []
    for _, sku_row in sku_df.iterrows():
        sku_id = sku_row["sku"]
        cat = sku_row["category"]
        supplier = sku_row["supplier"]
        unit_cost = sku_row["unit_cost"]
        unit_price = sku_row["unit_price"]
        lead_time = sku_row["lead_time_days"]
        product_name = sku_row["product_name"]

        # Draw base lambda again to vary by SKU
        base_lambda = rng.uniform(0.5, 4.0)
        if cat == "Domestic Beer":
            base_lambda *= 1.5
        elif cat == "Import & Craft Beer":
            base_lambda *= 1.2
        elif cat == "Hard Seltzer":
            base_lambda *= 1.1

        inventory = int(rng.integers(30, 150))
        reorder_point = int(inventory * 0.4)
        reorder_qty = int(inventory * 0.8)

        for d in dates:
            month = d.month
            dow = d.weekday()  # 0=Mon, 6=Sun

            season_factor = 1.0

            # Summer: seltzers/beer spike
            if month in [5, 6, 7, 8]:
                if cat in ["Hard Seltzer", "Domestic Beer", "Import & Craft Beer"]:
                    season_factor *= 1.7
                elif cat in ["Cider", "Wine", "Spirits"]:
                    season_factor *= 0.9

            # Late fall / winter: spirits & wine spike
            if month in [11, 12, 1]:
                if cat in ["Wine", "Spirits"]:
                    season_factor *= 1.6
                elif cat in ["Hard Seltzer"]:
                    season_factor *= 0.6

            # Weekend bump
            if dow in [4, 5]:  # Fri, Sat
                season_factor *= 1.8
            elif dow == 3:  # Thu
                season_factor *= 1.2
            elif dow == 6:  # Sun
                season_factor *= 1.1

            lam = max(base_lambda * season_factor, 0.01)
            units_sold = int(rng.poisson(lam))

            # make sure we don't sell more than we have
            units_sold = min(units_sold, inventory)
            inventory -= units_sold

            # if low inventory, reorder
            if inventory < reorder_point:
                inventory += reorder_qty + rng.integers(10, 40)

            revenue = units_sold * unit_price

            rows.append(
                {
                    DATE_COL: d,
                    SKU_COL: sku_id,
                    NAME_COL: product_name,
                    CAT_COL: cat,
                    SUPPLIER_COL: supplier,
                    UNITS_COL: units_sold,
                    REV_COL: revenue,
                    INV_COL: inventory,
                    COST_COL: unit_cost,
                    LEADTIME_COL: lead_time,
                }
            )

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ============================================================
# DATA LOADING & BASIC UTILITIES
# ============================================================

@st.cache_data(show_spinner="Loading sales & inventory data…")
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")

    df = pd.read_csv(path)

    required_cols = [
        DATE_COL,
        SKU_COL,
        NAME_COL,
        CAT_COL,
        SUPPLIER_COL,
        UNITS_COL,
        REV_COL,
        INV_COL,
        COST_COL,
        LEADTIME_COL,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    numeric_cols = [UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def get_file_last_updated(path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"


def get_date_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    return df[DATE_COL].min(), df[DATE_COL].max()


def add_sidebar_settings():
    st.sidebar.markdown("## Global Settings")

    st.session_state["history_days"] = st.sidebar.slider(
        "History window (days)",
        min_value=30,
        max_value=365,
        value=st.session_state["history_days"],
        step=15,
    )
    st.session_state["forecast_days"] = st.sidebar.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=60,
        value=st.session_state["forecast_days"],
        step=7,
    )
    st.session_state["safety_factor"] = st.sidebar.slider(
        "Safety stock factor (0.0–1.5)",
        min_value=0.0,
        max_value=1.5,
        value=float(st.session_state["safety_factor"]),
        step=0.1,
    )
    st.session_state["target_service_days"] = st.sidebar.slider(
        "Target service coverage (days)",
        min_value=7,
        max_value=60,
        value=st.session_state["target_service_days"],
        step=7,
    )
    st.session_state["min_slow_daily_units"] = st.sidebar.number_input(
        "Min daily units for slow-mover ranking",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["min_slow_daily_units"]),
        step=0.01,
    )
    st.session_state["fast_top_n"] = st.sidebar.number_input(
        "Top N fast movers",
        min_value=5,
        max_value=50,
        value=int(st.session_state["fast_top_n"]),
        step=5,
    )
    st.session_state["slow_top_n"] = st.sidebar.number_input(
        "Top N slow movers",
        min_value=5,
        max_value=50,
        value=int(st.session_state["slow_top_n"]),
        step=5,
    )


# ============================================================
# CORE METRICS & ANALYTICS
# ============================================================

def compute_inventory_metrics(
    df: pd.DataFrame,
    history_days: int,
    forecast_days: int,
    safety_factor: float,
    target_service_days: int,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    max_date = df[DATE_COL].max()
    history_start = max_date - timedelta(days=history_days)

    recent = df[df[DATE_COL] >= history_start].copy()

    daily_units = recent.groupby(SKU_COL)[UNITS_COL].sum() / max(history_days, 1)

    current_inventory = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)[INV_COL]
        .last()
        .fillna(0)
    )

    base_info = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)
        .agg(
            {
                NAME_COL: "last",
                CAT_COL: "last",
                SUPPLIER_COL: "last",
                COST_COL: "last",
                LEADTIME_COL: "last",
            }
        )
    )

    metrics = base_info.copy()
    metrics["avg_daily_units"] = daily_units.fillna(0)
    metrics["current_inventory"] = current_inventory.fillna(0)
    metrics["forecast_demand"] = metrics["avg_daily_units"] * forecast_days

    metrics[LEADTIME_COL] = metrics[LEADTIME_COL].replace(0, np.nan).fillna(7)
    metrics["safety_stock"] = (
        metrics["avg_daily_units"] * metrics[LEADTIME_COL] * safety_factor
    )

    metrics["weeks_on_hand"] = np.where(
        metrics["avg_daily_units"] > 0,
        metrics["current_inventory"] / (metrics["avg_daily_units"] * 7),
        np.inf,
    )

    metrics["inventory_value"] = metrics["current_inventory"] * metrics[COST_COL]
    metrics["dead_inventory_value"] = np.where(
        metrics["avg_daily_units"] < 0.01, metrics["inventory_value"], 0
    )

    metrics["projected_balance"] = (
        metrics["current_inventory"]
        - metrics["forecast_demand"]
        + metrics["safety_stock"]
    )

    status_list = []
    for _, row in metrics.iterrows():
        inv = row["current_inventory"]
        demand = row["forecast_demand"]
        pbalance = row["projected_balance"]

        if inv <= 0 or pbalance < 0:
            status_list.append("🔥 Stockout Risk")
        elif inv < demand:
            status_list.append("🟡 Low Inventory")
        elif inv > demand * 2:
            status_list.append("🔵 Overstock")
        else:
            status_list.append("✅ Healthy")

    metrics["status"] = status_list

    target_days = forecast_days + metrics[LEADTIME_COL] + target_service_days
    metrics["target_inventory"] = (
        metrics["avg_daily_units"] * target_days + metrics["safety_stock"]
    )

    metrics["recommended_order_qty"] = np.where(
        metrics["status"].isin(["🔥 Stockout Risk", "🟡 Low Inventory"]),
        np.maximum(metrics["target_inventory"] - metrics["current_inventory"], 0),
        0,
    ).round().astype(int)

    recent_rev = recent.groupby(SKU_COL)[REV_COL].sum()
    total_rev = recent_rev.sum()
    share = recent_rev / total_rev if total_rev > 0 else recent_rev * 0
    sorted_share = share.sort_values(ascending=False)
    cum_share = sorted_share.cumsum()

    abc_map: Dict[str, str] = {}
    for sku, cs in cum_share.items():
        if cs <= 0.8:
            abc_map[sku] = "A"
        elif cs <= 0.95:
            abc_map[sku] = "B"
        else:
            abc_map[sku] = "C"

    metrics["abc_class"] = metrics.index.map(lambda x: abc_map.get(x, "C"))
    metrics.reset_index(inplace=True)
    return metrics


def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    temp = df.copy()
    temp["month"] = temp[DATE_COL].dt.month

    monthly = temp.groupby([SKU_COL, "month"])[UNITS_COL].sum().reset_index()
    summary = monthly.groupby(SKU_COL)[UNITS_COL].agg(["mean", "std"]).reset_index()
    summary["seasonality_score"] = np.where(
        summary["mean"] > 0,
        summary["std"] / summary["mean"],
        0,
    )
    return summary[[SKU_COL, "seasonality_score"]]


def compute_category_summary(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if df.empty or metrics.empty:
        return pd.DataFrame()

    recent = df.copy()
    max_date = recent[DATE_COL].max()
    hist_start = max_date - timedelta(days=st.session_state["history_days"])
    recent = recent[recent[DATE_COL] >= hist_start]

    cat_rev = recent.groupby(CAT_COL)[REV_COL].sum()
    cat_units = recent.groupby(CAT_COL)[UNITS_COL].sum()
    cat_inv_value = metrics.groupby(CAT_COL)["inventory_value"].sum()
    cat_dead_value = metrics.groupby(CAT_COL)["dead_inventory_value"].sum()

    total_rev = cat_rev.sum()
    total_units = cat_units.sum()
    total_inv_value = cat_inv_value.sum()

    cat_df = pd.DataFrame(
        {
            "revenue": cat_rev,
            "units": cat_units,
            "inventory_value": cat_inv_value,
            "dead_inventory_value": cat_dead_value,
        }
    ).reset_index()

    cat_df["revenue_share"] = np.where(
        total_rev > 0, cat_df["revenue"] / total_rev, 0
    )
    cat_df["units_share"] = np.where(
        total_units > 0, cat_df["units"] / total_units, 0
    )
    cat_df["inventory_share"] = np.where(
        total_inv_value > 0, cat_df["inventory_value"] / total_inv_value, 0
    )

    return cat_df.sort_values("revenue", ascending=False)


def get_slow_fast_movers(
    metrics: pd.DataFrame,
    slow_n: int,
    fast_n: int,
    min_slow_daily_units: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    slow = (
        metrics[metrics["avg_daily_units"] >= min_slow_daily_units]
        .sort_values("avg_daily_units", ascending=True)
        .head(slow_n)
    )
    fast = (
        metrics.sort_values("avg_daily_units", ascending=False)
        .head(fast_n)
    )
    return slow, fast


def build_purchase_order(metrics: pd.DataFrame) -> pd.DataFrame:
    po = metrics[
        (metrics["recommended_order_qty"] > 0)
        & (metrics["status"].isin(["🔥 Stockout Risk", "🟡 Low Inventory"]))
    ].copy()
    if po.empty:
        return po

    po["estimated_cost"] = po["recommended_order_qty"] * po[COST_COL]
    po = po[
        [
            SKU_COL,
            NAME_COL,
            CAT_COL,
            SUPPLIER_COL,
            "status",
            "abc_class",
            "avg_daily_units",
            "current_inventory",
            "forecast_demand",
            "recommended_order_qty",
            COST_COL,
            "estimated_cost",
            LEADTIME_COL,
        ]
    ]
    return po.sort_values([SUPPLIER_COL, "status", "estimated_cost"], ascending=[True, True, False])


def simulate_discount(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_skus: list,
    discount_pct: float,
    price_elasticity: float = 1.3,
) -> Tuple[pd.DataFrame, float, float]:
    if not selected_skus:
        return pd.DataFrame(), 0.0, 0.0

    df_sim = df[df[SKU_COL].isin(selected_skus)].copy()
    if df_sim.empty:
        return pd.DataFrame(), 0.0, 0.0

    df_sim["price"] = np.where(
        df_sim[UNITS_COL] > 0,
        df_sim[REV_COL] / df_sim[UNITS_COL],
        0,
    )

    base_revenue = df_sim[REV_COL].sum()

    df_sim["new_price"] = df_sim["price"] * (1 - discount_pct)
    df_sim["new_units"] = df_sim[UNITS_COL] * (1 + price_elasticity * discount_pct)
    df_sim["new_revenue"] = df_sim["new_units"] * df_sim["new_price"]

    sku_cost = metrics.set_index(SKU_COL)[COST_COL].to_dict()
    df_sim["unit_cost"] = df_sim[SKU_COL].map(sku_cost).fillna(0)
    df_sim["base_margin"] = (df_sim["price"] - df_sim["unit_cost"]) * df_sim[UNITS_COL]
    df_sim["new_margin"] = (df_sim["new_price"] - df_sim["unit_cost"]) * df_sim["new_units"]

    total_new_revenue = df_sim["new_revenue"].sum()
    delta_revenue = total_new_revenue - base_revenue

    base_margin_total = df_sim["base_margin"].sum()
    new_margin_total = df_sim["new_margin"].sum()
    delta_margin = new_margin_total - base_margin_total

    result = (
        df_sim.groupby(SKU_COL)
        .agg(
            base_revenue=(REV_COL, "sum"),
            new_revenue=("new_revenue", "sum"),
            base_units=(UNITS_COL, "sum"),
            new_units=("new_units", "sum"),
            base_margin=("base_margin", "sum"),
            new_margin=("new_margin", "sum"),
        )
        .reset_index()
    )

    return result, float(delta_revenue), float(delta_margin)


# ============================================================
# KPI + UI HELPERS
# ============================================================

def kpi_header(df: pd.DataFrame, metrics: pd.DataFrame):
    total_rev = df[REV_COL].sum()
    total_units = df[UNITS_COL].sum()
    unique_skus = df[SKU_COL].nunique()
    total_inv_value = metrics["inventory_value"].sum()
    dead_value = metrics["dead_inventory_value"].sum()
    stockout_count = (metrics["status"] == "🔥 Stockout Risk").sum()
    overstock_count = (metrics["status"] == "🔵 Overstock").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Units Sold", f"{total_units:,.0f}")
    c3.metric("Active SKUs", f"{unique_skus:,}")
    c4.metric("On-Hand Inventory Value", f"${total_inv_value:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Dead Inventory Value", f"${dead_value:,.0f}")
    c6.metric("🔥 Stockout Risks", f"{stockout_count}")
    c7.metric("🔵 Overstock SKUs", f"{overstock_count}")


# ============================================================
# PAGE FUNCTIONS
# ============================================================

def page_overview(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Overview")
    kpi_header(df, metrics)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Revenue Trend by Month & Category")
        df_month = (
            df.set_index(DATE_COL)
            .groupby([pd.Grouper(freq="M"), CAT_COL])[REV_COL]
            .sum()
            .reset_index()
        )
        if df_month.empty:
            st.info("No revenue data available.")
        else:
            df_month["month"] = df_month[DATE_COL].dt.to_period("M").astype(str)
            pivot = df_month.pivot(index="month", columns=CAT_COL, values=REV_COL).fillna(0)
            st.area_chart(pivot)

    with col2:
        st.markdown("#### Inventory Status Breakdown")
        status_counts = metrics["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        if status_counts.empty:
            st.info("No inventory metrics available.")
        else:
            st.bar_chart(status_counts.set_index("status"))

        st.markdown("#### ABC Revenue Classes")
        abc_counts = metrics["abc_class"].value_counts().reindex(["A", "B", "C"]).fillna(0)
        st.bar_chart(abc_counts)


def page_inventory_forecast(metrics: pd.DataFrame):
    st.subheader("Inventory Forecast & Risk")

    status_filter = st.multiselect(
        "Filter by Status",
        options=["🔥 Stockout Risk", "🟡 Low Inventory", "🔵 Overstock", "✅ Healthy"],
        default=["🔥 Stockout Risk", "🟡 Low Inventory"],
    )
    cat_filter = st.multiselect(
        "Filter by Category",
        options=sorted(metrics[CAT_COL].dropna().unique()),
        default=None,
    )
    abc_filter = st.multiselect(
        "Filter by ABC Class",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
    )

    df_view = metrics.copy()
    if status_filter:
        df_view = df_view[df_view["status"].isin(status_filter)]
    if cat_filter:
        df_view = df_view[df_view[CAT_COL].isin(cat_filter)]
    if abc_filter:
        df_view = df_view[df_view["abc_class"].isin(abc_filter)]

    if df_view.empty:
        st.success("No SKUs match these filters – that might actually be good news 😄")
        return

    st.dataframe(
        df_view[
            [
                SKU_COL,
                NAME_COL,
                CAT_COL,
                SUPPLIER_COL,
                "status",
                "abc_class",
                "avg_daily_units",
                "current_inventory",
                "forecast_demand",
                "weeks_on_hand",
                "recommended_order_qty",
                "inventory_value",
            ]
        ].sort_values(["status", "weeks_on_hand"]),
        use_container_width=True,
    )


def page_slow_fast(metrics: pd.DataFrame):
    st.subheader("Slow & Fast Movers")
    slow_n = st.session_state["slow_top_n"]
    fast_n = st.session_state["fast_top_n"]
    min_slow_daily = st.session_state["min_slow_daily_units"]

    slow, fast = get_slow_fast_movers(metrics, slow_n, fast_n, min_slow_daily)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🐢 Slow Movers")
        if slow.empty:
            st.info("No slow movers detected for the current settings. Great job staying lean.")
        else:
            st.dataframe(
                slow[
                    [
                        SKU_COL,
                        NAME_COL,
                        CAT_COL,
                        "abc_class",
                        "avg_daily_units",
                        "current_inventory",
                        "weeks_on_hand",
                        "inventory_value",
                    ]
                ],
                use_container_width=True,
            )

    with col2:
        st.markdown("#### ⚡ Fast Movers")
        if fast.empty:
            st.info("No sales data yet to identify fast movers.")
        else:
            st.dataframe(
                fast[
                    [
                        SKU_COL,
                        NAME_COL,
                        CAT_COL,
                        "abc_class",
                        "avg_daily_units",
                        "current_inventory",
                        "weeks_on_hand",
                        "inventory_value",
                    ]
                ],
                use_container_width=True,
            )


def page_purchase_orders(metrics: pd.DataFrame):
    st.subheader("Purchase Order Builder")
    po = build_purchase_order(metrics)

    if po.empty:
        st.success("No stockout or low-inventory SKUs requiring purchase orders right now.")
        return

    st.markdown(
        "These SKUs are flagged as **🔥 Stockout Risk** or **🟡 Low Inventory**. "
        "Filter by supplier/category to export a ready-to-send PO."
    )

    suppliers = sorted(po[SUPPLIER_COL].dropna().unique())
    supplier_filter = st.multiselect(
        "Filter by Supplier",
        options=suppliers,
        default=suppliers,
    )
    cat_filter = st.multiselect(
        "Filter by Category",
        options=sorted(po[CAT_COL].dropna().unique()),
        default=None,
    )

    po_view = po.copy()
    if supplier_filter:
        po_view = po_view[po_view[SUPPLIER_COL].isin(supplier_filter)]
    if cat_filter:
        po_view = po_view[po_view[CAT_COL].isin(cat_filter)]

    if po_view.empty:
        st.warning("No items match the current filters.")
        return

    st.dataframe(po_view, use_container_width=True)
    total_cost = po_view["estimated_cost"].sum()
    st.markdown(f"**Total Estimated PO Cost:** ${total_cost:,.0f}")

    csv_bytes = po_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Purchase Order (CSV)",
        data=csv_bytes,
        file_name=f"AlcIQ_purchase_order_{datetime.today().date()}.csv",
        mime="text/csv",
    )


def page_sku_explorer(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("SKU Explorer")
    sku_options = sorted(metrics[SKU_COL].unique())
    selected_sku = st.selectbox("Select a SKU", options=sku_options)

    m = metrics[metrics[SKU_COL] == selected_sku].iloc[0]
    df_sku = df[df[SKU_COL] == selected_sku].sort_values(DATE_COL)

    st.markdown(f"### {m[NAME_COL]} ({m[SKU_COL]})")
    st.caption(f"Category: {m[CAT_COL]} • Supplier: {m[SUPPLIER_COL]} • ABC: {m['abc_class']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", m["status"])
    c2.metric("Avg Daily Units", f"{m['avg_daily_units']:.2f}")
    c3.metric("Weeks on Hand", "∞" if np.isinf(m["weeks_on_hand"]) else f"{m['weeks_on_hand']:.1f}")
    c4.metric("Current Inventory", f"{m['current_inventory']:,.0f}")

    st.markdown("#### Sales History (Units per Week)")
    if df_sku.empty:
        st.info("No sales history for this SKU.")
        return
    df_units = df_sku.set_index(DATE_COL)[UNITS_COL].resample("W").sum()
    st.line_chart(df_units)

    st.markdown("#### Revenue History (Per Week)")
    df_rev = df_sku.set_index(DATE_COL)[REV_COL].resample("W").sum()
    st.line_chart(df_rev)

    st.markdown("#### Raw Daily Records")
    st.dataframe(
        df_sku[[DATE_COL, UNITS_COL, REV_COL, INV_COL]],
        use_container_width=True,
    )


def page_category_supplier(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Category & Supplier Analytics")

    cat_summary = compute_category_summary(df, metrics)
    if cat_summary.empty:
        st.info("Not enough data to build category summary.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Category KPIs")
        st.dataframe(
            cat_summary[
                [
                    CAT_COL,
                    "revenue",
                    "revenue_share",
                    "units",
                    "units_share",
                    "inventory_value",
                    "inventory_share",
                    "dead_inventory_value",
                ]
            ],
            use_container_width=True,
        )
    with col2:
        st.markdown("#### Revenue Share by Category")
        st.bar_chart(cat_summary.set_index(CAT_COL)["revenue_share"])
        st.markdown("#### Inventory Value by Category")
        st.bar_chart(cat_summary.set_index(CAT_COL)["inventory_value"])

    st.markdown("---")
    st.markdown("#### Supplier Exposure")
    supplier_inv = (
        metrics.groupby(SUPPLIER_COL)["inventory_value"]
        .sum()
        .sort_values(ascending=False)
    )
    supplier_rev = (
        df.groupby(SUPPLIER_COL)[REV_COL]
        .sum()
        .sort_values(ascending=False)
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("Inventory Value by Supplier")
        st.bar_chart(supplier_inv)
    with col4:
        st.markdown("Revenue by Supplier")
        st.bar_chart(supplier_rev)


def page_seasonality(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Seasonality & Trends")
    seasonality_df = compute_seasonality(df)
    if seasonality_df.empty:
        st.info("Not enough data to compute seasonality.")
        return

    merged = pd.merge(
        metrics[[SKU_COL, NAME_COL, CAT_COL]],
        seasonality_df,
        on=SKU_COL,
        how="left",
    )

    st.markdown("#### Most Seasonal SKUs (Higher Score = Stronger Seasonality)")
    top_n = st.slider("How many to show", 10, 100, 25, step=5)
    top_seasonal = merged.sort_values("seasonality_score", ascending=False).head(top_n)
    st.dataframe(
        top_seasonal[[SKU_COL, NAME_COL, CAT_COL, "seasonality_score"]],
        use_container_width=True,
    )

    st.markdown("#### Seasonality Examples: Pick a SKU")
    sku_options = top_seasonal[SKU_COL].tolist()
    if not sku_options:
        st.info("No seasonal SKUs detected.")
        return

    selected_sku = st.selectbox("Select seasonal SKU", options=sku_options)
    df_sku = df[df[SKU_COL] == selected_sku].copy()
    if df_sku.empty:
        st.info("No sales data for this SKU.")
        return

    df_sku["month"] = df_sku[DATE_COL].dt.month
    month_summary = df_sku.groupby("month")[UNITS_COL].sum()
    st.line_chart(month_summary)


def page_discount_simulator(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Margin & Discount Simulator")
    st.markdown(
        "Pick one or more SKUs, choose a discount, and see the impact on revenue and margin "
        "using a simple price-elasticity model."
    )

    sku_options = sorted(metrics[SKU_COL].unique())
    selected_skus = st.multiselect("Select SKUs", options=sku_options)

    discount_pct = st.slider(
        "Discount (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    ) / 100.0
    elasticity = st.slider(
        "Price Elasticity (how sensitive demand is to price)",
        min_value=0.0,
        max_value=3.0,
        value=1.3,
        step=0.1,
    )

    if st.button("Run Simulation"):
        result, delta_revenue, delta_margin = simulate_discount(
            df,
            metrics,
            selected_skus,
            discount_pct=discount_pct,
            price_elasticity=elasticity,
        )

        if result.empty:
            st.warning("No data available for the selected SKUs.")
            return

        st.markdown("#### Per-SKU Impact")
        st.dataframe(result, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Δ Revenue (Total)",
                f"${delta_revenue:,.0f}",
                delta=f"${delta_revenue:,.0f}",
            )
        with col2:
            st.metric(
                "Δ Margin (Total)",
                f"${delta_margin:,.0f}",
                delta=f"${delta_margin:,.0f}",
            )


def page_settings(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Model Settings & Assumptions")
    st.markdown("These settings control how AlcIQ classifies risk and generates purchase orders.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Demand & Risk Settings")
        st.session_state["history_days"] = st.number_input(
            "History window (days)",
            min_value=30,
            max_value=365,
            value=int(st.session_state["history_days"]),
            step=15,
        )
        st.session_state["forecast_days"] = st.number_input(
            "Forecast horizon (days)",
            min_value=7,
            max_value=60,
            value=int(st.session_state["forecast_days"]),
            step=7,
        )
        st.session_state["safety_factor"] = st.slider(
            "Safety stock factor",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state["safety_factor"]),
            step=0.1,
        )

    with col2:
        st.markdown("### Service & Slow/Fast Definitions")
        st.session_state["target_service_days"] = st.number_input(
            "Target service coverage (days)",
            min_value=7,
            max_value=60,
            value=int(st.session_state["target_service_days"]),
            step=7,
        )
        st.session_state["min_slow_daily_units"] = st.number_input(
            "Min daily units for slow-mover ranking",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["min_slow_daily_units"]),
            step=0.01,
        )
        st.session_state["slow_top_n"] = st.number_input(
            "Slow movers shown",
            min_value=5,
            max_value=50,
            value=int(st.session_state["slow_top_n"]),
            step=5,
        )
        st.session_state["fast_top_n"] = st.number_input(
            "Fast movers shown",
            min_value=5,
            max_value=50,
            value=int(st.session_state["fast_top_n"]),
            step=5,
        )

    st.markdown("---")
    st.markdown("### Data Info")
    start_date, end_date = get_date_range(df)
    st.write(f"**Data Date Range:** {start_date.date()} → {end_date.date()}")
    st.write(f"**Rows:** {len(df):,} • **Unique SKUs:** {df[SKU_COL].nunique():,}")

    if st.button("Reset Settings to Defaults"):
        st.session_state["history_days"] = DEFAULT_HISTORY_DAYS
        st.session_state["forecast_days"] = DEFAULT_FORECAST_DAYS
        st.session_state["safety_factor"] = DEFAULT_SAFETY_FACTOR
        st.session_state["target_service_days"] = DEFAULT_TARGET_SERVICE_DAYS
        st.session_state["min_slow_daily_units"] = DEFAULT_MIN_SLOW_DAILY_UNITS
        st.session_state["fast_top_n"] = DEFAULT_FAST_TOP_N
        st.session_state["slow_top_n"] = DEFAULT_SLOW_TOP_N
        st.success("Settings reset. Reloading metrics…")
        st.cache_data.clear()
        st.rerun()


def page_raw_data(df: pd.DataFrame):
    st.subheader("Raw Data Explorer")
    st.dataframe(df, use_container_width=True)
    st.caption("Use this to verify uploads are correct and spot anomalies.")


def page_help():
    st.subheader("Help & FAQ")
    st.markdown(
        f"""
### What is AlcIQ?

AlcIQ is an **inventory intelligence tool for beverage retailers and drive-thrus**.
It helps you:

- Catch **stockout risks** before they happen  
- Flag **slow movers** and **dead inventory** that are tying up cash  
- Identify **fast movers** that deserve more shelf space  
- Build **purchase orders** in seconds instead of hours  
- Understand **seasonality** in your categories  
- Experiment with **discounts** and see their impact on revenue and margin  

---

### How does AlcIQ classify risk?

For each SKU, AlcIQ looks at:

1. **Average daily units sold** over the last `History window (days)`  
2. **Forecast demand** over the `Forecast horizon (days)`  
3. **Safety stock**, a buffer based on lead time and a safety factor  
4. **Current inventory**, from your latest counts  

From these, it assigns:

- 🔥 **Stockout Risk** – projected to run out within the forecast window  
- 🟡 **Low Inventory** – not enough to comfortably cover demand  
- 🔵 **Overstock** – way more inventory than expected demand  
- ✅ **Healthy** – inventory is in a good range  

You can adjust these assumptions under **Settings** to match your store’s risk tolerance.

---

### What data do I need to feed AlcIQ?

At minimum, a CSV with:

- Date (`{DATE_COL}`)  
- SKU (`{SKU_COL}`)  
- Product name (`{NAME_COL}`)  
- Category (`{CAT_COL}`)  
- Supplier (`{SUPPLIER_COL}`)  
- Units sold (`{UNITS_COL}`)  
- Revenue (`{REV_COL}`)  
- Inventory on hand (`{INV_COL}`)  
- Unit cost (`{COST_COL}`)  
- Lead time in days (`{LEADTIME_COL}`)  

Right now, if that file doesn't exist, AlcIQ auto-creates a realistic **sample dataset** at `data/alcIQ_sample.csv`.

---

### Daily workflow

1. Open AlcIQ and enter access code  
2. Check **Overview**  
3. Go to **Inventory Forecast** → filter 🔥 / 🟡  
4. Go to **Purchase Orders** → download CSV and send to suppliers  
5. Weekly: check **Slow & Fast Movers** and **Category Analytics**  
6. Use **Discount Simulator** before promos

        """
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("AlcIQ – Inventory Intelligence for Beverage Retailers")

    # --- Simple access code gate ---
    access = st.text_input("Enter access code to continue:", type="password")
    if access != ACCESS_CODE:
        st.warning("Please enter the correct access code to access AlcIQ.")
        st.stop()

    # Auto-generate sample data if file missing
    if not DATA_PATH.exists():
        st.info("No data file found. Generating a realistic sample dataset for a drive-thru beverage store…")
        generate_sample_dataset(DATA_PATH)

    # Top bar: refresh button + last updated
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with top_col2:
        st.caption(
            f"Data source: `{DATA_PATH}` • Last updated: {get_file_last_updated(DATA_PATH)}"
        )

    # Load data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar navigation + settings
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "",
        [
            "Overview",
            "Inventory Forecast",
            "Slow & Fast Movers",
            "Purchase Orders",
            "SKU Explorer",
            "Category & Supplier Analytics",
            "Seasonality & Trends",
            "Margin & Discount Simulator",
            "Settings & Assumptions",
            "Raw Data",
            "Help & FAQ",
        ],
    )

    st.sidebar.markdown("---")
    add_sidebar_settings()

    # Compute metrics using current settings
    metrics = compute_inventory_metrics(
        df,
        history_days=int(st.session_state["history_days"]),
        forecast_days=int(st.session_state["forecast_days"]),
        safety_factor=float(st.session_state["safety_factor"]),
        target_service_days=int(st.session_state["target_service_days"]),
    )

    # Route
    if page == "Overview":
        page_overview(df, metrics)
    elif page == "Inventory Forecast":
        page_inventory_forecast(metrics)
    elif page == "Slow & Fast Movers":
        page_slow_fast(metrics)
    elif page == "Purchase Orders":
        page_purchase_orders(metrics)
    elif page == "SKU Explorer":
        page_sku_explorer(df, metrics)
    elif page == "Category & Supplier Analytics":
        page_category_supplier(df, metrics)
    elif page == "Seasonality & Trends":
        page_seasonality(df, metrics)
    elif page == "Margin & Discount Simulator":
        page_discount_simulator(df, metrics)
    elif page == "Settings & Assumptions":
        page_settings(df, metrics)
    elif page == "Raw Data":
        page_raw_data(df)
    elif page == "Help & FAQ":
        page_help()


if __name__ == "__main__":
    main()









