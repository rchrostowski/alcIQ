# alcIQ – Liquor Inventory & Order Optimizer

**alcIQ** is a Streamlit-based decision support tool for **liquor and beverage retailers**.  
It helps store owners and managers place **smarter, data-driven purchase orders** by turning raw POS exports into clear reorder recommendations.

---

## ✨ Key capabilities

- Ingests recent **sales**, **inventory**, and **product master** data.
- Estimates **average daily demand** and demand variability for each SKU.
- Computes **reorder points** and **target stock levels** using lead times and a configurable service level.
- Generates a **recommended purchase order** by SKU and vendor.
- Highlights:
  - SKUs at **high risk of stockout**.
  - **Slow-moving / overstocked** SKUs that may tie up cash.

The goal is to give independent retailers access to the kind of inventory intelligence typically used by large chains.

---

## 🧱 App structure

The app is implemented as a single Streamlit script: `streamlit_app.py`.

### Sidebar – data & configuration

- **Use bundled sample data**  
  Quickly explore the app using the included CSVs in the `data/` folder.

- **Upload store data** (when not using sample data)  
  - Sales CSV  
  - Inventory CSV  
  - Products CSV  

- **Forecast & policy assumptions**  
  - Demand lookback window (days)  
  - Service level / safety factor (Z-score)  
  - Order review period (days between typical orders)

### Main interface

The main area is organized into two tabs:

#### 1. 📦 Recommended Order

- Top-level KPIs:
  - Total recommended order cost
  - Estimated profit on this order (based on recent selling prices and cost)
  - Number of SKUs with a positive recommended order quantity
- Vendor filter and an option to:
  - Show only SKUs with non-zero recommended order, or
  - Show all SKUs in the catalog.
- Detailed, sortable table of line items.
- **Download button** to export the recommended order as a CSV ready to send to distributors or upload to portals.

#### 2. 📊 Inventory Health

- “High stockout risk” panel:
  - Fast-moving SKUs with low on-hand inventory relative to their reorder points.
- “Overstocked / very slow movers” panel:
  - SKUs with very low average daily demand but significant inventory on hand.
- Designed to support decisions around:
  - Promotions, markdowns, and delisting
  - Aggressively protecting key items from stockouts

---

## 📂 Data formats

The app expects three CSV files with the following structures.  
Sample files are included in the `data/` directory.

### 1. Sales CSV (`data/sales_sample.csv`)

```csv
date,sku,product_name,qty_sold,unit_price
2025-11-01,1001,Titos 1.75L,3,32.99
2025-11-02,1001,Titos 1.75L,4,32.99
2025-11-03,1001,Titos 1.75L,2,32.99
2025-11-01,2001,Modelo 12pk,5,17.99
...

