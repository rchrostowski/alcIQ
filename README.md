# Pavlish Inventory & Order Optimizer

A prototype Streamlit app built for **Pavlish Beverage Company** to optimize inventory and recommend what to order.

## What it does

- Uses recent sales data to estimate **average daily demand** by SKU  
- Calculates **reorder points** and **target stock levels** using lead times & safety stock  
- Generates a **recommended order** by SKU and vendor  
- Highlights:
  - High stockout risk items (fast movers with low inventory)
  - Overstocked items (slow movers with lots of inventory)

## App layout

- **Sidebar**
  - Toggle: use sample data or upload your own CSVs  
  - Controls for:
    - Lookback window (days)
    - Safety factor (service level)
    - Review period (days between orders)

- **Main view**
  - KPIs:
    - Total recommended order cost
    - Estimated profit from the order
    - Number of SKUs to order
  - Table of recommended orders (filterable by vendor)
  - Downloadable CSV of the order
  - Inventory health snapshot (stockout risks and overstocked SKUs)

## Data format

### 1. Sales CSV

Example: `data/sales_sample.csv`

```csv
date,sku,product_name,qty_sold,unit_price
2025-11-01,1001,Titos 1.75L,3,32.99
2025-11-02,1001,Titos 1.75L,4,32.99
2025-11-03,1001,Titos 1.75L,2,32.99
...
