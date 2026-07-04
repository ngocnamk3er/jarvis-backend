---
name: data-analysis
description: Data analysis workflow using pandas, numpy, and matplotlib. Use when the user asks to analyze data, generate statistics, clean datasets, or create charts from data files.
---

# data-analysis

## Setup

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

## Loading data

```python
import pandas as pd, os

upload = os.environ["UPLOAD"]
output = os.environ["OUTPUT"]

# CSV
df = pd.read_csv(f"{upload}/data.csv")

# Excel
df = pd.read_excel(f"{upload}/data.xlsx", sheet_name=0)

# JSON
df = pd.read_json(f"{upload}/data.json")
```

## Quick overview

```python
print(df.shape)           # rows, cols
print(df.dtypes)          # column types
print(df.describe())      # stats for numeric cols
print(df.isnull().sum())  # missing values per column
print(df.head())
```

## Cleaning

```python
# Drop rows with any null
df = df.dropna()

# Fill nulls
df["col"] = df["col"].fillna(df["col"].median())

# Convert types
df["date"] = pd.to_datetime(df["date"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Rename columns
df = df.rename(columns={"old": "new"})

# Remove duplicates
df = df.drop_duplicates()
```

## Aggregation

```python
# Group and aggregate
summary = df.groupby("category").agg(
    total=("amount", "sum"),
    avg=("amount", "mean"),
    count=("amount", "count"),
).reset_index()

# Pivot table
pivot = df.pivot_table(values="amount", index="date", columns="category", aggfunc="sum")
```

## Visualization

Always save to `/output` and call `represent_file` afterwards:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(summary["category"], summary["total"])
ax.set_title("Total by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Total")
fig.tight_layout()
fig.savefig(f"{os.environ['OUTPUT']}/chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

Then show it:
```
represent_file("/output/chart.png")
```

## Saving results

```python
# CSV
df.to_csv(f"{output}/result.csv", index=False)

# Excel with multiple sheets
with pd.ExcelWriter(f"{output}/report.xlsx") as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
```
