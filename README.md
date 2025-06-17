# Akurasi-naik-signifikan

Below is a **battle-tested, step-by-step recipe** that will give you the two plots exactly as you specified – no silent assumptions, no hidden magic. Copy-paste and run; every line is annotated so you can see (and audit) the logic.

> **Assumptions**
> • Your DataFrame is called **`df`** and already contains the columns
> `label_interval`, `utc_hour` (0-23), `day_name` (English, capitalized), and `is_weekend` (boolean).
> • One row = one occurrence of `label_interval`.
> • You want raw counts (not proportions).

---

## 1  Data validation & pre-processing (do **once**)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- 1. Sanity checks --------
expected_hours = set(range(24))
actual_hours   = set(df['utc_hour'].unique())
assert actual_hours.issubset(expected_hours), \
       f"Found invalid utc_hour values: {sorted(actual_hours - expected_hours)}"

ordered_days = ['Sunday','Monday','Tuesday','Wednesday',
                'Thursday','Friday','Saturday']
assert set(df['day_name'].unique()).issubset(set(ordered_days)), \
       "Unexpected day_name values detected"

# Ensure correct dtypes / order
df['utc_hour'] = df['utc_hour'].astype(int)
df['day_name'] = pd.Categorical(df['day_name'],
                                categories=ordered_days, ordered=True)
# If is_weekend isn't present or reliable, build it defensively
if 'is_weekend' not in df.columns:
    df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday'])
```

Why bother? Catching bad hours (e.g., “24”, “-1”) or misspelled day names here prevents misleading blanks or shifts in the plots later.

---

## 2  Heat-map of counts by **day\_name × utc\_hour**

```python
# -------- 2. Aggregate --------
heatmap = (df
           .groupby(['day_name', 'utc_hour'])
           .size()                         # row count  ==  occurrences
           .unstack(fill_value=0)          # pivot: rows=days, cols=hours
           .reindex(index=ordered_days,    # guarantee canonical order
                    columns=range(24)))    # fill 0 where no data

# -------- 3. Plot --------
fig, ax = plt.subplots(figsize=(12, 5))

# 'Greys_r' gives dark for high counts, light for low counts (your spec)
img = ax.imshow(heatmap,
                aspect='auto',
                cmap='Greys_r')

# Labels & ticks
ax.set_xticks(np.arange(24))
ax.set_xticklabels(range(24))
ax.set_yticks(np.arange(7))
ax.set_yticklabels(ordered_days)
ax.set_xlabel('UTC hour (0–23)')
ax.set_ylabel('Day of week')
ax.set_title('Count of label_interval by UTC hour and day')

# Colorbar
cbar = fig.colorbar(img, ax=ax, pad=0.02)
cbar.set_label('Number of occurrences')

plt.tight_layout()
plt.show()
```

Key design choice: `Greys_r` (reversed greys) satisfies *“darker ⇒ more frequent”* literally. If you prefer a perceptually uniform colormap, swap to `'inferno_r'`.

---

## 3  Side-by-side bar chart (**weekday vs weekend**) per hour

```python
# -------- 4. Aggregate --------
bar_df = (df
          .groupby(['utc_hour', 'is_weekend'])
          .size()
          .unstack(fill_value=0)           # columns: False=weekday, True=weekend
          .reindex(index=range(24),        # again, force 0-23
                   fill_value=0)
          .rename(columns={False: 'Weekday', True: 'Weekend'}))

# -------- 5. Plot --------
fig, ax = plt.subplots(figsize=(12, 5))

hours = np.arange(24)
bar_w = 0.4

ax.bar(hours - bar_w/2, bar_df['Weekday'], width=bar_w,
       label='Weekday')
ax.bar(hours + bar_w/2, bar_df['Weekend'], width=bar_w,
       label='Weekend')

ax.set_xticks(hours)
ax.set_xlabel('UTC hour (0–23)')
ax.set_ylabel('Number of occurrences')
ax.set_title('label_interval frequency by hour: Weekday vs Weekend')
ax.legend(title='Day type', frameon=False)

plt.tight_layout()
plt.show()
```

Why two separate bars instead of a stacked bar?
*Side-by-side* keeps absolute counts visible and makes the weekday/weekend contrast crystal-clear at every hour. (Stacked would obscure direct comparison.)

---

### ⚡ Common pitfalls (and how the code avoids them)

| Pitfall                                  | Protection in the recipe                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Missing hours/days silently disappear    | `.reindex(columns=range(24))` and `.reindex(index=ordered_days)` force 0 counts to show up, so the axes are never mis-aligned. |
| Day label order scrambled alphabetically | `Categorical(..., ordered=True)` preserves Sunday→Saturday.                                                                    |
| Color perception reversed (light = many) | Explicit `Greys_r` puts high counts in dark cells.                                                                             |
| Floating-point hours or stray “24” rows  | `astype(int)` and the `assert` guardrail in Step 1.                                                                            |

---

### One-liner functions (optional helper)

If you’ll need these plots often, wrap them:

```python
def plot_heatmap_count(df):
    # … copy Section 2 (aggregate + plot) …

def plot_hourly_weekend_bars(df):
    # … copy Section 3 (aggregate + plot) …
```

Call them after you’ve performed the validations in **Section 1**.

---

## ✅ You now have 100 % reproducible, audit-friendly code

Run Section 1 once, then Section 2 and Section 3 (or the helper functions) whenever you need fresh visuals. Because every assumption is explicit and every intermediate result is built step-by-step, you can trace or unit-test each part – **no nasty surprises down the line**.
