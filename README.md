# Celebal_5


# House Prices — Linear Regression Walkthrough

> **One‑file approach to the classic Kaggle “Advanced Regression Techniques” competition.**  
> Everything lives in `HousePrices_EDA_Linear.py`, no notebooks required.

---

## 1.  What’s inside?

| File | Purpose |
|------|---------|
| `HousePrices_EDA_Linear.py` | Cleans the data, engineers a handful of features, trains Linear / Ridge / Lasso, and spits out `submission_linreg.csv`. |
| `submission_linreg.csv` | Generated when you run the script – ready for Kaggle upload. |

That’s it.  No hidden folders, no thirty‑layer ensemble.  I wanted something you can **read in a coffee break** and still learn a trick or two.

---

## 2.  Quick start

The script will:

1. Print dataset shapes plus a few metrics (RMSE, \(R^2\)).  
2. Create `submission_linreg.csv` in the same folder.  
3. Tell you which model (OLS, Ridge or Lasso) looked best on a hold‑out split.

---

## 3.  Modelling decisions (in plain English)

* **Missing “None”** — A bunch of columns are genuinely “not applicable” (`PoolQC`, `Alley`, …).  Leaving them blank confuses the model, so we replace NA with the string `"None"`.
* **LotFrontage** — Median‑impute per neighbourhood; frontage behaves like land price: location, location, location.
* **One‑hot encoding** — Vanilla `pd.get_dummies`, because the goal was a transparent linear model, not leaderboard domination.
* **Skew fix** — `log1p` on skewed numeric columns (anything with |skew| > 0.75).  Linear models like symmetry.
* **Variance filter** — Drops near‑constant columns; prevents OLS blow‑ups and speeds things up.

The idea is to _stay readable_.  If you need the last bit of juice, swap in CatBoost or toss the whole thing into a stacked ensemble.

---

## 4.  Results

On my side the Ridge model usually lands around **RMSE ≈ 31–35 k** on a 20 % validation split and **R² ≈ 0.82–0.84**.  
Public leaderboard RMSLE hovers around **0.16** – not medal‑worthy, but respectable for half a page of code.

---

## 5.  Room for improvement

* Tune `alpha` grids more aggressively.
* Add polynomial interactions (`OverallQual × GrLivArea` is a famous one).
* Swap variance filter for PCA if you don’t mind losing interpretability.
* Try Gradient Boosting / XGB / LightGBM and blend.

---

## 6.  License

MIT.  Use it, break it, share it, just keep the credit line please.

---

_Questions or ideas?  Open an issue or ping me – I’m always up for a geeky chat._
