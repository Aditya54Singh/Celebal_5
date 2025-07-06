
# House Prices - Linear Regression Analysis for Internship Assignment

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, scipy.stats as stats, warnings, os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import VarianceThreshold

warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = (8,5)

# 1 — Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['SalePrice']
train = train.drop('SalePrice', axis=1)
full = pd.concat([train, test], axis=0, ignore_index=True)
print(f'Train shape: {train.shape}, Test shape: {test.shape}')

# 2 — Handle missing values
cat_na = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish',
          'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
          'BsmtFinType2','MasVnrType']
full[cat_na] = full[cat_na].fillna('None')

zero_na = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
           'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
full[zero_na] = full[zero_na].fillna(0)

full['LotFrontage'] = full.groupby('Neighborhood')['LotFrontage'].transform(
    lambda s: s.fillna(s.median()))

for col in full.columns:
    if full[col].isna().sum():
        if full[col].dtype=='object':
            full[col] = full[col].fillna(full[col].mode()[0])
        else:
            full[col] = full[col].fillna(full[col].median())

# 3 — Feature Engineering
full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']
full['Age'] = 2025 - full['YearBuilt']

# 4 — Skew correction
numeric_feats = full.select_dtypes(include=['number']).columns
skew = full[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewed = skew[skew.abs() > 0.75].index
full[skewed] = np.log1p(full[skewed])

# 5 — One-hot encoding
full = pd.get_dummies(full, drop_first=True)
print('Post‑dummies shape:', full.shape)

# 6 — Feature selection
selector = VarianceThreshold(threshold=0.01)
full_reduced = selector.fit_transform(full)
print('After variance filter:', full_reduced.shape)

# Split back
X = full_reduced[:train.shape[0], :]
X_test = full_reduced[train.shape[0]:, :]

# 7 — Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 8 — Model training and evaluation
Xtr, Xval, ytr, yval = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'OLS': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 50)),
    'Lasso': LassoCV(alphas=np.logspace(-4, 1, 30), max_iter=5000)
}

for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xval)
    rmse = np.sqrt(mean_squared_error(yval, pred))
    r2 = r2_score(yval, pred)
    print(f'{name:<6} RMSE: {rmse:8.0f}  R²: {r2:5.3f}')

# 9 — Final prediction and submission
best = models['Ridge']
best.fit(X_scaled, y)
pred_test = best.predict(X_test_scaled)

sub = pd.DataFrame({'Id': test.Id, 'SalePrice': pred_test})
sub.to_csv('submission_linreg.csv', index=False)
print('✅ submission_linreg.csv saved')
