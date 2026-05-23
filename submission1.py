import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import re
import warnings
warnings.filterwarnings('ignore')




path = '/kaggle/input/datasets/ans002/options/new.csv'
df = pd.read_csv(path)
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')

iv_cols = [c for c in df.columns if 'NIFTY' in c]
strike_map = {}
for col in iv_cols:
    match = re.search(r'(\d+)(CE|PE)', col)
    if match:
        strike_map[col] = {'k': int(match.group(1)), 'type': 1 if match.group(2) == 'CE' else 0}

expiry = pd.to_datetime('27-01-2026 15:30', format='%d-%m-%Y %H:%M')

# ==========================================

print("Preparing spatial-temporal coordinate data...")
long_data = []

for idx, row in df.iterrows():
    tte = (expiry - row['datetime']).total_seconds() / 60
    spot = row['underlying_price']
    
    # Global level anchor for the current timestamp
    row_ivs = row[iv_cols].values.astype(float)
    mean_iv = np.nanmean(row_ivs) if not np.all(np.isnan(row_ivs)) else 0.15 
    
    for col in iv_cols:
        actual = row[col]
        long_data.append({
            'datetime': row['datetime'],
            'strike_col': col,
            'tte': tte,
            'spot': spot,
            'strike': strike_map[col]['k'],
            'moneyness': strike_map[col]['k'] / spot,
            'opt_type': strike_map[col]['type'],
            'surface_level': mean_iv,
            'actual': actual
        })

full_df = pd.DataFrame(long_data)
train_set = full_df.dropna(subset=['actual']).copy()
predict_set = full_df[full_df['actual'].isnull()].copy()

# ==========================================

print("Training Pure LightGBM on Raw IV...")
features = ['tte', 'spot', 'strike', 'moneyness', 'opt_type', 'surface_level']

X = train_set[features]
y = train_set['actual']


model = lgb.LGBMRegressor(
    n_estimators=4000,       
    learning_rate=0.03,      
    num_leaves=255,          
    min_child_samples=2,     
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ==========================================

predict_set['final_iv'] = model.predict(predict_set[features])


predict_set['final_iv'] = np.clip(predict_set['final_iv'], 0.001, None)

output_df = df.copy()
for _, row in predict_set.iterrows():
    output_df.loc[output_df['datetime'] == row['datetime'], row['strike_col']] = row['final_iv']


train_preds = model.predict(X)
final_mse = mean_squared_error(y, train_preds)

print(f"Final Direct GBM MSE (Training): {final_mse:.8f}")

output_df.to_csv('submission_pure_gbm.csv', index=False)
print("File successfully saved as 'submission_pure_gbm.csv'.")