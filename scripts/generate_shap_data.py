import pandas as pd
import numpy as np
import os, joblib, shap

base = os.path.dirname(os.path.abspath(__file__))

SOIL_PROPS = ['pH','OC','EC','K2O','available_P','available_N',
              'CEC','bulk_density','CaCO3']
FEATURES   = ['dem','slope','aspect','lulc','lithology',
              'geomorphology','NDVI','NDBI','SAVI','BSI']

# Load training data
data_path = os.path.join(base,'..','data','master_training_data.csv')
df = pd.read_csv(data_path)

avail = [f for f in FEATURES if f in df.columns]
X    = df[avail].dropna()
print(f"Features found : {avail}")
print(f"Rows available : {len(X)}")

results = {}
for prop in SOIL_PROPS:
    model_path = os.path.join(base,'..','models',f'rf_real_{prop}.pkl')
    if not os.path.exists(model_path):
        print(f"  SKIP {prop} — model file not found")
        continue
    model = joblib.load(model_path)
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        mean_shap  = np.abs(shap_vals).mean(axis=0)
        results[prop] = dict(zip(avail, mean_shap))
        print(f"  OK  {prop}")
    except Exception as e:
        print(f"  ERR {prop} — {e}")

if not results:
    print("\nNo models found. Check models/ folder.")
else:
    out_df = pd.DataFrame(results).T
    out_df.index.name = 'soil_property'
    out_path = os.path.join(base,'..','data','shap_importance.csv')
    out_df.to_csv(out_path)
    print(f"\nSaved → {out_path}")
    print(out_df.round(4))