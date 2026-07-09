# ============================================
# DSS - KHAMANON BLOCK
# Script 2: Random Forest Model Training
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ============================================
# LOAD FAKE DATA
# ============================================

data_path = os.path.join('..', 'data', 'soil_samples_fake.csv')
df = pd.read_csv(data_path)

print("=" * 50)
print("  RANDOM FOREST MODEL - KHAMANON BLOCK")
print("=" * 50)
print(f"\nData loaded: {len(df)} samples, {len(df.columns)} columns")

# ============================================
# PREPARE FEATURES (inputs to the model)
# These are your environmental covariates
# exactly as described in your synopsis
# ============================================

# Convert LULC text to numbers (RF needs numbers)
le = LabelEncoder()
df['LULC_encoded'] = le.fit_transform(df['LULC'])

# Save the encoder for later use in dashboard
joblib.dump(le, os.path.join('..', 'models', 'lulc_encoder.pkl'))

# Features = spectral indices + terrain + LULC
# Exactly what your Objective 2 methodology describes
features = [
    'NDVI', 'NDBI', 'SAVI', 'BSI',      # Sentinel-2 spectral indices
    'elevation', 'slope', 'aspect',       # Cartosat DEM derivatives
    'LULC_encoded'                        # Land use class
]

# Soil properties we want to predict
targets = [
    'pH',
    'organic_carbon',
    'EC',
    'available_N',
    'available_P',
    'available_K',
    'CEC'
]

X = df[features]

# ============================================
# TRAIN ONE MODEL PER SOIL PROPERTY
# Store results for all properties
# ============================================

results = {}
models  = {}

print("\nTraining models...")
print("-" * 50)

for target in targets:

    y = df[target]

    # Split: 70% training, 30% validation
    # Exactly as your methodology states
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.30,
        random_state = 42
    )

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators = 100,   # 100 decision trees
        max_depth    = 8,
        random_state = 42,
        n_jobs       = -1     # use all CPU cores
    )
    rf.fit(X_train, y_train)

    # Predict on validation set
    y_pred = rf.predict(X_test)

    # Calculate accuracy metrics
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results[target] = {
        'R2'  : round(r2,   3),
        'RMSE': round(rmse, 3),
        'MAE' : round(mae,  3)
    }

    models[target] = rf

    print(f"  {target:<20} R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

# ============================================
# SAVE ALL MODELS
# One .pkl file per soil property
# ============================================

for target, model in models.items():
    filename = f"rf_{target}.pkl"
    joblib.dump(model, os.path.join('..', 'models', filename))

print("-" * 50)
print(f"\nAll 7 models saved to: models/")

# ============================================
# SAVE VALIDATION RESULTS
# ============================================

results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join('..', 'data', 'model_validation.csv'))
print(f"Validation results saved to: data/model_validation.csv")

# ============================================
# PLOT 1: Actual vs Predicted for pH
# (most important property in your research)
# ============================================

X_train_pH, X_test_pH, y_train_pH, y_test_pH = train_test_split(
    X, df['pH'], test_size=0.30, random_state=42
)
y_pred_pH = models['pH'].predict(X_test_pH)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pH, y_pred_pH,
            color='steelblue', alpha=0.7,
            edgecolors='black', linewidth=0.5, s=80)
plt.plot([y_test_pH.min(), y_test_pH.max()],
         [y_test_pH.min(), y_test_pH.max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual pH', fontsize=12)
plt.ylabel('Predicted pH', fontsize=12)
plt.title('Random Forest: Actual vs Predicted pH\nKhamanon Block',
          fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('..', 'maps', 'validation_pH.png'), dpi=150)
plt.close()

# ============================================
# PLOT 2: Feature Importance for pH
# Shows which covariate matters most
# ============================================

importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': models['pH'].feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 6))
colors = ['#2ecc71' if i >= len(features)-3
          else '#3498db'
          for i in range(len(features))]
plt.barh(importance_df['Feature'],
         importance_df['Importance'],
         color=colors)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Variable Importance for pH Prediction\nKhamanon Block',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('..', 'maps', 'feature_importance_pH.png'), dpi=150)
plt.close()

# ============================================
# PLOT 3: Validation metrics for all properties
# ============================================

fig, ax = plt.subplots(figsize=(10, 5))
r2_values = [results[t]['R2'] for t in targets]
colors    = ['#2ecc71' if v >= 0.7
             else '#e67e22' if v >= 0.5
             else '#e74c3c'
             for v in r2_values]

bars = ax.bar(targets, r2_values, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.7, color='green',  linestyle='--',
           linewidth=1.5, label='Good (R²=0.7)')
ax.axhline(y=0.5, color='orange', linestyle='--',
           linewidth=1.5, label='Moderate (R²=0.5)')
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Model Accuracy for All Soil Properties\nKhamanon Block (Fake Data)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend()

for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join('..', 'maps', 'model_accuracy_all.png'), dpi=150)
plt.close()

print(f"3 charts saved to: maps/")
print("\n" + "=" * 50)
print("  STAGE 2 COMPLETE")
print("=" * 50)
print("\nWhat was built:")
print("  - 7 RF models (one per soil property)")
print("  - Validation metrics (R², RMSE, MAE)")
print("  - 3 charts saved to maps/ folder")
print("  - Models saved to models/ folder")
print("\nNext: Spatial prediction maps")