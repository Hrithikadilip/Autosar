import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  AUTOSAR TSN vs Non-TSN ML Analysis")
print("=" * 60)

FILE_PATHS = {
    "driving1_tsn":     r"C:\Users\Lenovo\OneDrive\Desktop\autosarproj\driving 1 orginal.csv",
    "driving1_nontsn":  r"C:\Users\Lenovo\OneDrive\Desktop\autosarproj\driving 1 injected.csv",
    "driving2_tsn":     r"C:\Users\Lenovo\OneDrive\Desktop\autosarproj\driving 2 original.csv",
    "driving2_nontsn":  r"C:\Users\Lenovo\OneDrive\Desktop\autosarproj\driving 2 injected.csv",
}

print("\n[1/7] Loading datasets...")
dfs = {}
for name, path in FILE_PATHS.items():
    try:
        df = pd.read_csv(path, low_memory=False)
        label = 0 if 'tsn' in name and 'non' not in name else 1   # 0=TSN, 1=Non-TSN
        df['label'] = label
        df['source'] = name
        dfs[name] = df
        print(f"  ✓ {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    except FileNotFoundError:
        print(f"  ✗ {name}: FILE NOT FOUND – skipping")

if not dfs:
    raise SystemExit("No files loaded. Check paths and try again.")

# Combine
combined = pd.concat(list(dfs.values()), ignore_index=True, sort=False)
print(f"\n  Combined dataset: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
print(f"  TSN (0):     {(combined['label']==0).sum():,} rows")
print(f"  Non-TSN (1): {(combined['label']==1).sum():,} rows")

# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────
print("\n[2/7] Preprocessing...")

# Drop non-feature columns
drop_cols = ['label', 'source']
feature_cols = [c for c in combined.columns if c not in drop_cols]

# Keep only numeric columns for ML
numeric_cols = combined[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"  Numeric features: {len(numeric_cols)}")

X_raw = combined[numeric_cols].copy()
y = combined['label'].values

# Handle missing values
missing_pct = X_raw.isnull().mean()
high_missing = missing_pct[missing_pct > 0.5].index.tolist()
X_raw.drop(columns=high_missing, inplace=True)
X_raw.fillna(X_raw.median(), inplace=True)

# Handle infinities
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw.fillna(X_raw.median(), inplace=True)

# Remove constant columns
non_const = X_raw.std() > 0
X_raw = X_raw.loc[:, non_const]

feature_names = X_raw.columns.tolist()
print(f"  Features after cleaning: {len(feature_names)}")

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ──────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[3/7] Splitting train/test (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ──────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────
print("\n[4/7] Training models...")

models = {
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                  SVC(probability=True, random_state=42, kernel='rbf'),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_f1 = -1

for name, model in models.items():
    print(f"  Training {name}...", end='', flush=True)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)

    results[name] = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "auc_roc":   round(float(auc),  4),
        "cv_mean":   round(float(cv_scores.mean()), 4),
        "cv_std":    round(float(cv_scores.std()),  4),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
    }

    if f1 > best_f1:
        best_f1    = f1
        best_model = (name, model)

    print(f"  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

print(f"\n  ★ Best model: {best_model[0]} (F1={best_f1:.4f})")

# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (best model if RF/GBM)
# ──────────────────────────────────────────────
print("\n[5/7] Feature importance...")
top_features = []
bname, bmodel = best_model

if hasattr(bmodel, 'feature_importances_'):
    imp = bmodel.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': imp})
    fi_df.sort_values('importance', ascending=False, inplace=True)
    top_features = fi_df.head(20).to_dict('records')

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 20))
    bars = ax.barh(fi_df.head(20)['feature'][::-1],
                   fi_df.head(20)['importance'][::-1],
                   color=colors[::-1])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top 20 Feature Importances — {bname}', fontsize=14, fontweight='bold')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines[:].set_color('#334155')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_importance.png")

# ──────────────────────────────────────────────
# 6. ANOMALY DETECTION (Isolation Forest)
# ──────────────────────────────────────────────
print("\n[6/7] Anomaly detection (Isolation Forest)...")
iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
iso.fit(X_train[y_train == 0])   # train on TSN only
anomaly_scores = iso.decision_function(X_test)
anomaly_pred   = iso.predict(X_test)           # -1 = anomaly, 1 = normal
anomaly_binary = (anomaly_pred == -1).astype(int)

iso_acc = accuracy_score(y_test, anomaly_binary)
iso_f1  = f1_score(y_test, anomaly_binary, zero_division=0)
print(f"  Isolation Forest → Acc={iso_acc:.4f}  F1={iso_f1:.4f}")

# ──────────────────────────────────────────────
# 7. PCA for 2D visualisation data
# ──────────────────────────────────────────────
print("\n[7/7] PCA projection for dashboard visualisation...")
# Sample for speed
sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X[sample_idx])
pca_data = [
    {"x": round(float(X_2d[i, 0]), 4),
     "y": round(float(X_2d[i, 1]), 4),
     "label": int(y[sample_idx[i]])}
    for i in range(len(sample_idx))
]
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ──────────────────────────────────────────────
# SAVE RESULTS JSON
# ──────────────────────────────────────────────
dataset_stats = {
    "total_rows":   int(combined.shape[0]),
    "tsn_rows":     int((combined['label'] == 0).sum()),
    "nontsn_rows":  int((combined['label'] == 1).sum()),
    "num_features": len(feature_names),
    "train_size":   int(X_train.shape[0]),
    "test_size":    int(X_test.shape[0]),
}

output = {
    "dataset_stats":     dataset_stats,
    "model_results":     results,
    "best_model":        bname,
    "top_features":      top_features[:15],
    "anomaly_detection": {
        "accuracy": round(float(iso_acc), 4),
        "f1":       round(float(iso_f1),  4),
    },
    "pca_data": pca_data,
    "pca_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
}

with open('/home/claude/results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ results.json saved")
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
for mname, mres in results.items():
    flag = " ★" if mname == bname else ""
    print(f"  {mname:<22} Acc={mres['accuracy']:.4f}  F1={mres['f1']:.4f}  AUC={mres['auc_roc']:.4f}{flag}")
print("="*60)
print("Done ✓")