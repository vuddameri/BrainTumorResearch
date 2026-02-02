import os
import pandas as pd
import numpy as np
from mrmr import mrmr_classif  # from mrmr-selection
import pacmap
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ===== USER SETTINGS =====
path = '/home/vuddameri/Ameya/Code' # copied from ameya/archive/greproc folder
csv_file = "features2.csv"  # Path to your CSV
label_col = "Label"                     # Adjust to your label column name
n_mrmr_options = [30, 50, 70,80,100]            # mRMR feature counts to test
n_pacmap_options = [5, 8, 12]            # PaCMAP output dimensions to test
cv_folds = 5                             # Cross-validation folds
random_state = 42
# =========================

# 1. Load CSV
# Load data and preprocessing (same as before)
os.chdir(path)
labelcol = "Label"           # Column name for labels
df = pd.read_csv(csv_file)
labelx = df.Label  # Labels for later use
X_full = df.drop(columns=[labelcol,"Filename","Min"])
Y = df[[labelcol]].values
encoder = OrdinalEncoder()
df[[labelcol]] = encoder.fit_transform(Y)
y = df[labelcol].values

# 2. Store results
results = []
# 3. Loop over mRMR feature counts
for n_mrmr in n_mrmr_options:
    # Select top features
    top_features = mrmr_classif(X=X_full, y=y, K=n_mrmr)
    X_top = X_full[top_features]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)

    # 4. Loop over PaCMAP dimensions
    for n_dim in n_pacmap_options:
        embedding = pacmap.PaCMAP(
            n_components=n_dim,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=random_state
        )
        X_pacmap = embedding.fit_transform(X_scaled, init="pca")

        # Model
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        )

        # Cross-validation
        scores = cross_val_score(clf, X_pacmap, y, cv=cv_folds, scoring='accuracy')
        mean_score = np.mean(scores)

        results.append({
            "n_mrmr_features": n_mrmr,
            "n_pacmap_dims": n_dim,
            "cv_accuracy": mean_score
        })
        print(f"mRMR={n_mrmr}, PaCMAP dims={n_dim}, CV acc={mean_score:.4f}")

# 5. Sort and save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="cv_accuracy", ascending=False)
print("\n=== Sorted Results ===")
print(results_df)

results_df.to_csv("mrmr_pacmap_gridsearch_results.csv", index=False)
print("\nResults saved to mrmr_pacmap_gridsearch_results.csv")

# Final Processed Features
opt_features = mrmr_classif(X=X_full, y=y, K=100)
X_opt = X_full[opt_features]
 # Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_opt)
embedding = pacmap.PaCMAP(
    n_components=12,
    MN_ratio=0.5,
    FP_ratio=2.0,
    random_state=random_state
)
X_pacmap = embedding.fit_transform(X_scaled, init="pca")

#Write Final Datasets
findat = pd.DataFrame(X_opt)
findat.columns = opt_features
findat['y'] = y
findat["Label"] = labelx
findat['FileName'] = df["Filename"]
findat.to_csv('transdataraw.csv',index=False)

colnames =  [f"pcmap{i}" for i in range(12)]
fintrn = pd.DataFrame(X_pacmap)
fintrn.columns = colnames
fintrn['y'] = y
fintrn["Label"] = labelx
fintrn['FileName'] = df["Filename"]
fintrn.to_csv('transdata.csv',index=False)