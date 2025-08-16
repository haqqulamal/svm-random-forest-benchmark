# SVM vs Random Forest — scikit-learn Benchmark

Benchmark sederhana membandingkan **SVM** (linear/RBF) dan **Random Forest** pada tugas klasifikasi.
Fokus: pipeline preprocessing, validasi yang benar, dan evaluasi metrik standar.

Notebook utama: `notebooks/svm_rf.ipynb`.

## Metodologi
- **EDA singkat**: distribusi fitur/target, missing values, imbalance check.
- **Preprocessing**:
  - Imputasi (SimpleImputer), encoding (OneHot/Ordinal).
  - **Scaling** (StandardScaler) diperlukan untuk SVM; dibungkus via `Pipeline`.
- **Model**:
  - SVM (kernel `linear` dan/atau `rbf`, grid kecil `C`, `gamma`).
  - Random Forest (grid kecil `n_estimators`, `max_depth`, `min_samples_split`).
  - **Cross-validation** (`StratifiedKFold`) + (opsional) `GridSearchCV`.
  - Imbalance: `class_weight="balanced"` atau `SMOTE` (opsional).
- **Evaluasi**:
  - Accuracy, Precision, Recall, F1, **ROC-AUC** (biner).
  - Confusion Matrix, ROC/PR curves.
  - Simpan ke `results/metrics.json` & `results/plots/`.
