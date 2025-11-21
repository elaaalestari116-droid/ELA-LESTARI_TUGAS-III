# Preprocessing Kredit Nasabah

Repository contains a full preprocessing pipeline applied to the kredit nasabah dataset.

**Files included**:
- DATASET KREDIT NASABAH.xlsx (original data) - stored externally, path provided in notebook
- preprocessing_notebook.ipynb - notebook reproducing steps
- preprocessing.py - example script
- PNG screenshots of each step (01_... to 10_...)
- CSV outputs: X_train_scaled.csv, X_test_scaled.csv, y_train.csv, y_test.csv

**Steps performed**:
1. Data inspection
2. Imputation (median/mode)
3. Rare category grouping (<1% -> 'Lain-Lain')
4. One-Hot Encoding (drop='first')
5. IQR-based outlier capping (multiplier=3)
6. Train-test split (80/20)
7. StandardScaler fitted on train

To create a GitHub repository, upload the contents of this folder and push to your account.
