# preprocessing.py - simplified pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel(r"/mnt/data/DATASET KREDIT NASABAH.xlsx")

# Impute
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns
df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

# Rare grouping
threshold = 0.01
for col in cat_cols:
    freqs = df[col].value_counts(normalize=True)
    rare = freqs[freqs < threshold].index.tolist()
    if rare:
        df[col] = df[col].apply(lambda x: 'Lain-Lain' if x in rare else x)

# One-hot encoding
enc = OneHotEncoder(drop='first', sparse=False)
enc_arr = enc.fit_transform(df[cat_cols])
enc_cols = enc.get_feature_names_out(cat_cols)
df_enc = pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), pd.DataFrame(enc_arr, columns=enc_cols)], axis=1)

# Outlier capping
for col in df_enc.select_dtypes(include=['number']).columns:
    Q1 = df_enc[col].quantile(0.25)
    Q3 = df_enc[col].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 3*IQR
    upper = Q3 + 3*IQR
    df_enc[col] = df_enc[col].clip(lower=lower, upper=upper)

# Train-test and scaling
X_train, X_test = train_test_split(df_enc, test_size=0.2, random_state=42)
num_cols_X = X_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[num_cols_X] = scaler.fit_transform(X_train[num_cols_X])
X_test[num_cols_X] = scaler.transform(X_test[num_cols_X])

X_train.to_csv('X_train_scaled.csv', index=False)
X_test.to_csv('X_test_scaled.csv', index=False)
