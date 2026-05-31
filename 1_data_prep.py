import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("winequality_combined.csv", sep=";")
print(f"Датасет загружен. Форма: {df.shape}")

print("\n Пропущенные значения:\n", df.isnull().sum().sum())
print(df.describe().round(2))

le = LabelEncoder()
df["wine_type"] = le.fit_transform(df["wine_type"])  # red=0, white=1
print(f"wine_type закодирован. Уникальные значения: {df['wine_type'].unique()}")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_test.npy", X_test_scaled)
np.save("data/processed/y_train.npy", y_train.values)
np.save("data/processed/y_test.npy", y_test.values)

print("\nДанные и скалер сохранены в папки data/processed и models/")