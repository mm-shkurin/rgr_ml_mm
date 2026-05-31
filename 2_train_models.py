import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from catboost import CatBoostRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import r2_score

tf.get_logger().setLevel('ERROR') 

print(" Загрузка подготовленных данных...")
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print(f" Данные загружены: Train={X_train.shape}, Test={X_test.shape}")

results = []

print("Обучение Ridge Regression...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
r2_ridge = r2_score(y_test, ridge.predict(X_test))
joblib.dump(ridge, "models/ridge.pkl")
results.append({"Model": "Ridge", "R²": r2_ridge})

print("Обучение Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
gb.fit(X_train, y_train)
r2_gb = r2_score(y_test, gb.predict(X_test))
joblib.dump(gb, "models/gradient_boosting.pkl")
results.append({"Model": "GradientBoosting", "R²": r2_gb})

print("Обучение CatBoost...")
cb = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, verbose=False, random_seed=42)
cb.fit(X_train, y_train)
r2_cb = r2_score(y_test, cb.predict(X_test))
cb.save_model("models/catboost.cbm")
results.append({"Model": "CatBoost", "R²": r2_cb})

print("Обучение Bagging...")
bag = BaggingRegressor(n_estimators=50, random_state=42)
bag.fit(X_train, y_train)
r2_bag = r2_score(y_test, bag.predict(X_test))
joblib.dump(bag, "models/bagging.pkl")
results.append({"Model": "Bagging", "R²": r2_bag})

print("Обучение Stacking...")
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]
stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(), cv=5)
stack.fit(X_train, y_train)
r2_stack = r2_score(y_test, stack.predict(X_test))
joblib.dump(stack, "models/stacking.pkl")
results.append({"Model": "Stacking", "R²": r2_stack})

print("📊 Обучение MLP (Fully Connected Neural Network)...")
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True
)
mlp.fit(X_train, y_train)
r2_mlp = r2_score(y_test, mlp.predict(X_test))
joblib.dump(mlp, "models/mlp.pkl")
results.append({"Model": "MLP (FCNN)", "R²": r2_mlp})

df_res = pd.DataFrame(results).sort_values(by="R²", ascending=False)
print("Итоговые метрики (R²) на тестовой выборке:")
print(df_res.to_string(index=False))
print("Все 6 моделей успешно сериализованы в папку `models/`")
