import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

mlp_model = joblib.load("models/mlp.pkl")
models["MLP (FCNN)"] = mlp_model

st.set_page_config(page_title="Прогнозирование", layout="wide")

st.title("Инференс моделей ML")

@st.cache_resource
def load_all():
    scaler = joblib.load("models/scaler.pkl")
    models = {
        "Ridge": joblib.load("models/ridge.pkl"),
        "GradientBoosting": joblib.load("models/gradient_boosting.pkl"),
        "Bagging": joblib.load("models/bagging.pkl"),
        "Stacking": joblib.load("models/stacking.pkl"),
        "CatBoost": CatBoostRegressor().load_model("models/catboost.cbm"),
        "FCNN": tf.keras.models.load_model("models/fcnn.keras")
    }
    return scaler, models

scaler, models = load_all()

features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type"
]

input_mode = st.radio("Выберите способ ввода данных:", ["Загрузить CSV", "Ручной ввод"])

input_df = None

if input_mode == "Загрузить CSV":
    uploaded = st.file_uploader("Загрузите .csv с колонками признаков", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            missing = [f for f in features if f not in df_up.columns]
            if missing:
                st.error(f"В файле отсутствуют колонки: {missing}")
            else:
                input_df = df_up[features]
                st.success("Файл успешно загружен!")
                st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Ошибка чтения: {e}")
else:
    st.markdown("Введите значения признаков (указаны единицы измерения):")
    cols = st.columns(3)
    data = {}
    for i, feat in enumerate(features):
        with cols[i % 3]:
            if feat == "wine_type":
                val = st.selectbox(f"{feat} (0=Red, 1=White)", [0, 1])
            else:
                val = st.number_input(f"{feat}", value=0.0, step=0.01, format="%.3f")
            data[feat] = val
    
    if st.button("Сформировать выборку"):
        input_df = pd.DataFrame([data])
        st.success("Данные готовы к прогнозу")
        st.dataframe(input_df)

if input_df is not None:
    st.divider()
    st.subheader("Результаты предсказания")
    
    X_scaled = scaler.transform(input_df)
    preds = {}
    for name, model in models.items():
        p = model.predict(X_scaled)
        if name == "FCNN" and p.ndim > 1:
            p = p.flatten()
        preds[name] = round(float(p[0]), 2)
    
    res_df = pd.DataFrame(list(preds.items()), columns=["Модель", "Прогноз качества"])
    
    def color_quality(val):
        if val >= 7: return "background-color: #4CAF50; color: white"
        elif val >= 5: return "background-color: #FFC107; color: black"
        else: return "background-color: #F44336; color: white"
        
    st.dataframe(res_df.style.applymap(color_quality, subset=["Прогноз качества"]), use_container_width=True)
    
    st.info("""
    **Интерпретация результата:**
    • `≥ 7.0` — Высокое качество (Premium)
    • `5.0 – 6.9` — Среднее качество (Table Wine)
    • `< 5.0` — Низкое качество (Possible defects)
    """)