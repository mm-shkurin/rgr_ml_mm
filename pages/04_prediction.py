import traceback

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

st.set_page_config(page_title="Прогнозирование", layout="wide")

st.title("Инференс моделей ML")

@st.cache_resource
def load_all():
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl") 
    models = {
        "Ridge": joblib.load("models/ridge.pkl"),
        "GradientBoosting": joblib.load("models/gradient_boosting.pkl"),
        "Bagging": joblib.load("models/bagging.pkl"),
        "Stacking": joblib.load("models/stacking.pkl"),
        "CatBoost": CatBoostRegressor().load_model("models/catboost.cbm"),
        "MLPRegressor": joblib.load("models/mlp.pkl"),
    }
    return scaler, models

scaler, models , le= load_all()

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
            df_up = pd.read_csv(uploaded, sep=";")
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
def preprocess_input(df, le, feature_list):
    """
    Предобработка входных данных для инференса.
    - Кодирует wine_type, если он в строковом формате
    - Гарантирует порядок колонок как при обучении
    """
    df = df.copy()
    
    # Обработка wine_type: если строки → кодируем, если числа → оставляем
    if df["wine_type"].dtype == object:
        df["wine_type"] = df["wine_type"].str.lower().str.strip()
        # Проверка: все значения есть в le.classes_?
        unknown = set(df["wine_type"]) - set(le.classes_)
        if unknown:
            raise ValueError(f"Неизвестные категории в wine_type: {unknown}. Ожидается: {list(le.classes_)}")
        df["wine_type"] = le.transform(df["wine_type"])
    elif df["wine_type"].dtype in [np.int64, np.int32, np.float64]:
        # Уже закодировано (0/1), просто приводим к int
        df["wine_type"] = df["wine_type"].astype(int)
    else:
        raise ValueError(f"Неподдерживаемый тип данных для wine_type: {df['wine_type'].dtype}")
    
    # Проверка наличия всех фич
    missing = set(feature_list) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Возвращаем строго в порядке обучения
    return df[feature_list]

if input_df is not None:
    st.divider()
    st.subheader("🔮 Результаты предсказания")
    
    try:
        # 1. Предобработка (кодирование + порядок колонок)
        input_processed = preprocess_input(input_df, le, features)
        
        # 2. Масштабирование
        X_scaled = scaler.transform(input_processed)
        
        # 3. Прогнозы по всем моделям
        preds = {}
        for name, model in models.items():
            p = model.predict(X_scaled)
            # Для нейросетей: flatten если нужно
            if hasattr(p, 'ndim') and p.ndim > 1:
                p = p.flatten()
            # Форматируем вывод
            if len(p) == 1:
                preds[name] = round(float(p[0]), 2)
            else:
                preds[name] = {
                    "mean": round(p.mean(), 2),
                    "min": round(p.min(), 2),
                    "max": round(p.max(), 2)
                }
        
        # 4. Отображение результатов
        res_data = []
        for name, val in preds.items():
            if isinstance(val, dict):
                res_data.append({
                    "Модель": name,
                    "Прогноз": f"{val['mean']:.2f}",
                    "Диапазон": f"{val['min']:.2f} – {val['max']:.2f}"
                })
            else:
                res_data.append({
                    "Модель": name,
                    "Прогноз": f"{val:.2f}",
                    "Диапазон": "-"
                })
        
        res_df = pd.DataFrame(res_data)
        
        # Цветовая индикация качества
        def highlight_quality(val):
            try:
                score = float(val)
                if score >= 7:
                    return "background-color: #4CAF50; color: white; font-weight: bold"
                elif score >= 5:
                    return "background-color: #FFC107; color: black"
                else:
                    return "background-color: #F44336; color: white"
            except:
                return ""
        
        st.dataframe(
            res_df.style.applymap(highlight_quality, subset=["Прогноз"]),
            use_container_width=True,
            hide_index=True
        )
        
        # 5. Интерпретация
        st.info("""
        **📊 Шкала качества вина:**
        | Баллы | Категория | Описание |
        |-------|-----------|----------|
        | ≥ 7.0 | 🟢 Premium | Высокое качество, рекомендуется |
        | 5.0–6.9 | 🟡 Standard | Обычное столовое вино |
        | < 5.0 | 🔴 Low | Возможные дефекты, низкое качество |
        """)
        
    except Exception as e:
        st.error(f"❌ Ошибка при прогнозировании: {type(e).__name__}: {e}")
        with st.expander("🔍 Детали ошибки"):
            st.code(traceback.format_exc())