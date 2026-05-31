import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Прогнозирование",
    layout="wide"
)

st.title("Инференс моделей ML")


@st.cache_resource
def load_all():
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    catboost_model = CatBoostRegressor()
    catboost_model.load_model("models/catboost.cbm")

    models = {
        "Ridge": joblib.load("models/ridge.pkl"),
        "GradientBoosting": joblib.load("models/gradient_boosting.pkl"),
        "Bagging": joblib.load("models/bagging.pkl"),
        "Stacking": joblib.load("models/stacking.pkl"),
        "CatBoost": catboost_model,
        "MLPRegressor": joblib.load("models/mlp.pkl"),
    }

    return scaler, label_encoder, models


scaler, label_encoder, models = load_all()

features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type"
]

input_mode = st.radio(
    "Выберите способ ввода данных:",
    ["Загрузить CSV", "Ручной ввод"]
)

input_df = None

if input_mode == "Загрузить CSV":

    uploaded = st.file_uploader(
        "Загрузите CSV-файл",
        type=["csv"]
    )

    if uploaded:

        try:
            df_up = pd.read_csv(uploaded, sep=";")

            missing = [col for col in features if col not in df_up.columns]

            if missing:
                st.error(f"В файле отсутствуют колонки: {missing}")
                st.stop()

            input_df = df_up[features].copy()

        # ====== обработка wine_type ======
            input_df["wine_type"] = (
                input_df["wine_type"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            input_df["wine_type"] = input_df["wine_type"].replace({
                "red": 0,
                "white": 1
            })

        # ====== обработка числовых колонок ======
            numeric_cols = [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol"
            ]

            for col in numeric_cols:
                input_df[col] = (
                    input_df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                )

                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce"
                )

        # ====== проверка wine_type ======
            if not input_df["wine_type"].isin([0, 1]).all():

                st.error(
                    f"Некорректные значения wine_type: "
                    f"{input_df['wine_type'].unique()}"
                )
                st.stop()

        # ====== проверка пропусков ======
            if input_df.isnull().any().any():

                st.error(
                    "После обработки появились NaN значения."
                )

                st.write(
                    input_df[input_df.isnull().any(axis=1)]
                )

                st.stop()

            st.success("Файл успешно загружен")
            st.dataframe(input_df.head())

            st.write("Типы данных после обработки:")
            st.write(input_df.dtypes)

        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")

else:

    st.markdown(
        "Введите значения признаков:"
    )

    cols = st.columns(3)

    data = {}

    for i, feature in enumerate(features):

        with cols[i % 3]:

            if feature == "wine_type":

                value = st.selectbox(
                    "wine_type",
                    [0, 1],
                    format_func=lambda x:
                    "Red" if x == 0 else "White"
                )

            else:

                value = st.number_input(
                    feature,
                    value=0.0,
                    step=0.01,
                    format="%.4f"
                )

            data[feature] = value

    if st.button("Сформировать выборку"):

        input_df = pd.DataFrame([data])

        st.success(
            "Данные готовы к прогнозу"
        )

        st.dataframe(input_df)

if input_df is not None:

    try:

        input_df = input_df.reindex(
            columns=features
        )

        X_scaled = scaler.transform(
            input_df
        )

        st.divider()
        st.subheader(
            "Результаты предсказания"
        )

        predictions = {}

        for name, model in models.items():

            pred = model.predict(
                X_scaled
            )

            pred = np.asarray(pred)

            if pred.ndim > 1:
                pred = pred.flatten()

            if len(pred) == 1:

                predictions[name] = round(
                    float(pred[0]),
                    2
                )

            else:

                predictions[name] = (
                    f"Среднее: {pred.mean():.2f} "
                    f"(мин: {pred.min():.2f}, "
                    f"макс: {pred.max():.2f})"
                )

        result_df = pd.DataFrame(
            list(predictions.items()),
            columns=[
                "Модель",
                "Прогноз качества"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.info(
            """
            Интерпретация результата:

            • ≥ 7.0 — Высокое качество (Premium)

            • 5.0 – 6.9 — Среднее качество (Table Wine)

            • < 5.0 — Низкое качество (Possible defects)
            """
        )

    except Exception as e:

        st.error(
            f"Ошибка при прогнозировании: {e}"
        )

        with st.expander(
            "Отладочная информация"
        ):
            st.write("Типы данных:")
            st.write(input_df.dtypes)

            st.write("Первые строки:")
            st.dataframe(
                input_df.head()
            )