import streamlit as st
import pandas as pd

st.set_page_config(page_title="О датасете", layout="wide")

st.title("📊 Описание набора данных Wine Quality")

st.markdown("""
### Предметная область
Датасет содержит результаты физико-химических анализов вин (красных и белых) 
и субъективные оценки их качества от экспертов (шкала от 0 до 10).

### Признаки (Features)
| Признак | Описание |
|---|---|
| `fixed acidity` | Фиксированная кислотность (г/дм³) |
| `volatile acidity` | Летучая кислотность (г/дм³) |
| `citric acid` | Содержание лимонной кислоты (г/дм³) |
| `residual sugar` | Остаточный сахар (г/дм³) |
| `chlorides` | Хлориды (г/дм³) |
| `free sulfur dioxide` | Свободный SO₂ (мг/дм³) |
| `total sulfur dioxide` | Общий SO₂ (мг/дм³) |
| `density` | Плотность (г/см³) |
| `pH` | Водородный показатель |
| `sulphates` | Сульфаты (г/дм³) |
| `alcohol` | Содержание алкоголя (% об.) |
| `wine_type` | Тип вина: 0 = Red, 1 = White |
""")

st.divider()
st.header("Предобработка и EDA")
st.info("""
1. **Загрузка**: объединение двух файлов (red & white) с добавлением колонки `wine_type`.
2. **Пропуски**: отсутствуют (датасет очищен изначально).
3. **Кодирование**: `LabelEncoder` для `wine_type` (red→0, white→1).
4. **Масштабирование**: `StandardScaler` применён ко всем числовым признакам 
   (критично для Ridge, FCNN и дистанционных метрик).
5. **Разделение**: `train_test_split(test_size=0.2, random_state=42)`.
""")

df_raw = pd.read_csv("winequality_combined.csv", sep=";")
st.subheader("👀 Первые 5 записей датасета")
st.dataframe(df_raw.head(), use_container_width=True)