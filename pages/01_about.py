import streamlit as st

st.set_page_config(page_title="О разработчике", layout="wide")

st.title("Информация о разработчике")

col1, col2 = st.columns([1, 3])
with col1:
    st.image("photo.jpg", width=250)
with col2:
    st.header("ФИО: Шкурин Михаил Максимович")
    st.header("Группа: ФИТ-242") 
    st.divider()
    st.subheader("Тема РГР:")
    st.success("Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")
    st.subheader("Стек технологий:")
    st.code("Python 3.10+ | Streamlit | Scikit-learn | CatBoost | TensorFlow | Pandas | Matplotlib | Seaborn")