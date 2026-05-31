import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Визуализации", layout="wide")

st.title("Визуализация зависимостей в данных")

df = pd.read_csv("winequality_combined.csv", sep=";")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Распределение целевой переменной (`quality`)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["quality"], bins=8, kde=True, color="purple", ax=ax1)
    ax1.set_xlabel("Quality Score")
    st.pyplot(fig1)

with col2:
    st.subheader("2. Корреляционная матрица признаков")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Качество vs Алкоголь (Boxplot)")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="quality", y="alcohol", data=df, palette="viridis", ax=ax3)
    st.pyplot(fig3)

with col4:
    st.subheader("4. Качество vs Летучая кислотность (Scatter)")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x="volatile acidity", y="quality", data=df, alpha=0.5, color="crimson", ax=ax4)
    ax4.set_xlabel("Volatile Acidity")
    st.pyplot(fig4)