# --- CÓDIGO FINAL E COMPLETO PARA O app.py ---

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- TÍTULO ---
st.title("Dashboard de Precificação Imobiliária - Tarefa 2 Bônus")
st.markdown("Por: Nícolas Duarte Vasconcellos, 200042343")

# --- CONFIGURAÇÃO CENTRAL (COM NOMES 100% CORRETOS DO ARQUIVO) ---
# Esta lista é a "fonte da verdade", agora baseada nos nomes exatos do seu CSV.
FEATURES = ['Gr Liv Area', 'Overall Qual', 'Full Bath']
TARGET = 'SalePrice'

# --- CARREGAMENTO, LIMPEZA E TREINAMENTO ---
@st.cache_data
def load_and_train_model(features_list, target_name):
    # Carregando o dataset que você subiu para o GitHub
    df = pd.read_csv("train.csv")
    
    # Seleciona apenas as colunas necessárias desde o início
    colunas_necessarias = features_list + [target_name]
    df_reg = df[colunas_necessarias].copy()
    
    # Limpeza robusta
    df_reg = df_reg[df_reg[target_name] > 0]
    df_reg.dropna(inplace=True)
    
    # Preparação final e treinamento
    df_reg['Log_SalePrice'] = np.log(df_reg[target_name])
    X = df_reg[features_list]
    y = df_reg['Log_SalePrice']
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df_reg

# Executa a função para carregar e treinar
model, df_reg = load_and_train_model(FEATURES, TARGET)

# --- BARRA LATERAL COM FILTROS INTERATIVOS ---
st.sidebar.header("Simulador de Preço do Imóvel")

# Os widgets usam os nomes corretos das colunas
area = st.sidebar.slider("Área do Imóvel (Gr Liv Area)", int(df_reg['Gr Liv Area'].min()), int(df_reg['Gr Liv Area'].max()), int(df_reg['Gr Liv Area'].mean()))
qualidade = st.sidebar.selectbox("Qualidade Geral (Overall Qual)", sorted(df_reg['Overall Qual'].unique()))
banheiros = st.sidebar.selectbox("Banheiros (Full Bath)", sorted(df_reg['Full Bath'].unique()))

# --- PREDIÇÃO E RESULTADOS ---
input_data = pd.DataFrame([[area, qualidade, banheiros]], columns=FEATURES)
prediction_log = model.predict(input_data)
prediction = np.exp(prediction_log)

st.subheader(f"Preço Estimado: ${prediction[0]:,.2f}")

# --- GRÁFICOS E INTERPRETAÇÃO ---
st.markdown("---")
st.subheader("Análise Visual")
sns.scatterplot(data=df_reg, x='Gr Liv Area', y='SalePrice', hue='Overall Qual')
plt.title("Relação entre Área, Qualidade e Preço")
st.pyplot()

st.subheader("Interpretação do Modelo e Recomendações")
st.write("""
(Cole aqui o seu texto de interpretação final que preenchemos)
""")
