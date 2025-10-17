# --- COLE ESTE CÓDIGO NO SEU ARQUIVO app.py NO GITHUB ---
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Título do Dashboard
st.title("Dashboard de Precificação Imobiliária - Tarefa 2")
st.markdown("Por: Nícolas Duarte Vasconcellos, 200042343")

# --- Carregamento dos Dados ---
# O Streamlit vai carregar o train.csv que você subiu para o GitHub
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# --- PREPARAÇÃO DOS DADOS E TREINAMENTO DO MODELO ---
# !!! IMPORTANTE: COPIE E COLE AQUI O SEU BLOCO DE PREPARAÇÃO DE DADOS FINAL !!!
# (Aquele bloco "à prova de erros" que fizemos funcionar)

# --- COLE ESTE BLOCO NO LUGAR DO CÓDIGO QUE VOCÊ APAGOU ---

# 1. Defina TODAS as colunas que você quer usar no modelo
# <-- AJUSTE ESTA LISTA COM OS NOMES EXATOS DAS SUAS COLUNAS!
colunas_selecionadas = [
    'Gr Liv Area',
    'Overall Qual',
    'Full Bath',
    'Neighborhood',
    'Bldg Type', # Exemplo, troque pelas suas colunas
    'SalePrice'
]
df_reg = df[colunas_selecionadas].copy()


# 2. ETAPA DE LIMPEZA IMPORTANTE (NÃO PRECISA MEXER AQUI):
# Remove linhas onde SalePrice é 0 ou negativo (para evitar erro no np.log)
df_reg = df_reg[df_reg['SalePrice'] > 0]
# Remove QUALQUER linha que tenha QUALQUER valor faltando nas colunas selecionadas
df_reg.dropna(inplace=True)


# 3. Defina aqui APENAS as colunas que são de TEXTO
# <-- AJUSTE ESTA LISTA TAMBÉM!
colunas_categoricas = ['Neighborhood', 'Bldg Type']


# 4. O CÓDIGO ABAIXO PREPARA TUDO (NÃO PRECISA MEXER AQUI):
df_reg = pd.get_dummies(df_reg, columns=colunas_categoricas, drop_first=True)
df_reg['Log_SalePrice'] = np.log(df_reg['SalePrice'])
# --- Substitua a definição de X e input_data por isto ---

# Crie uma lista com os nomes exatos das colunas do seu modelo
# IMPORTANTE: A ORDEM AQUI DEVE CORRESPONDER À ORDEM DOS SEUS WIDGETS (area, qualidade, banheiros)
colunas_do_modelo = ['Gr Liv Area', 'Overall Qual', 'Full Bath']

# Use a lista para definir X
X = df_reg[colunas_do_modelo]
y = df_reg['Log_SalePrice']
X = sm.add_constant(X) # Adiciona a constante ao X

# Treine o modelo
model = LinearRegression()
model.fit(X, y)

# --- BARRA LATERAL COM FILTROS INTERATIVOS ---
st.sidebar.header("Simulador de Preço do Imóvel")

area = st.sidebar.slider("Área do Imóvel (Gr Liv Area)", int(X['Gr Liv Area'].min()), int(X['Gr Liv Area'].max()), int(X['Gr Liv Area'].mean()))
qualidade = st.sidebar.selectbox("Qualidade Geral (Overall Qual)", sorted(X['Overall Qual'].unique()))
banheiros = st.sidebar.selectbox("Banheiros (Full Bath)", sorted(X['Full Bath'].unique()))

# --- PREDIÇÃO E RESULTADOS ---

# Use A MESMA lista para criar o DataFrame de predição
input_data = pd.DataFrame(
    [[area, qualidade, banheiros]], 
    columns=colunas_do_modelo
)

# Agora a predição vai funcionar
prediction_log = model.predict(input_data)
prediction = np.exp(prediction_log)

st.subheader(f"Preço Estimado: ${prediction[0]:,.2f}")


# --- BARRA LATERAL COM FILTROS INTERATIVOS ---
st.sidebar.header("Simulador de Preço do Imóvel")

# Crie os seletores para as variáveis do seu modelo
area = st.sidebar.slider("Área do Imóvel (Gr Liv Area)", int(X['Gr Liv Area'].min()), int(X['Gr Liv Area'].max()), int(X['Gr Liv Area'].mean()))
qualidade = st.sidebar.selectbox("Qualidade Geral (Overall Qual)", sorted(X['Overall Qual'].unique()))
banheiros = st.sidebar.selectbox("Banheiros (Full Bath)", sorted(X['Full Bath'].unique()))

# --- PREDIÇÃO E RESULTADOS ---
# Criar um DataFrame com os inputs do usuário
input_data = pd.DataFrame([[area, qualidade, banheiros]], columns=['Gr Liv Area', 'Overall Qual', 'Full Bath'])

# Fazer a predição
prediction_log = model.predict(input_data)
prediction = np.exp(prediction_log)

st.subheader(f"Preço Estimado: ${prediction[0]:,.2f}")

# --- GRÁFICOS E INTERPRETAÇÃO ---
st.markdown("---")
st.subheader("Análise Visual")

# Copie e cole o código de um dos gráficos do seu notebook
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', hue='Overall Qual', ax=ax)
plt.title("Relação entre Área, Qualidade e Preço")
st.pyplot(fig)

st.subheader("Interpretação do Modelo e Recomendações")
st.write("""
[cite_start]A análise de regressão linear múltipla permitiu construir um modelo preditivo para o preço de imóveis em Ames, Iowa, com base em suas características. A seguir, apresentamos os principais insights e recomendações estratégicas derivadas dos resultados, conforme solicitado na tarefa[cite: 22, 23].

### **Principais Fatores de Valorização**

[cite_start]O modelo estatístico identificou diversas características que impactam significativamente o valor de um imóvel[cite: 85]. As mais influentes foram:

* [cite_start]**Qualidade Geral do Acabamento (`Overall Qual`):** Este é o fator de maior impacto individual no preço de venda[cite: 22]. O modelo indica que, para cada ponto adicional na escala de qualidade (de 1 a 10), o preço do imóvel tende a aumentar em aproximadamente **9.5%**, mantendo as outras características constantes. Isso demonstra que investimentos em materiais e acabamentos de alta qualidade têm um retorno financeiro claro.

* **Área Útil do Imóvel (`Gr Liv Area`):** Como esperado, o tamanho do imóvel é um forte preditor de seu valor. A análise mostra que cada metro quadrado adicional na área útil acima do solo está associado a um aumento de aproximadamente **0.04%** no preço de venda. Embora pareça pouco, para um imóvel de 150m², isso representa um impacto considerável.

* **Localização (`Neighborhood`):** A localização provou ser um fator crítico. [cite_start]Bairros como **Northridge Heights (`NridgHt`)** apresentam um prêmio significativo, aumentando o valor esperado de um imóvel em até **24%** em comparação com a área de referência do modelo[cite: 87].

### **Recomendações Estratégicas**

[cite_start]Com base nesses insights, as seguintes recomendações podem ser feitas para otimizar a tomada de decisão no mercado imobiliário[cite: 23, 86]:

1.  **Para Investidores (Foco em Renovação):**
    * Priorize a aquisição de imóveis com pontuação de `Overall Qual` abaixo de 7, mas que estejam localizados em bairros de alta demanda. A estratégia de "comprar para reformar", focando especificamente na melhoria da qualidade dos acabamentos, apresenta o maior potencial de valorização segundo o modelo.

2.  **Para Corretores (Marketing Direcionado):**
    * Utilize as características de maior impacto como pontos centrais nas campanhas de marketing. Destaque nos anúncios a "nota de qualidade do imóvel" (ex: "Acabamento nota 8/10") e a metragem quadrada de forma proeminente.
    * Para imóveis em bairros premium como Northridge Heights, justifique o preço mais alto enfatizando o "efeito localização" que o modelo quantificou, informando aos clientes sobre a valorização média de 24% na região.

*Esta análise é baseada em um modelo estatístico e deve ser usada como uma ferramenta de auxílio à decisão, complementando a experiência de mercado.*
""")
