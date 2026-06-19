import streamlit as st
import tensorflow as tf
import numpy as np

# Configuração visual da página
st.set_page_config(page_title="Preditor de Imóveis IA", page_icon="🏠")

st.title("🏠 IA Preditora de Preço de Imóveis")
st.write("Insira o tamanho do imóvel em metros quadrados para que a IA do TensorFlow calcule o preço estimado.")

# Função para carregar o modelo de forma otimizada (só carrega uma vez na memória)
@st.cache_resource
def carregar_modelo_ia():
    return tf.keras.models.load_model('modelo_imoveis.keras')

# Carrega o modelo criado no passo 1
try:
    modelo = carregar_modelo_ia()
    
    # Campo de entrada para o usuário digitar o tamanho
    tamanho = st.number_input("Tamanho do imóvel (m²):", min_value=10.0, max_value=500.0, value=75.0, step=5.0)

    # Botão para ativar a previsão
    if st.button("Calcular Preço Estimado"):
        # Prepara o dado no formato que o TensorFlow exige (Array de 2 dimensões)
        dados_entrada = np.array([[tamanho]], dtype=float)
        
        # Faz a previsão
        previsao = modelo.predict(dados_entrada)
        preco_final = previsao[0][0]
        
        # Exibe o resultado de forma destacada
        st.success(f"🎯 O valor estimado para um imóvel de {tamanho}m² é de **R$ {preco_final:.2f} mil**.")

except Exception as e:
    st.error("Erro: O arquivo 'modelo_imoveis.keras' não foi encontrado. Certifique-se de rodar o script 'treinar_modelo.py' primeiro!")