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
        # Cria uma animação de carregamento bonita enquanto o código roda
        with st.spinner("🤖 A Inteligência Artificial está calculando... Aguarde."):
            
            # Prepara o dado no formato que o TensorFlow exige
            dados_entrada = np.array([[tamanho]], dtype=float)
            
            # Faz a previsão
            previsao = modelo.predict(dados_entrada)
            preco_final = previsao[0][0]
            
        # O resultado só aparece quando o cálculo termina, limpando a animação
        st.success(f"🎯 O valor estimado para um imóvel de {tamanho}m² é de **R$ {preco_final:.2f} mil**.")