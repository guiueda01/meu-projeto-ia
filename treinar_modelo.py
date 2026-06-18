import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. Definindo o Problema e os Dados...")
# Dados de exemplo: Tamanho do imóvel (m²) e Preço (em milhares de R$)
# Exemplo: 50m² = 150 mil, 60m² = 180 mil...
X_treino = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=float)
y_treino = np.array([150.0, 180.0, 210.0, 240.0, 270.0, 300.0], dtype=float)

print("2. Construindo a arquitetura do modelo TensorFlow...")
modelo = models.Sequential([
    layers.Dense(units=1, input_shape=[1])  # Uma camada simples para relação linear
])

# Compilando o modelo com otimizador Adam e função de perda de erro quadrático médio
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

print("3. Treinando a Inteligência Artificial...")
# Treina o modelo por 500 épocas (iterações)
modelo.fit(X_treino, y_treino, epochs=500, verbose=0)

print("4. Salvando o modelo treinado...")
# Exporta o cérebro da IA para ser usado na interface web
modelo.save('modelo_imoveis.keras')
print("Sucesso! O arquivo 'modelo_imoveis.keras' foi gerado.")