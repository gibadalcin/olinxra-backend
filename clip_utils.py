import os
import onnxruntime as ort
import numpy as np
from PIL import Image

# O caminho do seu modelo ONNX quantizado.
# Certifique-se de que este arquivo foi adicionado ao seu projeto e ao deploy.
QUANTIZED_MODEL_PATH = "quantized_clip_model.onnx"

# Carregamos a sessão do ONNX Runtime uma única vez na inicialização do servidor.
# O try/except é para lidar com o caso onde o arquivo não existe
# e evitar que a aplicação quebre durante o startup.
def initialize_onnx_session():
    try:
        ort_session = ort.InferenceSession(QUANTIZED_MODEL_PATH)
        return ort_session
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo ONNX: {e}. Verifique o arquivo '{QUANTIZED_MODEL_PATH}'.")

# A sessão é inicializada na função startup_event no main.py
# e passada como um argumento para a função de extração
def extract_clip_features(image_path, ort_session):
    # O pré-processamento manual da imagem é necessário para o ONNX.
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1)) # Transforma para o formato (C, H, W)
    
    # O ONNX espera uma dimensão de batch.
    ort_input = {ort_session.get_inputs()[0].name: np.expand_dims(image_np, axis=0)}
    
    # Executa a inferência no modelo ONNX
    features = ort_session.run(None, ort_input)
    
    # Retorna o vetor de features
    return features[0].flatten().tolist()