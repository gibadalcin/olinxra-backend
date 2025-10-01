import torch
import clip
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Carrega o modelo CLIP original do PyTorch
print("Carregando o modelo CLIP original...")
# Carrega o modelo na CPU para garantir a exportação correta
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()

# 2. Exporta o modelo para o formato ONNX (FP32)
onnx_path = "clip_model.onnx"
dummy_image_input = torch.randn(1, 3, 224, 224)
dummy_text_input = torch.randint(0, 100, (1, 77))

# Note: opse_version 14 é um bom padrão
torch.onnx.export(
    model,
    (dummy_image_input, dummy_text_input),
    onnx_path,
    opset_version=14,
    input_names=['image_input', 'text_input'],
    output_names=['output'],
    dynamic_axes={'image_input': {0: 'batch_size'}, 'text_input': {0: 'batch_size'}}
)
print(f"Modelo ONNX FP32 exportado com sucesso para '{onnx_path}'.")

# 3. Quantiza o modelo ONNX usando Quantização Dinâmica (UINT8)
# Esta quantização é mais compatível com o provedor de CPU padrão
print("Iniciando a quantização dinâmica do modelo (UINT8)...")
quantized_model_path = "quantized_clip_model.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    # ALTERADO para UINT8, que é mais compatível com CPUs padrão
    weight_type=QuantType.QUInt8, 
    per_channel=False,
    reduce_range=False,
)
print(f"Modelo quantizado (UINT8) salvo em '{quantized_model_path}'.")
print("Processo concluído.")