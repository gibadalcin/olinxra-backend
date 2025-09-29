import torch
import clip
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Carrega o modelo CLIP original do PyTorch
print("Carregando o modelo CLIP original...")
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()

# 2. Exporta o modelo para o formato ONNX
onnx_path = "clip_model.onnx"
dummy_image_input = torch.randn(1, 3, 224, 224)
dummy_text_input = torch.randint(0, 100, (1, 77))

torch.onnx.export(
    model,
    (dummy_image_input, dummy_text_input),
    onnx_path,
    opset_version=14,
    input_names=['image_input', 'text_input'],
    output_names=['output'],
    dynamic_axes={'image_input': {0: 'batch_size'}, 'text_input': {0: 'batch_size'}}
)
print(f"Modelo ONNX exportado com sucesso para '{onnx_path}'.")

# 3. Quantiza o modelo ONNX
print("Iniciando a quantização do modelo...")
quantized_model_path = "quantized_clip_model.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False,
)
print(f"Modelo quantizado salvo em '{quantized_model_path}'.")
print("Processo concluído.")
