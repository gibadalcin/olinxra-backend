import torch
import clip
from onnxruntime.quantization import quantize_dynamic, QuantType

# Wrapper para exportar apenas o encoder de imagem
class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model.encode_image(x)

print("Carregando o modelo CLIP original...")
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()
image_encoder = CLIPImageEncoder(model)

onnx_path = "clip_image_encoder.onnx"
dummy_image_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    image_encoder,
    dummy_image_input,
    onnx_path,
    opset_version=14,
    input_names=['image_input'],
    output_names=['image_features'],
    dynamic_axes={'image_input': {0: 'batch_size'}}
)
print(f"Encoder de imagem exportado com sucesso para '{onnx_path}'.")

print("Iniciando a quantização dinâmica do encoder de imagem (UINT8)...")
quantized_model_path = "quantized_clip_model.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    reduce_range=False,
)
print(f"Encoder quantizado (UINT8) salvo em '{quantized_model_path}'.")
print("Processo concluído.")