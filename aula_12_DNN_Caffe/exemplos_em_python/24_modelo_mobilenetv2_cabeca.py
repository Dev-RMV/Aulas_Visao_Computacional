from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 24 - Construir MobileNetV2 congelada + cabeca Dense/Dropout.
"""
from utils.tf_utils import build_mobilenetv2_gender
model = build_mobilenetv2_gender(input_shape=(160,160,3), dropout=0.3)
model.summary()
print("Camadas treinaveis:", sum(1 for l in model.layers if l.trainable))
print("Parametros treinaveis:", sum(w.numpy().size for w in model.trainable_weights))
