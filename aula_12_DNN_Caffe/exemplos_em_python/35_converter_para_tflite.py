from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 35 - Converter MobileNetV2 fine-tuned para TensorFlow Lite.
Opcional para sistemas embarcados.
"""
import tensorflow as tf
from pathlib import Path
model_path = Path("resultados/25_mobilenetv2_genero.keras")
if not model_path.exists():
    print("Treine o modelo no Exemplo 25 antes de converter.")
    raise SystemExit
model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
out = Path("resultados/35_genero_mobilenetv2.tflite")
out.write_bytes(tflite_model)
print("Modelo TFLite salvo em", out)
print(f"Tamanho TFLite: {out.stat().st_size / (1024*1024):.2f} MB")
