"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from dnn_utils import MODELS, ensure_dirs

ensure_dirs()
model = MobileNetV2(weights='imagenet')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite_model = converter.convert()

out = MODELS / 'mobilenetv2_imagenet.tflite'
out.write_bytes(tflite_model)
print('Modelo TensorFlow Lite salvo em:', out)
print('Tamanho aproximado [MB]:', out.stat().st_size / (1024*1024))

# DESAFIO DO ALUNO:
# Ative converter.optimizations = [tf.lite.Optimize.DEFAULT]
# e compare o tamanho do arquivo gerado.
