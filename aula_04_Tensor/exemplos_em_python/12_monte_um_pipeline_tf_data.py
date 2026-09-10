import tensorflow as tf

arquivos = tf.data.Dataset.from_tensor_slices(
    ["assets/carro_real.png"]
)

def carregar(caminho):
    dados = tf.io.read_file(caminho)
    imagem = tf.io.decode_png(dados, channels=3)
    imagem = tf.image.resize(imagem, (224, 224))
    return imagem / 255.0

dataset = arquivos.map(carregar).batch(1)
for lote in dataset.take(1):
    print(lote.shape, lote.dtype)
