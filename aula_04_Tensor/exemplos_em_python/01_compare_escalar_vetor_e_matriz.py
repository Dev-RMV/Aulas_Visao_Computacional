import tensorflow as tf

itens = {
    "escalar": tf.constant(7),
    "vetor": tf.constant([10, 20, 30]),
    "matriz": tf.constant([[1, 2], [3, 4]]),
}

for nome, tensor in itens.items():
    print(nome, "shape=", tensor.shape,
          "rank=", tf.rank(tensor).numpy(),
          "dtype=", tensor.dtype)
