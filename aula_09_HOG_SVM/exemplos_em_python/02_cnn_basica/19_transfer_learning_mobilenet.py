import tensorflow as tf
from tensorflow.keras import layers,Model
base=tf.keras.applications.MobileNetV2(input_shape=(160,160,3),include_top=False,weights='imagenet'); base.trainable=False
i=layers.Input((160,160,3)); x=tf.keras.applications.mobilenet_v2.preprocess_input(i); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); o=layers.Dense(2,activation='softmax')(x); m=Model(i,o); m.summary()
