import matplotlib.pyplot as plt
import os
import re
import shutil
import ssl
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset = tf.keras.utils.get_file("stack_overflow", url, untar=True, cache_dir='.keras', cache_subdir='')

dataset_dir = os.path.dirname(dataset)

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

binary_vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

int_vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)


def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label


def int_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label


# 生成训练数据集、验证数据集和测试数据集
binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(ds):
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

binary_model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')])

binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'])

history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=10, callbacks=[es_callback])

int_model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True),
    layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
    layers.GlobalMaxPooling1D(),
    layers.Dense(4)
])

int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)

binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
int_loss, int_accuracy = int_model.evaluate(int_test_ds)

print("Binary model accuracy: {:2.2%}".format(binary_accuracy))
print("Int model accuracy: {:2.2%}".format(int_accuracy))

export_model = tf.keras.Sequential(
    [binary_vectorize_layer, binary_model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])


def get_string_labels(predicted_scores_batch):
    predicted_int_labels = tf.math.argmax(predicted_scores_batch, axis=1)
    return tf.gather(raw_train_ds.class_names, predicted_int_labels)


inputs = [
    "how do I extract keys from a dict into a list?",  # 'python'
    "debug public static void main(string[] args) {...}",  # 'java'
]
predicted_scores = export_model.predict(tf.expand_dims(inputs, axis=1))
predicted_labels = get_string_labels(predicted_scores)
for question, label in zip(inputs, predicted_labels):
    print("Question: ", question)
    print("Predicted label: ", label.numpy())
