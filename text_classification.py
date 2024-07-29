# https://www.tensorflow.org/tutorials/keras/text_classification?hl=zh-cn
import matplotlib.pyplot as plt
import os
import re
import shutil
import ssl
import string
import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.keras', cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# 移除无标签数据
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    '.keras/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    '.keras/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    '.keras/aclImdb/test',
    batch_size=batch_size)


# 自定义标准化函数来移除HTML
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
# 调用 adapt 方法以构建词汇表
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# 生成训练数据集、验证数据集和测试数据集
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


def build_model(hp):
    hp_embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=512, step=16)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = keras.Sequential()
    model.add(layers.Embedding(max_features, hp_embedding_dim))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.1)])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_binary_accuracy',
                     max_epochs=10,
                     directory='.keras',
                     project_name='text_classification')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_ds, validation_data=val_ds, epochs=50, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
The optimal number of embedding vector dimension is {best_hps.get('embedding_dim')}.
""")

model = tf.keras.Sequential([layers.Embedding(max_features + 1, best_hps.get('embedding_dim')),
                             layers.GlobalAveragePooling1D(),
                             layers.Dropout(0.2),
                             layers.Dense(1)])

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.1)])

epochs = 10
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[stop_early])

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

export_model = tf.keras.Sequential([vectorize_layer,
                                    model,
                                    layers.Activation('sigmoid')
                                    ])

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

examples = ["The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
            ]

print(export_model.predict(tf.expand_dims(examples, axis=1)))
