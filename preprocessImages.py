import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from PIL import features
from keras.src.callbacks import EarlyStopping

print(tf.__version__)
# print(features.pilinfo())

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                    fname=os.getcwd() + '/.keras/flower_photos',
#                                    untar=True)
# data_dir = pathlib.Path(data_dir)

data_dir = pathlib.Path('.keras/flower_photos')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

input_shape = (180, 180, 3)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=input_shape),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# 打印模型结构
model.summary()

es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[es_callback]
)

for images, labels in val_ds.take(1):  # 只取一个批次

    predictions = model.predict(images[:1])
    print(labels[:1].numpy())

    probabilities = tf.nn.softmax(predictions, axis=1)

    # 每个类别预测概率的数组
    print(f'predict: {predictions[0]}')

    # 获取最有可能的类别的索引
    predicted_class_index = np.argmax(probabilities[0])

    # 将索引转换为类别名称
    predicted_class_name = class_names[predicted_class_index]
    print(predicted_class_name)

    plt.figure(figsize=(20, 20))
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(predicted_class_name)
    plt.axis("off")
    plt.show()
