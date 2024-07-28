# https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

batch_size = 32
img_height = 180
img_width = 180

# ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

model = tf.keras.models.load_model('.keras/preprocess_images_model.keras')
model.summary()

image_raw = tf.io.read_file("dandelion.jpg")
image = tf.image.decode_jpeg(image_raw, channels=3)
image = tf.image.resize(image, [img_width, img_height])

predictions = model.predict(tf.expand_dims(image, axis=0))
probabilities = tf.nn.softmax(predictions, axis=1)
predicted_class_index = np.argmax(probabilities[0])
predicted_class_name = class_names[predicted_class_index]
print(predicted_class_name)

plt.figure(figsize=(20, 20))
plt.imshow(image.numpy().astype("uint8"))
plt.axis("off")
plt.show()

tfjs.converters.save_keras_model(model, ".keras")
