# https://www.tensorflow.org/tutorials/load_data/csv?hl=zh-cn
import ssl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)
ssl._create_default_https_context = ssl._create_unverified_context

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()

# 分离特征和标签
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# 数据预处理（这里仅展示标准化，根据你的数据可能需要其他预处理）
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(abalone_features)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, abalone_labels, test_size=0.2, random_state=42)

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)  # 输出层，1个节点（年龄）
])

# 编译模型
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

# 打印模型结构
model.summary()

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True)

# 训练模型
model.fit(X_train, y_train, epochs=30, callbacks=[es_callback])

# 评估模型
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 使用模型进行预测
predictions = model.predict(X_test)
