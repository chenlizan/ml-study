import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 定义预测列
predict_col = 'ZA'

va_df = pd.read_csv('FULL_3_LONG.csv',
                    usecols=[
                        'SID',  # 学生ID
                        'TEST',  # 考试序号
                        'ZA',  # 每次考试归一化成绩
                        'NOICM',  # 全班排除自己的平均成绩
                        'NOICSD',  # 全班排除自己的成绩标准差
                        'L',  # 所在班级低分人
                        'Q',  # 所在班的区分度
                        'PROM',  # 老师上次评职称到现在的间隔年限
                        'EXP',  # 老师工作年限
                        'EXPSQR',  # 老师工作年限平方项
                        'EDU',  # 老师教育年限
                        'TOP',  # 老师毕业学校性质 985、211(1)
                        'RK',  # 老师是否有职称
                        'SEX',  # 老师性别 女(0)
                        'CG',  # 高一高二换过老师 换过(1）
                        'SIZE',  # 班上的学生数
                        'KEY',  # 重点班
                        'SCIE',  # 理科班 （1）
                        'CHOSE',  # 科目选择 1物化生  2地物化 3 政物生  4 史地生 5 政史地
                        'SBJ',  # 1 语文 2 数学 3 外语 4 政治 5 历史  6 地理  7 物理  8 化学  9 生物
                        'LANG',  # 外语语种： 1.英语；2.日语；3.俄语
                        'TEST',  # 考试次数，0是高一的分班考试，1-5是高二的月考，其中2 3 5是区统考，其他的是校统考
                        'GRADE',  # 年级，1.一年级 2.二年级
                        'CLASS',  # 班级，高二重新分班，与高一的班级不一样，用GRADE CLASS两个变量才能识别
                        'PID',  # SID-TEST 二维面板识别码
                        'CID',  # GRADE-CLASS二维组合
                        'DF',  # 以第一次考试为基线的难度系数
                        'A',  # 考试成绩的原始分，经过难度系数调整后的值
                        'Q',  # 考卷的区分度
                        'UNIV',  # 教师的毕业院校
                        'M',  # 每科每次考试的年级均值
                        'SD',  # 每科每次考试的年级标准差，包括自己在内
                    ])

# 删除所有含有缺失值的行
va_df = va_df.dropna(
    subset=['SID', 'TEST', 'ZA', 'NOICM', 'NOICSD', 'L', 'Q', 'PROM', 'EXP', 'EXPSQR', 'EDU', 'TOP', 'RK', 'SEX', 'KEY',
            'SCIE'])

# 获取需要的数据列
dataset = va_df[
    ['SID', 'TEST', 'ZA', 'NOICM', 'NOICSD', 'L', 'Q', 'PROM', 'EXP', 'EXPSQR', 'EDU', 'TOP', 'RK', 'SEX', 'KEY',
     'SCIE']]

# 数据拆分为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 拆分训练集和测试集的预测值
train_labels = train_dataset.pop(predict_col)
test_labels = test_dataset.pop(predict_col)

# 打印训练集总体统计数据
train_stats = train_dataset.describe().transpose()
print(train_stats)

# 初始化特征规范化层
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))
print(normalizer.mean.numpy())

# 定义EarlyStopping回调
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)


# 定义超参模型
def model_builder(hp):
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)

    model = keras.Sequential([
        normalizer,
        layers.Dense(units=hp_units, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=['mae'])
    return model


# 实例化超参调谐器
tuner = kt.Hyperband(model_builder,
                     objective='val_mae',
                     max_epochs=10,
                     directory='.keras',
                     project_name='va_final_regression')

# 运行调谐器
tuner.search(train_dataset, train_labels, epochs=100, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
超参搜索完成.
最佳输出维度 {best_hps.get('units')}
最佳学习率 {best_hps.get('learning_rate')}.
""")

# 创建顺序模型
dnn_model = keras.Sequential([
    normalizer,
    layers.Dense(units=best_hps.get('units'), activation='relu'),
    layers.Dense(1)
])

# 编译模型
dnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=['mae'])

# 打印模型概述
dnn_model.summary()

# 训练模型
dnn_history = dnn_model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    epochs=100,
    callbacks=[stop_early])

# 回归模型的性能评估指标(MAE)
print('evaluate:', dnn_model.evaluate(test_dataset, test_labels))


# 绘制训练过程中损失(loss)和验证损失(val_loss)的图表
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0.6, 0.8])
    plt.xlabel('Epoch')
    plt.ylabel(f'Error [{predict_col}]')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(dnn_history)

# 绘制散点图和线性
test_predictions = dnn_model.predict(test_dataset).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel(f'True Values [{predict_col}]')
plt.ylabel(f'Predictions [{predict_col}]')
lims = [0, 4]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

# 绘制错误分布图
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel(f'Prediction Error [{predict_col}]')
plt.ylabel('Count')
plt.show()
