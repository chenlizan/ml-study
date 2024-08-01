import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

dataset = va_df[
    ['SID', 'TEST', 'ZA', 'NOICM', 'NOICSD', 'L', 'Q', 'PROM', 'EXP', 'EXPSQR', 'EDU', 'TOP', 'RK', 'SEX', 'KEY',
     'SCIE']]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('ZA')
test_labels = test_dataset.pop('ZA')

train_stats = train_dataset.describe().transpose()
print(train_stats)


def norm(x):
    # return (x - train_stats['mean']) / train_stats['std']
    return x


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

model = keras.Sequential([
    layers.Input(shape=train_dataset.shape[1:]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.summary()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [ZA]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$ZA^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

EPOCHS = 1000
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop])

plot_history(history)

test_predictions = model.predict(normed_test_data[:10]).flatten()

print(test_predictions)
print(test_labels[:10])
