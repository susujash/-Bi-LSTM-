import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model

# 读取数据
data = pd.read_csv("heart_sound_database.csv")

# 降噪
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

cutoff = 1000  # 降噪截止频率
fs = 2000  # 采样率
data['signal_filtered'] = butter_lowpass_filter(data['signal'], cutoff, fs)

# 包络提取及分段截取
segment_len = 1024  # 每个分段的长度
segments = []
for i in range(0, len(data['signal_filtered']) - segment_len, segment_len // 2):
    segment = data['signal_filtered'][i:i+segment_len]
    envelope = np.abs(hilbert(segment))
    segments.append(envelope)
segments = np.array(segments)

# 特征提取
X = []
for segment in segments:
    feature = [np.mean(segment), np.var(segment)]
    X.append(feature)
X = np.array(X)

# 构造标签
y = data['label'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
input_layer = Input(shape=(X_train.shape[1],))
x = Bidirectional(LSTM(32, return_sequences=True))(input_layer)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.2)(x)
output_layer = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# 评估模型
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f%%' % (accuracy*100))


#在运行以上代码前，需要先将本地的先心病杂音数据库放入同一目录下，并将其命名为 heart_sound_database.csv。该数据库应该包含以下两列数据：signal（包含心脏杂音信号的原始数据）和 label（指示该数据属于先心病或非先心病的标签数据）。

#以上代码将读取数据库中的数据，并按照先进行降


#以下是实现从本地导入先心病杂音数据库进行分类识别的 Python 代码所需要用到的库：

#numpy：用于进行数组运算。
#pandas：用于读取 CSV 格式的数据库文件。
#tensorflow：用于构建和训练深度学习模型。
#scipy：用于实现信号处理中的滤波和包络提取功能。
#sklearn：用于数据集的划分和评估。