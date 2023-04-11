import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 数据准备
data, sr = librosa.load('heart_sound.wav', sr=2000)
# 进行短时傅里叶变换并取绝对值
stft = np.abs(librosa.stft(data, n_fft=512, hop_length=256))
# 进行标准化
scaler = StandardScaler()
stft = scaler.fit_transform(stft.T).T
# 将数据转为PyTorch tensor，并添加一个维度作为batch维
x = torch.tensor(stft[np.newaxis, :, :], dtype=torch.float32)

# 定义模型和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM(input_size=257, hidden_size=128, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 进行训练
num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(x.to(device))
    labels = torch.tensor([0]).to(device)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 进行预测
with torch.no_grad():
    outputs = model(x.to(device))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted class:', predicted.item())