import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pyvqnet import kint64, kfloat32
from pyvqnet.tensor import QTensor, zeros, ones, reshape, max
from pyvqnet.nn import Module, Conv1D, MaxPool1D, Linear, ReLu, Dropout, Softmax, CrossEntropyLoss
from pyvqnet.optim import Adam
from pyvqnet.data import data_generator

# 数据加载和预处理
def load_and_preprocess_data(train_path, test_path):
    # 加载数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 2. 统计信息
    print("\n== 训练集统计信息 ==")
    print(train_data.describe().T)
    print("\n== 测试集统计信息 ==")
    print(test_data.describe().T)

    # 3. 缺失值情况检测
    missing1 = train_data.isnull().sum()
    missing2 = test_data.isnull().sum()
    print("\n== 缺失值情况 ==")
    print(missing1)
    print(missing2)
    if missing1.sum() == 0 & missing2.sum() == 0:
        print("\n训练集无缺失值情况出现")
    elif missing1.sum() > 0 & missing2.sum() == 0:
        print("\n== 缺失值情况 ==")
        print(missing1)
    elif missing1.sum() == 0 & missing2.sum() > 0:
        print("\n== 缺失值情况 ==")
        print(missing2)
    else:
        print("\n== 缺失值情况 ==")
        print(missing1)
        print(missing2)

    # 5. 将train除 Temperature 外的负值设为 0
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    non_temp_cols = [c for c in numeric_cols if c != 'Temperature']
    train_data[non_temp_cols] = train_data[non_temp_cols].clip(lower=0)
    print("\n已将除 Temperature 外的负值设置为 0。")

    # 6. 将 Humidity 大于 100 的值设为 100, in train_data
    if 'Humidity' in train_data.columns:
        train_data['Humidity'] = train_data['Humidity'].clip(upper=100)
        print("已将 Humidity 超过 100 的值设置为 100。")

    # 5. 将test除 Temperature 外的负值设为 0
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
    non_temp_cols = [c for c in numeric_cols if c != 'Temperature']
    test_data[non_temp_cols] = test_data[non_temp_cols].clip(lower=0)
    print("\n已将除 Temperature 外的负值设置为 0。")

    # 6. 将 Humidity 大于 100 的值设为 100, in test_data
    if 'Humidity' in test_data.columns:
        test_data['Humidity'] = test_data['Humidity'].clip(upper=100)
        print("已将 Humidity 超过 100 的值设置为 100。")
    
    # 分离特征和标签
    X_train = train_data.drop('Air Quality', axis=1)
    y_train = train_data['Air Quality']
    X_test = test_data.drop('Air Quality', axis=1)
    y_test = test_data['Air Quality']
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 编码标签
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    return X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, encoder.classes_

# 定义神经网络模型, 使用一维卷积神经网络
class AirQualityCNN(Module):
    def __init__(self, input_dim, output_dim):
        super(AirQualityCNN, self).__init__()
        self.conv1 = Conv1D(1, 4, kernel_size=2, padding=0, dtype=kfloat32)
        self.pool = MaxPool1D([2],[2], "valid")
        self.fc1 = Linear(4 * ((input_dim - 2 + 1) // 2), 8, dtype=kfloat32)  # 4 * ((input_dim - 2 + 1) // 2) 是卷积和池化后的特征数量
        self.fc2 = Linear(8, output_dim, dtype=kfloat32)
        self.relu = ReLu()

    def forward(self, x):
        x = reshape(x,(x.shape[0], 1, x.shape[1]))  # 转换为1D卷积输入格式
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x) # 不需要在这里使用softmax，因为CrossEntropyLoss会自动处理softmax
        return x

# 计算模型参数数量，自动加和
def count_parameters(model):
    return sum(p.size for p in model.parameters())

# 计算评估指标(accuracy和F1分数)
def compute_metrics(y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1_scores = []
    for label in range(len(classes)):
        precision = precision_score(y_true, y_pred, labels=[label], average='macro')
        recall = recall_score(y_true, y_pred, labels=[label], average='macro')
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    mean_f1 = np.mean(f1_scores)
    return accuracy, mean_f1

def main():
    # 数据路径
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    # 加载和预处理数据
    X_train, y_train, X_test, y_test, classes = load_and_preprocess_data(train_path, test_path)
    
    # 获取特征维度和类别数
    input_dim = X_train.shape[1]  # 特征数量
    output_dim = len(classes)     # 类别数量
    
   # 创建模型
    model = AirQualityCNN(input_dim, output_dim)
    
    # 检查参数数量
    params_count = count_parameters(model)
    print(f"模型参数数量: {params_count}")
    
    # 设置损失函数和优化器，优化可以重点关注这里
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 训练参数
    batch_size = 32
    epochs = 100

    train_samples = X_train.shape[0]  # 总样本数
    total_steps = (train_samples + batch_size - 1) // batch_size  # 向上取整计算总步数
    print(f"总步数: {total_steps}")
    print(f"训练样本数: {train_samples}")
    
    # 创建数据生成器（使用原始numpy数组）
    train_loader = data_generator(X_train, y_train, batch_size, shuffle=True)
    
    # 训练模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x_tensor = QTensor(batch_x)
            batch_y_tensor = QTensor(batch_y, dtype=kint64)

            # 前向传播
            outputs = model(batch_x_tensor)
            # 输出outputs和batch_y_tensor的形状
            # print(f"outputs shape: {outputs.shape}, batch_y_tensor shape: {batch_y_tensor.shape}")
            loss = loss_fn(batch_y_tensor, outputs)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每10个epoch打印一次训练信息
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / total_steps  # 使用预计算的总步数
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 评估模型
    model.eval()
    X_test_tensor = QTensor(X_test, dtype=kfloat32)
    y_test_tensor = QTensor(y_test, dtype=kfloat32)
    # probs = Softmax()(outputs)  # 手动添加 softmax
    # _, predicted = probs.data.max(axis=1)
    outputs = model(X_test_tensor)
    # 先获取最大值索引（注意 axis 格式）
    predicted = outputs.data.argmax(axis=[1], keepdims=False)
    predi = QTensor(predicted, requires_grad=True)

    # 将 QTensor 转换为 numpy 数组
    predicted_np = predi.to_numpy().astype(int)

    # # 打印调试信息
    # print("Predicted shape:", predicted_np.shape)
    # print("Predicted dtype:", predicted_np.dtype)
    # print("Predicted sample:", predicted_np[:5])
        
    accuracy, mean_f1 = compute_metrics(y_test, predicted_np, classes)
        
    print(f"准确率: {accuracy:.4f}")
    print(f"平均F1分数: {mean_f1:.4f}")
        
    # 保存结果到文本文件
    with open('results.txt', 'w') as f:
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"平均F1分数: {mean_f1:.4f}\n")

if __name__ == "__main__":
    main()

# Version one Output:
# == 训练集统计信息 ==
#                                 count        mean         std     min      25%     50%      75%     max
# Temperature                    4000.0   29.913175    6.676400   13.40   25.000   28.90   33.800   58.60
# Humidity                       4000.0   69.888925   15.802541   36.00   58.200   69.70   80.200  124.70
# PM2.5                          4000.0   19.738875   24.175401    0.00    4.500   11.60   25.425  295.00
# PM10                           4000.0   29.783425   26.993986   -0.20   12.100   21.30   37.525  315.80
# NO2                            4000.0   26.431825    8.900405    7.40   20.100   25.30   31.900   62.10
# SO2                            4000.0    9.942825    6.708436   -6.20    5.075    7.90   13.600   44.90
# CO                             4000.0    1.497030    0.547727    0.65    1.030    1.41    1.830    3.72
# Proximity_to_Industrial_Areas  4000.0    8.415300    3.600281    2.50    5.400    7.90   11.100   25.80
# Population_Density             4000.0  494.937250  152.599177  189.00  379.000  492.00  596.000  957.00

# == 测试集统计信息 ==
#                                 count       mean         std     min      25%      50%      75%     max
# Temperature                    1000.0   30.49240    6.878773   16.80   25.475   29.500   34.900   57.80
# Humidity                       1000.0   70.72490   16.096029   36.10   59.250   70.500   80.800  128.10
# PM2.5                          1000.0   21.75520   25.965977    0.00    5.000   13.500   28.525  240.10
# PM10                           1000.0   31.95810   28.674239    2.00   13.675   23.600   40.250  261.50
# NO2                            1000.0   26.33320    8.879141    9.30   20.000   25.100   31.600   64.90
# SO2                            1000.0   10.30280    6.911153   -1.90    5.200    8.400   14.200   39.60
# CO                             1000.0    1.51365    0.539237    0.72    1.040    1.435    1.880    3.37
# Proximity_to_Industrial_Areas  1000.0    8.46580    3.654823    2.50    5.400    8.100   11.100   24.80
# Population_Density             1000.0  507.37000  153.044869  188.00  389.750  512.000  614.000  934.00

# == 缺失值情况 ==
# Temperature                      0
# Humidity                         0
# PM2.5                            0
# PM10                             0
# NO2                              0
# SO2                              0
# CO                               0
# Proximity_to_Industrial_Areas    0
# Population_Density               0
# Air Quality                      0
# dtype: int64
# Temperature                      0
# Humidity                         0
# PM2.5                            0
# PM10                             0
# NO2                              0
# SO2                              0
# CO                               0
# Proximity_to_Industrial_Areas    0
# Population_Density               0
# Air Quality                      0
# dtype: int64

# 训练集无缺失值情况出现

# 已将除 Temperature 外的负值设置为 0。
# 已将 Humidity 超过 100 的值设置为 100。

# 已将除 Temperature 外的负值设置为 0。
# 已将 Humidity 超过 100 的值设置为 100。
# 模型参数数量: 184
# 总步数: 125
# 训练样本数: 4000
# Epoch 10/100, Loss: 0.0000
# Epoch 20/100, Loss: 0.0000
# Epoch 30/100, Loss: 0.0000
# Epoch 40/100, Loss: 0.0000
# Epoch 50/100, Loss: 0.0000
# Epoch 60/100, Loss: 0.0000
# Epoch 70/100, Loss: 0.0000
# Epoch 80/100, Loss: 0.0000
# Epoch 90/100, Loss: 0.0000
# Epoch 100/100, Loss: 0.0000
# 准确率: 0.9080
# 平均F1分数: 0.8853