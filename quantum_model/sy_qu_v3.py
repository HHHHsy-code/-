from pyvqnet.qnn.qlinear import QLinear
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from pyvqnet.data.data import data_generator
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss

from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayerV3
from pyvqnet.qnn.pq3.measure import ProbsMeasure
import pyqpanda3.core as pq 

from data_pre_v3_sy import data_preprocess
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import QTensor
from pyqpanda3.quantum_info import StateVector, DensityMatrix
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, f1_score


def load_data(train_csv, batch_size=32, shuffle: bool = True):
    """
    1. 读取数据并转换为张量
    """
    # 1. 数据预处理
    X_train, X_val, y_train, y_val, label_encoder, scaler = data_preprocess(train_csv)
    # X_train.shape = (3200, 9), X_val.shape = (800, 9)

    # 2. 转换为 VQNet 张量
    #    dtype 可选 Float32、Float64，取决于你的量子电路需求
    X_train_t = QTensor(X_train, dtype=kfloat32, requires_grad=False)
    y_train_t = QTensor(y_train.astype(np.int64), dtype=kfloat32, requires_grad=False)
    X_val_t   = QTensor(X_val,   dtype=kfloat32, requires_grad=False)
    y_val_t   = QTensor(y_val.astype(np.int64),   dtype=kfloat32, requires_grad=False)
    
    return X_train_t, y_train_t, X_val_t, y_val_t, label_encoder, scaler

class VariationQuantumCircuit():
    """
    量子电路设计
    """
    def __init__(self, num_qubits=9, depth=1):
        self.num_qubits = num_qubits
        self.depth      = depth
        # 量子模拟器后端
        self.machine = pq.CPUQVM()

    def __call__(self, x: np.ndarray, param):
        """
        x: 一维特征向量，长度 = num_qubits
        """
        qubits = range(self.num_qubits)
        circuit = pq.QCircuit()
        
        # 使用 Hadamard 门初始化量子比特
        for i in range(self.num_qubits):
            circuit << pq.H(qubits[i])

        # 迭代量子电路深度
        for i in range(self.depth):
            for j in range(self.num_qubits):   
                # 应用旋转门
                circuit << pq.RZ(qubits[j], x[j])  # 特征向量与量子门的作用
                circuit << pq.RX(qubits[j], param[j])  # 参数与量子电路的作用

        # 完成电路
        for i in range(self.num_qubits):
            circuit << pq.RZ(qubits[i], x[i])

        prog = pq.QProg()
        prog << circuit
        
        # 使用量子程序计算概率
        result_prob = ProbsMeasure(self.machine, prog, [0,1])
        return result_prob


def loss_function(output, y_train_t):
    predicted_class = np.argmax(output.to_numpy().astype(int), axis=1)
    loss = np.mean(predicted_class != y_train_t.to_numpy().astype(int))
    
    grad = np.zeros_like(output.to_numpy().astype(int))
    grad[np.arange(len(predicted_class)), y_train_t.to_numpy().astype(int)] = -1
    return loss, grad

class AirQualityQuantumNN(Module):
    """
    量子神经网络模型
    """
    def __init__(self):
        super(AirQualityQuantumNN, self).__init__()
        # 使用 QuantumLayerV3 替换 QuantumLayer
        quantum_circuit = VariationQuantumCircuit(num_qubits=9, depth=1)
        
        self.pqc = QuantumLayerV3(
            origin_qprog_func=quantum_circuit,  # 传递量子电路
            para_num=9,  # 量子电路中的参数数量
            qvm_type="cpu",  # 使用 cpu 类型模拟器
            shots=1000,  # 测量次数
            initializer=None,  # 参数初始化方法
            dtype=kfloat32  # 数据类型
        )

    def forward(self, x):
        # 确保传入的是 QTensor 类型
        x = QTensor(x, dtype=kfloat32, requires_grad=False)
        return self.pqc(x)

def quantum_model_train():
    """
    4. 在这里完成训练量子模型的代码
    """
    X_train_t, y_train_t, X_val_t, y_val_t, label_encoder, scaler = load_data("train_data.csv")
    print(X_train_t, y_train_t, X_val_t, y_val_t)
    
    model = AirQualityQuantumNN()
    
    # 训练循环
    for epoch in range(10):
        total_loss = 0
        # 使用 tqdm 包装训练循环，显示进度条
        for i in tqdm(range(X_train_t.shape[0]), desc=f"Epoch {epoch+1}"):  # 显示 epoch 号和进度条
            # 前向传播
            output = model(X_train_t)  # 输入特征数据到量子层
            
            # 计算损失和梯度
            loss, grad = loss_function(output, y_train_t[i])
            total_loss += loss
            
            # 反向传播
            output.backward(grad)  # 进行反向传播更新参数
        
        # 计算验证集上的损失和准确率
        val_output = model(X_val_t)
        val_loss, _ = loss_function(val_output, y_val_t)
        val_pred = np.argmax(val_output.to_numpy().astype(int), axis=1)
        val_acc = accuracy_score(y_val_t.to_numpy().astype(int), val_pred)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / X_train_t.shape[0]}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")  # 打印每个epoch的平均损失、验证损失和验证准确率

def quantum_model_test():
    """
    5. 使用测试数据集验证模型
    """
    # 加载测试数据
    X_train_t, y_train_t, X_val_t, y_val_t, label_encoder, scaler = load_data("train_data.csv")
    X_test_t, y_test_t, _, _, _, _ = load_data("test_data.csv")  # 加载测试数据
    
    model = AirQualityQuantumNN()
    
    # 使用测试集进行预测
    test_output = model(X_test_t)
    test_pred = np.argmax(test_output.to_numpy().astype(int), axis=1)
    
    # 计算测试集上的准确率和 F1 分数
    test_acc = accuracy_score(y_test_t.to_numpy().astype(int), test_pred)
    test_f1 = f1_score(y_test_t.to_numpy().astype(int), test_pred, average='weighted')
    
    print(f"Test Accuracy: {test_acc}")
    print(f"Test F1 Score: {test_f1}")
    

if __name__ == "__main__":
    quantum_model_train()
    quantum_model_test()
