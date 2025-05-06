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
from pyvqnet.qnn.pq3.measure import ProbsMeasure
import pyqpanda3.core as pq 

from data_pre_v3_sy import data_preprocess
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import QTensor
from pyqpanda3.quantum_info import StateVector, DensityMatrix


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

    # # 3. 构建数据迭代器
    # train_loader = data_generator(X_train_t, y_train_t,
    #                               batch_size=batch_size,
    #                               shuffle=shuffle)
    # val_loader   = data_generator(X_val_t,   y_val_t,
    #                               batch_size=batch_size,
    #                               shuffle=False)

    # # 返回训练/验证迭代器，以及 label_encoder（用于解码预测结果）和 scaler（如果需要对测试集做同样的标准化）
    # return train_loader, val_loader, label_encoder, scaler
    
    return X_train_t, y_train_t, X_val_t, y_val_t, label_encoder, scaler

class VariationQuantumCircuit():
    """
    2. 在这里完成量子线路的设计
    """
    def __init__(self, num_qubits=9, depth=1):
        self.num_qubits = num_qubits
        self.depth      = depth
        # 量子模拟器后端
        self.machine = pq.CPUQVM()
        
    def __call__(self, x: np.ndarray, param):
        """
        x:       一维特征向量，长度 = num_qubits
        """
        qubits = range(self.num_qubits)
        circuit = pq.QCircuit()
        for i in range(self.num_qubits):
            circuit << pq.H(qubits[i])
        
        for i in range(self.depth):
            for j in range(self.num_qubits):   
                circuit << pq.RZ(qubits[j], x
                                 [j])
                circuit << pq.RX(qubits[j], param[j])
        
        for i in range(self.num_qubits):
            circuit << pq.RZ(qubits[i], x[i])
        
        prog = pq.QProg()
        prog << circuit
        
        result_prob = ProbsMeasure(self.machine,prog, [0,1])
        
        return result_prob        

def loss_function(output, y_train_t):
    predicted_class = np.argmax(output.data, axis=1)
    loss = np.mean(predicted_class != y_train_t.data)
    
    grad = np.zeros_like(output.data)
    grad[np.arange(len(predicted_class)), y_train_t.data] = -1
    return loss, grad
    
    
class AirQualityQuantumNN(Module):
    """
    3. 在这里完成初始化量子神经网络模型的代码
    """
    def __init__(self):
        super(AirQualityQuantumNN, self).__init__()
        # 创建量子层实例
        quantum_circuit = VariationQuantumCircuit(num_qubits=9, depth=1)
        
        self.pqc = QuantumLayer(
            qprog_with_measure=quantum_circuit,  # 传递量子电路
            para_num=9,  # 量子电路中的参数数量
            diff_method="parameter_shift",  # 默认差分方法，或选择其他方法
            delta=0.01,  # 计算梯度时的步长
            dtype=kfloat32  # 数据类型，默认是kfloat32
        )

    def forward(self, x):
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
        for i in range(X_train_t.shape[0]):
            # 前向传播
            
            output = model(X_train_t)  # 输入特征数据到量子层
            
            # 计算损失和梯度
            loss, grad = loss_function(output.data, y_train_t[i])
            total_loss += loss
            
            # 反向传播
            output.backward(grad)  # 进行反向传播更新参数
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / X_train_t.shape[0]}")  # 打印每个epoch的平均损失
    


def quantum_model_test():
    """
    5. 使用测试数据集验证模型
    """


if __name__ == "__main__":
    quantum_model_train()

    quantum_model_test()
