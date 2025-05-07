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

from data_pre_v3_sy import data_preprocess
from pyvqnet.qnn.pq3.measure import ProbsMeasure
import pyqpanda3 as pq
from pyvqnet.optim import sgd
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.data import data_generator as dataloader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pyqpanda3.core as pq1
from sklearn.metrics import f1_score


def load_data_train(train_csv, shuffle: bool = True):
    """
    1. 读取数据并转换为张量
    """
    # 1. 数据预处理
    X_train_t, X_val_t, y_train_t, y_val_t, label_encoder, scaler = data_preprocess(train_csv)
    # X_train.shape = (3200, 9), X_val.shape = (800, 9)

    # 2. 转换为 VQNet 张量
    #    dtype 可选 Float32、Float64，取决于你的量子电路需求
    # X_train_t = QTensor(X_train, dtype=kfloat32, requires_grad=False)
    # y_train_t = QTensor(y_train.astype(np.int64), dtype=kfloat32, requires_grad=False)
    # X_val_t   = QTensor(X_val,   dtype=kfloat32, requires_grad=False)
    # y_val_t   = QTensor(y_val.astype(np.int64),   dtype=kfloat32, requires_grad=False)
    
    return X_train_t, y_train_t, X_val_t, y_val_t, label_encoder, scaler

def load_data_test(test_csv, shuffle:bool = True):
    """
    1. 读取数据并转换为张量
    """
    data = pd.read_csv(test_csv)
    le = LabelEncoder()
    y = data['Air Quality']
    y_encoded = le.fit_transform(y)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_temp_cols = [c for c in numeric_cols if c != 'Temperature']
    data[non_temp_cols] = data[non_temp_cols].clip(lower=0)
    
    if 'Humidity' in data.columns:
        data['Humidity'] = data['Humidity'].clip(upper=100)
        
    X = data.drop(columns=['Air Quality'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le, scaler
    


class VariationQuantumCircuit():
    """
    2. 在这里完成量子线路的设计
    """
    def __init__(self, num_qubits=9, depth=1):
        self.num_qubits = num_qubits
        self.depth      = depth
        # 量子模拟器后端
        self.machine = pq1.CPUQVM()
        
    def __call__(self, input, params):
        qubits = range(self.num_qubits)
        qvc_cir = pq1.QCircuit()
        #参数化电路构造
        for i in range(self.num_qubits):
            qvc_cir << pq1.H(qubits[i])
            
        for j in range(self.depth):
            for i in range(self.num_qubits):
                qvc_cir << pq1.RZ(qubits[i], input[i])
                qvc_cir << pq1.RX(qubits[i], params[i+j*self.num_qubits])
                if i < self.num_qubits - 1:
                    qvc_cir << pq1.CNOT(qubits[i], qubits[i+1])
                if i == self.num_qubits - 1:
                    qvc_cir << pq1.CNOT(qubits[i], qubits[0])
                
        for k in range(self.num_qubits):
            qvc_cir << pq1.RZ(qubits[k], input[k])
            
        prog = pq1.QProg()
        prog << qvc_cir
        
        result_prob = ProbsMeasure(self.machine,prog, [0,1])
        
        return result_prob
        
class AirQualityQuantumNN(Module):
    """
    3. 在这里完成初始化量子神经网络模型的代码
    """
    def __init__(self):
        super(AirQualityQuantumNN, self).__init__()
        
        quantum_circuit = VariationQuantumCircuit(num_qubits=9, depth=1)
        
        self.qvc = QuantumLayer(quantum_circuit, 9)
        
    def forward(self, x):
        return self.qvc(x)


def get_accuracy(result_qt, label_qt):
    # 1. 预测部分 —— 如果是 QTensor 就 to_numpy，否则假定已是 ndarray
    if hasattr(result_qt, "to_numpy"):
        probs = result_qt.to_numpy()
    else:
        probs = result_qt
    pred = np.argmax(probs, axis=1)

    # 2. 真实标签 —— 同理
    if hasattr(label_qt, "to_numpy"):
        truth = label_qt.to_numpy().flatten().astype(int)
    else:
        truth = label_qt.flatten().astype(int)

    return np.sum(pred == truth)

def compute_f1(result_qt, label_qt, average='macro'):
    """
    计算 F1 分数。
    """
    # 确保是 numpy 数组
    if hasattr(result_qt, "to_numpy"):
        probs = result_qt.to_numpy()
    else:
        probs = result_qt
    pred = np.argmax(probs, axis=1)

    # 2. 真实标签 —— 同理
    if hasattr(label_qt, "to_numpy"):
        truth = label_qt.to_numpy().flatten().astype(int)
    else:
        truth = label_qt.flatten().astype(int)
    
    return f1_score(truth, probs, average=average)

def quantum_model_train():
    """
    4. 在这里完成训练量子模型的代码
    """
    X_train, y_train, X_val_t, y_val_t, le, scaler = load_data_train('train_data.csv')
    model = AirQualityQuantumNN()
    
    optimizer = sgd.SGD(model.parameters(), lr=0.5)
    batch_size = 4
    epoch = 10
    loss = CrossEntropyLoss()
    print("训练开始")
    model.train()
    
    for i in range(epoch):
        count = 0
        sum_loss = 0
        accuary = 0
        t = 0
        for data,label in dataloader(X_train, y_train, batch_size, False):
            optimizer.zero_grad()
            data, label = QTensor(data), QTensor(label)
            
            result = model(data)
            
            loss_b = loss(label, result)
            loss_b.backward()
            optimizer.step()
            sum_loss += loss_b.item()
            count += batch_size
            accuary += get_accuracy(result, label)
            t = t+1
    
        print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
    
def quantum_model_test():
    """
    5. 使用测试数据集验证模型
    """
    X_test, y_test, le, scaler = load_data_test('test_data.csv')
    model = AirQualityQuantumNN()
    
    model.eval()
    loss = CrossEntropyLoss()
    count = 0
    test_data, test_label = X_test, y_test
    test_batch_size = 1
    accuary = 0
    sum_loss = 0
    
    all_preds = []
    all_trues = []
    
    for testd, testl in dataloader(test_data, test_label, test_batch_size):
        testd = QTensor(testd)
        test_result = model(testd)
        test_loss = loss(testl, test_result)
        sum_loss += test_loss.item()
        count += test_batch_size
        accuary += get_accuracy(test_result, testl)
    print(f"测试结果:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")
    
if __name__ == "__main__":
    quantum_model_train()

    quantum_model_test()
