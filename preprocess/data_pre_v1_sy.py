import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_preprocess(data_csv):
    """
    1. 对数据的预处理，获取训练集和验证集数据的全部特征
    """
    # 1. 加载数据
    data = pd.read_csv(data_csv)
    
    # 2. 检查数据基本信息
    print("== 数据基本信息 ==")
    print(data.info())  # 数据维度、列名、非空数等
    print("\n== 缺失值情况 ==")
    print(data.isnull().sum())  # 每列缺失值数量

    # 3. 展示统计信息
    print("\n== 数据统计信息 ==")
    print(data.describe().T)  # 均值、方差、最大值、最小值等

    # 4. 特征与标签拆分
    feature_cols = data.columns.drop('Air Quality')
    X = data[feature_cols].copy()
    y = data['Air Quality'].copy()

    # 5. 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("\n标签映射:", dict(zip(le.classes_, le.transform(le.classes_))))

    # 6. 类别型特征 One-Hot 编码
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)
        print("\n已对以下类别特征进行 One-Hot 编码:", list(cat_cols))
    else:
        print("\n未检测到类别型特征，无需 One-Hot 编码。")

    # 7. 异常值检测（IQR 方法）
    print("\n== 异常值检测 (IQR Method) ==")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        cnt = ((X[col] < lower) | (X[col] > upper)).sum()
        print(f"Feature '{col}': {cnt} 个异常值")

    # 8. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"\n训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")

    return X_train, X_val, y_train, y_val, le, scaler

if __name__ == "__main__":
    data_csv = 'train_data.csv'  # 请替换为你的数据文件路径
    data_preprocess(data_csv)