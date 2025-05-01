import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_preprocess(data_csv):
    """
    数据预处理：
      1. 加载数据并检查基本信息
      2. 输出统计信息
      3. 缺失值情况检测，如无缺失则提示
      4. 标签编码
      5. 将除 Temperature 外的负值设为 0
      6. 将 Humidity 大于 100 的值设为 100
      7. 标准化
      8. 训练/验证集划分
    """
    # 1. 加载数据
    data = pd.read_csv(data_csv)
    print("== 数据基本信息 ==")
    print(data.info())

    # 2. 统计信息
    print("\n== 数据统计信息 ==")
    print(data.describe().T)

    # 3. 缺失值情况检测
    missing = data.isnull().sum()
    print("\n== 缺失值情况 ==")
    print(missing)
    if missing.sum() == 0:
        print("\n无缺失值情况出现")
    else:
        print("\n== 缺失值情况 ==")
        print(missing)

    # 4. 标签编码
    le = LabelEncoder()
    y = data['Air Quality']
    y_encoded = le.fit_transform(y)
    print("\n标签映射:", dict(zip(le.classes_, le.transform(le.classes_))))
    print(le)

    # 5. 将除 Temperature 外的负值设为 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_temp_cols = [c for c in numeric_cols if c != 'Temperature']
    data[non_temp_cols] = data[non_temp_cols].clip(lower=0)
    print("\n已将除 Temperature 外的负值设置为 0。")

    # 6. 将 Humidity 大于 100 的值设为 100
    if 'Humidity' in data.columns:
        data['Humidity'] = data['Humidity'].clip(upper=100)
        print("已将 Humidity 超过 100 的值设置为 100。")

    # 7. 特征与标签拆分
    X = data.drop(columns=['Air Quality'])

    # 8. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2,
        random_state=42, stratify=y_encoded
    )
    print(f"\n训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")

    return X_train, X_val, y_train, y_val, le, scaler

if __name__ == "__main__":
    csv_path = 'train_data.csv'
    data_preprocess(csv_path)