import pandas as pd
import numpy as np

class CKDDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """加载CKD数据集"""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def min_max_normalize(self, data):
        """手动实现min-max归一化"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    def preprocess_data(self):
        """预处理数据"""
        # 处理缺失值
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(
            self.data[numeric_columns].mean()
        )
        
        # 分离特征和标签
        print("原始标签值：", self.data['label'].value_counts())  # 显示每个类别的数量
        self.y = np.where(self.data['label'].str.lower() == 'ckd', 1, 0)  # 转换为小写再比较
        print("转换后的标签分布：")
        print("- 正样本(CKD)数量:", np.sum(self.y == 1))
        print("- 负样本(非CKD)数量:", np.sum(self.y == 0))
        
        self.X = self.data.drop(['label', 'ID'], axis=1)
        
        # 确保所有特征都是数值型
        self.X = self.X.astype(float)
        
        # 打印归一化前的范围
        print("\n归一化前的特征范围：")
        for col in self.X.columns:
            print(f"{col}: [{self.X[col].min():.3f}, {self.X[col].max():.3f}]")
        
        # 只对后8个特征进行归一化（前20个保持不变）
        X_normalized = self.X.copy()
        for col in self.X.columns[20:]:
            X_normalized[col] = self.min_max_normalize(self.X[col])
        
        self.X = X_normalized.values
        
        # 打印归一化后的范围
        print("\n归一化后的特征范围：")
        for i, col in enumerate(X_normalized.columns):
            print(f"{col}: [{self.X[:, i].min():.3f}, {self.X[:, i].max():.3f}]")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """分割训练集和测试集"""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # 创建数据加载器实例
    loader = CKDDataLoader("../CKD.csv")
    
    # 加载数据
    data = loader.load_data()
    print("数据集大小:", data.shape)
    print("\n数据类型:")
    print(data.dtypes)
    
    # 预处理数据
    X, y = loader.preprocess_data()
    print("\n特征矩阵形状:", X.shape)
    print("标签数量:", y.shape)
    
    # 分割数据
    X_train, X_test, y_train, y_test = loader.split_data()
    print("\n训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)