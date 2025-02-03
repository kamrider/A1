import numpy as np

class FeatureSelector:
    def __init__(self):
        self.feature_importances_ = None
        self.selected_features_ = None
    
    def calculate_entropy(self, x):
        """计算熵
        
        Args:
            x (ndarray): 输入数据
            
        Returns:
            float: 熵值
        """
        # 对连续值进行离散化（分箱）
        hist, bin_edges = np.histogram(x, bins='auto')
        # 计算概率
        p = hist / len(x)
        # 去除0概率（避免log(0)）
        p = p[p > 0]
        # 计算熵
        return -np.sum(p * np.log2(p))
    
    def calculate_mutual_information(self, x, y):
        """计算互信息
        
        Args:
            x (ndarray): 特征向量
            y (ndarray): 标签向量
            
        Returns:
            float: 互信息值
        """
        # 计算x的熵
        h_x = self.calculate_entropy(x)
        # 计算y的熵
        h_y = self.calculate_entropy(y)
        
        # 计算联合熵
        xy = np.vstack((x, y))
        h_xy = self.calculate_entropy(xy.T)
        
        # 互信息 = H(X) + H(Y) - H(X,Y)
        return h_x + h_y - h_xy
    
    def select_features(self, X, y, method='correlation', threshold=0.1, k=None):
        """选择特征
        
        Args:
            X (ndarray): 特征矩阵
            y (ndarray): 标签向量
            method (str): 特征选择方法，可选['correlation', 'mutual_info', 'variance']
            threshold (float): 特征重要性阈值
            k (int): 选择前k个重要特征
            
        Returns:
            ndarray: 选择后的特征矩阵
        """
        if method == 'correlation':
            # 计算每个特征与标签的相关系数
            importances = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) 
                                  for i in range(X.shape[1])])
        
        elif method == 'mutual_info':
            # 计算每个特征与标签的互信息
            importances = np.array([self.calculate_mutual_information(X[:, i], y) 
                                  for i in range(X.shape[1])])
        
        elif method == 'variance':
            # 使用方差作为特征重要性
            importances = np.var(X, axis=0)
            
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        self.feature_importances_ = importances
        
        # 根据重要性排序特征
        sorted_idx = np.argsort(importances)[::-1]
        
        if k is not None:
            selected_idx = sorted_idx[:k]
        else:
            selected_idx = sorted_idx[importances[sorted_idx] > threshold]
        
        self.selected_features_ = selected_idx
        
        # 打印特征重要性
        print("\n特征重要性排序:")
        for idx in sorted_idx:
            print(f"特征 {idx}: {importances[idx]:.4f}")
        
        print(f"\n选择了 {len(selected_idx)} 个特征")
        return X[:, selected_idx]

    def combine_features(self, X, y, methods=['correlation', 'mutual_info', 'variance']):
        """组合不同方法选择的特征"""
        selected_features = set()
        for method in methods:
            # 每个方法选择前5个特征
            self.select_features(X, y, method=method, k=5)
            selected_features.update(self.selected_features_)
        
        # 返回组合后的特征
        return X[:, list(selected_features)]

if __name__ == "__main__":
    # 测试代码
    from data_loader import CKDDataLoader
    
    print("开始加载数据...")
    loader = CKDDataLoader("../CKD.csv")
    loader.load_data()
    X, y = loader.preprocess_data()
    
    selector = FeatureSelector()
    
    # 使用相关系数方法
    print("\n使用相关系数方法:")
    X_selected = selector.select_features(X, y, method='correlation', k=10)
    
    # 使用自实现的互信息方法
    print("\n使用互信息方法:")
    X_selected_mi = selector.select_features(X, y, method='mutual_info', k=10)
    
    # 使用方差方法
    print("\n使用方差方法:")
    X_selected_var = selector.select_features(X, y, method='variance', k=10)
