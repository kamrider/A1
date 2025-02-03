import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.5, max_iterations=1000, decay_rate=0.95):
        """初始化逻辑回归模型
        
        Args:
            learning_rate (float): 初始学习率
            max_iterations (int): 最大迭代次数
            decay_rate (float): 学习率衰减率
        """
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None
        self.losses = []
        
    def _update_learning_rate(self, iteration):
        """更新学习率
        
        Args:
            iteration (int): 当前迭代次数
        """
        # 指数衰减
        self.learning_rate = self.initial_learning_rate * (self.decay_rate ** (iteration // 100))
        
    def sigmoid(self, z):
        """sigmoid激活函数
        
        Args:
            z (ndarray): 输入值
        
        Returns:
            ndarray: sigmoid(z)
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """训练逻辑回归模型
        
        Args:
            X (ndarray): 特征矩阵
            y (ndarray): 标签向量
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print("训练数据范围：", np.min(X), np.max(X))
        print("标签分布：", np.unique(y, return_counts=True))
        
        # 记录最佳参数和最小损失
        best_weights = self.weights.copy()
        best_bias = self.bias
        min_loss = float('inf')
        patience_counter = 0
        
        for i in range(self.max_iterations):
            # 更新学习率
            self._update_learning_rate(i)
            
            # 前向传播
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # 检查是否找到更好的参数
            if loss < min_loss:
                min_loss = loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停机制
            if patience_counter >= 5:  # 如果连续5次没有改善
                print(f"早停：在迭代{i}次后停止")
                break
            
            # 反向传播
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代打印一次损失
            if i % 100 == 0:
                print(f"迭代 {i}, 损失: {loss:.4f}, 学习率: {self.learning_rate:.6f}")
        
        # 使用最佳参数
        self.weights = best_weights
        self.bias = best_bias
        
    def predict(self, X):
        """预测类别
        
        Args:
            X (ndarray): 形状为(n_samples, n_features)的特征矩阵
            
        Returns:
            ndarray: 形状为(n_samples,)的预测标签
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        return (predictions >= 0.5).astype(int)
    
    def compute_loss(self, y_true, y_pred):
        """计算二元交叉熵损失
        
        Args:
            y_true (ndarray): 真实标签
            y_pred (ndarray): 预测概率
            
        Returns:
            float: 损失值
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # 添加调试信息
        if np.isnan(loss):
            print("警告：损失值为nan")
            print("y_true范围:", np.min(y_true), np.max(y_true))
            print("y_pred范围:", np.min(y_pred), np.max(y_pred))
        
        return loss

def cross_validate(X, y, n_splits=10):
    """实现十折交叉验证
    
    Args:
        X (ndarray): 特征矩阵
        y (ndarray): 标签向量
        n_splits (int): 折数，默认为10
        
    Returns:
        tuple: (mean_accuracy, std_accuracy)
    """
    # 打乱数据
    np.random.seed(42)  # 固定随机种子以确保结果可重现
    indices = np.random.permutation(len(X))
    fold_size = len(X) // n_splits
    accuracies = []
    
    print(f"\n开始{n_splits}折交叉验证...")
    
    for i in range(n_splits):
        # 选择测试集索引
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_splits-1 else len(X)
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([
            indices[:start_idx],
            indices[end_idx:]
        ])
        
        # 分割数据
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # 训练和评估
        model = LogisticRegression(learning_rate=0.5, max_iterations=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        
        print(f"第{i+1}折验证准确率: {accuracy:.4f}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n所有折的准确率: {accuracies}")
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"最高准确率: {np.max(accuracies):.4f}")
    print(f"最低准确率: {np.min(accuracies):.4f}")
    
    return mean_acc, std_acc

def split_data(X, y, test_size=0.2, random_state=42):
    """分割数据集为训练集和测试集
    
    Args:
        X (ndarray): 特征矩阵
        y (ndarray): 标签向量
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 生成随机索引
    indices = np.random.permutation(len(X))
    
    # 计算分割点
    split = int(len(X) * (1 - test_size))
    
    # 分割数据
    train_idx, test_idx = indices[:split], indices[split:]
    
    return (X[train_idx], X[test_idx], 
            y[train_idx], y[test_idx])

if __name__ == "__main__":
    # 测试代码
    from data_loader import CKDDataLoader
    from feature_selector import FeatureSelector
    
    print("开始加载数据...")
    # 加载数据
    loader = CKDDataLoader("../CKD.csv")
    loader.load_data()
    X, y = loader.preprocess_data()
    
    print("\n尝试不同的特征选择方法...")
    selector = FeatureSelector()
    
    # 使用相关系数方法
    print("\n使用相关系数方法:")
    X_selected_corr = selector.select_features(X, y, method='correlation', k=10)
    
    # 使用互信息方法
    print("\n使用互信息方法:")
    X_selected_mi = selector.select_features(X, y, method='mutual_info', k=10)
    
    # 使用方差方法
    print("\n使用方差方法:")
    X_selected_var = selector.select_features(X, y, method='variance', k=10)
    
    # 选择表现最好的特征集
    methods = {
        'correlation': X_selected_corr,
        'mutual_info': X_selected_mi,
        'variance': X_selected_var
    }
    
    best_accuracy = 0
    best_method = None
    best_X = None
    
    for method_name, X_selected in methods.items():
        print(f"\n测试 {method_name} 方法:")
        # 分割数据
        X_train, X_test, y_train, y_test = split_data(X_selected, y)
        
        # 训练模型
        model = LogisticRegression(learning_rate=0.5, max_iterations=1000)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"准确率: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method_name
            best_X = X_selected
    
    print(f"\n最佳方法: {best_method}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    # 对每种特征选择方法进行十折交叉验证
    print("\n对所有方法进行十折交叉验证:")
    cv_results = {}
    
    for method_name, X_selected in methods.items():
        print(f"\n评估 {method_name} 方法:")
        mean_acc, std_acc = cross_validate(X_selected, y, n_splits=10)
        cv_results[method_name] = (mean_acc, std_acc)
    
    # 打印所有方法的比较结果
    print("\n各方法交叉验证结果比较:")
    for method_name, (mean_acc, std_acc) in cv_results.items():
        print(f"{method_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # 打印最终损失
    print(f"最终损失: {model.losses[-1]:.4f}")
    
    # 打印每100次迭代的损失
    print("\n损失变化:")
    for i, loss in enumerate(model.losses[::100]):
        print(f"迭代 {i*100}: {loss:.4f}")
    
    # 打印每个选中特征的权重
    print("\n特征权重:")
    for idx, feature_idx in enumerate(selector.selected_features_):
        print(f"特征 {feature_idx}: {model.weights[idx]:.4f}")
    
    # 使用相关系数方法选择更多特征
    print("\n使用相关系数方法选择特征:")
    X_selected = selector.select_features(X, y, method='correlation', k=15)  # 增加到15个特征
    
    # 进行十折交叉验证
    mean_acc, std_acc = cross_validate(X_selected, y, n_splits=10)
