import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def AutoPrincipalComponentsAnalysis(X):

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # PCA 分析
    pca = PCA()
    pca.fit(Z)

    # 獲得解釋方差比率
    explained_variance_ratio = pca.explained_variance_ratio_

    # 繪製條形圖
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='blue', label='Individual explained variance')
    plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance by PCA Components')
    plt.axhline(y=0.85, color='r', linestyle='--')  # 85% 解釋方差線
    plt.legend(loc='best')
    plt.show()
    # -------------------------------------------------------------

    X_mean = X.mean()
    X_std = X.std()
    X_std.replace(0, 1, inplace=True)  # 将标准差为0的替换为1，避免除以零

    # 数据标准化
    X_std.replace(0, 1, inplace=True)  # 将标准差为0的替换为1，避免除以零
    Z = (X - X_mean) / X_std

    # 计算协方差矩阵
    c = Z.cov()

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(c)

    # 只取特征值和特征向量的实数部分
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 根据特征值的降序排序特征值和对应的特征向量
    idx = eigenvalues.argsort()[::-1]   # 降序索引
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 根据解释的方差确定组件数
    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(explained_var >= 0.50) + 1  # 解释至少50%的方差所需的最小组件数

    # 提取前 n_components 个主成分
    u = eigenvectors[:, :n_components]
    pca_component = pd.DataFrame(u,
                                index = X.columns,
                                columns = [f'PC{i+1}' for i in range(n_components)])

    # 绘制 PCA 分量热图
    plt.figure(figsize =(5, 7))
    sns.heatmap(pca_component, annot=False, cmap='coolwarm')
    plt.title('PCA Component Heatmap')
    plt.show()
    # -------------------------------------------------------------

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])

    # 训练模型并转换数据
    Z_pca = pipeline.fit_transform(X)

    # 保存流水线模型
    with open('pca_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    print("模型已保存!")

    # 创建包含 PCA 主成分的 DataFrame
    pca_columns = [f'PCA{i+1}' for i in range(pipeline.named_steps['pca'].n_components_)]
    df_pca = pd.DataFrame(Z_pca, columns=pca_columns)

    return df_pca

def PrincipalComponentsAnalysisPlot(X):

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # PCA 分析
    pca = PCA()
    pca.fit(Z)

    # 獲得解釋方差比率
    explained_variance_ratio = pca.explained_variance_ratio_

    # 繪製條形圖
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='blue', label='Individual explained variance')
    plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance by PCA Components')
    plt.axhline(y=0.85, color='r', linestyle='--')  # 85% 解釋方差線
    plt.legend(loc='best')
    plt.show()