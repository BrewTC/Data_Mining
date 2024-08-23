import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def LinearKernelPCA(X, y):

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kpca', KernelPCA(kernel='linear', n_jobs=6)),
        ('regressor', RandomForestRegressor(random_state=28, n_jobs=6))  # 使用回归器
    ])

    logspace_values = np.logspace(-2, 2, 15)
    print("Logspace values from 10^-2 to 10^2 with points:", logspace_values)

    param_grid = {
        'kpca__gamma': logspace_values,
        'kpca__n_components': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]       # 尝试不同的主成分数量
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=6)  # 使用 R^2 作为性能评价标准
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # 可视化第一主成分和第二主成分（如果适用）
    best_kpca = grid_search.best_estimator_.named_steps['kpca']
    print()
    # print(best_kpca)
    print()
    n_components_value = best_kpca.n_components
    print("The number of components in best_kpca is:", n_components_value)
    print()
    print()

    X_kpca = best_kpca.transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('Kernel PCA of Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    best_gamma = grid_search.best_params_['kpca__gamma']
    best_n_components = grid_search.best_params_['kpca__n_components']

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kpca', KernelPCA(n_components=best_n_components, kernel='linear', gamma=best_gamma))
    ])

    linear_kpca = pipeline.fit_transform(X)

    with open('./AI_MD/linear_kpca_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    print("模型已保存!")

    pca_columns = [f'Linear_KPCA{i+1}' for i in range(n_components_value)]
    df_linear_kpca = pd.DataFrame(linear_kpca, columns=pca_columns)
    # print(df_linear_kpca)

    # 计算累积解释方差比例
    explained_variance = np.var(linear_kpca, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # 绘制解释方差图
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Kernel PCA Explained Variance')
    plt.axhline(y=0.85, color='r', linestyle='--')  # 85% 解释方差线
    plt.grid(True)
    plt.show()

    return df_linear_kpca, cumulative_explained_variance
