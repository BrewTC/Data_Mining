import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import pickle

def find_optimal_components(V, max_components=15):
    # 如果输入是DataFrame，转换为numpy array
    if isinstance(V, pd.DataFrame):
        V = V.values
    
    scaler = MinMaxScaler()
    V_scaled = scaler.fit_transform(V)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    errors = []

    for n in range(1, max_components + 1):
        fold_errors = []
        for train_index, test_index in kf.split(V_scaled):
            V_train, V_test = V_scaled[train_index], V_scaled[test_index]
            model = NMF(n_components=n, init='random', random_state=0)
            W_train = model.fit_transform(V_train)
            H = model.components_
            W_test = model.transform(V_test)
            V_test_approx = np.dot(W_test, H)
            error = mean_squared_error(V_test, V_test_approx)
            fold_errors.append(error)
        errors.append(np.mean(fold_errors))

    optimal_n = np.argmin(errors) + 1
    initial_error = errors[0]
    final_error = errors[-1]
    target_error_reduction = initial_error - final_error
    target_error = initial_error - target_error_reduction * 0.85

    # 绘制错误图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), errors, marker='o', label='MSE per Component')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Reconstruction MSE')
    plt.title('Cross-Validated Reconstruction Error vs. Number of Components')
    # plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n_components = {optimal_n}')
    plt.axhline(y=target_error, color='red', linestyle='--', label='85% Reduction Target')
    plt.legend()
    plt.show()

    standardization_pipeline = Pipeline([
        ('standardscaler', StandardScaler())
    ])

    # 使用最佳成分数再次拟合模型并转换整个数据集
    nmf_pipeline = Pipeline([
    ('minmaxscaler', MinMaxScaler(feature_range=(1, 2))),
    ('nmf', NMF(n_components=optimal_n, init='random', random_state=0))
    ])

    # 训练模型并转换数据
    V_standardized = standardization_pipeline.fit_transform(V)
    W_final = nmf_pipeline.fit_transform(V_standardized)
    nmf_features = [f'NMF{i+1}' for i in range(optimal_n)]
    df_nmf = pd.DataFrame(W_final, columns=nmf_features)

    # 保存流水线模型
    with open('nmf_first_pipeline.pkl', 'wb') as file:
        pickle.dump(standardization_pipeline, file)
    print("模型已保存!")

    with open('nmf_second_pipeline.pkl', 'wb') as file:
        pickle.dump(nmf_pipeline, file)
    print("模型已保存!")

    return df_nmf

# # 示例代码，使用随机数据
# np.random.seed(0)
# data = np.random.rand(100, 10)  # 生成随机数据
# df = pd.DataFrame(data)
# df_nmf = find_optimal_components(df, max_components=10)
# print(df_nmf.head())
