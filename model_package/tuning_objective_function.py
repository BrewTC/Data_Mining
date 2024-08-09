# 引入必要套件
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def random_forest_regressor_tuning_objective_function(params, X_train, y_train, X_test, y_test):

    # 定義目標函數
    def objective_function(params):
        # 設定隨機森林參數
        n_estimators = int(params[0])
        max_depth = int(params[1])
        min_samples_split = int(params[2])
        min_samples_leaf = int(params[3])
        
        # 訓練隨機森林模型
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                random_state=28, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # 使用交叉驗證來評估模型性能
        scores = cross_val_score(rf, X_train, y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        loss = -scores.mean()  # 交叉驗證的平均 MSE
        return loss

    # 定義信賴區域約束
    def trust_region_constraint(params, params_old, radius):
        return radius - np.linalg.norm(params - params_old)

    # 初始參數
    params_initial = np.array([50, 10, 2, 1])  # 初始參數: n_estimators, max_depth, min_samples_split, min_samples_leaf
    params_old = params_initial.copy()
    radius = 5.0

    # 設置參數範圍限制
    bounds = [(10, 200),  # n_estimators 範圍
            (1, 30),    # max_depth 範圍
            (2, 20),    # min_samples_split 範圍
            (1, 20)]    # min_samples_leaf 範圍

    # 設置優化問題
    result = minimize(objective_function, params_initial,
                      constraints={'type': 'ineq', 'fun': trust_region_constraint, 'args': (params_old, radius)},
                      method='trust-constr', bounds=bounds, options={'gtol': 1e-6, 'disp': True})

    # 優化後的參數
    optimized_params = result.x
    print(f"優化後的參:        {optimized_params}")
    print(f"n_estimators:      {optimized_params[0].round(2)}")
    print(f"max_depth:         {optimized_params[1].round(2)}")
    print(f"min_samples_split: {optimized_params[2].round(2)}")
    print(f"min_samples_leaf:  {optimized_params[3].round(2)}")

    # 提取優化後的參數
    n_estimators_optimized = int(optimized_params[0])
    max_depth_optimized = int(optimized_params[1])
    min_samples_split_optimized = int(optimized_params[2])
    min_samples_leaf_optimized = int(optimized_params[3])

    # 訓練優化後的隨機森林模型
    rf_optimized = RandomForestRegressor(
        n_estimators=n_estimators_optimized, max_depth=max_depth_optimized, 
        min_samples_split=min_samples_split_optimized, min_samples_leaf=min_samples_leaf_optimized,
        random_state=28, n_jobs=-1
    )
    rf_optimized.fit(X_train, y_train)

    # 優化後模型的預測
    y_pred_optimized = rf_optimized.predict(X_test)
    mse_optimized = mean_squared_error(y_test, y_pred_optimized)
    
    return mse_optimized


# # 讀取資料集
# df = pd.read_csv('240715 Tai+MS_M2-M5_BS70-180_dis2_old40-75_clean_df_Train.csv')

# # 切割訓練集與測試集
# X = df.drop('BS_mg_dl', axis=1)
# y = df['BS_mg_dl']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# # 測試函數
# params = [50, 10, 2, 1]
# mse = random_forest_regressor_tuning_objective_function(params, X_train, y_train, X_test, y_test)
# print(f"優化後的MSE:       {mse.round(2)}")
