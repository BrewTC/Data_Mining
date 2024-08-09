import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def permutation_importance_feature_selection_and_visualization(data_path, target_column, n_estimators=50, test_size=0.2, random_state=28, n_repeats=10):

    def evaluate_model(X, y, features):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(rf, X[features], y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(scores)

    # 讀取資料集
    df = pd.read_csv(data_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 切割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 訓練隨機森林模型
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 計算 Permutation Importance
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    perm_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })

    # 尋找最佳的 importance_mean 和 importance_std 閾值
    best_score = -np.inf
    best_mean_threshold = None
    best_std_threshold = None

    mean_thresholds = np.linspace(perm_importance_df['importance_mean'].min(), perm_importance_df['importance_mean'].max(), 10)
    std_thresholds = np.linspace(perm_importance_df['importance_std'].min(), perm_importance_df['importance_std'].max(), 10)

    for mean_threshold in mean_thresholds:
        for std_threshold in std_thresholds:
            important_features = perm_importance_df[perm_importance_df['importance_mean'] > mean_threshold]
            selected_features = important_features[important_features['importance_std'] <= std_threshold]['feature']
            if not selected_features.empty:
                score = evaluate_model(X_train, y_train, selected_features)
                if score > best_score:
                    best_score = score
                    best_mean_threshold = mean_threshold
                    best_std_threshold = std_threshold

    selection_features = perm_importance_df[(perm_importance_df['importance_mean'] > best_mean_threshold) &
                                        (perm_importance_df['importance_std'] <= best_std_threshold)]

    print(f"最佳 mean 閾值: {best_mean_threshold}")
    print(f"最佳 std 閾值: {best_std_threshold}")
    print("最終保留的特徵：", len(selection_features))
    print(selection_features)

    # 視覺化
    plt.figure(figsize=(12, 8))
    plt.scatter(perm_importance_df['importance_mean'], perm_importance_df['importance_std'], c='blue', label='All Features')
    plt.scatter(selection_features['importance_mean'], selection_features['importance_std'], c='red', label='Selected Features')

    for i in range(selection_features.shape[0]):
        plt.text(selection_features['importance_mean'].iloc[i], selection_features['importance_std'].iloc[i], selection_features['feature'].iloc[i], fontsize=9)

    plt.axvline(x=best_mean_threshold, color='gray', linestyle='--', label='Best Mean Threshold')
    plt.axhline(y=best_std_threshold, color='gray', linestyle='--', label='Best Std Threshold')

    plt.xlabel('Importance Mean')
    plt.ylabel('Importance Std')
    plt.title('Feature Importance Mean vs Std')
    plt.legend()
    plt.show()
    
    return selection_features

# 使用範例
# data_path, target_column = '240715 Tai+MS_M2-M5_BS70-180_dis2_old40-75_clean_df_Train.csv', 'BS_mg_dl'
# selection_features = permutation_importance_feature_selection_and_visualization(data_path, target_column)
