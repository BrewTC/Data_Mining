import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np

# 限制只有feature_importances_的模型使用

def visualize_feature_importance(features_number, reg, columns, X_test, y_test):
    # 獲取特徵重要性
    feature_importance = reg.feature_importances_

    # 獲取特徵名稱
    feature_names = columns

    # 將特徵名稱與特徵重要性一一對應，然後排序
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 顯示排名前n個特徵的名稱和重要性
    top_n = features_number  # 前10個特徵

    # 提取前n個特徵和其對應的重要性值
    top_features = [item[0] for item in sorted_feature_importance[:top_n]]
    top_importance = [item[1] for item in sorted_feature_importance[:top_n]]
    # print(top_features, '\n', top_importance)

    # 創建水平條形圖
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 左邊的子圖：特徵重要性
    axs[0].barh(top_features, top_importance, align='center')
    axs[0].set_xlabel('Feature Importance')
    axs[0].set_ylabel('Feature')
    axs[0].set_title('Top {} Feature Importance in GradientBoostingRegressor'.format(top_n))
    axs[0].invert_yaxis()  # 逆轉y軸，使最重要的特徵位於頂部

    # 計算特徵的排列重要性
    result = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    # 右邊的子圖：排列重要性
    axs[1].boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    axs[1].set_title("Permutation Importance (test set)")
    axs[1].set_xlabel('Permutation Importance')
    axs[1].set_ylabel('Feature')

    # 調整子圖之間的間距
    plt.tight_layout()

    # 顯示圖表
    plt.show()

    return top_features
