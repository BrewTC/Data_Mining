'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Create a synthetic dataset
X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=28)

# Make three features highly correlated
X[:, 1] = X[:, 0] + 0.1 * np.random.normal(size=100)
X[:, 2] = X[:, 0] + 0.1 * np.random.normal(size=100)
X[:, 3] = np.random.normal(size=100)  # Independent feature

# Create a DataFrame for the dataset
columns = ['Feature1', 'Feature2', 'Feature3', 'IndependentFeature']
df = pd.DataFrame(X, columns=columns)
df['Target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Target', axis=1), df['Target'], test_size=0.2, random_state=42)
'''


# 1. 使用 DropHighPSIReg 移除高分佈偏移特徵，回傳保留的特徵
import pandas as pd
from feature_engine.selection import DropHighPSIFeatures

def DropHighPSIReg(X):
    '''
    Ref
    - https://feature-engine.trainindata.com/en/latest/user_guide/selection/DropHighPSIFeatures.html

    可以使用不同的閾值來評估分佈偏移的大小，具體取決於 更改為 PSI 值。最常用的閾值是：
    - 低於 10%，該變數沒有發生重大變化。
    - 在25%以上，該變數經歷了重大轉變。
    - 在這兩個值之間，偏移是中間的。
    - "auto"：閾值將根據基礎數據集和目標數據集的大小以及條柱的數量進行計算。
    '''
    # 回歸模型使用的分類法
    # Remove the features with high PSI values using a 60-40 split.
    transformer = DropHighPSIFeatures(split_frac=0.6)
    transformer.fit(X)

    print('各特徵的PSI值: \n', transformer.psi_values_)
    print('觀測值的切割點: \n', transformer.cut_off_)
    print(f'要移除的{len(transformer.features_to_drop_)}項特徵: \n', transformer.features_to_drop_)

    keep_list = [item for item in X.columns.tolist() if item not in transformer.features_to_drop_]
    print(f'要保留的{len(keep_list)}項特徵: \n', keep_list)
    
    # X['above_cut_off'] = X.index > transformer.cut_off_
    # sns.ecdfplot(data=X, x='old', hue='above_cut_off')
    return keep_list

# 2. 使用 VarianceInflationFactor 移除多重共線性特徵，回傳保留的特徵
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import pandas as pd

def VarianceInflationFactor(X, vif_number=5):
    '''
    方差膨脹因數 （VIF） 是衡量多重共線性的另一種方法。 它被測量為整體模型方差與每個獨立特徵的方差的比率。 一個特徵的高 VIF 表明它與一個或多個其他特徵相關。 根據經驗：
    VIF = 1 表示無相關性
    VIF = 1-5 中等相關性
    VIF >5 高相關
    VIF 是一種消除多重共線性特徵的有用技術。 對於我們的演示，將所有 VIF 高於 10 的刪除。

    Ref: 
    - https://zhuanlan.zhihu.com/p/507101225
    '''
    # calculate VIF 
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 
        index=X.columns) 
    
    # display VIFs in a table 
    index = X.columns.tolist() 
    vif_df = pd.DataFrame(vif, index = index, 
                          columns = ['VIF']).sort_values(by='VIF', 
                                                         ascending=False) 
    # 印出VIF係數小於10的特徵
    filtered_vif = vif_df[vif_df['VIF'] < vif_number] 
    print(filtered_vif, '\n')

    # 回傳 filtered_vif 中的所有 columns
    result_columns = filtered_vif.index.tolist()
    print('回傳的低共線性特徵:', result_columns)

    return result_columns

# 3. 使用 MulticollinearCorrelatedFeatures 移除多重共線性特徵，回傳保留的特徵
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def MulticollinearCorrelatedFeatures(X_train, X_test, y_train, y_test):
    '''
    考慮是否有多重共線性特徵(Multicollinear or Correlated Features)
    Ref:  
    - https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py  
    - https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.  
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X_train).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X_train.columns,
        ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()

    mse_values = []  # Initialize the list to store MSE values
    best_mse = float('inf')
    best_selected_features = None

    for i in range(len(X_train.columns)):
        cluster_ids = hierarchy.fcluster(dist_linkage, i, criterion="maxclust")  # Use a suitable cluster count
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

        X_train_sel = X_train.iloc[:, selected_features]
        X_test_sel = X_test.iloc[:, selected_features]

        clf_sel = RandomForestRegressor(n_estimators=100, random_state=28, n_jobs=-1)
        clf_sel.fit(X_train_sel, y_train)

        y_pred = clf_sel.predict(X_test_sel)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)  # Append MSE for this iteration to the list

        print(
            "The {} time:".format(i) + " Accuracy on test data with features removed: {:.2f}, MAE: {:.2f}".format(
                clf_sel.score(X_test_sel, y_test), mse
            )
        )

        # Update the best MSE and selected features
        if mse < best_mse:
            best_mse = mse
            best_selected_features = selected_features

    # Visualize MSE
    plt.plot(range(1, len(X_train.columns) + 1), mse_values, marker='o')
    plt.title('Test MSE for Different Cluster Counts')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.show()

    return best_selected_features

# 4. 使用 PermutationImportance 找出排列特徵 > threshold(預設0)的特徵
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

def PermutationImportance(X_train, X_test, y_train, y_test, threshold):
    '''
    考慮排列特徵重要性(Permutation Importance)
    Ref:
    - https://scikit-learn.org/stable/modules/permutation_importance.html  
    - https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py  
    '''
    clf = RandomForestRegressor(n_estimators=100, random_state=28, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Calculate permutation importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=28, n_jobs=-1)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    # 仅保留重要性大于 0 的特征
    selected_features = X_train.columns[perm_sorted_idx][result.importances_mean[perm_sorted_idx] > threshold]

    # Print or use selected_features as needed
    print("Features with permutation importance greater than 0:", selected_features.to_list())

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # 左侧图：决策树模型的特征重要性
    ax1.barh(tree_indices, clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X_train.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(clf.feature_importances_)))

    # 右侧图：Permutation Importance
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_train.columns[perm_sorted_idx]
    )

    # 仅显示重要性大于 0 的特征
    ax2.set_yticklabels(selected_features)
    fig.tight_layout()
    plt.show()

    return selected_features.to_list()

# 5. 使用 RfSequentialFeatureSelector 來設定要選用的特徵數量(個數or小數點百分比) 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector

def RfSequentialFeatureSelector(X_train, X_test, y_train, y_test, n_features_to_select):
    # Create a RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=28, n_jobs=-1)

    # Create a SequentialFeatureSelector
    # Forward selection (add one feature at a time)
    sfs = SequentialFeatureSelector(rf_reg, 
                                    direction='forward',  # 'forward' for forward selection, 'backward' for backward elimination
                                    n_features_to_select=n_features_to_select,  # Number of features to select
                                    scoring='neg_mean_squared_error',  # Scoring metric
                                    cv=5)  # Number of cross-validation folds

    # Fit the SequentialFeatureSelector on the training data
    sfs.fit(X_train, y_train)

    # Get the selected feature indices
    selected_feature_indices = np.where(sfs.get_support())[0]

    # Get the names of the selected features
    selected_feature_names = X_train.columns[selected_feature_indices].to_list()

    # Transform the data to include only the selected features
    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)

    # Train a RandomForestRegressor on the selected features
    rf_reg_selected = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg_selected.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred_selected = rf_reg_selected.predict(X_test_selected)

    # Calculate the Mean Squared Error (MSE)
    mse_selected = mean_squared_error(y_test, y_pred_selected)

    # Print the selected feature names and MSE
    print("Selected Feature Names:", selected_feature_names)
    print("Mean Squared Error on Test Set (selected features):", mse_selected)
    
    return selected_feature_names
