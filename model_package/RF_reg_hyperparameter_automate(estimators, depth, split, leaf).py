import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def find_best_n_estimators(X_train, y_train, X_test, y_test):

    n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300]
    train_scores = []
    test_scores = []

    for n_estimators in n_estimators_list:
        # 創建RandomForestRegressor模型
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=28, n_jobs=-1)

        # 使用交叉驗證計算訓練集和測試集上的MSE
        train_mse = -cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        test_mse = -cross_val_score(rf, X_test, y_test, cv=5, scoring='neg_mean_squared_error').mean()

        train_scores.append(train_mse)
        test_scores.append(test_mse)

    # 找到具有最佳測試集性能的n_estimators值
    best_n_estimators = n_estimators_list[np.argmin(test_scores)]

    # 繪製折線圖
    plt.figure(figsize=(5, 4))
    plt.plot(n_estimators_list, train_scores, label='Train MSE')
    plt.plot(n_estimators_list, test_scores, label='Test MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.title(f'RandomForestRegressor - Best n_estimators: {best_n_estimators}')
    plt.grid(True)
    plt.show()

    best_n_estimators = find_best_n_estimators(X_train, y_train, X_test, y_test)
    print(f'Best n_estimators: {best_n_estimators}')

    return best_n_estimators

def plot_max_depth_errors(X_train, y_train, X_test, y_test, best_n_estimators):
    max_depths = list(range(1, 10))
    train_errors = []
    test_errors = []

    for depth in max_depths:
        rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=depth, n_jobs=-1)
        rf.fit(X_train, y_train)

        # 計算訓練集和測試集的均方誤差
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)

        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    min_test_error = min(test_errors)
    min_test_error_index = test_errors.index(min_test_error)

    # 繪製訓練誤差和測試誤差隨max_depth的變化圖
    plt.figure(figsize=(5, 4))
    plt.plot(max_depths, train_errors, marker='o', linestyle='-', markersize=5, label='Train Error')
    plt.plot(max_depths, test_errors, marker='o', linestyle='-', markersize=5, label='Test Error')

    plt.xlabel('max_depth')
    plt.ylabel('MSE')
    plt.title(f'max_depth: {min_test_error_index}')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_max_depth = plot_max_depth_errors(X_train, y_train, X_test, y_test, best_n_estimators)
    print(f'Best max_depth: {best_max_depth}')

    return min_test_error_index

def plot_min_samples_split_errors(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth):

    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8]
    train_errors = []
    test_errors = []

    # 對每個min_samples_split值進行訓練和測試
    for min_split in min_samples_split_values:
        rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                  min_samples_split=min_split, random_state=28, n_jobs=-1)
        rf.fit(X_train, y_train)

        # 計算訓練集和測試集的MSE
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)

        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)
    
    best_min_samples_split = min_samples_split_values[np.argmin(test_errors)]
    print(f'Best min_samples_split: {best_min_samples_split}')

    # 繪製訓練誤差和測試誤差隨min_samples_split的變化圖
    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_split_values, train_errors, marker='o', linestyle='-', markersize=5, label='Train MSE')
    plt.plot(min_samples_split_values, test_errors, marker='o', linestyle='-', markersize=5, label='Test MSE')

    plt.xlabel('min_samples_split')
    plt.ylabel('MSE')
    plt.title(f'min_samples_split: {best_min_samples_split}')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_min_samples_split = plot_min_samples_split_errors(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth)
    print(f'Best min_samples_split: {best_min_samples_split}')

    return best_min_samples_split

def find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, best_min_samples_split):

    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8]
    train_errors = []
    test_errors = []

    # 對每個min_samples_leaf值進行訓練和測試
    for min_leaf in min_samples_leaf_values:
        rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, 
                                  min_samples_split=best_min_samples_split, min_samples_leaf=min_leaf,
                                  random_state=28, n_jobs=-1)
        rf.fit(X_train, y_train)

        # 計算訓練集和測試集的MSE
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)

        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    best_min_samples_leaf = min_samples_leaf_values[np.argmin(test_errors)]
    print(f'Best min_samples_split: {best_min_samples_leaf}')

    # 繪製訓練誤差和測試誤差隨min_samples_leaf的變化圖
    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_leaf_values, train_errors, marker='o', linestyle='-', markersize=5, label='Train MSE')
    plt.plot(min_samples_leaf_values, test_errors, marker='o', linestyle='-', markersize=5, label='Test MSE')

    plt.xlabel('min_samples_leaf')
    plt.ylabel('MSE')
    plt.title(f'min_samples_split: {best_min_samples_leaf}')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_min_samples_leaf = find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, best_min_samples_split)
    print(f'Best min_samples_leaf: {best_min_samples_leaf}')

    return best_min_samples_leaf



