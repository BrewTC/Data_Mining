import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def find_best_n_estimators(X_train, y_train, X_test, y_test, model_type={'GB', 'RF'}):
    if model_type == 'RF':
        model = RandomForestRegressor(n_jobs=-1, random_state=28)
    elif model_type == 'GB':
        model = GradientBoostingRegressor(random_state=28)
    else:
        raise ValueError("Invalid model_type. Use 'RF' or 'GB'.")

    n_estimators_list = [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300]
    train_mse_scores = []
    test_mse_scores = []

    for n_estimators in n_estimators_list:
        model.n_estimators = n_estimators

        # Calculate the mean squared error using cross-validation for Train and Test sets
        train_mse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        test_mse = -cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error').mean()

        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)

    # Calculate the average of Train and Test MSE for each n_estimators
    average_mse_scores = [(train_mse + test_mse) / 2 for train_mse, test_mse in zip(train_mse_scores, test_mse_scores)]

    # Find the number of estimators associated with the lowest average MSE
    best_n_estimators = n_estimators_list[np.argmin(average_mse_scores)]
    print(f'Best n_estimators: {best_n_estimators}')

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.plot(n_estimators_list, train_mse_scores, label='Train MSE')
    plt.plot(n_estimators_list, test_mse_scores, label='Test MSE')
    # plt.plot(n_estimators_list, average_mse_scores, label='Average MSE (Train and Test)')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.title(f'{model_type} - Best n_estimators: {best_n_estimators}')
    plt.grid(True)
    plt.show()

    return best_n_estimators

def find_best_max_depth(X_train, y_train, X_test, y_test, best_n_estimators, model_type={'GB', 'RF'}):
    max_depths = list(range(1, 15))
    train_errors = []
    test_errors = []
    # mse_average_scores = []

    for depth in max_depths:
        train_depth_errors = []
        test_depth_errors = []
        for _ in range(10):  # Repeat 10 times and take the average
            if model_type == 'RF':
                model = RandomForestRegressor(n_estimators=best_n_estimators, 
                                              max_depth=depth, 
                                              random_state=28,
                                              n_jobs=-1)
            elif model_type == 'GB':
                model = GradientBoostingRegressor(n_estimators=best_n_estimators,
                                                  max_depth=depth, 
                                                  random_state=28)
            else:
                raise ValueError("Invalid model_type. Use 'RF' or 'GB'.")

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_error = mean_squared_error(y_train, train_pred)
            test_error = mean_squared_error(y_test, test_pred)

            train_depth_errors.append(train_error)
            test_depth_errors.append(test_error)

        train_errors.append(np.mean(train_depth_errors))
        test_errors.append(np.mean(test_depth_errors))

    min_test_error = min(test_errors)
    best_max_depth = max_depths[test_errors.index(min_test_error)]

    plt.figure(figsize=(5, 4))
    plt.plot(max_depths, train_errors, marker='o', linestyle='-', markersize=5, label='Train Error')
    plt.plot(max_depths, test_errors, marker='o', linestyle='-', markersize=5, label='Test Error')

    plt.xlabel('max_depth')
    plt.ylabel('MSE')
    plt.title(f'max_depth: {best_max_depth}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_max_depth

def find_min_samples_split(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, model_type={'GB', 'RF'}):
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    train_errors = []
    test_errors = []

    for min_split in min_samples_split_values:
        if model_type == 'RF':
            model = RandomForestRegressor(n_estimators=best_n_estimators, 
                                          max_depth=best_max_depth, 
                                          min_samples_split=min_split, 
                                          random_state=28,
                                          n_jobs=-1)
        elif model_type == 'GB':
            model = GradientBoostingRegressor(n_estimators=best_n_estimators,
                                              max_depth=best_max_depth,
                                              min_samples_split=min_split,
                                              random_state=28)
        else:
            raise ValueError("Invalid model_type. Use 'RF' or 'GB'.")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    best_min_samples_split = min_samples_split_values[np.argmin(test_errors)]
    print(f'Best min_samples_split: {best_min_samples_split}')

    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_split_values, train_errors, marker='o', linestyle='-', markersize=5, label='Train MSE')
    plt.plot(min_samples_split_values, test_errors, marker='o', linestyle='-', markersize=5, label='Test MSE')

    plt.xlabel('min_samples_split')
    plt.ylabel('MSE')
    plt.title(f'min_samples_split: {best_min_samples_split}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_min_samples_split

def find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, best_min_samples_split,
                               model_type={'GB', 'RF'}):

    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8]
    train_errors = []
    test_errors = []

    for min_leaf in min_samples_leaf_values:
        if model_type == 'RF':
            model = RandomForestRegressor(n_estimators=best_n_estimators, 
                                          max_depth=best_max_depth, 
                                          min_samples_split=best_min_samples_split,
                                          min_samples_leaf=min_leaf,
                                          random_state=28,
                                          n_jobs=-1)
        elif model_type == 'GB':
            model = GradientBoostingRegressor(n_estimators=best_n_estimators, 
                                              max_depth=best_max_depth, 
                                              min_samples_split=best_min_samples_split,
                                              random_state=28)
        else:
            raise ValueError("Invalid model_type. Use 'RF' or 'GB'.")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    best_min_samples_leaf = min_samples_leaf_values[np.argmin(test_errors)]
    print(f'Best min_samples_leaf: {best_min_samples_leaf}')

    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_leaf_values, train_errors, marker='o', linestyle='-', markersize=5, label='Train MSE')
    plt.plot(min_samples_leaf_values, test_errors, marker='o', linestyle='-', markersize=5, label='Test MSE')

    plt.xlabel('min_samples_leaf')
    plt.ylabel('MSE')
    plt.title(f'min_samples_leaf: {best_min_samples_leaf}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_min_samples_leaf
