import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def find_best_n_estimators(X_train, y_train, X_test, y_test):
    model_type = 'GradientBoostingRegressor'
    model = GradientBoostingRegressor(random_state=28)
    
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

def find_best_learning_rate(X_train, y_train, X_test, y_test, best_n_estimators):
    learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # List of learning rates to test
    train_mse_scores = []
    test_mse_scores = []

    best_train_mse = float('inf')
    best_test_mse = float('inf')
    best_learning_rate = None

    for learning_rate in learning_rates:
        # Create a GradientBoostingRegressor model with the current learning rate
        gb = GradientBoostingRegressor(learning_rate=learning_rate, 
                                       n_estimators=best_n_estimators, 
                                       random_state=28)

        # Calculate the mean squared error using cross-validation for Train and Test sets
        train_mse = -cross_val_score(gb, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        test_mse = -cross_val_score(gb, X_test, y_test, cv=5, scoring='neg_mean_squared_error').mean()

        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)

        # Check if this learning rate gives a better average MSE
        if train_mse < best_train_mse and test_mse < best_test_mse:
            best_train_mse = train_mse
            best_test_mse = test_mse
            best_learning_rate = learning_rate
    
    print(f'Best learning_rate: {best_learning_rate}')

    # Plot the results for Train and Test MSE
    plt.figure(figsize=(5, 4))
    plt.plot(learning_rates, train_mse_scores, label='Train MSE')
    plt.plot(learning_rates, test_mse_scores, label='Test MSE')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.title('GradientBoostingRegressor - Train and Test MSE')
    plt.grid(True)
    plt.show()

    return best_learning_rate

def find_best_max_depth(X_train, y_train, X_test, y_test, best_n_estimators, best_learning_rate):
    max_depths = list(range(1, 15))
    train_errors = []
    test_errors = []
    # mse_average_scores = []

    for depth in max_depths:
        train_depth_errors = []
        test_depth_errors = []
        for _ in range(10):  # Repeat 10 times and take the average

            model = GradientBoostingRegressor(n_estimators=best_n_estimators, 
                                              learning_rate=best_learning_rate,
                                              max_depth=depth, 
                                              random_state=28)

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
    print(f'Best max_depth: {best_max_depth}')

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

def find_min_samples_split(X_train, y_train, X_test, y_test, best_n_estimators, best_learning_rate, best_max_depth):
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    train_errors = []
    test_errors = []

    for min_split in min_samples_split_values:

        model = GradientBoostingRegressor(n_estimators=best_n_estimators,
                                          learning_rate=best_learning_rate,
                                          max_depth=best_max_depth,
                                          min_samples_split=min_split,
                                          random_state=28)

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

def find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split):

    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8]
    train_errors = []
    test_errors = []

    for min_leaf in min_samples_leaf_values:
        
        model = GradientBoostingRegressor(n_estimators=best_n_estimators,
                                          learning_rate=best_learning_rate,
                                          max_depth=best_max_depth,
                                          min_samples_split=best_min_samples_split,
                                          min_samples_leaf=min_leaf,
                                          random_state=28)
        
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
