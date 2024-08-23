import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def find_best_n_estimators(X_train, y_train, X_test, y_test):
    model_type = 'RandomForestClassifier'
    model = RandomForestClassifier(random_state=28, 
                                  n_jobs=-1)

    n_estimators_list = [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300]
    train_accuracy_scores = []
    test_accuracy_scores = []

    for n_estimators in n_estimators_list:
        model.n_estimators = n_estimators

        # Calculate the accuracy using cross-validation for Train and Test sets
        train_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        test_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()

        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)

    # Find the number of estimators associated with the highest average accuracy
    best_n_estimators = n_estimators_list[np.argmax(test_accuracy_scores)]
    print(f'Best n_estimators: {best_n_estimators}')

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.plot(n_estimators_list, train_accuracy_scores, label='Train Accuracy')
    plt.plot(n_estimators_list, test_accuracy_scores, label='Test Accuracy')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_type} - Best n_estimators: {best_n_estimators}')
    plt.grid(True)
    plt.show()

    return best_n_estimators

def find_best_max_depth(X_train, y_train, X_test, y_test, best_n_estimators):
    max_depths = list(range(1, 15))
    train_accuracy_scores = []
    test_accuracy_scores = []

    for depth in max_depths:
        model = RandomForestClassifier(n_estimators=best_n_estimators,
                                       max_depth=depth, 
                                       random_state=28,
                                       n_jobs=-1)
        
        # Calculate accuracy using cross-validation for Train and Test sets
        train_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        test_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()

        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)

    # Find the depth associated with the highest average accuracy
    best_max_depth = max_depths[np.argmax(test_accuracy_scores)]

    plt.figure(figsize=(5, 4))
    plt.plot(max_depths, train_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Train Accuracy')
    plt.plot(max_depths, test_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Test Accuracy')

    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title(f'max_depth: {best_max_depth}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_max_depth

def find_min_samples_split(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth):
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    train_accuracy_scores = []
    test_accuracy_scores = []

    for min_split in min_samples_split_values:
        model = RandomForestClassifier(n_estimators=best_n_estimators, 
                                        max_depth=best_max_depth, 
                                        min_samples_split=min_split, 
                                        random_state=28,
                                        n_jobs=-1)
        
        # Calculate accuracy using cross-validation for Train and Test sets
        train_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        test_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()

        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)

    # Find the min_samples_split associated with the highest average accuracy
    best_min_samples_split = min_samples_split_values[np.argmax(test_accuracy_scores)]
    print(f'Best min_samples_split: {best_min_samples_split}')

    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_split_values, train_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Train Accuracy')
    plt.plot(min_samples_split_values, test_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Test Accuracy')

    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')
    plt.title(f'min_samples_split: {best_min_samples_split}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_min_samples_split

def find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, best_min_samples_split):

    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8]
    train_accuracy_scores = []
    test_accuracy_scores = []

    for min_leaf in min_samples_leaf_values:
        model = RandomForestClassifier(n_estimators=best_n_estimators, 
                                        max_depth=best_max_depth, 
                                        min_samples_split=best_min_samples_split,
                                        min_samples_leaf=min_leaf,
                                        random_state=28,
                                        n_jobs=-1)
        
        # Calculate accuracy using cross-validation for Train and Test sets
        train_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        test_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()

        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)

    best_min_samples_leaf = min_samples_leaf_values[np.argmax(test_accuracy_scores)]
    print(f'Best min_samples_leaf: {best_min_samples_leaf}')

    plt.figure(figsize=(5, 4))
    plt.plot(min_samples_leaf_values, train_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Train Accuracy')
    plt.plot(min_samples_leaf_values, test_accuracy_scores, marker='o', linestyle='-', markersize=5, label='Test Accuracy')

    plt.xlabel('min_samples_leaf')
    plt.ylabel('Accuracy')
    plt.title(f'min_samples_leaf: {best_min_samples_leaf}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_min_samples_leaf

# Similarly, you can adjust find_min_samples_split and find_best_min_samples_leaf functions accordingly.

# Example usage:
# X_train, X_test, y_train, y_test are assumed to be defined elsewhere
# best_n_estimators = find_best_n_estimators(X_train, y_train, X_test, y_test)
# best_max_depth = find_best_max_depth(X_train, y_train, X_test, y_test, best_n_estimators)
# best_min_samples_split = find_min_samples_split(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth)
# best_min_samples_leaf = find_best_min_samples_leaf(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth, best_min_samples_split)
