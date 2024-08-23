import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_deviance_curve(reg, X_train, y_train, X_test, y_test, params):
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = mean_squared_error(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

# Example usage:
# params = {
#     "n_estimators": 500,
#     "max_depth": None,
#     "min_samples_split": 2,
#     "learning_rate": 0.01,
#     "loss": "squared_error",
# }

# plot_deviance_curve(reg, X_train, y_train, X_test, y_test, params)
