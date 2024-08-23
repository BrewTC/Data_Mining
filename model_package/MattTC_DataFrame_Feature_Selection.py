import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# 特徵選擇(RF feature_importances)
def RF_reg_feature_importances(X_train, y_train, csv_file_name='RF_reg_feature_importances'):
    reg = RandomForestRegressor(n_estimators=100, random_state=28, n_jobs=-1)
    reg.fit(X_train, y_train)

    # Get feature importances
    feature_importances = reg.feature_importances_

    # Create a DataFrame to store feature names and their importances
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

    # Filter features with importance > 0
    feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0.001]

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print the sorted feature importances
    print(feature_importance_df)

    feature_importance_list = feature_importance_df['Feature'].to_list()
    pd.Series(feature_importance_list, name='RF_reg_feature_imp').to_csv(csv_file_name + '.csv', index=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances, tick_label=X_train.columns)
    plt.xlabel('Feature Name')
    plt.ylabel('Feature Importance Score')
    plt.title('Feature Importance in Random Forest')
    plt.xticks(rotation=45)  # Set the rotation of the x-axis labels to 45 degrees
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()

    return feature_importance_df, feature_importance_list

# 特徵選擇(SVR features_selection)
def SVR_poly_Sc_features_selection(df, feature_importance_list, y_label='BS_mg_dl', features_number=30):
    
    # 將目標變量 y 分離出來，並從 DataFrame 中刪除
    y = df[y_label].copy()
    df = df[feature_importance_list]

    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    float_columns = df.select_dtypes(include=['float']).columns.tolist()
    int_columns = df.select_dtypes(include=['int']).columns.tolist()
    print(object_columns)
    print("Float Columns:", float_columns)
    print("Int Columns:", int_columns)
    print(len(float_columns+int_columns))


    X_train, X_test, y_train, y_test = train_test_split(
        df[float_columns + int_columns], y , test_size=0.1, random_state=28
    )

    # 假设您已经定义了cat_linear_processor和num_linear_processor
    cat_linear_processor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) # handle_unknown="use_encoded_value", unknown_value=-1
    num_linear_processor = StandardScaler()

    # 列变换器，将cat_linear_processor应用于整数列，将num_linear_processor应用于浮点数列
    linear_preprocessor = make_column_transformer(
        (cat_linear_processor, int_columns),  # int_columns是包含整数列名称的列表
        (num_linear_processor, float_columns)  # float_columns是包含浮点数列名称的列表
    )

    # X_train_transformed = linear_preprocessor.fit_transform(X_train)
    # X_test_transformed = linear_preprocessor.transform(X_test)
    # X_train_transformed.shape, X_test_transformed.shape 

    degree_values = list(range(1, 15))  # 尝试1到5次多项式
    mse_values = []
    for degree in degree_values:
        svr_poly_pipeline = make_pipeline(linear_preprocessor, 
                                            SVR(kernel='poly', degree=degree))
        svr_poly_pipeline.fit(X_train, y_train)
        y_pred = svr_poly_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)
    # 找到具有最小MSE的degree值
    best_degree = degree_values[mse_values.index(min(mse_values))]

    new_reg = make_pipeline(linear_preprocessor, 
                            SVR(kernel='poly', degree=best_degree)) 

    new_reg.fit(X_train, y_train)
    result = permutation_importance(new_reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    # sorted_idx = result.importances_mean.argsort()

    # 获取特征名字
    feature_names = X_train.columns.to_list()

    # 根据特征重要性排序
    sorted_feature_importance = sorted(zip(feature_names, result.importances_mean), key=lambda x: x[1], reverse=True)

    # 显示排名前n个特征的名字和重要性
    top_n = features_number  # 前n个特征

    # 提取前n个特征和其对应的重要性值
    top_features = [item[0] for item in sorted_feature_importance[:top_n]]
    # top_importance = [item[1] for item in sorted_feature_importance[:top_n]]
    return top_features
