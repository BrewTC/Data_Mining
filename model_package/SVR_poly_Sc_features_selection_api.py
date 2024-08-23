import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

def SVR_poly_Sc_features_selection_api(df, y, features_number):
    # y = df['BS_mg_dl']

    if 'BS_mg_dl' in df.columns:
        df = df.drop('BS_mg_dl', axis=1)

    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    float_columns = df.select_dtypes(include=['float']).columns.tolist()
    int_columns = df.select_dtypes(include=['int']).columns.tolist()
    print(object_columns)
    print("Float Columns:", float_columns)
    print("Int Columns:", int_columns)
    print(len(float_columns+int_columns))


    X_train, X_test, y_train, y_test = train_test_split(
        df[float_columns + int_columns], y, test_size=0.1, random_state=28
    )

    # 假设您已经定义了cat_linear_processor和num_linear_processor
    cat_linear_processor = OrdinalEncoder() # handle_unknown="use_encoded_value", unknown_value=-1
    num_linear_processor = StandardScaler()

    # 列变换器，将cat_linear_processor应用于整数列，将num_linear_processor应用于浮点数列
    linear_preprocessor = make_column_transformer(
        (cat_linear_processor, int_columns),  # int_columns是包含整数列名称的列表
        (num_linear_processor, float_columns)  # float_columns是包含浮点数列名称的列表
    )

    X_train_transformed = linear_preprocessor.fit_transform(X_train)
    X_test_transformed = linear_preprocessor.transform(X_test)
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
