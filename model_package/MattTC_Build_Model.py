from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from model_package import columns_select_dtypes_api
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(df, top_features, df_processed_y_label, test_size=0.1, random_state=28):
    # Separating the features and target
    X = df[top_features]
    #y = df[target_col]
    
    X, y = X.reset_index(drop=True), df_processed_y_label.reset_index(drop=True)
    
    # Select data types and find duplicates
    float_columns = X.select_dtypes(include=['float']).columns.tolist()
    int_columns = X.select_dtypes(include=['int']).columns.tolist()
    
    duplicate_float_columns = set([col for col in float_columns if float_columns.count(col) > 1])
    duplicate_int_columns = set([col for col in int_columns if int_columns.count(col) > 1])
    
    print("重複的浮點數型態列名稱:", duplicate_float_columns)
    print("重複的整數型態列名稱:", duplicate_int_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Check for infinite and NaN values
    train_inf = np.isinf(X_train).values.any() or np.isinf(y_train).values.any()
    test_inf = np.isinf(X_test).values.any() or np.isinf(y_test).values.any()
    train_nan = np.isnan(X_train).values.any() or np.isnan(y_train).values.any()
    test_nan = np.isnan(X_test).values.any() or np.isnan(y_test).values.any()
    
    print("訓練集中是否包含無窮大值：", train_inf)
    print("測試集中是否包含無窮大值：", test_inf)
    print("訓練集中是否包含 NaN 值 ：", train_nan)
    print("測試集中是否包含 NaN 值 ：", test_nan)
    
    # Column transformer
    cat_linear_processor = OrdinalEncoder() # handle_unknown="use_encoded_value", unknown_value=-1
    num_linear_processor = StandardScaler()
    
    linear_preprocessor = make_column_transformer(
        (cat_linear_processor, int_columns),
        (num_linear_processor, float_columns)
    )
    
    # Transform data
    X_train_transformed = linear_preprocessor.fit_transform(X_train)
    X_test_transformed = linear_preprocessor.transform(X_test)
    
    print("Transformed shapes:", X_train_transformed.shape, X_test_transformed.shape)
    
    return X_train, X_test, y_train, y_test, float_columns, int_columns

# ----------------------------------------------------------------------------
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def train_evaluate(model, X_train, y_train, X_test, y_test):
    # Cross-validation
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Making predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluation
    train_mae, train_mse, train_rmse, train_r2 = evaluate(y_train, train_pred)
    test_mae, test_mse, test_rmse, test_r2 = evaluate(y_test, test_pred)
    
    # Printing evaluation results
    print(f'Model: {model}')
    print('Cross-Validation Score:', cv_score)
    print('Train set evaluation:')
    print('MAE:', train_mae)
    print('MSE:', train_mse)
    print('RMSE:', train_rmse)
    print('R2 Square', train_r2)
    print('__________________________________')
    
    print('Test set evaluation:')
    print('MAE:', test_mae)
    print('MSE:', test_mse)
    print('RMSE:', test_rmse)
    print('R2 Square', test_r2)
    
    # Creating a DataFrame to hold the results
    results_df = pd.DataFrame(
        data=[[model.__class__.__name__, cv_score, test_mae, test_mse, test_rmse, test_r2]], 
        columns=['Model', 'CV Score', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
    
    return results_df

# Helper functions used in the main function
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

# ----------------------------------------------------------------------------

import pandas as pd
import pickle
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RANSACRegressor, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
import scipy as sp
import joblib  # 使用 joblib 进行序列化

def create_transformer_funcs(X_train, y_train):
    def ransac_lasso_transformer_func(x):
        model = RANSACRegressor(base_estimator=Lasso(random_state=28, max_iter=1000), min_samples=2, random_state=28) 
        return model.fit(X_train, y_train).predict(x).reshape(-1, 1)
    
    def ransac_elasticnet_transformer_func(x):
        model = RANSACRegressor(base_estimator=ElasticNet(random_state=28, max_iter=1000), min_samples=2, random_state=28) 
        return model.fit(X_train, y_train).predict(x).reshape(-1, 1)
    
    def huber_transformer_func(x):
        model = HuberRegressor(max_iter=1000)  # 增加迭代次数以促进收敛
        return model.fit(X_train, y_train).predict(x).reshape(-1, 1)
    
    def theilsen_transformer_func(x):
        model = TheilSenRegressor(random_state=28)
        return model.fit(X_train, y_train).predict(x).reshape(-1, 1)
    
    return ransac_lasso_transformer_func, ransac_elasticnet_transformer_func, huber_transformer_func, theilsen_transformer_func

def train_and_evaluate(rf_model, X_train, y_train, X_test, y_test, cc_int_columns, cc_float_columns):

    cat_linear_processor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num_linear_processor = StandardScaler()
    linear_preprocessor = make_column_transformer(
        (cat_linear_processor, cc_int_columns),
        (num_linear_processor, cc_float_columns)
    )
    
    rf_pipeline = TransformedTargetRegressor(
        regressor=make_pipeline(linear_preprocessor, rf_model), 
        #transformer=QuantileTransformer(n_quantiles=min(1000, len(X_train)), output_distribution='normal', random_state=28),
        #func=np.log1p, inverse_func=np.expm1
        func=np.log10, inverse_func=sp.special.exp10
    )

    # rf_pipeline = rf_model
    rf_pipeline.fit(X_train, y_train)
    train_pred = rf_pipeline.predict(X_train)
    test_pred = rf_pipeline.predict(X_test)
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    
    print('RF Pipeline Score:', rf_pipeline.score(X_train, y_train))
    print('Train MSE:', train_error)
    print('Test MSE:', test_error)
    
    joblib.dump(rf_pipeline, 'pipeline_with_preprocessor_and_RFmodel.joblib')
    
    return train_pred, test_pred, train_error, test_error

# ----------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.linear_model import RANSACRegressor, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle

class ModelFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, models):
        self.models = models
    def fit(self, X, y=None):
        for model in self.models:
            model.fit(X, y)
        return self
    def transform(self, X):
        outputs = [model.predict(X).reshape(-1, 1) for model in self.models]
        return np.hstack(outputs)
    
def train_and_evaluate_ModelFeatureUnion(rf_model, X_train, y_train, X_test, y_test, cc_int_columns, cc_float_columns):
    # 设置数据预处理器
    linear_preprocessor = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cc_int_columns),
        (StandardScaler(), cc_float_columns)
    )
    
    # 定义模型
    ransac_model = RANSACRegressor(base_estimator=Lasso(random_state=28), # , max_iter=10000
                                    min_samples=2, random_state=28)
    huber_model = HuberRegressor() # max_iter=10000
    theilsen_model = TheilSenRegressor(random_state=28)
    models = [ransac_model, huber_model, theilsen_model]
    
    # 集成模型特征生成
    model_union = ModelFeatureUnion(models=models)
    
    # 组合预处理器和模型特征生成
    model_pipeline = Pipeline([
        ('preprocessor', linear_preprocessor),
        ('feature_union', model_union),
        ('final_model', rf_model)
    ])
    
    # 设置目标变换器
    target_transformer = QuantileTransformer(n_quantiles=min(1000, len(X_train)), output_distribution='normal', random_state=28)
    
    # 构建最终的 TransformedTargetRegressor
    transformed_regressor = TransformedTargetRegressor(regressor=model_pipeline,
        #transformer=target_transformer,
        #func=np.log1p, inverse_func=np.expm1
        #func=np.log10, inverse_func=sp.special.exp10,
    )
    
    # 训练模型
    transformed_regressor.fit(X_train, y_train)
    
    # 进行预测和评估
    train_pred = transformed_regressor.predict(X_train)
    test_pred = transformed_regressor.predict(X_test)
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    
    print('RF Pipeline Score: ', transformed_regressor.score(X_train, y_train))
    print('Train MSE: ', train_error)
    print('Test MSE: ', test_error)

    # 使用pickle序列化
    try:
        with open('pipeline_with_preprocessor_and_RFmodel.pkl', 'wb') as f:
            pickle.dump(transformed_regressor, f)
    except Exception as e:
        print("Pickle error:", e)
        
# ----------------------------------------------------------------------------
# 這邊等等放所有排列組合的算法

# ----------------------------------------------------------------------------
        print("Pickle error:", e)

    return train_pred, test_pred, train_error, test_error
# ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor, Lasso, HuberRegressor, TheilSenRegressor, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import special

def run_tests(X_train, y_train, X_test, y_test, linear_preprocessor, models, transformers):
    results = []
    for model in models:
        for transformer in transformers:
            # Setup the transformed target regressor with the current model and transformer
            pipeline = TransformedTargetRegressor(
                regressor=make_pipeline(linear_preprocessor, model),
                transformer=transformer
            )

            # Train the model
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)

            # Calculate errors
            train_error = mean_squared_error(y_train, train_pred)
            test_error = mean_squared_error(y_test, test_pred)

            # Store results
            results.append({
                'model': model.__class__.__name__,
                'transformer': transformer.__class__.__name__ if not isinstance(transformer, tuple) else transformer[0].__class__.__name__ + ' with function',
                'train_error': train_error,
                'test_error': test_error
            })
    return pd.DataFrame(results), train_pred, test_pred, train_error, test_error


'''
# 定义模型
models = [
    RandomForestRegressor(n_estimators=10, random_state=28),  # Simplified for quick run
    RANSACRegressor(base_estimator=Lasso(random_state=28)),
    HuberRegressor(),
    TheilSenRegressor(random_state=28)
]

# 定义转换器
quantile_transformer = QuantileTransformer(n_quantiles=min(1000, len(X_train)), output_distribution='normal', random_state=28)
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
log10_transformer = FunctionTransformer(func=np.log10, inverse_func=special.exp10)

transformers = [
    quantile_transformer,
    log_transformer,
    log10_transformer
]

# 运行测试
df_results = run_tests(X_train, y_train, X_test, y_test, linear_preprocessor, models, transformers)
print(df_results)
'''