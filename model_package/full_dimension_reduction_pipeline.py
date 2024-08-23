import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def full_dimension_reduction_pipeline(df_processed, target_column, top_features, n_components_kpca=20, n_components_pca=20, test_size=0.2, random_state=28, n_repeats=10):

    def LinearKernelPCA(X, n_components):
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('kpca', KernelPCA(n_components=n_components, kernel='linear', random_state=random_state, n_jobs=-1))
        ])
        X_kpca = pipeline.fit_transform(X)
        with open('./AI_MD/linear_kpca_pipeline.pkl', 'wb') as file:
            pickle.dump(pipeline, file)
        print("KPCA模型已保存!")
        pca_columns = [f'Linear_KPCA{i+1}' for i in range(n_components)]
        df_linear_kpca = pd.DataFrame(X_kpca, columns=pca_columns)
        explained_variance = np.var(X_kpca, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        return pipeline, df_linear_kpca, cumulative_explained_variance

    def AutoPrincipalComponentsAnalysis(X, n_components):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=random_state))
        ])
        X_pca = pipeline.fit_transform(X)
        with open('./AI_MD/linear_pca_pipeline.pkl', 'wb') as file:
            pickle.dump(pipeline, file)
        print("PCA模型已保存!")
        pca_columns = [f'PCA{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)
        explained_variance = np.var(X_pca, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        return pipeline, df_pca, cumulative_explained_variance

    def evaluate_dimension_reduction(X, y, df_kpca, df_pca, df_combined, cum_explained_kpca, cum_explained_pca):
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6]
        results_kpca = []
        results_pca = []
        results_combined = []

        for threshold in thresholds:
            # KPCA
            n_components_kpca = np.argmax(cum_explained_kpca >= threshold) + 1
            selected_features_kpca = df_kpca.iloc[:, :n_components_kpca]

            X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca = train_test_split(selected_features_kpca, y, test_size=0.3, random_state=random_state)
            model_kpca = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            model_kpca.fit(X_train_kpca, y_train_kpca)

            y_pred_kpca = model_kpca.predict(X_test_kpca)
            r2_kpca = r2_score(y_test_kpca, y_pred_kpca)
            mse_kpca = mean_squared_error(y_test_kpca, y_pred_kpca)
            results_kpca.append((n_components_kpca, r2_kpca, mse_kpca))

            # PCA
            n_components_pca = np.argmax(cum_explained_pca >= threshold) + 1
            selected_features_pca = df_pca.iloc[:, :n_components_pca]

            X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(selected_features_pca, y, test_size=0.3, random_state=random_state)
            model_pca = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            model_pca.fit(X_train_pca, y_train_pca)
            y_pred_pca = model_pca.predict(X_test_pca)
            r2_pca = r2_score(y_test_pca, y_pred_pca)
            mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
            results_pca.append((n_components_pca, r2_pca, mse_pca))

            # Combined
            selected_features_combined = df_combined.iloc[:, :n_components_kpca + n_components_pca]
            
            X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(selected_features_combined, y, test_size=0.3, random_state=random_state)
            model_combined = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            model_combined.fit(X_train_combined, y_train_combined)
            y_pred_combined = model_combined.predict(X_test_combined)
            r2_combined = r2_score(y_test_combined, y_pred_combined)
            mse_combined = mean_squared_error(y_test_combined, y_pred_combined)
            results_combined.append((n_components_kpca + n_components_pca, r2_combined, mse_combined))

        # Find the best number of components and R² for each method
        best_index_kpca = np.argmax([r2 for _, r2, _ in results_kpca])
        best_n_kpca, best_r2_kpca = results_kpca[best_index_kpca][0], results_kpca[best_index_kpca][1]
        print(f'KPCA:     n={best_n_kpca}, R²={best_r2_kpca:.2f}')

        best_index_pca = np.argmax([r2 for _, r2, _ in results_pca])
        best_n_pca, best_r2_pca = results_pca[best_index_pca][0], results_pca[best_index_pca][1]
        print(f'PCA:      n={best_n_pca}, R²={best_r2_pca:.2f}')

        best_index_combined = np.argmax([r2 for _, r2, _ in results_combined])
        best_n_combined, best_r2_combined = results_combined[best_index_combined][0], results_combined[best_index_combined][1]
        print(f'Combined: n={best_n_combined}, R²={best_r2_combined:.2f}')

        return best_n_kpca, best_n_pca, results_kpca, results_pca, results_combined

    def combine_features(n_components_kpca, n_components_pca, X, y):
        linear_kpca_pipeline, df_linear_kpca, cum_explained_kpca = LinearKernelPCA(X, n_components_kpca)
        pca_pipeline, df_pca, cum_explained_pca = AutoPrincipalComponentsAnalysis(X, n_components_pca)

        # 选择 KPCA 前 n 项特征和 PCA 前 n 项特征
        selected_features_kpca = df_linear_kpca.iloc[:, :n_components_kpca]
        selected_features_pca = df_pca.iloc[:, :n_components_pca]
        
        df_combined = pd.concat([selected_features_kpca, selected_features_pca], axis=1)
        print()
        print("FeatureUnion转换前的数据大小:", df_combined.shape)

        feature_pipeline = Pipeline([
            ('combined_features', FeatureUnion([
                ('linear_kpca', Pipeline([('linear_kpca', linear_kpca_pipeline)])),
                ('pca', Pipeline([('pca', pca_pipeline)]))
                ]))
        ])

        X_combined = feature_pipeline.fit_transform(df_combined)
        union_columns = [f'union{i+1}' for i in range(n_components_kpca + n_components_pca)]
        X_combined_df = pd.DataFrame(X_combined, columns=union_columns)
        print("FeatureUnion转换后的数据大小:", X_combined_df.shape)

        with open('./AI_MD/combined_features_pipeline.pkl', 'wb') as file:
            pickle.dump(feature_pipeline, file)
        print("FeatureUnion模型已保存！")

        return X_combined_df, feature_pipeline, df_linear_kpca, df_pca, cum_explained_kpca, cum_explained_pca

    # Load dataset
    X = df_processed[top_features]
    y = df_processed[target_column].values
    print("数据集大小:", X.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform KPCA and PCA
    linear_kpca_pipeline, df_linear_kpca, cum_explained_kpca = LinearKernelPCA(X_train, n_components_kpca)
    pca_pipeline, df_pca, cum_explained_pca = AutoPrincipalComponentsAnalysis(X_train, n_components_pca)

    # Combine the features
    df_combined = pd.concat([df_linear_kpca, df_pca], axis=1)

    # Evaluate dimension reduction
    best_n_kpca, best_n_pca, results_kpca, results_pca, results_combined = evaluate_dimension_reduction(X_train, y_train, df_linear_kpca, df_pca, df_combined, cum_explained_kpca, cum_explained_pca)
    print(best_n_kpca, best_n_pca)

    # Combine features based on best n_components
    X_combined_df, feature_pipeline, df_linear_kpca, df_pca, cum_explained_kpca, cum_explained_pca = combine_features(best_n_kpca, best_n_pca, X_train, y_train)

    return X_combined_df, feature_pipeline, df_linear_kpca, df_pca, cum_explained_kpca, cum_explained_pca, best_n_kpca, best_n_pca

# # 使用範例
# data_path = 'df_processed.csv'
# target_column = 'BS_mg_dl'
# top_features = ['feature1', 'feature2', 'feature3', 'feature4']  # 替换为实际的特征名称
# X_combined_df, feature_pipeline, df_linear_kpca, df_pca, cum_explained_kpca, cum_explained_pca, best_n_kpca, best_n_pca = full_dimension_reduction_pipeline(data_path, target_column, top_features, n_components_kpca=20, n_components_pca=20)
