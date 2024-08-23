
import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from optbinning import ContinuousOptimalPWBinning
from model_package import columns_select_dtypes_api

def df_kbins_optb_preprocess_api(csv_file, n_bins, max_n_bins):

    # csv_file = '0101_0831 Time_all_dia_12_user_0'
    df = pd.read_csv(str(csv_file) + '.csv')
    df.columns = [s.replace('/', '_') for s in df.columns]
    if 'ecg_filename' in df.columns:
        df = df.drop('ecg_filename', axis=1)

    raw_X, raw_y = df.drop('BS_mg_dl', axis=1), df['BS_mg_dl']
    raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
        raw_X, raw_y, test_size=0.1, random_state=28
    )
    print(raw_X_train.shape, raw_y_train.shape)

    raw_float_columns, raw_int_columns = columns_select_dtypes_api.df_select_dtypes_list(raw_X)
    print('Raw_float_columns: ', len(raw_float_columns))
    print('Raw_int_columns: ', len(raw_int_columns))

    # 初始化標準化器
    scaler = StandardScaler()
    scaler.fit(df[raw_float_columns])
    df_numeric_scaled = scaler.transform(df[raw_float_columns])
    # print(df_numeric_scaled)
    # 初始化 KBinsDiscretizer
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    est.fit(df_numeric_scaled)
    df_continuous_binned = est.transform(df_numeric_scaled)
    # print(df_continuous_binned)

    # 儲存模型
    directory = './AI_MD/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the scaler model
    scaler_filename = os.path.join(directory, 'scaler_model.joblib')
    joblib.dump(scaler, scaler_filename)
    # Save the KBinsDiscretizer model
    est_filename = os.path.join(directory, 'discretizer_model.joblib')
    joblib.dump(est, est_filename)

    # 取得轉換後的特徵名稱
    kbin_names = [f"{col}_kbins5" for col in raw_float_columns]
    # 創建包含 bin_names 的 DataFrame
    df_kbinned = pd.DataFrame(df_continuous_binned, columns=kbin_names)


    # 使用Sc後的資料表 
    df_numeric_scaled_df = pd.DataFrame(df_numeric_scaled, columns=raw_float_columns)

    index_object = pd.Index(raw_float_columns)

    df_optbinned = pd.DataFrame()
    optb_models = {}

    for col in index_object:
        optb = ContinuousOptimalPWBinning(name=col, max_n_bins=max_n_bins, 
                                        solver="highs", objective="l1")
        optb.fit(df_numeric_scaled_df[col], raw_y)
        optbins = optb.transform(df_numeric_scaled_df[col])

        optb_name = f"{col}_optb10"
        df_optbinned[optb_name] = optbins # .flatten()
        # 存模
        optb_filename = os.path.join(directory, f'{col}_optb_model.joblib')
        joblib.dump(optb, optb_filename)
        optb_models[col] = optb_filename
        
    # Save the dictionary containing the mapping of feature names to optb models
    optb_models_filename = os.path.join(directory, 'optb_models_dict.joblib')
    joblib.dump(optb_models, optb_models_filename)

    # print(df_optbinned)
        
    # 合併標準化後的數值型特徵和離散化後的特徵
    df_processed = pd.concat([df, df_kbinned, df_optbinned], axis=1)
    # df_processed.to_csv('231124 df_processed(kbins5, optbins10)(correct ver).csv', index=False)
    # print(df_processed)
    return df_processed