# 乾 protobuf==4.21.0之後 OPTB套件會出錯 .......
# !pip install protobuf==4.20.0
import pandas as pd
import numpy as np
import os, joblib, pickle
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, MinMaxScaler
from optbinning import ContinuousOptimalPWBinning
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from model_package import columns_select_dtypes_api

# 移除不使用的特徵項目
def csv_file_preprocess(filename):

    df = pd.read_csv(str(filename)+'.csv')
    print("Initial memory usage:")
    df.info(memory_usage='deep')
    print(df.info())
    print(df.shape)

    columns_to_fill = ['stability', 'HT_symptom', 'userstatus']
    # 檢查這些列是否存在於 DataFrame 中
    existing_columns = [col for col in columns_to_fill if col in df.columns]
    # 如果存在，則對這些列進行填充操作
    if existing_columns:
        df[existing_columns] = df[existing_columns].fillna(-1)

    columns_to_remove = [
        'ecg_filename', # 'realBS', 'mealstatus', 
        'year','month','day','hour','minute','second','time_date',
        'sample','id','User_name','file_name','lp4','id_num','cha_2',
        'cha_url', 'Probable_Lead', 'HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5',
        'HRV_SDNNI5','HRV_ULF','HRV_VLF', 'HRV_LF', 'HRV_LFHF','HRV_LFn','HRV_MSEn','HRV_CMSEn','HRV_RCMSEn',
        'FD_Higushi','RRV_RMSSD','RRV_MeanBB', 'RRV_SDBB', 'RRV_CVBB','RRV_CVSD','RRV_MedianBB','RRV_MadBB','RRV_MCVBB',
        'RRV_VLF','RRV_LF','RRV_HF', 'RRV_LFHF', 'RRV_LFn','RRV_HFn','RRV_SD1','RRV_SD2','RRV_SD2SD1', 'RRV_ApEn','RRV_SampEn',
        'Sample_Entropy', 'User Name', 'file name', 'cha', 
        'realSYS', 'realDIA', 'realHR',	'realHR2', 'realSYS2', 'realDIA2', 'realBS2',
        'neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 
        'AI_BS',
        # 'AI_SYS', 'AI_DIA', 'AI_MEDIC', 'AI_EAT_pa', 'AI_DIS', 'AI_DIS_pa',
        # 'AI_Heart_age', 'AI_Depression', 'AI_Sqc', 'AI_BScut140',
        # 'AI_bshl', 'AI_bshl_pa', 'AI_dis', 'AI_dis_pa','AI_medic',
        'fatigue', 
        'HRV_SampEn', 'qtc_state', 
        'eat_last_T', 'waistline', 'drink_w', 'low_bs', 'sport', 'sleep', 'family_sym',	'eat_last_T',
        'VLF/P','ULF/P',
        # 'medicationstatus', 
        'CHA', 'dev_type', 'mood_state', 'md_num', 
        # 'BPc_dia', 'BPc_sys', 
        'BSPS3',
        'R_height', 'T_height',
        'ai_sec_bs', 'ai_sec_dia', 'ai_sec_sys', 'dis0bs1_0', 'dis0bs1_1', 'dis1bs1_0', 'dis1bs1_1', 
        # 'ecg_Arr', 'ecg_Arr%', 'ecg_PVC', 'ecg_PVC%',
        'ecg_QTc_state', 'ecg_Rbv', 'ecg_Tbv',
        'skin_touch', 'sym_score_shift066',	'sys', 't_error', 'unhrv',
        'waybp1_0_dia', 'waybp1_0_sys', 'waybp1_1_dia', 'waybp1_1_sys', 'waybs1_0', 'waybs1_1', 
        'AI_EAT',
        'HRV_Cd', 'HRV_Cd.1', 'Unnamed: 0', 'phone_nm',
        'RRV_SDSD', 'DFA_1',
        'AI_3d_meal', 'AI_3d_meal_pa', 'AI_P10_meal_2c', 'AI_P10_meal_2c_pa', 'AI_P15_meal_2c', 'AI_P15_meal_2c_pa', 'AI_P8_meal_2c', 'AI_P8_meal_2c_01pa'
    ]

    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    df.drop(columns=columns_to_remove, inplace=True)
    
    print("Initial memory usage:")
    df.info(memory_usage='deep')
    # df.dropna(axis=0, how='any', inplace=True)

    df = df[df.ne('no_data').all(axis=1)]
    df = df[df.ne('error').all(axis=1)]
    # df = df[df['sex'].apply(lambda x: isinstance(x, int))]
    # df = df[df['old'].apply(lambda x: isinstance(x, int))]

    """ 
    label_encoder = LabelEncoder()
    for column_name in df.columns:
        
        if df[column_name].dtype == 'object':
            
            df[column_name]= label_encoder.fit_transform(df[column_name])

            mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            data = pd.Series(mapping, name=column_name)
            data.to_csv("{}_df_label_encoder.csv".format(column_name), index=True) 
    """
    print(df.shape)
    df.to_csv(filename + "_clean_df.csv", index=False)

    return df
'''
# mealstatus126, 34)->(0, 1), add 'meal_time', 'old_cate', 'BMI_cate'
def process_data(df):
    # 篩選mealstatus
    mealstatus_12634 = df['mealstatus'].isin([1, 2, 3, 4, 6])
    df = df[mealstatus_12634]

    # 替換mealstatus的值
    replacement_mapping = {1: 0, 2: 0, 6: 0, 3: 1, 4: 1}
    df['mealstatus'] = df['mealstatus'].replace(replacement_mapping)

    # 切分時間並替換
    df['meal_time'] = pd.cut(
        df['recode_T'],
        bins=[0, 400, 1031, 1631, 2231, 2359], 
        labels=['0-400', '400-1031', '1031-1631', '1631-2231', '2231-2359'],
        right=False
    )
    replace_dict = {
        '0-400': 3, '400-1031': 0, '1031-1631': 1, '1631-2231': 2, '2231-2359': 3
    }
    df['meal_time'] = df['meal_time'].replace(replace_dict)
    df = df[df['meal_time'].isin([0, 1, 2, 3])]

    # 切分年齡並替換
    df['old_cate'] = pd.cut(
        df['old'],
        bins=[18, 45, 66, 101], 
        labels=['18-44', '45-65', '65-101'],
        right=False
    )
    replace_dict = {'18-44': 0, '45-65': 1, '65-101': 2}
    df['old_cate'] = df['old_cate'].replace(replace_dict)
    df = df[df['old_cate'].isin([0, 1, 2])]

    # 切分BMI並替換
    df['BMI_cate'] = pd.cut(
        df['BMI'],
        bins=[18, 26, 31, 41], 
        labels=['18-25', '25-30', '30-40'],
        right=False
    )
    replace_dict = {'18-25': 0, '25-30': 1, '30-40': 2}
    df['BMI_cate'] = df['BMI_cate'].replace(replace_dict)
    df = df[df['BMI_cate'].isin([0, 1, 2])]

    return df
'''

# ----------------------------------------------------------------------------
def check_nan_and_inf(df):
    print("Memory usage at start of check_nan_and_inf:")
    df.info(memory_usage='deep')

    unnecessary_columns = ['filename', 'ecg_filename', 'AI_EAT', 'group', 'HRV_Cd', 'miny_local_total'] # , 'BSc'
    df.drop(columns=unnecessary_columns, inplace=True, errors='ignore')
    df.dropna(inplace=True)  # Remove rows with NaN values

    df.columns = [s.replace('/', '_').replace('"', '_') for s in df.columns]
    print('Raw data shape: ')
    print(df.shape)

    columns_to_remove = df.columns[df.isin([np.inf, -np.inf, np.nan]).any()]

    df = df.drop(columns=columns_to_remove)
    print('Columns to remove: ', len(columns_to_remove))
    print(df.shape)

    print("Memory usage after removing problematic columns:")
    df.info(memory_usage='deep')

    # 检查 DataFrame 中是否存在空值
    has_nan = df.isna().any().any()

    # 检查 DataFrame 中是否存在 'inf'
    has_inf = df.applymap(lambda x: isinstance(x, (int, float)) and (x == float('inf') or x == float('-inf'))).any().any()

    # 使用 applymap 和 lambda 函數來檢查 'error'，並使用 any 來找到包含 'error' 的行
    error_rows = df.apply(lambda row: row.astype(str).str.contains('error').any(), axis=1)

    # error_rows 是一個布林序列，True 表示行包含 'error'
    error_indices = error_rows[error_rows].index.tolist()

    print('----------------------------')
    if has_nan:
        print("DataFrame 中存在 NaN")
    else:
        print("DataFrame 中不存在 NaN")

    if has_inf:
        print("DataFrame 中存在 'inf'")
    else:
        print("DataFrame 中不存在 'inf'")
    
    if error_indices:
        print("DataFrame 中存在 'error':\n", error_indices)
    else:
        print("DataFrame 中不存在 'error'")

    # 移除包含 'error' 的行
    df = df.loc[~error_rows]

    # 如果需要，打印更新後的 DataFrame，以確認 'error' 行已被移除
    
    print(df.shape)

    return df

def compute_float_integer_columns(df):
    # 直接在DataFrame中排除 'BS_mg_dl' 列，並分類其它列的數據類型
    df_select = df.loc[:, df.columns != 'BS_mg_dl']
    raw_float_columns, raw_int_columns = columns_select_dtypes_api.df_select_dtypes_list(df_select)
    print(f"Float columns: {len(raw_float_columns)}\n", raw_float_columns, f"\nInteger columns: {len(raw_int_columns)}\n", raw_int_columns)
    return raw_float_columns, raw_int_columns

# 標準化連續型特徵
def standardize_and_save(df, raw_float_columns, directory='./AI_MD/', scaler_filename='standard_scaler_model.joblib'):
    """
    Standardize the specified columns of a DataFrame and save the scaler model.

    Parameters:
    - df: DataFrame to standardize.
    - columns: List of column names to standardize.
    - directory: Directory to save the scaler model.
    - scaler_filename: Filename for the saved scaler model.

    Returns:
    - DataFrame with standardized columns.
    """
    print("Memory usage before standardization:")
    df.info(memory_usage='deep')

    # 檢查並創建存儲目錄
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 資料進行標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[raw_float_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=raw_float_columns)
    
    # 儲存標準化模型
    scaler_path = os.path.join(directory, scaler_filename)
    joblib.dump(scaler, scaler_path)

    print("Memory usage after standardization:")
    scaled_df.info(memory_usage='deep')
    
    return scaled_df

# 類別型分箱方法進行KBins(不用使用y)
def categorical_binning(scaled_df, raw_float_columns, directory='./AI_MD/', discretizer_filename='discretizer_model.joblib'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans') # quantile
    est.fit(scaled_df)
    df_continuous_binned = est.transform(scaled_df)

    kbin_names = [f"{col}_kbins5" for col in raw_float_columns]
    df_kbinned = pd.DataFrame(df_continuous_binned, columns=kbin_names)

    est_path = os.path.join(directory, discretizer_filename)
    joblib.dump(est, est_path)

    return df_kbinned

# 連續型分箱方法進行Optb轉換(需要使用y)
def continuous_binning(df, scaled_df, raw_float_columns, target_column, directory='./AI_MD/', max_n_bins=5):
    if not os.path.exists(directory):
        os.makedirs(directory)

    optb_data = []
    optb_models = {}

    for col in raw_float_columns:
        optb = ContinuousOptimalPWBinning(name=col, max_n_bins=max_n_bins, solver="highs", objective="l1")
        optb.fit(scaled_df[col], df[target_column])
        optbins = optb.transform(scaled_df[col])

        optb_name = f"{col}_optb{max_n_bins}"
        optb_series = pd.Series(optbins, name=optb_name)
        optb_data.append(optb_series)

        optb_filename = os.path.join(directory, f'{col}_optb_model.joblib')
        joblib.dump(optb, optb_filename)
        optb_models[col] = optb_filename

    df_optbinned = pd.concat(optb_data, axis=1)
    optb_models_filename = os.path.join(directory, 'optb_models_dict.joblib')
    joblib.dump(optb_models, optb_models_filename)

    return df_optbinned

# 連續型分群方法
def cluster_columns(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
    # 确保模型存储目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)

    def silhouette_score_method(data, max_k=15):
        silhouettes = []
        for k in range(5, max_k + 1):  # 从群数为5开始
            kmeans = KMeans(n_clusters=k, random_state=28)
            kmeans.fit(data.reshape(-1, 1))
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette = silhouette_score(data.reshape(-1, 1), kmeans.labels_)
                silhouettes.append(silhouette)
            else:
                silhouettes.append(-1)
        best_k = np.argmax(silhouettes) + 5  # 加5是因为从5开始
        return best_k

    cluster_data = []  # 创建一个空列表以收集所有聚类列

    # 遍历每一个列名
    for column_name in raw_float_columns:
        best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
        kmeans = KMeans(n_clusters=best_k_silhouette, random_state=28)
        kmeans.fit(scaled_df[column_name].values.reshape(-1, 1))
        kmeans_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_model.joblib')
        joblib.dump(kmeans, kmeans_filename)
        cluster_data.append(pd.Series(kmeans.labels_, name=f"{column_name}_cluster{best_k_silhouette}"))

    # 使用 pd.concat 将所有列合并成一个 DataFrame
    cluster_df = pd.concat(cluster_data, axis=1)
    return cluster_df

from sklearn.mixture import GaussianMixture
def cluster_columns_gmm(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
    if not os.path.exists(directory):
        os.makedirs(directory)

    def silhouette_score_method(data, max_k=15):
        silhouettes = []
        for k in range(5, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=28)
            gmm.fit(data.reshape(-1, 1))
            labels = gmm.predict(data.reshape(-1, 1))
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data.reshape(-1, 1), labels)
                silhouettes.append(silhouette)
            else:
                silhouettes.append(-1)
        best_k = np.argmax(silhouettes) + 5
        return best_k

    cluster_data = []

    for column_name in raw_float_columns:
        best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
        gmm = GaussianMixture(n_components=best_k_silhouette, random_state=28)
        gmm.fit(scaled_df[column_name].values.reshape(-1, 1))
        labels = gmm.predict(scaled_df[column_name].values.reshape(-1, 1))
        gmm_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_gmm_model.joblib')
        joblib.dump(gmm, gmm_filename)
        cluster_data.append(pd.Series(labels, name=f"{column_name}_cluster{best_k_silhouette}_gmm"))

    cluster_df = pd.concat(cluster_data, axis=1)
    return cluster_df

# def cluster_columns_hierarchical(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     def silhouette_score_method(data, max_k=15):
#         silhouettes = []
#         for k in range(2, max_k + 1):
#             hierarchical = AgglomerativeClustering(n_clusters=k)
#             labels = hierarchical.fit_predict(data.reshape(-1, 1))
#             if len(np.unique(labels)) > 1:
#                 silhouette = silhouette_score(data.reshape(-1, 1), labels)
#                 silhouettes.append(silhouette)
#             else:
#                 silhouettes.append(-1)
#         best_k = np.argmax(silhouettes) + 2
#         return best_k

#     cluster_data = []

#     for column_name in raw_float_columns:
#         best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
#         hierarchical = AgglomerativeClustering(n_clusters=best_k_silhouette)
#         labels = hierarchical.fit_predict(scaled_df[column_name].values.reshape(-1, 1))
#         hierarchical_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_hierarchical_model.joblib')
#         joblib.dump(hierarchical, hierarchical_filename)
#         cluster_data.append(pd.Series(labels, name=f"{column_name}_cluster{best_k_silhouette}_hierarchical"))

#     cluster_df = pd.concat(cluster_data, axis=1)
#     return cluster_df

# from sklearn.cluster import SpectralClustering
# def cluster_columns_spectral(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     def silhouette_score_method(data, max_k=15):
#         silhouettes = []
#         for k in range(5, max_k + 1):
#             spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=28)
#             labels = spectral.fit_predict(data.reshape(-1, 1))
#             if len(np.unique(labels)) > 1:
#                 silhouette = silhouette_score(data.reshape(-1, 1), labels)
#                 silhouettes.append(silhouette)
#             else:
#                 silhouettes.append(-1)
#         best_k = np.argmax(silhouettes) + 5
#         return best_k

#     cluster_data = []

#     for column_name in raw_float_columns:
#         best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
#         spectral = SpectralClustering(n_clusters=best_k_silhouette, affinity='nearest_neighbors', random_state=28)
#         labels = spectral.fit_predict(scaled_df[column_name].values.reshape(-1, 1))
#         spectral_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_spectral_model.joblib')
#         joblib.dump(spectral, spectral_filename)
#         cluster_data.append(pd.Series(labels, name=f"{column_name}_cluster{best_k_silhouette}_spectral"))

#     cluster_df = pd.concat(cluster_data, axis=1)
#     return cluster_df

from sklearn.cluster import Birch
def cluster_columns_birch(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
    if not os.path.exists(directory):
        os.makedirs(directory)

    def silhouette_score_method(data, max_k=15):
        silhouettes = []
        for k in range(2, max_k + 1):
            birch = Birch(n_clusters=k)
            labels = birch.fit_predict(data.reshape(-1, 1))
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data.reshape(-1, 1), labels)
                silhouettes.append(silhouette)
            else:
                silhouettes.append(-1)
        best_k = np.argmax(silhouettes) + 2
        return best_k

    cluster_data = []

    for column_name in raw_float_columns:
        best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
        birch = Birch(n_clusters=best_k_silhouette)
        labels = birch.fit_predict(scaled_df[column_name].values.reshape(-1, 1))
        birch_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_birch_model.joblib')
        joblib.dump(birch, birch_filename)
        cluster_data.append(pd.Series(labels, name=f"{column_name}_cluster{best_k_silhouette}_birch"))

    cluster_df = pd.concat(cluster_data, axis=1)
    return cluster_df

from sklearn.cluster import AgglomerativeClustering
def cluster_columns_agglomerative(scaled_df, raw_float_columns, directory='./AI_MD/', max_k=15):
    if not os.path.exists(directory):
        os.makedirs(directory)

    def silhouette_score_method(data, max_k=15):
        silhouettes = []
        for k in range(2, max_k + 1):
            agglomerative = AgglomerativeClustering(n_clusters=k)
            labels = agglomerative.fit_predict(data.reshape(-1, 1))
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data.reshape(-1, 1), labels)
                silhouettes.append(silhouette)
            else:
                silhouettes.append(-1)
        best_k = np.argmax(silhouettes) + 2
        return best_k

    cluster_data = []

    for column_name in raw_float_columns:
        best_k_silhouette = silhouette_score_method(scaled_df[column_name].values, max_k)
        agglomerative = AgglomerativeClustering(n_clusters=best_k_silhouette)
        labels = agglomerative.fit_predict(scaled_df[column_name].values.reshape(-1, 1))
        agglomerative_filename = os.path.join(directory, f'{column_name}_cluster{best_k_silhouette}_agglomerative_model.joblib')
        joblib.dump(agglomerative, agglomerative_filename)
        cluster_data.append(pd.Series(labels, name=f"{column_name}_cluster{best_k_silhouette}_agglomerative"))

    cluster_df = pd.concat(cluster_data, axis=1)
    return cluster_df


def minmax_scaler(df, raw_float_columns, directory='./AI_MD/'):
    # 確認每個列名都在原始DataFrame中
    assert all(column in df.columns for column in raw_float_columns), "Some columns listed in 'raw_float_columns' are not in the dataframe."

    df_columns = [column + '_MinMaxSc' for column in raw_float_columns]
    print("raw_float_columns:", raw_float_columns)  # 列印原始浮點列名
    print("Number of raw_float_columns:", len(raw_float_columns))  # 列印原始浮點列數量
    print("Number of df_columns:", len(df_columns))  # 列印預期的轉換後列數量

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[raw_float_columns])
    print("Shape of scaled_data:", scaled_data.shape)  # 列印轉換後的數據形狀

    scaler_path = os.path.join(directory, 'scaler_MinMaxSc.pkl')
    joblib.dump(scaler, scaler_path)

    if len(df_columns) != scaled_data.shape[1]:
        raise ValueError("Mismatch between the number of provided column names and the number of columns in scaled data")

    scaled_df = pd.DataFrame(scaled_data, columns=df_columns)
    # 使用字典來儲存新增的列，最後再合併
    feature_cut_points = {}
    new_columns = {}
    bin_columns = {}
    
    for column in df_columns:
        log_column = f'{column}_log2'
        new_columns[log_column] = scaled_df[column].apply(lambda x: np.log2(x) if x > 0 else 0)

        raw_min = scaled_df[column].min()
        raw_max = scaled_df[column].max()
        raw_interval = (raw_max - raw_min) / 5
        log_cut_points = [np.log2(raw_min + i * raw_interval) for i in range(1, 5)]
        log_cut_points = np.unique(log_cut_points)
        feature_cut_points[log_column] = log_cut_points
        
        # 計算 bin 分類
        cut_points = feature_cut_points[log_column]
        bins = np.concatenate(([-np.inf], cut_points, [np.inf]))
        label_column = f"{log_column}_bin5"
        bin_columns[label_column] = pd.cut(new_columns[log_column], bins=bins, labels=False, include_lowest=True)

    # 新增所有計算好的列
    log_df = pd.DataFrame(new_columns)
    bin_df = pd.DataFrame(bin_columns)
    minmax_log2_df = pd.concat([scaled_df, log_df, bin_df], axis=1)

    feature_cut_points_path = os.path.join(directory, 'feature_cut_points_model.pkl')
    joblib.dump(feature_cut_points, feature_cut_points_path)

    return minmax_log2_df

def input_reg_csv_to_preprocess(csv_file_name):
    # csv_file_name = 'dia_03_user_0_cate1'
    df = pd.read_csv(csv_file_name + '.csv')
    df = check_nan_and_inf(df)
    # df = process_data(df)
    raw_float_columns, raw_int_columns = compute_float_integer_columns(df)

    scaled_df = standardize_and_save(df, raw_float_columns)
    
    df_kbinned = categorical_binning(scaled_df, raw_float_columns)
    df_optbinned = continuous_binning(df, scaled_df, raw_float_columns, 'BS_mg_dl')
    cluster_df = cluster_columns(scaled_df, raw_float_columns) 
    minmax_log2_df = minmax_scaler(df, raw_float_columns)
    cluster_gmm_df = cluster_columns_gmm(scaled_df, raw_float_columns)
    # cluster_hierarchical_df = cluster_columns_hierarchical(scaled_df, raw_float_columns)
    # cluster_spectral_df = cluster_columns_spectral(scaled_df, raw_float_columns)
    cluster_birch_df = cluster_columns_birch(scaled_df, raw_float_columns)
    # cluster_agglomerative_df = cluster_columns_agglomerative(scaled_df, raw_float_columns)

    # 重設索引
    df.reset_index(drop=True, inplace=True)
    # scaled_df.reset_index(drop=True, inplace=True)
    df_kbinned.reset_index(drop=True, inplace=True)
    df_optbinned.reset_index(drop=True, inplace=True)
    cluster_df.reset_index(drop=True, inplace=True)
    minmax_log2_df.reset_index(drop=True, inplace=True)
    cluster_gmm_df.reset_index(drop=True, inplace=True)
    # cluster_hierarchical_df.reset_index(drop=True, inplace=True)
    # cluster_spectral_df.reset_index(drop=True, inplace=True)
    cluster_birch_df.reset_index(drop=True, inplace=True)
    # cluster_agglomerative_df.reset_index(drop=True, inplace=True)

    print('scaled_data:', scaled_df.shape)
    print('kbinned_data:', df_kbinned.shape)
    print('optbinned_data:', df_optbinned.shape)
    print('cluster_data:', cluster_df.shape)
    print('minmax_log2_data:', minmax_log2_df.shape)
    print('cluster_gmm_data:', cluster_gmm_df.shape)
    # print('cluster_hierarchical_data:', cluster_hierarchical_df.shape)
    # print('cluster_spectral_data:', cluster_spectral_df.shape)
    print('cluster_birch_data:', cluster_birch_df.shape)
    # print('cluster_agglomerative_data:', cluster_agglomerative_df.shape)

    # 合併標準化後的原始資料+類別型特徵+連續型型特徵+分群特徵
    df_processed = pd.concat([
        df, 
        df_kbinned, 
        df_optbinned, 
        cluster_df, 
        minmax_log2_df, 
        cluster_gmm_df,
        # cluster_hierarchical_df,
        # cluster_spectral_df,
        cluster_birch_df,
        # cluster_agglomerative_df
    ], axis=1)

    # 添加頻率編碼
    frequency_encodings = {}
    for col in raw_int_columns:
        freq_series = df[col].value_counts(normalize=True)
        frequency_encodings[col] = freq_series
        df_processed[col + '_freq'] = df[col].map(freq_series)

    # 儲存頻率映射字典
    with open('./AI_MD/frequency_encodings.pkl', 'wb') as f:
        pickle.dump(frequency_encodings, f)

    df_processed.to_csv('df_processed.csv', index=False)
    print(df_processed.shape)

    return df_processed

# ----------------------------------------------------------------------------