import re
import pandas as pd

def filtered_top_features_to_raw_float_columns(df_processed, top_features):
    
    print('top_features len:', len(top_features))

    def top_features_remove_suffixes(top_features):
        # 定義後綴列表
        suffixes = ['_kbin5', '_optb5', '_MinMaxSc', '_MinMaxSc_log2']
        
        unique_features = set()
        # 準備一個regex模式，匹配'_cluster'後接數字
        cluster_pattern = re.compile(r'_cluster\d+')
        
        for feature in top_features:
            # 先處理_cluster後接數字的情況
            feature = cluster_pattern.sub('', feature)
            # 處理其它後綴
            for suffix in suffixes:
                if feature.endswith(suffix):
                    feature = feature[:feature.rfind(suffix)]
                    break
            unique_features.add(feature)


        return list(unique_features)

    # 移除後綴並找出不重複的特徵名稱
    unique_features = top_features_remove_suffixes(top_features)
    print('unique_features len: ', len(unique_features), '\n', unique_features, '\n')

    # 使用字典推導創建一個字典，其中包含從 DataFrame 中取得的列
    df_dict = {feature: df_processed[feature] for feature in unique_features}
    # 轉換成 DataFrame
    df_dict_dataframe = pd.DataFrame(df_dict)
    print('Before dataframe shape:', df_processed.shape)
    print('After dataframe shape:', df_dict_dataframe.shape)
    # df_dict_dataframe.to_csv('filtered_top_features_to_raw_float_columns.csv', index=False)

    return df_dict_dataframe