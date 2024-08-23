from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

def df_get_fpgrowth_cols(raw_float_columns, df, max_len):
    float_columns_add_BS_mg_dl = raw_float_columns + ['BS_mg_dl']

    scaler = StandardScaler()
    scaler.fit(df[float_columns_add_BS_mg_dl])
    df_numeric_scaled = scaler.transform(df[float_columns_add_BS_mg_dl])
    # print(df_numeric_scaled)

    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    est.fit(df_numeric_scaled)
    df_continuous_binned = est.transform(df_numeric_scaled)
    # print(df_continuous_binned)

    # 取得轉換後的特徵名稱
    kbin_names = [f"{col}_kbins5" for col in float_columns_add_BS_mg_dl]
    # 創建包含 bin_names 的 DataFrame
    df_kbinned = pd.DataFrame(df_continuous_binned, columns=kbin_names)
    # df_kbinned.to_csv('231211 float_columns_kbins.csv', index=False)

    df_str = df_kbinned.astype(str)
    df_dum = pd.get_dummies(df_str)
    df_dum_int = df_dum.astype('int')
    # df_dum_int.to_csv('231211 df_get_dummies_kbins.csv', index=False)
    # df_dum_int.describe().T

    max_len = max_len
    frequent_itemsets = fpgrowth(df_dum_int, min_support=0.1, use_colnames=True, max_len=max_len)

    # 仅选择具有两个或更多项的频繁项集
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)>=max_len)]
    print(frequent_itemsets)
    # frequent_itemsets.to_csv('231211 fpgrowth_len_3.csv', index = None)

    # 选择包含任何 'BS_mg_dl_bins5_x.x' 项的频繁项集
    frequent_itemsets_with_BS = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: any('BS_mg_dl_kbins5' in item for item in x))]

    # 打印包含 'BS_mg_dl_bins5_x.x' 项的频繁项集
    print(frequent_itemsets_with_BS)
    # frequent_itemsets_with_BS.to_csv('231211 frequent_itemsets_with_BS.csv', index=False)

    itemsets_series = frequent_itemsets_with_BS['itemsets']

    # 將 'itemsets' 中的所有 frozenset 併在一起
    all_itemsets = set().union(*itemsets_series)

    # 移除包含 'BS_mg_dl_kbins5_' 的值
    filtered_itemsets = {item for item in all_itemsets if 'BS_mg_dl_kbins5_' not in item}

    # 創建一個新的 DataFrame，每一行只包含一個 frozenset
    result_df = pd.DataFrame({'filtered_itemsets': list(filtered_itemsets)})
    
    # 移除values中的右側四個string
    result_df['filtered_itemsets'] = result_df['filtered_itemsets'].str[:-4]

    print(result_df)
    return result_df