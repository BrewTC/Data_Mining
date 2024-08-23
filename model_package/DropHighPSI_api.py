'''
Ref
https://feature-engine.trainindata.com/en/latest/user_guide/selection/DropHighPSIFeatures.html

可以使用不同的閾值來評估分佈偏移的大小，具體取決於 更改為 PSI 值。最常用的閾值是：
- 低於 10%，該變數沒有發生重大變化。
- 在25%以上，該變數經歷了重大轉變。
- 在這兩個值之間，偏移是中間的。
- "auto"：閾值將根據基礎數據集和目標數據集的大小以及條柱的數量進行計算。
'''
import pandas as pd
from feature_engine.selection import DropHighPSIFeatures
# import seaborn as sns

def DropHighPSI_reg_api(df, y):

    drop_target = str(y)

    if drop_target in df.columns:
        df = df.drop(drop_target, axis=1)
    else:
        pass

    # 回歸模型使用的分類法
    # Remove the features with high PSI values using a 60-40 split.
    transformer = DropHighPSIFeatures(split_frac=0.6)
    transformer.fit(df)

    print('各特徵的PSI值: \n', transformer.psi_values_)
    print('觀測值的切割點: \n', transformer.cut_off_)

    print(f'要移除的{len(transformer.features_to_drop_)}項特徵: \n', transformer.features_to_drop_)

    keep_list = [item for item in df.columns.tolist() if item not in transformer.features_to_drop_]

    print(f'要保留的{len(keep_list)}項特徵: \n', keep_list)
    
    # df['above_cut_off'] = df.index > transformer.cut_off_
    # sns.ecdfplot(data=df, x='old', hue='above_cut_off')
    return keep_list