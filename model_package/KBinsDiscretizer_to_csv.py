from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

def KBinsDiscretizer_to_csv(n_bins, df, column_list):
    #　n_bins = n_bins # 5 or 10
    encode_options = ['ordinal']
    strategy_options = ['uniform', 'quantile']

    for encode_option in encode_options:
        for strategy_option in strategy_options:

            # 创建KBinsDiscretizer对象并拟合/转换数据
            est = KBinsDiscretizer(n_bins, encode=encode_option, strategy=strategy_option)
            transformed_data = est.fit_transform(df[column_list])
            # print('transformed_data:', transformed_data.shape)

            # 为新生成的特征创建名称，包括原始列名和n_bins信息
            new_feature_names = [f"{column}_bins{n_bins}" for column in column_list]

            transformed_df = pd.DataFrame(transformed_data, columns=new_feature_names)
            print(transformed_df)

            combine = pd.concat([df, transformed_df], axis=1)

            combine.to_csv(f'KBins transformed_df encode_{encode_option}_strategy{strategy_option}_bins{n_bins}.csv', index=False)
            print(f'encode_{encode_option}_strategy{strategy_option}_bins{n_bins}')
