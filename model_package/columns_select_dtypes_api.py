
def df_select_dtypes_list(df_X):

    if 'BS_mg_dl' in df_X.columns:
        # If present, drop the 'BS_mg_dl' column
        df_X = df_X.drop('BS_mg_dl', axis=1)

    # 提取object类型的列为列表(希望是沒有)
    object_columns = df_X.select_dtypes(include=['object']).columns.tolist()

    # 提取float类型的列为列表
    float_columns = df_X.select_dtypes(include=['float']).columns.tolist()

    # 提取int类型的列为列表
    int_columns = df_X.select_dtypes(include=['int']).columns.tolist()

    print("Object Columns:", object_columns)
    print("Float Columns:", float_columns)
    print("Int Columns:", int_columns)
    print("All columns:", len(float_columns + int_columns))

    return float_columns, int_columns