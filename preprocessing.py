# preprocessing.py 数据预处理

import pandas as pd

def data_preprocessing(data):
    """
    数据预处理函数：
    - 删除无用列
    - 去除重复数据
    - 去除缺失数据
    """
    # 去除无用数据
    columns_to_drop = ["payType", "deviceID"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    if existing_columns_to_drop:
        data.drop(existing_columns_to_drop, axis=1, inplace=True)
        print(f"已删除列: {existing_columns_to_drop}")
    else:
        print("无无用列需要删除。")
    
    if data is None or data.empty:
        print("数据为空，返回None。")
        return None
    
    flag = False  # 初始化 flag
    
    # 去除重复数据
    duplicate = data.duplicated().sum()
    print("检查重复数据...")
    if duplicate > 0:
        flag = True
        print(f"发现重复数据 {duplicate} 条，正在删除...")
        data.drop_duplicates(inplace=True)
    else:
        print("未发现重复数据")
    
    # 去除缺失数据
    null = data.isnull().sum().sum()
    print("检查缺失数据...")
    if null > 0:
        flag = True
        print(f"发现缺失数据 {null} 个，正在删除...")
        data.dropna(inplace=True)
    else:
        print("未发现缺失数据")
    
    if flag:
        print("数据预处理完成！")
    else:
        print("数据很干净")
    
    return data
