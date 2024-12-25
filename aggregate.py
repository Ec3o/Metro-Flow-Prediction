# aggregate.py 此代码用于聚合原始数据集
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def aggregate_raw_data(input_file, output_file, freq='10T'):
    """
    将原始数据聚合为每个站点每10分钟的进出人数。
    
    参数：
        input_file (str): 原始数据CSV文件路径。
        output_file (str): 聚合后数据的CSV文件路径。
        freq (str): 时间频率，默认每10分钟（'10T'）。
    """
    # 读取原始数据
    try:
        df = pd.read_csv(input_file, parse_dates=['time'])
        print(f"成功加载文件 {input_file}")
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 未找到。")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生错误: {e}")
        return
    
    # 过滤状态码，假设1为进站，0为出站，其他忽略
    df_filtered = df[df['status'].isin([0, 1])].copy()
    print(f"过滤后的数据条目数: {len(df_filtered)}")
    
    # 创建进站和出站的事件列
    df_filtered['in_event'] = df_filtered['status'].apply(lambda x: 1 if x == 1 else 0)
    df_filtered['out_event'] = df_filtered['status'].apply(lambda x: 1 if x == 0 else 0)
    
    # 设置时间为索引
    df_filtered.set_index('time', inplace=True)
    
    # 按站点和时间段聚合进站和出站人数
    aggregation_in = df_filtered.groupby('stationID').resample(freq)['in_event'].sum().reset_index()
    aggregation_out = df_filtered.groupby('stationID').resample(freq)['out_event'].sum().reset_index()
    
    # 合并进站和出站数据
    aggregation = pd.merge(aggregation_in, aggregation_out, on=['stationID', 'time'], how='outer')
    
    # 生成一个完整的时间范围，确保所有时间段都有数据
    all_times = pd.date_range(start=df_filtered.index.min().replace(hour=0, minute=0, second=0), 
                              end=df_filtered.index.max().replace(hour=23, minute=59, second=59),
                              freq=freq)
    
    # 创建一个完整的时间段 DataFrame
    full_time_df = pd.DataFrame(all_times, columns=['time'])
    
    # 对每个站点，确保所有时间段都有记录，如果没有数据就填充0
    aggregation_full = pd.DataFrame()
    for station in df_filtered['stationID'].unique():
        # 获取该站点的数据
        station_data = aggregation[aggregation['stationID'] == station]
        
        # 重新设置时间为索引，进行时间重索引以填充缺失的时间段
        station_data.set_index('time', inplace=True)
        station_data_resampled = station_data.reindex(all_times, fill_value=0).reset_index()
        station_data_resampled.rename(columns={'index': 'time'}, inplace=True)
        
        # 添加站点ID列
        station_data_resampled['stationID'] = station
        
        # 合并到最终的DataFrame
        aggregation_full = pd.concat([aggregation_full, station_data_resampled])
    
    # 生成 'startTime' 和 'endTime'
    aggregation_full['startTime'] = aggregation_full['time']
    aggregation_full['endTime'] = aggregation_full['time'] + pd.to_timedelta(freq)
    
    # 只保留必要的列
    aggregation_full = aggregation_full[['stationID', 'startTime', 'endTime', 'in_event', 'out_event']]
    
    # 将 'in_event' 和 'out_event' 重命名为 'inNums' 和 'outNums'
    aggregation_full.rename(columns={'in_event': 'inNums', 'out_event': 'outNums'}, inplace=True)
    
    # 排序：先按站点ID排序，再按时间排序
    aggregation_full = aggregation_full.sort_values(by=['stationID', 'startTime']).reset_index(drop=True)
    
    # 保存为CSV
    aggregation_full.to_csv(output_file, index=False)
    print(f"聚合后的数据已保存到 {output_file}")

if __name__ == "__main__":
    for i in range(1, 26):
        print(f"当前进行第{i}个文件的处理")
        if len(str(i)) == 1:
            tmp = "0" + str(i)
        else:
            tmp = str(i)
        input_file = f"data/raw/record_2019-01-{tmp}.csv"
        output_file = f"data/aggregated/aggregated_record_2019-01-{tmp}.csv"
        aggregate_raw_data(input_file, output_file)
