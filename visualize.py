#visualize.py 此代码用于数据可视化
def data_visualization_split_graph(df, station_mapping, line_mapping):
    import pandas as pd
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    from scipy.interpolate import make_interp_spline  # 导入 make_interp_spline
    import numpy as np
    # 解析时间列
    df['time'] = pd.to_datetime(df['time'])
    
    # 过滤上车记录（status == 1 代表上车）
    df_filtered = df[df['status'] == 1]
    
    # 映射 stationID 到 stationName
    df_filtered['stationName'] = df_filtered['stationID'].map(station_mapping)
    
    # 映射 lineID 到 lineName
    df_filtered['lineName'] = df_filtered['lineID'].map(line_mapping)
    
    # 检查是否有未映射的 stationID 或 lineID
    unmapped_stations = df_filtered[df_filtered['stationName'].isnull()]['stationID'].unique()
    if len(unmapped_stations) > 0:
        print(f"警告: 以下 stationID 未找到对应的站点名称: {unmapped_stations}")
    
    unmapped_lines = df_filtered[df_filtered['lineName'].isnull()]['lineID'].unique()
    if len(unmapped_lines) > 0:
        print(f"警告: 以下 lineID 未找到对应的线路名称: {unmapped_lines}")
    
    # 计算人流量
    passenger_flow = df_filtered.groupby(['lineName', 'stationName']).size().reset_index(name='count')
    
    # 确保站点顺序
    # 我们根据 stationID 确保站点在图中的顺序
    station_order = df_filtered[['stationID', 'stationName']].drop_duplicates().sort_values('stationID')
    station_order_dict = dict(zip(station_order['stationName'], station_order['stationID']))
    
    # 添加 station_order 列用于排序
    passenger_flow['station_order'] = passenger_flow['stationName'].map(station_order_dict)
    passenger_flow = passenger_flow.sort_values(['lineName', 'station_order'])
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 获取所有线路名称
    lines = passenger_flow['lineName'].unique()
    
    # 遍历每条线路，分别绘制图表
    for line in lines:
        line_data = passenger_flow[passenger_flow['lineName'] == line]
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='stationName', y='count', data=line_data, marker='o', label=line)
        
        # 添加标题和标签
        plt.title(f'{line} 每个站点的人流量', fontsize=16)
        plt.xlabel('站点名称', fontsize=14)
        plt.ylabel('人流量', fontsize=14)
        
        # 调整x轴标签旋转角度以防重叠
        plt.xticks(rotation=45, ha='right')
        
        # 显示图例
        plt.legend(title='线路', fontsize=12, title_fontsize=12)
        
        # 调整布局以防止标签被截断
        plt.tight_layout()
        
        # 显示图表
        # plt.show()
        plt.savefig(f'img/{line} 每个站点的人流量.png')# 主流程
def data_visualization_realtime_graph(df, station_mapping, line_mapping):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    from scipy.interpolate import make_interp_spline  # 导入 make_interp_spline
    import numpy as np
    # 解析时间列
    df['time'] = pd.to_datetime(df['time'])
    
    # 过滤上车记录（status == 1 代表上车）
    df_filtered = df[df['status'] == 1].copy()
    
    # 映射 stationID 到 stationName
    df_filtered['stationName'] = df_filtered['stationID'].map(station_mapping)
    
    # 映射 lineID 到 lineName
    df_filtered['lineName'] = df_filtered['lineID'].map(line_mapping)
    
    # 检查是否有未映射的 stationID 或 lineID
    unmapped_stations = df_filtered[df_filtered['stationName'].isnull()]['stationID'].unique()
    if len(unmapped_stations) > 0:
        print(f"警告: 以下 stationID 未找到对应的站点名称: {unmapped_stations}")
    
    unmapped_lines = df_filtered[df_filtered['lineName'].isnull()]['lineID'].unique()
    if len(unmapped_lines) > 0:
        print(f"警告: 以下 lineID 未找到对应的线路名称: {unmapped_lines}")
    
    # 设置时间为索引
    df_filtered.set_index('time', inplace=True)
    
    # 计算人流量：按线路和小时分组，计算每条线路每小时的人数
    passenger_flow = df_filtered.groupby('lineName').resample('H').size().reset_index(name='count')
    
    # 可选：应用滚动平均以平滑曲线（例如，窗口为3小时）
    passenger_flow['count_smoothed'] = passenger_flow.groupby('lineName')['count'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # 提取日期和星期几
    unique_dates = passenger_flow['time'].dt.date.unique()
    if len(unique_dates) == 1:
        date = unique_dates[0]
        day_of_week = pd.Timestamp(date).day_name()
        title_date = f"{date} ({day_of_week})"
    else:
        # 如果有多天的数据，可以根据需要调整
        title_date = "多天数据"
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 创建绘图对象
    plt.figure(figsize=(15, 8))
    
    # 获取所有线路名称
    lines = passenger_flow['lineName'].unique()
    
    # 为每条线路绘制光滑曲线
    for line in lines:
        line_data = passenger_flow[passenger_flow['lineName'] == line]
        
        # 将时间转换为数字格式以便插值
        x = mdates.date2num(line_data['time'])
        y = line_data['count_smoothed']
        
        # 检查数据点数量是否足够进行插值
        if len(x) < 4:
            print(f"警告: 线路 {line} 的数据点不足，无法进行插值。")
            plt.plot(line_data['time'], y, marker='o', label=line)
            continue
        
        # 创建更密集的x值用于插值
        x_new = np.linspace(x.min(), x.max(), 300)  # 300个点用于平滑
        
        # 创建插值函数
        spl = make_interp_spline(x, y, k=3)  # k=3表示三次样条
        y_new = spl(x_new)
        
        # 将密集的x_new转换回datetime格式
        time_new = mdates.num2date(x_new)
        
        # 绘制插值后的平滑曲线
        plt.plot(time_new, y_new, label=line)
    
    # 添加标题和标签
    plt.title(f'杭州地铁线实时人流量图{title_date}', fontsize=18)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('人流量', fontsize=14)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 设置x轴的主刻度为每小时
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 每小时一个主刻度
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))  # 格式化时间显示为小时:分钟
    
    # 调整x轴标签旋转角度以防重叠
    plt.xticks(rotation=45, ha='right')
    
    # 显示图例
    plt.legend(title='线路', fontsize=12, title_fontsize=12)
    
    # 调整布局以防止标签被截断
    plt.tight_layout()
    
    # 显示图表
    # plt.show()
    plt.savefig(f'img/杭州地铁线实时人流量图{title_date}.png')
