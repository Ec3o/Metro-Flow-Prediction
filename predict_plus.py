# predict_plus.py: 通过机器学习预测热门站点的流量,并生成可视化图表
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

def load_and_process_data(start_date, end_date, target_stations):
    """
    加载指定日期范围内的数据并进行预处理
    """
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:#遍历日期范围内的每一天
        date_str = current_date.strftime('%Y-%m-%d')
        filename = f'data/aggregated/aggregated_record_{date_str}.csv'
        
        try:
            df = pd.read_csv(filename)
            df['date'] = current_date
            all_data.append(df)
        except:
            print(f"错误：无法加载日期{date_str}的数据")
            
        current_date += datetime.timedelta(days=1)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['stationID'].isin(target_stations)]
    
    combined_df['startTime'] = pd.to_datetime(combined_df['startTime']) #将startTime列转换为日期时间格式
    combined_df['hour'] = combined_df['startTime'].dt.hour #提取小时
    combined_df['minute'] = combined_df['startTime'].dt.minute #提取分钟
    combined_df['weekday'] = combined_df['date'].dt.weekday #提取星期几
    
    is_weekend = combined_df['weekday'].isin([5, 6]) #判断是否为周末
    weekday_data = combined_df[~is_weekend] #取出工作日数据
    weekend_data = combined_df[is_weekend] #取出周末数据
    
    return weekday_data, weekend_data

def prepare_features(data, target_type):
    """
    准备模型训练所需的特征
    """
    X = pd.DataFrame({
        'hour': data['hour'],
        'minute': data['minute'],
        'weekday': data['weekday'],
        'stationID': data['stationID']
    })
    
    X = pd.get_dummies(X, columns=['stationID'], prefix='station')
    y = data[target_type].values  # 确保使用数值数组
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    训练和评估三种模型
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'SVR': SVR(kernel='rbf'),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred) #均方误差
        r2 = r2_score(y_test, y_pred) #决定系数
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2
        }
    
    return results

def generate_prediction_dates(base_date, num_days):
    """
    生成预测日期序列
    """
    return [base_date + datetime.timedelta(days=i) for i in range(num_days)]

def predict_single_flow(model, scaler, feature_data, feature_columns):
    """
    预测单个时间点的流量
    """
    # 确保特征列顺序一致
    feature_data = feature_data.reindex(columns=feature_columns, fill_value=0)
    # 标准化特征
    feature_scaled = scaler.transform(feature_data)
    # 预测并确保非负
    flow = model.predict(feature_scaled)[0]
    return max(0, int(round(flow)))

def create_prediction_data(date, station, hour, minute):
    """
    创建预测用的特征数据
    """
    feature_data = pd.DataFrame({
        'hour': [hour],
        'minute': [minute],
        'weekday': [date.weekday()],
        'stationID': [station]
    })
    return pd.get_dummies(feature_data, columns=['stationID'], prefix='station') #独热编码

def predict_station_flows(model_info, date, station, feature_columns):
    """
    预测某个站点一天的流量
    """
    flows = []
    for hour in range(24):
        for minute in range(0, 60, 10):
            current_time = datetime.datetime(
                date.year, date.month, date.day, hour, minute
            )
            end_time = current_time + datetime.timedelta(minutes=10)
            
            feature_data = create_prediction_data(date, station, hour, minute)
            flow = predict_single_flow(
                model_info['model'],
                model_info['scaler'],
                feature_data,
                feature_columns
            )
            
            flows.append({
                'stationID': station,
                'startTime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'endTime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'flow': flow
            })
    return flows
def visualize_station_flows(predictions_df, station_names):
    """
    为每个站点创建每日流量图表
    
    Args:
        predictions_df: 包含预测结果的DataFrame
        station_names: 字典，站点ID到站点名称的映射
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 获取唯一的日期列表
    dates = predictions_df['startTime'].dt.date.unique()
    
    # 为每个站点和每一天创建图表
    for station_id in predictions_df['stationID'].unique():
        station_name = station_names.get(station_id, f'站点{station_id}')
        
        for date in dates:
            # 获取当天该站点的数据
            daily_data = predictions_df[
                (predictions_df['stationID'] == station_id) & 
                (pd.to_datetime(predictions_df['startTime']).dt.date == date)
            ].copy()
            
            if len(daily_data) == 0:
                continue
                
            # 转换时间格式
            daily_data['startTime'] = pd.to_datetime(daily_data['startTime'])
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            # 绘制入站和出站流量
            plt.plot(daily_data['startTime'], daily_data['inNums'], 
                    label='入站人数', color='blue', marker='o', markersize=4)
            plt.plot(daily_data['startTime'], daily_data['outNums'], 
                    label='出站人数', color='red', marker='s', markersize=4)
            
            # 设置x轴显示格式
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
            
            # 添加标题和标签
            plt.title(f'{station_name} {date} 客流量预测', fontsize=14)
            plt.xlabel('时间', fontsize=12)
            plt.ylabel('人数', fontsize=12)
            
            # 添加网格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加图例
            plt.legend(loc='best')
            
            # 旋转x轴标签
            plt.xticks(rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(f'img/{station_name}_{date}_预测流量图.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已生成{station_name} {date}的流量图表")

if __name__ == "__main__":
    # 设置参数
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2019, 1, 25)
    target_stations = [10, 12, 15, 20, 24] #凤起路、西湖文化广场、火车东站、客运中心、文泽路
    
    print("加载并处理数据...")
    weekday_data, weekend_data = load_and_process_data(start_date, end_date, target_stations)
    
    # 存储模型信息
    models_info = {
        'weekday': {'inNums': {}, 'outNums': {}},
        'weekend': {'inNums': {}, 'outNums': {}}
    }
    
    # 训练模型
    print("\n训练模型中...")
    for data_type, data in [('weekday', weekday_data), ('weekend', weekend_data)]:
        for flow_type in ['inNums', 'outNums']:
            print(f"\n训练 {data_type} {flow_type} 模型中...")
            X, y = prepare_features(data, flow_type)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
            best_model = min(models.items(), key=lambda x: x[1]['mse'])
            
            print(f"最佳 {data_type} {flow_type} 表现模型:", best_model[0])
            print(f"均方误差MSE: {best_model[1]['mse']:.2f}")
            print(f"R2 系数得分: {best_model[1]['r2']:.4f}")
            
            models_info[data_type][flow_type] = {
                'model': best_model[1]['model'],
                'scaler': best_model[1]['scaler'],
                'feature_columns': X_train.columns
            }
    
    # 生成预测日期
    prediction_dates = generate_prediction_dates(
        datetime.datetime(2019, 1, 26),
        5
    )
    
    # 预测流量
    print("\n预测流量中...")
    all_predictions = []
    for date in prediction_dates:
        print(f"正在预测日期: {date.date()}")
        data_type = 'weekday' if date.weekday() < 5 else 'weekend'
        
        for station in target_stations:
            # 预测入站流量
            in_flows = predict_station_flows(
                models_info[data_type]['inNums'],
                date,
                station,
                models_info[data_type]['inNums']['feature_columns']
            )
            
            # 预测出站流量
            out_flows = predict_station_flows(
                models_info[data_type]['outNums'],
                date,
                station,
                models_info[data_type]['outNums']['feature_columns']
            )
            
            # 合并入站和出站流量
            for in_flow, out_flow in zip(in_flows, out_flows):
                prediction = {
                    'stationID': station,
                    'startTime': in_flow['startTime'],
                    'endTime': in_flow['endTime'],
                    'inNums': in_flow['flow'],
                    'outNums': out_flow['flow']
                }
                all_predictions.append(prediction)
    
    # 创建最终预测数据框
    final_predictions = pd.DataFrame(all_predictions)
    final_predictions = final_predictions.sort_values(['stationID', 'startTime'])
    
    # 保存预测结果
    print("\n正在保存预测结果...")
    final_predictions.to_csv('flow_predictions.csv', index=False)
    print("预测结果已保存为'flow_predictions.csv'")
    
    # 输出预测结果的基本统计信息
    print("\n预测结果统计信息:")
    print("全量预测数据:", len(final_predictions))
    print("\n入流量预测统计数据:")
    print(final_predictions['inNums'].describe())
    print("\n出流量预测统计数据:")
    print(final_predictions['outNums'].describe())
    station_names = {
        10: '凤起路站',
        12: '西湖文化广场站',
        15: '火车东站',
        20: '客运中心站',
        24: '文泽路站'
    }
    
    print("\n生成可视化图表...")
    # 转换时间列为datetime类型
    final_predictions['startTime'] = pd.to_datetime(final_predictions['startTime'])
    visualize_station_flows(final_predictions, station_names)
    print("图表生成完成！")