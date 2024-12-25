# predict.py 已废弃，使用新的代码 predict_plus.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings("ignore")

class MetroFlowAnalysis:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.daily_data = None
        self.predictions = None
        self.scaler = StandardScaler()
        
        # 初始化优化后的模型
        self.models = {
            'RandomForest': {
                'in': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=42
                ),
                'out': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=42
                )
            },
            'LinearRegression': {
                'in': LinearRegression(),
                'out': LinearRegression()
            },
            'SVR': {
                'in': SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1,
                    gamma='scale'
                ),
                'out': SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1,
                    gamma='scale'
                )
            }
        }
        
        self.best_model = {
            'in': None,
            'out': None,
            'in_name': None,
            'out_name': None
        }
    
    def load_data(self):
        """加载数据并添加特征"""
        daily_totals = []
        
        for day in range(1, 26):
            date_str = f"2019-01-{str(day).zfill(2)}"
            file_name = f"aggregated_record_{date_str}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            
            try:
                df = pd.read_csv(file_path)
                daily_total = {
                    'date': f"2019-01-{str(day).zfill(2)}",
                    'inNums': df['inNums'].sum(),
                    'outNums': df['outNums'].sum()
                }
                
                # 添加时间特征
                date_obj = pd.to_datetime(daily_total['date'])
                daily_total.update({
                    'dayOfWeek': date_obj.dayofweek,
                    'dayOfMonth': date_obj.day,
                    'isWeekend': 1 if date_obj.dayofweek >= 5 else 0,
                    'week_number': date_obj.isocalendar()[1]
                })
                
                # 添加移动平均特征
                if daily_totals:
                    daily_total['prev_day_in'] = daily_totals[-1]['inNums']
                    daily_total['prev_day_out'] = daily_totals[-1]['outNums']
                else:
                    daily_total['prev_day_in'] = daily_total['inNums']
                    daily_total['prev_day_out'] = daily_total['outNums']
                
                daily_totals.append(daily_total)
                
            except Exception as e:
                print(f"Error loading data for day {day}: {e}")
        
        self.daily_data = pd.DataFrame(daily_totals)
        
        # 添加滚动平均特征
        self.daily_data['rolling_mean_in_7d'] = self.daily_data['inNums'].rolling(window=7, min_periods=1).mean()
        self.daily_data['rolling_mean_out_7d'] = self.daily_data['outNums'].rolling(window=7, min_periods=1).mean()
        
        return self.daily_data
        
    def prepare_features(self, data):
        """准备特征"""
        feature_columns = [
            'dayOfWeek', 'dayOfMonth', 'isWeekend', 'week_number',
            'prev_day_in', 'prev_day_out',
            'rolling_mean_in_7d', 'rolling_mean_out_7d'
        ]
        return data[feature_columns]
    def plot_results(self):
        """绘制实际值和预测值的对比图"""
        combined_data = pd.concat([
            self.daily_data.assign(isPrediction=False),
            self.predictions
        ]).reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 绘制进站人数
        ax1.plot(combined_data[~combined_data['isPrediction']]['date'], 
                combined_data[~combined_data['isPrediction']]['inNums'],
                'o-', label='实际进站人数', color='blue')
        if not self.predictions.empty:
            ax1.plot(self.predictions['date'], self.predictions['inNums'],
                    's--', label=f'预测进站人数({self.best_model["in_name"]})', color='red')
        ax1.set_title('地铁站进站人数趋势与预测', fontsize=12)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('人数')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 绘制出站人数
        ax2.plot(combined_data[~combined_data['isPrediction']]['date'],
                combined_data[~combined_data['isPrediction']]['outNums'],
                'o-', label='实际出站人数', color='blue')
        if not self.predictions.empty:
            ax2.plot(self.predictions['date'], self.predictions['outNums'],
                    's--', label=f'预测出站人数({self.best_model["out_name"]})', color='red')
        ax2.set_title('地铁站出站人数趋势与预测', fontsize=12)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('人数')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    def train_and_compare_models(self):
        """训练并比较不同模型的性能，使用交叉验证"""
        X = self.prepare_features(self.daily_data)
        y_in = self.daily_data['inNums']
        y_out = self.daily_data['outNums']
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 存储各模型性能
        performance = {
            'in': {},
            'out': {}
        }
        
        # 定义评分指标
        rmse_scorer = make_scorer(lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred)))
        
        print("\n模型评估指标比较：")
        print("\n进站模型比较（5折交叉验证）：")
        print("-" * 70)
        print(f"{'模型名称':<15}{'R²分数(均值±标准差)':<25}{'RMSE(均值±标准差)':<25}")
        print("-" * 70)
        
        # 评估进站模型
        best_r2_mean_in = -float('inf')
        for name, model_pair in self.models.items():
            model_in = model_pair['in']
            
            # 进行交叉验证
            cv_r2 = cross_val_score(model_in, X_scaled, y_in, cv=5, scoring='r2')
            cv_rmse = cross_val_score(model_in, X_scaled, y_in, cv=5, scoring=rmse_scorer)
            
            # 计算均值和标准差
            r2_mean, r2_std = cv_r2.mean(), cv_r2.std()
            rmse_mean, rmse_std = -cv_rmse.mean(), cv_rmse.std()
            
            performance['in'][name] = {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            }
            
            print(f"{name:<15}{r2_mean:>.4f}±{r2_std:>.4f}{' '*5}{rmse_mean:>.2f}±{rmse_std:>.2f}")
            
            # 选择最佳模型并在全部数据上训练
            if r2_mean > best_r2_mean_in:
                best_r2_mean_in = r2_mean
                model_in.fit(X_scaled, y_in)  # 在全部数据上训练最佳模型
                self.best_model['in'] = model_in
                self.best_model['in_name'] = name
        
        print("\n出站模型比较（5折交叉验证）：")
        print("-" * 70)
        print(f"{'模型名称':<15}{'R²分数(均值±标准差)':<25}{'RMSE(均值±标准差)':<25}")
        print("-" * 70)
        
        # 评估出站模型
        best_r2_mean_out = -float('inf')
        for name, model_pair in self.models.items():
            model_out = model_pair['out']
            
            # 进行交叉验证
            cv_r2 = cross_val_score(model_out, X_scaled, y_out, cv=5, scoring='r2')
            cv_rmse = cross_val_score(model_out, X_scaled, y_out, cv=5, scoring=rmse_scorer)
            
            # 计算均值和标准差
            r2_mean, r2_std = cv_r2.mean(), cv_r2.std()
            rmse_mean, rmse_std = -cv_rmse.mean(), cv_rmse.std()
            
            performance['out'][name] = {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            }
            
            print(f"{name:<15}{r2_mean:>.4f}±{r2_std:>.4f}{' '*5}{rmse_mean:>.2f}±{rmse_std:>.2f}")
            
            # 选择最佳模型并在全部数据上训练
            if r2_mean > best_r2_mean_out:
                best_r2_mean_out = r2_mean
                model_out.fit(X_scaled, y_out)  # 在全部数据上训练最佳模型
                self.best_model['out'] = model_out
                self.best_model['out_name'] = name
        
        print("\n最佳模型选择：")
        print(f"进站预测最佳模型: {self.best_model['in_name']}")
        print(f"出站预测最佳模型: {self.best_model['out_name']}")
        
        return performance
    def generate_report(self):
        """生成数据分析报告"""
        combined_data = pd.concat([
            self.daily_data.assign(isPrediction=False),
            self.predictions
        ]).reset_index(drop=True)
        
        weekday_map = {
            0: '周一', 1: '周二', 2: '周三', 3: '周四',
            4: '周五', 5: '周六', 6: '周日'
        }
        combined_data['weekday'] = combined_data['dayOfWeek'].map(weekday_map)
        
        report_data = combined_data.copy()
        report_data['inNums'] = report_data['inNums'].apply(lambda x: f"{x:,}")
        report_data['outNums'] = report_data['outNums'].apply(lambda x: f"{x:,}")
        report_data['数据类型'] = report_data['isPrediction'].map({True: '预测值', False: '实际值'})
        
        columns = ['date', 'weekday', 'inNums', 'outNums', '数据类型']
        report_data = report_data[columns]
        report_data.columns = ['日期', '星期', '进站人数', '出站人数', '数据类型']
        
        return report_data
    def predict_future_flow(self, start_day=26, end_day=31):
        """使用最佳模型预测未来客流量"""
        predictions = []
        last_data = self.daily_data.iloc[-1]
        
        for day in range(start_day, end_day + 1):
            prediction_date = pd.to_datetime(f"2019-01-{day}")
            
            # 准备预测特征
            pred_features = {
                'date': prediction_date.strftime('%Y-%m-%d'),
                'dayOfWeek': prediction_date.dayofweek,
                'dayOfMonth': prediction_date.day,
                'isWeekend': 1 if prediction_date.dayofweek >= 5 else 0,
                'week_number': prediction_date.isocalendar()[1],
                'prev_day_in': last_data['inNums'],
                'prev_day_out': last_data['outNums'],
                'rolling_mean_in_7d': self.daily_data['inNums'].tail(7).mean(),
                'rolling_mean_out_7d': self.daily_data['outNums'].tail(7).mean()
            }
            
            # 转换特征为DataFrame
            X_pred = pd.DataFrame([pred_features])
            X_pred_scaled = self.scaler.transform(self.prepare_features(X_pred))
            
            # 使用最佳模型预测
            in_pred = round(self.best_model['in'].predict(X_pred_scaled)[0])
            out_pred = round(self.best_model['out'].predict(X_pred_scaled)[0])
            
            prediction = {
                'date': pred_features['date'],
                'dayOfWeek': pred_features['dayOfWeek'],
                'inNums': in_pred,
                'outNums': out_pred,
                'isPrediction': True
            }
            predictions.append(prediction)
            
            # 更新last_data用于下一次预测
            last_data = pd.Series({
                'inNums': in_pred,
                'outNums': out_pred
            })
        
        self.predictions = pd.DataFrame(predictions)
        return self.predictions

def main():
    analyzer = MetroFlowAnalysis("./data/aggregated")
    
    print("加载数据...")
    daily_data = analyzer.load_data()
    print("数据加载完成！")
    
    print("\n训练和比较模型...")
    performance = analyzer.train_and_compare_models()
    print("模型训练完成！")
    
    print("\n使用最佳模型预测未来客流量...")
    predictions = analyzer.predict_future_flow()
    print("预测完成！")
    
    print("\n生成可视化图表...")
    fig = analyzer.plot_results()
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig.savefig('img/地铁进出人流预测图.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 地铁进出人流预测图.png")
    
    print("\n生成数据报告...")
    report = analyzer.generate_report()
    print("\n数据分析报告：")
    print(report.to_string(index=False))
    
    report.to_csv('metro_flow_report.csv', index=False, encoding='utf-8-sig')
    print("\n报告已保存为 metro_flow_report.csv")

if __name__ == "__main__":
    main()