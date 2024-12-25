# analyze_visualize.py 此代码用于生成人流量折线及散点图
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
import numpy as np
from preprocessing import data_preprocessing
from mapping import station_mapping, line_mapping
from visualize import data_visualization_split_graph,data_visualization_realtime_graph
import warnings
warnings.filterwarnings("ignore")
def main():
    path = "data/raw/record_2019-01-02.csv"
    print(f"加载数据集 {path} 中...")
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"错误: 文件 {path} 未找到。请确保文件路径正确。")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生错误: {e}")
        return
    
    print("原始数据前五行数据如下所示:")
    print(data.head())
    
    # 数据预处理
    data = data_preprocessing(data)
    print("\n数据预处理后前五行数据如下所示:")
    print(data.head())
    
    # 基础数据可视化
    data_visualization_split_graph(data, station_mapping, line_mapping)
    data_visualization_realtime_graph(data, station_mapping, line_mapping)
    

if __name__ == "__main__":
    main()
