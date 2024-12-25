# 杭电数据挖掘大作业
用于预测杭州市热门站点的流量数据，数据集来自阿里云天池
## 安装依赖
```bash
pip install -r requirements.txt
```
## 运行
将数据集放置于`data/raw`文件夹下，并运行`aggregate.py`以压缩数据集大小用于训练.  
运行`predict_plus.py`以预测本月剩余日期的流量数据，基于三种模型（随机森林、线性回归、支持向量机）进行最佳模型评估  
选择最佳模型进行数据预测，最终进行数据可视化.