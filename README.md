# 机器学习课程设计竞赛项目
## 项目简介
本项目包含赛道A两套任务代码，全部使用传统机器学习：KMeans聚类、决策树、XGB/LightGBM，无深度学习。
1. A1.py：图节点多分类
   - 图拓扑特征+标签传播特征
   - KMeans节点聚类扩充特征
   - 决策树作为对比模型，XGB为主预测模型
2. A2.py：商品推荐精排
   - 用户/物品共现相似度特征
   - LightGBM Lambdarank + XGB Pairwise双模型融合排序

## 运行截图
<div align="center">

</div>

## 环境依赖
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm
