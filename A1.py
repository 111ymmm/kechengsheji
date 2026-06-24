import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# ==================== 路径（保持不变） ====================
PATH_A1_NPZ = r"D:\机器学习课程设计\A分类\A分类\A1.npz"
PATH_SUBMIT_TPL = r"D:\机器学习课程设计\A分类\A分类\sample_submission.csv"
PATH_SAVE_A1 = r"D:\机器学习课程设计\prediction\A1.csv"

RANDOM_SEED = 42
NUM_CLASS = 10
CLUSTER_NUM = 12   # 节点聚类数，可微调 10~15

# ==================== 原有高分图特征（完全保留） ====================
def extract_graph_features(adj_csr):
    n = adj_csr.shape[0]
    degree = np.array(adj_csr.sum(axis=1)).flatten()
    indegree = np.array(adj_csr.sum(axis=0)).flatten()
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = adj_csr[i].indices
        if len(neighbors) <= 1:
            continue
        sub = adj_csr[neighbors][:, neighbors]
        clustering[i] = sub.nnz / (len(neighbors)**2 + 1e-6)
    return np.vstack([degree, indegree, clustering]).T

# ==================== 原有标签传播特征（完全保留） ====================
def label_propagation(adj, labels, train_idx, n_class=10, alpha=0.85, iters=10):
    from sklearn.preprocessing import normalize
    n = adj.shape[0]
    A = normalize(adj, norm="l1", axis=1)
    Y = np.zeros((n, n_class))
    Y[train_idx, labels[train_idx]] = 1
    F = Y.copy()
    for _ in range(iters):
        F = alpha * A.dot(F) + (1 - alpha) * Y
    return F

# ==================== 加载数据（原版不变） ====================
data = np.load(PATH_A1_NPZ, allow_pickle=True)
adj = csr_matrix((data["adj_data"], data["adj_indices"], data["adj_indptr"]),
                 shape=data["adj_shape"])
attr = csr_matrix((data["attr_data"], data["attr_indices"], data["attr_indptr"]),
                  shape=data["attr_shape"])
labels = data["labels"]
train_idx = data["train_idx"]
test_idx = data["test_idx"]

node_attr = attr.toarray()
graph_feat = extract_graph_features(adj)
lp_feat = label_propagation(adj, labels, train_idx)

# ==================== 仅新增：KMeans节点聚类特征（不破坏原有特征） ====================
all_base_feat = np.hstack([node_attr, graph_feat, lp_feat])
scaler = StandardScaler()
feat_scaled = scaler.fit_transform(all_base_feat)

# 节点聚类，作为补充特征
kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=RANDOM_SEED)
cluster_label = kmeans.fit_predict(feat_scaled)
# 拼接聚类特征（低维特征，不会导致分数暴跌）
X = np.hstack([all_base_feat, cluster_label.reshape(-1, 1)])

# ==================== 数据集划分（原版不变） ====================
tr_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
X_train, y_train = X[tr_idx], labels[tr_idx]
X_val, y_val = X[val_idx], labels[val_idx]

# ==================== 1. 主模型：XGBoost（沿用你原版高分参数，最终用它输出结果） ====================
print("===== 训练主模型 XGBoost =====")
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=NUM_CLASS,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_SEED
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=20
)
xgb_val_pred = xgb_model.predict(X_val)
xgb_acc = accuracy_score(y_val, xgb_val_pred)
print(f"XGBoost 验证集准确率: {xgb_acc:.4f}")

# ==================== 2. 额外新增：独立决策树模型（仅做对比，不参与融合，满足作业要求） ====================
print("\n===== 训练对比模型 决策树 DecisionTree =====")
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    random_state=RANDOM_SEED
)
dt_model.fit(X_train, y_train)
dt_val_pred = dt_model.predict(X_val)
dt_acc = accuracy_score(y_val, dt_val_pred)
print(f"决策树 验证集准确率: {dt_acc:.4f}")

# ==================== 预测 & 输出（核心：依旧用XGB主模型输出，保证高分） ====================
print("\n使用 XGBoost 模型生成预测结果...")
test_pred = xgb_model.predict(X[test_idx])

# 输出文件格式严格遵循赛事要求
sub = pd.read_csv(PATH_SUBMIT_TPL)
sub["label"] = test_pred
sub.to_csv(PATH_SAVE_A1, index=False)
print("✅ A1 分类任务完成，文件已保存")