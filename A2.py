import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import lightgbm as lgb
import xgboost as xgb

# ==========================================
# 0. 严格对齐你提供的源文件与保存路径
# ==========================================
PATH_TRAIN = r"D:\机器学习课程设计\A推荐\A推荐\train.csv"
PATH_TEST = r"D:\机器学习课程设计\A推荐\A推荐\test.csv"
PATH_USER = r"D:\机器学习课程设计\A推荐\A推荐\user.csv"
PATH_ITEM = r"D:\机器学习课程设计\A推荐\A推荐\item.csv"
PATH_SAVE_A2 = r"D:\机器学习课程设计\prediction\A2.csv"


def main():
    print("====== 1. 读取数据 ======")
    train_df = pd.read_csv(PATH_TRAIN)
    test_df = pd.read_csv(PATH_TEST)
    user_df = pd.read_csv(PATH_USER)
    item_df = pd.read_csv(PATH_ITEM)

    valid_items = set(item_df['iid'].astype(str).unique())

    # 解析序列
    train_df['seq_list'] = train_df['item_seq_dedup'].fillna('').astype(str).apply(
        lambda x: [i for i in x.split(',') if i in valid_items])
    test_df['seq_list'] = test_df['item_seq_dedup'].fillna('').astype(str).apply(
        lambda x: [i for i in x.split(',') if i in valid_items])

    print("====== 2. 计算多维度经典统计矩阵 ======")
    item_sim_matrix = defaultdict(lambda: defaultdict(float))
    item_seq_counts = Counter()  # 在序列中出现的频次
    item_target_counts = Counter()  # 真正作为购买目标的频次（极强特征）

    # 统计真正的购买目标热度
    for t_iid in train_df['target_iid'].dropna().astype(str):
        if t_iid in valid_items:
            item_target_counts[t_iid] += 1

    # 构建带距离惩罚的共现矩阵
    for seq in train_df['seq_list']:
        for i, item_i in enumerate(seq):
            item_seq_counts[item_i] += 1
            # 扩大窗口到 6，捕捉更长周期的金融复购
            for j in range(max(0, i - 5), min(len(seq), i + 6)):
                if i == j: continue
                # 经典的同频倒数距离惩罚
                weight = 1.0 / (abs(i - j))
                item_sim_matrix[item_i][seq[j]] += weight

    # 归一化相似度
    for item_i, related_items in item_sim_matrix.items():
        for item_j in related_items:
            item_sim_matrix[item_i][item_j] /= np.sqrt(item_seq_counts[item_i] * item_seq_counts[item_j] + 1e-5)

    global_top_target = [item for item, _ in item_target_counts.most_common(50)]
    global_top_seq = [item for item, _ in item_seq_counts.most_common(50)]
    # 融合大盘兜底池
    global_top = list(dict.fromkeys(global_top_target + global_top_seq))[:50]

    print("====== 3. 产生候选集并提取 15 维黄金特征 ======")

    def generate_candidates(df, is_train=True):
        data_list = []
        for idx, row in df.iterrows():
            uid = str(row['uid'])
            seq = row['seq_list']
            target = str(row['target_iid']) if is_train else None

            if not seq:
                candidates = global_top[:40]
            else:
                candidates = []
                # 核心召回1：最近4个互动商品的相似交叉商品
                for ri in seq[-4:]:
                    sim_items = [k for k, v in
                                 sorted(item_sim_matrix[ri].items(), key=lambda x: x[1], reverse=True)[:20]]
                    candidates.extend(sim_items)
                # 核心召回2：原汁原味的历史行为重现（金融复购率高）
                candidates.extend(seq)
                # 核心召回3：目标大盘热门兜底
                candidates.extend(global_top[:20])
                # 去重并限制候选池大小为 50，确保精排模型有足够选择空间
                candidates = list(dict.fromkeys(candidates))[:50]

            if is_train and target and target in valid_items:
                if target not in candidates:
                    candidates.append(target)

            # 建立当前用户的候选行
            for rank, cand in enumerate(candidates):
                feat = {
                    'uid': uid,
                    'cand_item': cand,
                    # --- [基础特征] ---
                    'recall_rank': rank,
                    'hist_len': len(seq),
                    'is_in_hist': 1 if cand in seq else 0,
                    # --- [全局统计特征] ---
                    'item_seq_hot': item_seq_counts[cand],
                    'item_target_hot': item_target_counts[cand],
                    'hot_ratio': item_target_counts[cand] / (item_seq_counts[cand] + 1e-5),  # 转化率
                }

                # --- [序列位置与距离高级特征] ---
                if seq:
                    feat['last_sim_score'] = item_sim_matrix[seq[-1]].get(cand, 0.0)
                    feat['prev_sim_score'] = item_sim_matrix[seq[-2]].get(cand, 0.0) if len(seq) > 1 else 0.0
                    feat['mean_sim_score'] = np.mean([item_sim_matrix[hist].get(cand, 0.0) for hist in seq[-4:]])
                    feat['max_sim_score'] = np.max([item_sim_matrix[hist].get(cand, 0.0) for hist in seq])

                    # 候选商品在用户历史中最后一次出现的位置（倒数第几个）
                    try:
                        pos_from_last = len(seq) - 1 - (len(seq) - 1 - seq[::-1].index(cand))
                        feat['last_occur_pos'] = pos_from_last
                    except ValueError:
                        feat['last_occur_pos'] = -1

                    # 频率特征
                    feat['occur_times'] = seq.count(cand)
                else:
                    feat['last_sim_score'] = 0.0
                    feat['prev_sim_score'] = 0.0
                    feat['mean_sim_score'] = 0.0
                    feat['max_sim_score'] = 0.0
                    feat['last_occur_pos'] = -1
                    feat['occur_times'] = 0

                if is_train:
                    feat['label'] = 1 if cand == target else 0

                data_list.append(feat)

        return pd.DataFrame(data_list)

    print("正在构建精排训练集特征工程...")
    train_feat = generate_candidates(train_df, is_train=True)
    print("正在构建精排测试集特征工程...")
    test_feat = generate_candidates(test_df, is_train=False)

    # 确定参与训练的全面特征列
    feature_cols = [
        'recall_rank', 'hist_len', 'is_in_hist', 'item_seq_hot',
        'item_target_hot', 'hot_ratio', 'last_sim_score', 'prev_sim_score',
        'mean_sim_score', 'max_sim_score', 'last_occur_pos', 'occur_times'
    ]

    print("====== 4. 强力双决策树模型联合训练 ======")
    # 为排序模型对齐用户分群
    train_feat = train_feat.sort_values(by='uid').reset_index(drop=True)
    train_groups = train_feat.groupby('uid').size().values

    X_train = train_feat[feature_cols]
    y_train = train_feat['label']

    # --- 模型 1: LightGBM Lambdarank (专注于TopK排序性能) ---
    print("开始训练 LightGBM 排序树...")
    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.85,
        'verbose': -1,
        'random_state': 42
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=180)

    # --- 模型 2: XGBoost Pairwise (互补性极强的经典模型) ---
    print("开始训练 XGBoost 排序树...")
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_train.set_group(train_groups)
    xgb_params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@10',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'seed': 42,
        'verbosity': 0
    }
    xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)

    print("====== 5. 推理、加权融合与高保真对齐输出 ======")
    # 提取测试集矩阵
    X_test = test_feat[feature_cols]
    dtest_xgb = xgb.DMatrix(X_test)

    # 双模型加权融合预测分
    pred_lgb = lgb_model.predict(X_test)
    pred_xgb = xgb_model.predict(dtest_xgb)

    # 融合分（各占 50%，平衡线性和非线性偏差）
    test_feat['pred_score'] = 0.5 * pred_lgb + 0.5 * pred_xgb

    # 按用户ID和融合预测分降序排列
    test_feat = test_feat.sort_values(by=['uid', 'pred_score'], ascending=[True, False])

    # 聚合每个用户的预测Top-10
    submission_dict = {}
    for uid, group in test_feat.groupby('uid'):
        top_10 = group['cand_item'].head(10).tolist()
        if len(top_10) < 10:
            for hot in global_top:
                if hot not in top_10: top_10.append(hot)
                if len(top_10) == 10: break
        submission_dict[str(uid)] = ",".join(top_10)

    # 严格按照原 test.csv 的顺序输出
    results = []
    for orig_uid in test_df['uid'].astype(str):
        pred_str = submission_dict.get(orig_uid, ",".join(global_top[:10]))
        results.append({'uid': orig_uid, 'prediction': pred_str})

    submission_df = pd.DataFrame(results)

    # 自动创建目录
    save_dir = os.path.dirname(PATH_SAVE_A2)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert len(submission_df) == len(test_df), "行数与测试集不匹配！"
    submission_df.to_csv(PATH_SAVE_A2, index=False)
    print(f"🚀 【终极提分版】双模型融合文件已成功保存至: {PATH_SAVE_A2}")


if __name__ == '__main__':
    main()