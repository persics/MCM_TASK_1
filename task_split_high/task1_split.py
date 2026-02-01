# ===================== 1. 基础设置与导入 =====================
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = np.int_

import emcee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from warnings import filterwarnings

filterwarnings('ignore')

# 绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
np.random.seed(42)

# ===================== 2. 读取数据 =====================
def read_your_data():
    file_path = "2026_MCM_Problem_C_Data.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    total_samples = len(df)
    total_seasons = 34

    # 自动生成season字段
    season_sample_counts = [12]*21 + [13]*13
    df['season'] = np.repeat(range(1, total_seasons+1), season_sample_counts)[:total_samples]

    # 生成建模标签
    df['actual_eliminate'] = df['results'].apply(lambda x: 0 if 'Place' in str(x) else 1)
    df['final_rank'] = df['placement'].astype(int)

    # 生成week字段
    if 'week' not in df.columns:
        df['week'] = df.groupby('season').cumcount() + 1
        df['week'] = df['week'].apply(lambda x: min(x, 5))

    # 生成player_id
    if 'player_id' not in df.columns:
        df['player_id'] = [f'C{i+1:03d}' for i in range(total_samples)]

    return df

df = read_your_data()

# ===================== 3. 数据预处理（全局编码，局部拆分） =====================
def preprocess_data_split(df):
    # 定义特征
    cat_feats = ['celebrity_homecountry/region', 'celebrity_homestate', 'celebrity_industry']
    cont_feats = ['celebrity_age_during_season'] # 注意：final_rank是Y，不放入X

    # 1. 全局拟合Encoder和Scaler（保证两组模型的特征维度一致，便于对比）
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    scaler = StandardScaler()

    # 填充缺失值
    df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(
        df['celebrity_age_during_season'].mean()
    ).astype(int)
    for col in cat_feats:
        df[col] = df[col].fillna('Unknown').astype(str)

    # 拟合转换
    X_cat = encoder.fit_transform(df[cat_feats])
    X_cont = scaler.fit_transform(df[cont_feats])
    
    # 生成列名
    cat_cols = []
    for i, feat in enumerate(cat_feats):
        unique_vals = encoder.categories_[i][1:]
        cat_cols.extend([f'{feat}_{str(val).replace("/", "-").replace(" ", "_")}' for val in unique_vals])
    
    feature_names = ['intercept'] + cat_cols + cont_feats
    
    # 构建全局X矩阵
    X_all = np.hstack([np.ones((len(df), 1)), X_cat, X_cont])
    y_all = df['final_rank'].values

    # 2. 拆分数据集：排名法赛季 vs 百分比法赛季
    # 排名法赛季：1, 2, 28-34
    rank_seasons = [1, 2] + list(range(28, 35))
    
    mask_rank = df['season'].isin(rank_seasons)
    mask_percent = ~df['season'].isin(rank_seasons)

    data_split = {
        'rank': {
            'X': X_all[mask_rank],
            'y': y_all[mask_rank],
            'indices': df[mask_rank].index
        },
        'percent': {
            'X': X_all[mask_percent],
            'y': y_all[mask_percent],
            'indices': df[mask_percent].index
        }
    }
    
    print(f"\n📊 数据拆分完成：")
    print(f"  - 排名法数据（Rank Model）：{mask_rank.sum()} 样本 (Seasons: 1-2, 28-34)")
    print(f"  - 百分比法数据（Percent Model）：{mask_percent.sum()} 样本 (Seasons: 3-27)")
    print(f"  - 特征维度：{X_all.shape[1]}")

    return df, data_split, feature_names

df, data_split, feature_names = preprocess_data_split(df)

# ===================== 4. 通用贝叶斯MCMC训练函数 =====================
def run_mcmc_model(X, y, model_name="Model"):
    print(f"\n🚀 开始训练 {model_name} ...")
    
    # 先验
    def log_prior(theta):
        # 弱信息先验
        if np.abs(theta[0]) > 100: return -np.inf # 截距约束
        return -0.5 * np.sum(theta**2 / 25) # N(0, 5)

    # 似然 (回归模型：latent_score ~ Normal)
    def log_likelihood(theta, X, y):
        mu = np.dot(X, theta)
        sigma = 1.2 # 固定噪声，也可设为参数
        return np.sum(stats.norm.logpdf(y, mu, sigma))

    # 后验
    def log_probability(theta, X, y):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + log_likelihood(theta, X, y)

    # MCMC 设置
    n_params = X.shape[1]
    n_walkers = max(32, 2 * n_params)
    initial = np.random.normal(0, 0.1, (n_walkers, n_params))
    
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(X, y))
    sampler.run_mcmc(initial, 4000, progress=True)
    
    samples = sampler.get_chain(discard=1500, flat=True)
    return samples

# ===================== 5. 分别训练两个模型 =====================

# 1. 训练排名法模型 (Rank Seasons)
print("--- 正在拟合排名法模型 (Rank Model) ---")
samples_rank = run_mcmc_model(data_split['rank']['X'], data_split['rank']['y'], "Rank_Model")

# 2. 训练百分比法模型 (Percent Seasons)
print("--- 正在拟合百分比法模型 (Percent Model) ---")
samples_percent = run_mcmc_model(data_split['percent']['X'], data_split['percent']['y'], "Percent_Model")

# ===================== 6. 后验推断与结果合并 =====================
def infer_and_merge(df, data_split, samples_rank, samples_percent):
    # 初始化列
    df['est_rank'] = 0
    df['vote_posterior'] = None
    df['vote_posterior'] = df['vote_posterior'].astype(object) # 允许存列表

    # 辅助推断函数
    def infer_subset(X, samples, indices):
        rank_posterior_list = []
        # 抽样 1000 次
        subset_samples = samples[np.random.choice(len(samples), 1000, replace=False)]
        
        # 批量计算
        # X: (N_subset, n_feat), Theta: (1000, n_feat) -> Mu: (N_subset, 1000)
        mu_mat = np.dot(X, subset_samples.T)
        
        # 添加噪声并取整
        pred_mat = np.round(mu_mat + np.random.normal(0, 1.2, mu_mat.shape))
        pred_mat[pred_mat < 1] = 1 # 截断
        
        # 存回 DataFrame
        est_ranks = np.mean(pred_mat, axis=1).astype(int)
        
        # 更新df
        df.loc[indices, 'est_rank'] = est_ranks
        
        # 这种方式稍慢但安全：逐行赋值posterior
        # 构造一个也就是 (N_subset,) 的 object 数组
        post_objs = [row for row in pred_mat]
        df.loc[indices, 'vote_posterior'] = pd.Series(post_objs, index=indices)

    # 推断 Rank 部分
    infer_subset(data_split['rank']['X'], samples_rank, data_split['rank']['indices'])
    
    # 推断 Percent 部分
    infer_subset(data_split['percent']['X'], samples_percent, data_split['percent']['indices'])
    
    print("\n✅ 双模型推断完成，结果已合并至主DataFrame")
    return df

df = infer_and_merge(df, data_split, samples_rank, samples_percent)

# ===================== 7. 按机制计算淘汰概率 (复用逻辑) =====================
# 注意：这里逻辑与之前相同，但输入数据源自两个不同训练出来的模型
def calculate_eliminate_mixed(df):
    df['est_eliminate'] = 0
    df['eliminate_prob'] = 0.0
    n_sim = 1000

    rank_seasons = [1, 2] + list(range(28, 35))
    
    for season in sorted(df['season'].unique()):
        # 判定规则
        rule = 'rank' if season in rank_seasons else 'percent'
        
        df_season = df[df['season'] == season]
        for week in df_season['week'].unique():
            idx_week = (df['season'] == season) & (df['week'] == week)
            df_week = df[idx_week]
            n_player = len(df_week)
            if n_player <= 1: continue

            # 获取后验样本 (N_player, 1000)
            # 注意：vote_posterior 已经是通过各自模型生成的了
            vote_posterior_week = np.vstack(df_week['vote_posterior'].values)
            
            score = df_week['final_rank'].values # 评委名次(代理)
            
            # 模拟循环
            elim_count = np.zeros(n_player)
            
            if rule == 'rank':
                # 排名法：(评委排名 + 粉丝排名) 最大者淘汰
                rank_score = stats.rankdata(score, method='min')
                for s in range(n_sim):
                    vote_s = vote_posterior_week[:, s]
                    rank_vote_s = stats.rankdata(vote_s, method='min')
                    total = rank_score + rank_vote_s
                    # 标记最大值（可能有并列）
                    elim_count[total == total.max()] += 1
                    
            elif rule == 'percent':
                # 百分比法：(评委占比 + 粉丝占比) 最小者淘汰
                # 转换：名次 -> 分数 (简单反转)
                raw_score = df_week['final_rank'].max() - score + 1
                p_score = raw_score / raw_score.sum()
                
                for s in range(n_sim):
                    vote_s = vote_posterior_week[:, s]
                    # 转换：后验名次 -> 虚拟票数
                    raw_vote = df_week['est_rank'].max() - vote_s + 1
                    raw_vote = np.maximum(raw_vote, 0.1) # 避免除0
                    p_vote = raw_vote / raw_vote.sum()
                    
                    total_p = p_score + p_vote
                    elim_count[total_p == total_p.min()] += 1

            df.loc[idx_week, 'eliminate_prob'] = elim_count / n_sim
            
            # 硬分类（概率>0.5或最大者）
            best_guess_idx = df_week.index[np.argmax(elim_count)]
            df.loc[idx_week, 'est_eliminate'] = 0
            df.loc[best_guess_idx, 'est_eliminate'] = 1

    return df

print("\n⚙️ 正在应用各自的淘汰规则计算概率...")
df = calculate_eliminate_mixed(df)

# ===================== 8. 验证与导出 =====================
def validate_and_export(df):
    # 验证
    acc = (df['actual_eliminate'] == df['est_eliminate']).mean()
    try:
        auc = roc_auc_score(df['actual_eliminate'], df['eliminate_prob'])
    except:
        auc = 0.5
        
    print(f"\n📊 综合模型性能：")
    print(f"  - 准确率 (Accuracy): {acc:.2%}")
    print(f"  - AUC: {auc:.4f}")

    # 对比不同模型下的系数（可选）
    print("\n🔍 两个模型对'年龄'特征的影响对比 (Standardized Beta):")
    # 找到年龄对应的索引
    age_idx = feature_names.index('celebrity_age_during_season')
    beta_rank = np.mean(samples_rank[:, age_idx])
    beta_pct = np.mean(samples_percent[:, age_idx])
    print(f"  - 排名法时代 (Rank Era) 年龄系数: {beta_rank:.4f}")
    print(f"  - 百分比时代 (Percent Era) 年龄系数: {beta_pct:.4f}")
    if abs(beta_rank - beta_pct) > 0.1:
        print("    -> 发现显著差异：年龄在不同时期的影响权重不同。")

    # 导出
    df.to_excel("34季_双模型分层建模结果.xlsx", index=False)
    print("\n📁 结果已导出：34季_双模型分层建模结果.xlsx")

validate_and_export(df)