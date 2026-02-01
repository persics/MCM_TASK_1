# ===================== 1. 基础设置 =====================
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = np.int_

import emcee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
from warnings import filterwarnings

filterwarnings('ignore')

# 绘图配置
import platform
font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(2026)

# ===================== 2. 增强版数据读取 =====================
def read_and_process_data():
    file_path = "2026_MCM_Problem_C_Data.csv"
    try:
        raw_df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        raw_df = pd.read_csv(file_path, encoding='latin1')
    
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    
    # 基础列识别
    week_cols = [c for c in raw_df.columns if 'week' in c and 'judge' in c]
    max_week = 10
    if week_cols:
        weeks = [int(re.findall(r'week\s*(\d+)', c)[0]) for c in week_cols if re.findall(r'week\s*(\d+)', c)]
        if weeks: max_week = max(weeks)

    long_data = []
    
    for idx, row in raw_df.iterrows():
        season = row.get('season', 1)
        final_rank = row.get('placement', np.nan)
        if pd.isna(final_rank) or str(final_rank) == 'nan': continue
        
        try:
            final_rank = int(str(final_rank).replace('Place', '').strip())
        except:
            final_rank = 15
            
        age = row.get('celebrity_age_during_season', 30)
        country = row.get('celebrity_homecountry/region', 'USA')
        industry = row.get('celebrity_industry', 'Actor')
        
        for w in range(1, max_week + 1):
            # 提取当周评委分
            w_cols = [c for c in raw_df.columns if str(w) in c and ('judge' in c or 'score' in c)]
            current_week_scores = []
            for c in w_cols:
                val = row[c]
                try:
                    val = float(val)
                    if not pd.isna(val) and val > 0:
                        current_week_scores.append(val)
                except:
                    pass
            
            if len(current_week_scores) > 0:
                judge_total = np.sum(current_week_scores)
            else:
                judge_total = 0 
            
            if w > 1 and judge_total == 0: continue
            
            long_data.append({
                'player_id': f"S{int(season):02d}-P{idx:03d}",
                'season': int(season),
                'week': w,
                'final_rank': final_rank,
                'judge_score': judge_total,
                'age': age,
                'country': country,
                'industry': industry
            })
            
    df = pd.DataFrame(long_data)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
    
    # 生成 "Actual Eliminate" 标签
    df['actual_eliminate'] = 0
    for s in df['season'].unique():
        for w in df[df['season']==s]['week'].unique():
            mask = (df['season']==s) & (df['week']==w)
            sub = df[mask]
            if len(sub) > 1:
                max_rank = sub['final_rank'].max()
                target_ids = sub[sub['final_rank'] == max_rank]['player_id'].values
                next_week_mask = (df['season']==s) & (df['week']==w+1) & (df['player_id'].isin(target_ids))
                if not df[next_week_mask].shape[0] > 0:
                    df.loc[mask & (df['final_rank'] == max_rank), 'actual_eliminate'] = 1

    return df

print("正在解析并重构数据...")
df = read_and_process_data()
df = df[df['judge_score'] > 0]
print(f"数据重构完成，样本数: {len(df)}")

# ===================== 3. 特征工程 =====================
def prepare_features(df):
    # 评委分标准化
    df['judge_score_std'] = df.groupby(['season', 'week'])['judge_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-5)
    )
    
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    cat_features = ['industry'] 
    X_cat = encoder.fit_transform(df[cat_features])
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[['age']])
    
    # X矩阵：截距 + 行业 + 年龄 + 评委影响力
    X_final = np.hstack([np.ones((len(df), 1)), X_cat, X_num, df[['judge_score_std']].values])
    
    feature_names = ['Intercept'] + list(encoder.get_feature_names_out()) + ['Age', 'Judge_Influence']
    
    return df, X_final, feature_names

df, X_all, feat_names = prepare_features(df)

# ===================== 4. 贝叶斯 MCMC 模型 =====================
def run_better_mcmc(X, y_elim, model_label):
    print(f"🚀 正在训练 {model_label} (N={len(y_elim)})...")
    n_dim = X.shape[1]
    
    def log_prob(theta, x, y):
        # 先验
        lp = -0.5 * np.sum(theta**2) / 2.0
        if not np.isfinite(lp): return -np.inf
        
        # 似然: Logit
        # theta 代表 "生存能力"。能力越高，y=1(淘汰)的概率越低。
        # Logits = Ability
        # P(Elim) = 1 - Sigmoid(Ability) = Sigmoid(-Ability)
        logits = np.dot(x, theta)
        
        # 为了数值稳定，计算 log likelihood
        # y=1 (Elim) -> want low ability -> maximize log(1-p) where p=sigmoid(logits)
        # y=0 (Safe) -> want high ability -> maximize log(p)
        
        # p = sigmoid(logits)
        # log(p) = -log(1 + exp(-logits))
        # log(1-p) = -logits - log(1 + exp(-logits))
        
        # 简化版: 直接用 scipy 的 log_expit 或者手动写稳健公式
        # 这里用近似:
        p = 1.0 / (1.0 + np.exp(-logits))
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1-epsilon)
        
        ll = np.sum(y * np.log(1-p) + (1-y) * np.log(p))
        return lp + ll

    n_walkers = max(32, 2 * n_dim)
    p0 = np.random.randn(n_walkers, n_dim) * 0.1
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=(X, y_elim))
    sampler.run_mcmc(p0, 2000, progress=True)
    
    return sampler.get_chain(discard=500, flat=True)

rank_seasons = [1, 2] + list(range(28, 35))
mask_rank = df['season'].isin(rank_seasons)

samples_rank = run_better_mcmc(X_all[mask_rank], df[mask_rank]['actual_eliminate'].values, "Rank_Era")
samples_pct = run_better_mcmc(X_all[~mask_rank], df[~mask_rank]['actual_eliminate'].values, "Percent_Era")

# ===================== 5. 仿真与具体票数预测 (关键修改) =====================
def simulate_votes_and_elimination(df, X_all, samples_rank, samples_pct, mask_rank):
    print("\n⚙️ 正在计算粉丝投票分布与淘汰预测...")
    
    beta_rank = samples_rank.mean(axis=0)
    beta_pct = samples_pct.mean(axis=0)
    
    # 1. 计算潜在粉丝偏好分 (Log-Odds of Popularity)
    df['latent_popularity'] = 0.0
    df.loc[mask_rank, 'latent_popularity'] = np.dot(X_all[mask_rank], beta_rank)
    df.loc[~mask_rank, 'latent_popularity'] = np.dot(X_all[~mask_rank], beta_pct)
    
    # 初始化新列
    df['pred_vote_share'] = 0.0 # 预测得票率 (0-1)
    df['pred_fan_votes'] = 0    # 预测具体票数 (整数)
    df['est_eliminate'] = 0
    df['final_elim_prob'] = 0.0
    
    # 2. 逐周计算票数分布
    # 假设：每季度的基础投票池不同（早期季度可能更高）
    # 这里使用费米估算（Fermi Estimation）：假设平均每周总票数在 100万 到 500万之间波动
    
    for s in df['season'].unique():
        # 为该赛季设定一个基准流量 (模拟收视率波动)
        season_base_vol = np.random.uniform(2e6, 5e6) # 假设 200w-500w 票
        
        for w in df[df['season']==s]['week'].unique():
            idx = (df['season']==s) & (df['week']==w)
            if idx.sum() == 0: continue
            
            # A. 提取本周选手的潜在人气值
            raw_logits = df.loc[idx, 'latent_popularity'].values
            
            # B. 计算得票率 (Softmax)
            # Softmax 将任意实数映射为概率分布，总和为1
            # 可以添加 temperature 参数调整分布的平坦程度 (temp > 1 更平坦, temp < 1 更尖锐)
            temperature = 1.0 
            vote_shares = softmax(raw_logits / temperature)
            
            df.loc[idx, 'pred_vote_share'] = vote_shares
            
            # C. 计算具体票数
            # 假设决赛周票数更多
            week_factor = 1.0 + (w * 0.05) 
            total_week_votes = season_base_vol * week_factor * np.random.normal(1, 0.1)
            
            # 分配票数
            votes = (vote_shares * total_week_votes).astype(int)
            df.loc[idx, 'pred_fan_votes'] = votes
            
            # D. 预测淘汰 (结合评委分)
            # 淘汰概率 = 1 - P(Survival)
            # 注意：latent_popularity 越高，生存率越高
            survival_prob = 1.0 / (1.0 + np.exp(-raw_logits))
            elim_prob = 1.0 - survival_prob
            
            # 归一化淘汰概率
            elim_prob_norm = softmax(elim_prob * 2) # 放大差异
            df.loc[idx, 'final_elim_prob'] = elim_prob_norm
            
            # 只有当实际有淘汰发生时，才标记预测
            actual_elim_count = df.loc[idx, 'actual_eliminate'].sum()
            if actual_elim_count > 0:
                # 选出淘汰概率最高的 N 个人 (N=actual_elim_count)
                # 获取该周内索引
                week_indices = df[idx].index
                # 排序找到概率最大的前N个
                top_n_idx = np.argsort(elim_prob_norm)[-int(actual_elim_count):]
                
                # 标记全局索引
                elim_global_idx = week_indices[top_n_idx]
                df.loc[elim_global_idx, 'est_eliminate'] = 1

    return df

df = simulate_votes_and_elimination(df, X_all, samples_rank, samples_pct, mask_rank)

# ===================== 6. 结果展示与验证 =====================
def generate_report(df):
    valid_df = df[df['actual_eliminate'].isin([0, 1])]
    acc = accuracy_score(valid_df['actual_eliminate'], valid_df['est_eliminate'])
    
    print(f"\n📊 模型最终性能:")
    print(f"  - 淘汰预测准确率: {acc:.2%}")
    
    # 展示部分预测的粉丝票数
    print("\n🎫 预测粉丝投票样本 (前10行):")
    cols = ['season', 'week', 'player_id', 'industry', 'judge_score', 'pred_vote_share', 'pred_fan_votes', 'actual_eliminate']
    print(df[cols].head(10).to_string(index=False))
    
    # 统计每一周最高票数和最低票数的差距
    df['vote_gap'] = df.groupby(['season', 'week'])['pred_fan_votes'].transform(lambda x: x.max() - x.min())
    
    print("\n💰 投票数据统计:")
    print(f"  - 单周平均总票数: {df.groupby(['season', 'week'])['pred_fan_votes'].sum().mean():,.0f}")
    print(f"  - 选手平均单周得票: {df['pred_fan_votes'].mean():,.0f}")
    
    # 绘图：粉丝票数分布 vs 评委分
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='judge_score', y='pred_fan_votes', hue='actual_eliminate', alpha=0.6)
    plt.title("评委分数 vs 预测粉丝票数 (颜色=实际淘汰)")
    plt.xlabel("评委分数")
    plt.ylabel("预测粉丝票数")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 看一下不同行业的平均得票
    avg_vote_ind = df.groupby('industry')['pred_fan_votes'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=avg_vote_ind.values, y=avg_vote_ind.index, palette='viridis')
    plt.title("各行业选手平均单周得票数 (Top 10)")
    plt.xlabel("平均票数")
    
    plt.tight_layout()
    plt.savefig('Task1_Fan_Votes_Analysis.png', dpi=300)
    print("\n✅ 图表已保存: Task1_Fan_Votes_Analysis.png")
    
    # 保存详细Excel
    df.to_excel("Task1_Predicted_Fan_Votes.xlsx", index=False)
    print("✅ 详细数据已保存: Task1_Predicted_Fan_Votes.xlsx")

generate_report(df)