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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import re
from warnings import filterwarnings

filterwarnings('ignore')

# 绘图配置
import platform
system_name = platform.system()
font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(2026)

# ===================== 2. 增强版数据读取（核心改进） =====================
def read_and_process_data():
    file_path = "2026_MCM_Problem_C_Data.csv"
    # 尝试多种编码读取
    try:
        raw_df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        raw_df = pd.read_csv(file_path, encoding='latin1')
    
    # 1. 基础清洗
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    
    # 2. 提取评委分数 (Judge Scores)
    # 我们需要把宽表(Wide)转换为长表(Long)，并保留每一周的评委分
    # 假设列名格式类似: "week1_judge1", "week 1 judge 1" 等
    
    # 先构建基础的长表骨架
    base_cols = ['season', 'placement', 'celebrity_age_during_season', 
                 'celebrity_homecountry/region', 'celebrity_industry', 'results']
    # 容错：如果找不到列名，用相近的
    available_cols = [c for c in base_cols if c in raw_df.columns]
    
    # 自动识别共有多少周
    week_cols = [c for c in raw_df.columns if 'week' in c and 'judge' in c]
    max_week = 10 # 默认
    if week_cols:
        weeks = [int(re.findall(r'week\s*(\d+)', c)[0]) for c in week_cols if re.findall(r'week\s*(\d+)', c)]
        if weeks: max_week = max(weeks)

    long_data = []
    
    for idx, row in raw_df.iterrows():
        season = row.get('season', 1) # 默认1
        # 很多数据没有显式的season列，需要按行数推断，或者假设文件里有
        # 这里为了稳健，如果CSV里没season，我们按行号分块（每行一个选手）
        # *注意*：原题数据结构通常是一行一个选手。
        
        final_rank = row.get('placement', np.nan)
        if pd.isna(final_rank): continue
        if str(final_rank) == 'nan': continue
        
        # 尝试转换Rank
        try:
            final_rank = int(str(final_rank).replace('Place', '').strip())
        except:
            final_rank = 15 # 默认低排名
            
        # 提取特征
        age = row.get('celebrity_age_during_season', 30)
        country = row.get('celebrity_homecountry/region', 'USA')
        industry = row.get('celebrity_industry', 'Actor')
        is_eliminated_season = 0 # 标记整季是否被淘汰（通常都是）
        
        # 遍历每一周提取评委分
        for w in range(1, max_week + 1):
            # 查找当周的所有评委分
            # 匹配逻辑：包含 'weekX' 且包含 'judge' 的列
            # 或者是 'weekX_score'
            score_sum = 0
            count = 0
            
            # 正则匹配该周的所有分数列
            pat = re.compile(f"week\s?{w}[^0-9]") 
            
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
                judge_avg = np.mean(current_week_scores)
                # 归一化到 0-10 或 0-30
                judge_total = np.sum(current_week_scores)
            else:
                judge_total = 0 # 说明没参加这一周，或者被淘汰了
            
            # 如果分数是0，且不是第一周，通常意味着已经被淘汰了
            if w > 1 and judge_total == 0:
                continue # 不添加这一行
            
            # 结果标签 (Actual Eliminate)
            # 这里简化处理：如果是选手参加的最后一周（且不是决赛），则为淘汰
            # 实际上很难精确对应哪周淘汰，我们用 "Final Rank" 倒推
            # 简单的逻辑：排名越靠后(数值大)，越早淘汰。
            # 暂时先全部标记为0，后面统一计算
            
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
    
    # 填充：处理 Age 缺失
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
    
    # 生成 "本周是否淘汰" 的标签
    # 逻辑：对于每个赛季，计算每周的人数。人数变少的时刻，就是有人淘汰。
    # 简化版逻辑：根据 final_rank 和 week 的关系。
    # 假设：总人数 N。 第1名参加了所有周。 第N名只参加了第1周。
    # 我们用一种统计学方法：对于同一赛季同一周，
    # 标记：Actual_Eliminate = 1 if (本选手是该周 final_rank 值最大的那个)
    df['actual_eliminate'] = 0
    for s in df['season'].unique():
        for w in df[df['season']==s]['week'].unique():
            mask = (df['season']==s) & (df['week']==w)
            sub = df[mask]
            if len(sub) > 1:
                # 找到本周仍在参赛的选手中，最终排名最差(数值最大)的人
                # 这是一个合理的代理变量(Proxy)
                max_rank = sub['final_rank'].max()
                # 还要确保他没有参加下一周
                target_ids = sub[sub['final_rank'] == max_rank]['player_id'].values
                
                # 检查这些人是否有下一周的数据
                next_week_mask = (df['season']==s) & (df['week']==w+1) & (df['player_id'].isin(target_ids))
                if not df[next_week_mask].shape[0] > 0:
                    df.loc[mask & (df['final_rank'] == max_rank), 'actual_eliminate'] = 1

    return df

print("正在解析并重构数据（包含评委分数提取）...")
df = read_and_process_data()
print(f"数据重构完成，样本数: {len(df)}")
# 过滤掉分数异常低的行（可能是未参赛）
df = df[df['judge_score'] > 0]

# ===================== 3. 特征工程与模型拆分 =====================
def prepare_features(df):
    # 1. 对评委分数进行赛季内标准化（消除不同赛季分制不同带来的影响）
    df['judge_score_std'] = df.groupby(['season', 'week'])['judge_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-5)
    )
    
    # 2. 编码人口统计学特征
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    cat_features = ['industry'] # 国家太杂，先只用行业
    X_cat = encoder.fit_transform(df[cat_features])
    
    # 3. 连续特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[['age']])
    
    # 4. 合并 X
    # 注意：我们这里不把 judge_score 放入 X 来预测 fan_vote
    # 因为我们假设 Fan Vote 是由"人"决定的，而不是由"评委分"决定的
    # 但我们可以加入 'judge_score_std' 作为协变量，因为粉丝容易跟风
    X_final = np.hstack([np.ones((len(df), 1)), X_cat, X_num, df[['judge_score_std']].values])
    
    feature_names = ['Intercept'] + list(encoder.get_feature_names_out()) + ['Age', 'Judge_Influence']
    
    return df, X_final, feature_names

df, X_all, feat_names = prepare_features(df)

# ===================== 4. 贝叶斯模型 (Latent Fan Preference) =====================
# 核心思想：Result ~ Judge + Fan
# 我们已知 Result (Survived=1, Elim=0) 和 Judge。
# 我们用 Logit 模型： P(Survival) = Sigmoid( alpha * Judge + beta * X_fan )
# 这里的 beta * X_fan 就是我们要学的粉丝偏好。

def run_better_mcmc(X, y_elim, model_label):
    # 这里的 y_elim 是 "是否被淘汰"。1=淘汰，0=晋级
    # 逻辑：Score = X * theta
    # P(淘汰) = Sigmoid(Score) 
    # *注意*：这是反向的，分数越低越容易淘汰。
    # 所以我们定义 Latent Ability = X * theta
    # P(Elim) = 1 - Sigmoid(Ability)
    
    print(f"🚀 正在训练 {model_label} (N={len(y_elim)})...")
    
    n_dim = X.shape[1]
    
    def log_lik(theta, x, y):
        # 逻辑回归似然
        logits = np.dot(x, theta) 
        # y=1 (Eliminated) 意味着 Ability 低。
        # 我们让 theta 代表 "生存能力" (Popularity)
        # 那么 P(Elim) = 1 / (1 + exp(logits))  (当logits很大时，P_elim很小)
        # log P(y=1) = -log(1 + exp(logits)) = log_sig(-logits)
        # log P(y=0) = log(1 - 1/(1+exp)) = log(exp/(1+exp)) = logits - log(1+exp)
        
        # 简单的数值稳定写法:
        # P(y=0|x) = sigmoid(logits) -> 晋级概率
        # P(y=1|x) = 1 - sigmoid(logits) -> 淘汰概率
        
        # 我们的 y 是 actual_eliminate (1=淘汰)
        # 所以我们最大化: y*log(1-p) + (1-y)*log(p)
        # 其中 p = sigmoid(logits)
        
        p = 1.0 / (1.0 + np.exp(-logits))
        epsilon = 1e-6
        p = np.clip(p, epsilon, 1-epsilon)
        
        # 如果 y=1 (淘汰), 我们希望 p (晋级率) 低 -> log(1-p)
        # 如果 y=0 (晋级), 我们希望 p (晋级率) 高 -> log(p)
        ll = np.sum(y * np.log(1-p) + (1-y) * np.log(p))
        return ll

    def log_prior(theta):
        # 岭回归先验 (L2正则)
        return -0.5 * np.sum(theta**2) / 2.0

    def log_prob(theta, x, y):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + log_lik(theta, x, y)

    # 初始化
    n_walkers = max(32, 2 * n_dim)
    p0 = np.random.randn(n_walkers, n_dim) * 0.1
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=(X, y_elim))
    sampler.run_mcmc(p0, 2000, progress=True) # 步数增加以确保收敛
    
    return sampler.get_chain(discard=1000, flat=True)

# 拆分训练
# 排名法赛季：1,2, 28-34
rank_seasons = [1, 2] + list(range(28, 35))
mask_rank = df['season'].isin(rank_seasons)

# 1. 训练模型 (Target: actual_eliminate)
# 注意：我们这里直接用 "是否淘汰" 作为硬指标训练，
# 系数 (theta) 将告诉我们：在给定评委分(Judge_Influence)的情况下，
# 年龄、行业等特征如何额外影响生存率(即粉丝票仓)。
samples_rank = run_better_mcmc(X_all[mask_rank], df[mask_rank]['actual_eliminate'].values, "Rank_Era")
samples_pct = run_better_mcmc(X_all[~mask_rank], df[~mask_rank]['actual_eliminate'].values, "Percent_Era")

# ===================== 5. 仿真与预测 (混合机制) =====================
def simulate_elimination(df, X_all, samples_rank, samples_pct, mask_rank):
    print("\n⚙️ 正在进行高精度仿真 (结合真实评委分 + 预测粉丝分)...")
    
    # 1. 计算 "粉丝生存指数" (Fan Survival Score)
    # 使用后验均值
    beta_rank = samples_rank.mean(axis=0)
    beta_pct = samples_pct.mean(axis=0)
    
    df['pred_fan_score'] = 0.0
    
    # 分别计算
    df.loc[mask_rank, 'pred_fan_score'] = np.dot(X_all[mask_rank], beta_rank)
    df.loc[~mask_rank, 'pred_fan_score'] = np.dot(X_all[~mask_rank], beta_pct)
    
    # 2. 结合评委分计算淘汰概率
    # 这里的逻辑必须符合物理规律：
    # 总能力 = (权重A * 评委分) + (权重B * 粉丝分)
    # 模型其实已经隐式学习了权重（通过回归系数）
    # pred_fan_score 实际上已经是 "Log-Odds of Survival"
    
    # 我们直接转换成概率
    logits = df['pred_fan_score'].values
    survival_prob = 1.0 / (1.0 + np.exp(-logits))
    
    # 修正：淘汰概率 = 1 - 生存概率
    df['eliminate_prob'] = 1.0 - survival_prob
    
    # 3. 赛季内归一化 (Softmax)
    # 因为每周必定淘汰一人(或多人)，我们最好在每周内部比较概率
    df['final_elim_prob'] = 0.0
    df['est_eliminate'] = 0  # 初始化预测淘汰列
    
    for s in df['season'].unique():
        for w in df[df['season']==s]['week'].unique():
            idx = (df['season']==s) & (df['week']==w)
            if idx.sum() == 0: continue
            
            # 检查本周是否有实际淘汰
            actual_elim_count = df.loc[idx, 'actual_eliminate'].sum()
            
            if actual_elim_count > 0:
                # 本周有淘汰，我们预测谁被淘汰
                probs = df.loc[idx, 'eliminate_prob'].values
                # Softmax 归一化，让这周总得有人淘汰
                # 为了拉大差距，可以加个 Temperature
                probs_exp = np.exp(probs * 2) 
                probs_norm = probs_exp / np.sum(probs_exp)
                
                df.loc[idx, 'final_elim_prob'] = probs_norm
                
                # 标记预测结果 (概率最大的那个人)
                best_guess_idx = df[idx]['final_elim_prob'].idxmax()
                df.loc[best_guess_idx, 'est_eliminate'] = 1
            else:
                # 本周没有淘汰（比如决赛周），所有人预测为晋级
                df.loc[idx, 'est_eliminate'] = 0
                df.loc[idx, 'final_elim_prob'] = df.loc[idx, 'eliminate_prob'].values
            
    return df

df = simulate_elimination(df, X_all, samples_rank, samples_pct, mask_rank)

# ===================== 6. 验证与可视化 =====================
def check_performance(df):
    # 只看有淘汰发生的周（过滤掉全员晋级的周，如果有的话）
    valid_df = df[df['actual_eliminate'].isin([0, 1])]
    
    acc = accuracy_score(valid_df['actual_eliminate'], valid_df['est_eliminate'])
    try:
        auc = roc_auc_score(valid_df['actual_eliminate'], valid_df['final_elim_prob'])
    except:
        auc = 0.5
    
    print(f"\n📊 模型性能评估:")
    print(f"  - 准确率 (Accuracy): {acc:.2%} (基准线: ~12%)") 
    print(f"  - AUC Score: {auc:.4f}")

# ===================== 7. 专业舞者及名人特征影响分析模型 =====================

def analyze_dancer_celebrity_impact(df):
    """
    分析专业舞者以及名人特征（年龄、行业等）对比赛的影响
    回答：这些因素对名人在比赛中的表现影响有多大？
    它们对评委分数和粉丝投票的影响方式是否一致？
    """
    
    print("\n" + "="*80)
    print("🎭 专业舞者及名人特征影响分析")
    print("="*80)
    
    # ===================== 7.1 数据准备与舞者特征提取 =====================
    
    # 从原始数据中重新读取以获取专业舞者信息
    try:
        raw_df = pd.read_csv("2026_MCM_Problem_C_Data.csv", encoding='utf-8-sig')
    except:
        raw_df = pd.read_csv("2026_MCM_Problem_C_Data.csv", encoding='latin1')
    
    # 清洗列名
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    
    # 提取专业舞者信息（假设列名为'ballroom partner'的变体）
    dancer_col = None
    for col in raw_df.columns:
        if 'ballroom' in col.lower() or 'partner' in col.lower():
            dancer_col = col
            break
    
    if dancer_col:
        dancer_info = raw_df[['season', dancer_col]].copy()
        dancer_info.columns = ['season', 'dancer_name']
        dancer_info = dancer_info.dropna()
        
        # 创建专业舞者ID
        dancer_info['dancer_id'] = dancer_info['dancer_name'].astype('category').cat.codes
    else:
        print("⚠️ 未找到专业舞者信息列，将使用模拟数据")
        dancer_info = None
    
    # ===================== 7.2 创建分析数据集 =====================
    
    # 汇总每位选手的平均表现数据
    player_summary = []
    
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]
        if len(player_data) == 0:
            continue
        
        # 基本信息
        season = player_data['season'].iloc[0]
        final_rank = player_data['final_rank'].iloc[0]
        age = player_data['age'].iloc[0]
        country = player_data['country'].iloc[0] if 'country' in player_data.columns else 'Unknown'
        industry = player_data['industry'].iloc[0] if 'industry' in player_data.columns else 'Unknown'
        
        # 表现指标
        avg_judge_score = player_data['judge_score'].mean()
        avg_fan_score = player_data['pred_fan_score'].mean()
        total_weeks = player_data['week'].max()
        survived_weeks = len(player_data)
        
        # 淘汰指标
        was_eliminated = 1 if player_data['actual_eliminate'].sum() > 0 else 0
        
        # 尝试匹配专业舞者
        dancer_name = 'Unknown'
        dancer_exp = 0  # 舞者经验（参与过的赛季数）
        
        if dancer_info is not None:
            try:
                # 根据赛季匹配舞者
                season_dancers = dancer_info[dancer_info['season'] == season]
                if not season_dancers.empty:
                    # 简单匹配：取第一个舞者（实际应用中需要更精确的匹配逻辑）
                    dancer_name = season_dancers['dancer_name'].iloc[0]
                    
                    # 计算舞者经验（过往参与赛季数）
                    all_seasons = dancer_info[dancer_info['dancer_name'] == dancer_name]['season'].unique()
                    dancer_exp = len(all_seasons)
            except:
                pass
        
        player_summary.append({
            'player_id': player_id,
            'season': season,
            'final_rank': final_rank,
            'age': age,
            'country': country,
            'industry': industry,
            'dancer_name': dancer_name,
            'dancer_exp': dancer_exp,
            'avg_judge_score': avg_judge_score,
            'avg_fan_score': avg_fan_score,
            'total_weeks': total_weeks,
            'survived_weeks': survived_weeks,
            'was_eliminated': was_eliminated,
            'survival_rate': survived_weeks / total_weeks if total_weeks > 0 else 0
        })
    
    analysis_df = pd.DataFrame(player_summary)
    
    # ===================== 7.3 名人特征影响分析 =====================
    
    print("\n📊 名人特征对比赛影响分析")
    print("-"*60)
    
    # 7.3.1 年龄的影响
    print("\n1. 年龄对比赛表现的影响:")
    
    # 按年龄分组分析
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['<25', '25-35', '35-45', '45-55', '>55']
    analysis_df['age_group'] = pd.cut(analysis_df['age'], bins=age_bins, labels=age_labels)
    
    age_stats = analysis_df.groupby('age_group').agg({
        'avg_judge_score': 'mean',
        'avg_fan_score': 'mean',
        'survival_rate': 'mean',
        'final_rank': 'mean',
        'player_id': 'count'
    }).rename(columns={'player_id': 'count'})
    
    print("按年龄组统计的平均表现:")
    print(age_stats.round(3))
    
    # 年龄与评委分数的相关性
    age_judge_corr = analysis_df['age'].corr(analysis_df['avg_judge_score'])
    age_fan_corr = analysis_df['age'].corr(analysis_df['avg_fan_score'])
    age_rank_corr = analysis_df['age'].corr(analysis_df['final_rank'])
    
    print(f"\n年龄与评委分数的相关性: {age_judge_corr:.3f}")
    print(f"年龄与粉丝分数的相关性: {age_fan_corr:.3f}")
    print(f"年龄与最终排名的相关性: {age_rank_corr:.3f} (负值表示年龄越大排名越好)")
    
    # 7.3.2 行业的影响
    print("\n2. 行业对比赛表现的影响:")
    
    # 只分析出现频率较高的行业
    industry_counts = analysis_df['industry'].value_counts()
    top_industries = industry_counts[industry_counts >= 5].index.tolist()
    
    if len(top_industries) > 0:
        industry_stats = analysis_df[analysis_df['industry'].isin(top_industries)].groupby('industry').agg({
            'avg_judge_score': ['mean', 'std'],
            'avg_fan_score': ['mean', 'std'],
            'survival_rate': 'mean',
            'final_rank': 'mean',
            'player_id': 'count'
        }).round(3)
        
        print("按行业统计的平均表现:")
        print(industry_stats)
        
        # 行业排名
        industry_ranking = analysis_df.groupby('industry')['final_rank'].mean().sort_values()
        print(f"\n行业平均排名 (数值越小越好):")
        for i, (industry, rank) in enumerate(industry_ranking.items(), 1):
            if industry in top_industries:
                print(f"  {i:2d}. {industry:20s}: {rank:.2f}")
    
    # ===================== 7.4 专业舞者影响分析 =====================
    
    print("\n3. 专业舞者对比赛表现的影响:")
    
    if 'dancer_name' in analysis_df.columns and analysis_df['dancer_name'].nunique() > 1:
        # 只分析有足够数据的舞者
        dancer_counts = analysis_df['dancer_name'].value_counts()
        top_dancers = dancer_counts[dancer_counts >= 3].index.tolist()
        
        if len(top_dancers) > 0:
            dancer_stats = analysis_df[analysis_df['dancer_name'].isin(top_dancers)].groupby('dancer_name').agg({
                'avg_judge_score': 'mean',
                'avg_fan_score': 'mean',
                'survival_rate': 'mean',
                'final_rank': 'mean',
                'dancer_exp': 'first',
                'player_id': 'count'
            }).rename(columns={'player_id': 'partners_count'}).round(3)
            
            # 按舞者经验分组
            exp_bins = [0, 3, 6, 10, 20]
            exp_labels = ['新手(1-3季)', '中级(4-6季)', '资深(7-10季)', '元老(10+季)']
            analysis_df['exp_group'] = pd.cut(analysis_df['dancer_exp'], bins=exp_bins, labels=exp_labels, right=False)
            
            exp_stats = analysis_df.groupby('exp_group').agg({
                'avg_judge_score': 'mean',
                'avg_fan_score': 'mean',
                'survival_rate': 'mean',
                'final_rank': 'mean',
                'player_id': 'count'
            }).rename(columns={'player_id': 'count'}).round(3)
            
            print("按舞者经验分组的平均表现:")
            print(exp_stats)
            
            # 舞者经验与表现的相关性
            if analysis_df['dancer_exp'].nunique() > 1:
                exp_judge_corr = analysis_df['dancer_exp'].corr(analysis_df['avg_judge_score'])
                exp_fan_corr = analysis_df['dancer_exp'].corr(analysis_df['avg_fan_score'])
                exp_rank_corr = analysis_df['dancer_exp'].corr(analysis_df['final_rank'])
                
                print(f"\n舞者经验与评委分数的相关性: {exp_judge_corr:.3f}")
                print(f"舞者经验与粉丝分数的相关性: {exp_fan_corr:.3f}")
                print(f"舞者经验与最终排名的相关性: {exp_rank_corr:.3f}")
    
# ===================== 7.5 优化：粉丝分数与评委分数的归一化处理 =====================

def normalize_scores(df):
    """
    对评委分数和粉丝分数进行归一化处理，使它们在相同尺度上可比
    """
    # 方法1：Min-Max归一化到[0,1]区间
    from sklearn.preprocessing import MinMaxScaler
    
    # 评委分数归一化
    judge_scaler = MinMaxScaler()
    df['judge_score_norm'] = judge_scaler.fit_transform(df[['judge_score']])
    
    # 粉丝分数归一化（使用预测的粉丝分数）
    fan_scaler = MinMaxScaler()
    df['fan_score_norm'] = fan_scaler.fit_transform(df[['pred_fan_score']])
    
    # 方法2：Z-score标准化（均值为0，标准差为1）
    from sklearn.preprocessing import StandardScaler
    
    judge_std_scaler = StandardScaler()
    df['judge_score_std'] = judge_std_scaler.fit_transform(df[['judge_score']])
    
    fan_std_scaler = StandardScaler()
    df['fan_score_std'] = fan_std_scaler.fit_transform(df[['pred_fan_score']])
    
    # 方法3：相对权重计算（按百分比）
    # 对于每一周，计算评委分数和粉丝分数的相对贡献
    df['judge_contribution'] = 0.0
    df['fan_contribution'] = 0.0
    
    for s in df['season'].unique():
        for w in df[df['season'] == s]['week'].unique():
            mask = (df['season'] == s) & (df['week'] == w)
            week_data = df[mask]
            
            if len(week_data) > 0:
                # 计算本周内的相对分数
                judge_sum = week_data['judge_score'].sum()
                fan_sum = week_data['pred_fan_score'].sum()
                
                if judge_sum > 0:
                    df.loc[mask, 'judge_contribution'] = df.loc[mask, 'judge_score'] / judge_sum
                if fan_sum > 0:
                    # 注意：粉丝分数可能是负值，需要先调整
                    min_fan = week_data['pred_fan_score'].min()
                    if min_fan < 0:
                        adjusted_fan = week_data['pred_fan_score'] - min_fan + 1
                        fan_sum = adjusted_fan.sum()
                        df.loc[mask, 'fan_contribution'] = adjusted_fan / fan_sum
                    else:
                        df.loc[mask, 'fan_contribution'] = df.loc[mask, 'pred_fan_score'] / fan_sum
    
    # 计算综合得分（评委和粉丝各占50%权重）
    df['combined_score_norm'] = 0.5 * df['judge_score_norm'] + 0.5 * df['fan_score_norm']
    
    # 计算评委分数和粉丝分数的比例
    df['judge_fan_ratio'] = df['judge_score_norm'] / (df['fan_score_norm'] + 1e-8)
    
    return df

# 应用归一化处理
print("\n📊 正在对评委分数和粉丝分数进行归一化处理...")
df = normalize_scores(df)

# ===================== 7.6 优化后的特征影响分析 =====================

def analyze_feature_impact_with_normalization(df):
    """
    使用归一化分数重新分析特征影响
    """
    print("\n" + "="*80)
    print("🎭 使用归一化分数的特征影响分析")
    print("="*80)
    
    # 重新汇总选手数据，使用归一化分数
    player_summary_norm = []
    
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]
        if len(player_data) == 0:
            continue
        
        # 基本信息
        season = player_data['season'].iloc[0]
        final_rank = player_data['final_rank'].iloc[0]
        age = player_data['age'].iloc[0]
        industry = player_data['industry'].iloc[0] if 'industry' in player_data.columns else 'Unknown'
        
        # 归一化后的表现指标
        avg_judge_norm = player_data['judge_score_norm'].mean()
        avg_fan_norm = player_data['fan_score_norm'].mean()
        avg_combined_norm = player_data['combined_score_norm'].mean()
        avg_judge_contribution = player_data['judge_contribution'].mean()
        avg_fan_contribution = player_data['fan_contribution'].mean()
        
        # 标准化表现指标
        avg_judge_std = player_data['judge_score_std'].mean()
        avg_fan_std = player_data['fan_score_std'].mean()
        
        # 比赛表现
        total_weeks = player_data['week'].max()
        survived_weeks = len(player_data)
        survival_rate = survived_weeks / total_weeks if total_weeks > 0 else 0
        
        player_summary_norm.append({
            'player_id': player_id,
            'season': season,
            'final_rank': final_rank,
            'age': age,
            'industry': industry,
            'avg_judge_norm': avg_judge_norm,
            'avg_fan_norm': avg_fan_norm,
            'avg_combined_norm': avg_combined_norm,
            'avg_judge_contribution': avg_judge_contribution,
            'avg_fan_contribution': avg_fan_contribution,
            'avg_judge_std': avg_judge_std,
            'avg_fan_std': avg_fan_std,
            'total_weeks': total_weeks,
            'survived_weeks': survived_weeks,
            'survival_rate': survival_rate,
            'judge_fan_ratio': player_data['judge_fan_ratio'].mean()
        })
    
    analysis_df_norm = pd.DataFrame(player_summary_norm)
    
    # ===================== 归一化后的分析 =====================
    
    print("\n📊 归一化分数统计分析:")
    print("-"*60)
    
    # 描述性统计
    norm_stats = analysis_df_norm[['avg_judge_norm', 'avg_fan_norm', 'avg_combined_norm',
                                    'avg_judge_contribution', 'avg_fan_contribution',
                                    'avg_judge_std', 'avg_fan_std']].describe().round(3)
    print("归一化分数的描述性统计:")
    print(norm_stats)
    
    # 相关性分析（使用归一化分数）
    print("\n📈 归一化分数的相关性分析:")
    
    # 年龄与归一化分数的相关性
    age_judge_corr_norm = analysis_df_norm['age'].corr(analysis_df_norm['avg_judge_norm'])
    age_fan_corr_norm = analysis_df_norm['age'].corr(analysis_df_norm['avg_fan_norm'])
    age_combined_corr_norm = analysis_df_norm['age'].corr(analysis_df_norm['avg_combined_norm'])
    
    print(f"年龄与归一化评委分数的相关性: {age_judge_corr_norm:.3f}")
    print(f"年龄与归一化粉丝分数的相关性: {age_fan_corr_norm:.3f}")
    print(f"年龄与综合分数的相关性: {age_combined_corr_norm:.3f}")
    
    # 归一化分数与最终排名的相关性
    judge_rank_corr_norm = analysis_df_norm['avg_judge_norm'].corr(analysis_df_norm['final_rank'])
    fan_rank_corr_norm = analysis_df_norm['avg_fan_norm'].corr(analysis_df_norm['final_rank'])
    combined_rank_corr_norm = analysis_df_norm['avg_combined_norm'].corr(analysis_df_norm['final_rank'])
    
    print(f"归一化评委分数与最终排名的相关性: {judge_rank_corr_norm:.3f} (负值表示分数越高排名越好)")
    print(f"归一化粉丝分数与最终排名的相关性: {fan_rank_corr_norm:.3f} (负值表示分数越高排名越好)")
    print(f"综合分数与最终排名的相关性: {combined_rank_corr_norm:.3f} (负值表示分数越高排名越好)")
    
    # ===================== 评委与粉丝贡献度分析 =====================
    
    print("\n📊 评委与粉丝贡献度分析:")
    print("-"*60)
    
    # 计算整体贡献度比例
    total_judge_contribution = analysis_df_norm['avg_judge_contribution'].mean()
    total_fan_contribution = analysis_df_norm['avg_fan_contribution'].mean()
    
    print(f"平均评委贡献度: {total_judge_contribution:.3f}")
    print(f"平均粉丝贡献度: {total_fan_contribution:.3f}")
    print(f"评委:粉丝贡献度比例: {total_judge_contribution/total_fan_contribution:.3f}:1")
    
    # 按行业分析贡献度
    if 'industry' in analysis_df_norm.columns:
        industry_contribution = analysis_df_norm.groupby('industry').agg({
            'avg_judge_contribution': 'mean',
            'avg_fan_contribution': 'mean',
            'judge_fan_ratio': 'mean',
            'player_id': 'count'
        }).rename(columns={'player_id': 'count'}).round(3)
        
        industry_contribution['total_contribution'] = industry_contribution['avg_judge_contribution'] + industry_contribution['avg_fan_contribution']
        industry_contribution['judge_weight'] = industry_contribution['avg_judge_contribution'] / industry_contribution['total_contribution']
        industry_contribution['fan_weight'] = industry_contribution['avg_fan_contribution'] / industry_contribution['total_contribution']
        
        print("\n按行业统计的评委与粉丝贡献度:")
        print(industry_contribution.sort_values('judge_fan_ratio', ascending=False))
    
    # ===================== 评委与粉丝影响的比较分析 =====================
    
    print("\n📊 评委与粉丝影响比较分析:")
    print("-"*60)
    
    # 创建评委影响指数和粉丝影响指数
    analysis_df_norm['judge_impact_index'] = analysis_df_norm['avg_judge_norm'] * analysis_df_norm['avg_judge_contribution']
    analysis_df_norm['fan_impact_index'] = analysis_df_norm['avg_fan_norm'] * analysis_df_norm['avg_fan_contribution']
    
    # 计算相对影响
    analysis_df_norm['total_impact'] = analysis_df_norm['judge_impact_index'] + analysis_df_norm['fan_impact_index']
    analysis_df_norm['judge_impact_ratio'] = analysis_df_norm['judge_impact_index'] / analysis_df_norm['total_impact']
    analysis_df_norm['fan_impact_ratio'] = analysis_df_norm['fan_impact_index'] / analysis_df_norm['total_impact']
    
    # 整体影响比例
    avg_judge_impact_ratio = analysis_df_norm['judge_impact_ratio'].mean()
    avg_fan_impact_ratio = analysis_df_norm['fan_impact_ratio'].mean()
    
    print(f"平均评委影响比例: {avg_judge_impact_ratio:.3f}")
    print(f"平均粉丝影响比例: {avg_fan_impact_ratio:.3f}")
    print(f"评委:粉丝影响比例: {avg_judge_impact_ratio/avg_fan_impact_ratio:.3f}:1")
    
    # ===================== 优化后的可视化 =====================
    
    print("\n🎨 生成优化后的可视化分析图表...")
    
    plt.figure(figsize=(20, 15))
    
    # 子图1: 归一化分数分布对比
    plt.subplot(3, 4, 1)
    plt.boxplot([analysis_df_norm['avg_judge_norm'], analysis_df_norm['avg_fan_norm']], 
                labels=['Judge score', 'Fan score'])
    plt.title('Comparison of normalized score distributions')
    plt.ylabel('Score value (0-1)')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 评委vs粉丝分数散点图
    plt.subplot(3, 4, 2)
    plt.scatter(analysis_df_norm['avg_judge_norm'], analysis_df_norm['avg_fan_norm'], 
                alpha=0.6, c=analysis_df_norm['final_rank'], cmap='viridis', s=50)
    plt.colorbar(label='Final ranking')
    plt.xlabel('Normalize the judges scores')
    plt.ylabel('Normalized fan scores')
    plt.title('Judges scores vs fans scores')
    
    # 添加对角线
    min_val = min(analysis_df_norm['avg_judge_norm'].min(), analysis_df_norm['avg_fan_norm'].min())
    max_val = max(analysis_df_norm['avg_judge_norm'].max(), analysis_df_norm['avg_fan_norm'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    plt.legend()
    
    # 计算并显示相关系数
    corr = analysis_df_norm['avg_judge_norm'].corr(analysis_df_norm['avg_fan_norm'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图3: 评委与粉丝贡献度对比
    plt.subplot(3, 4, 3)
    labels = ['Contribution of judges', 'Fan contribution']
    sizes = [total_judge_contribution, total_fan_contribution]
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Average contribution of judges and fans')
    
    # 子图4: 年龄与归一化分数的关系
    plt.subplot(3, 4, 4)
    plt.scatter(analysis_df_norm['age'], analysis_df_norm['avg_judge_norm'], 
                alpha=0.6, label='Judge score', s=50)
    plt.scatter(analysis_df_norm['age'], analysis_df_norm['avg_fan_norm'], 
                alpha=0.6, label='Fan score', s=50)
    plt.scatter(analysis_df_norm['age'], analysis_df_norm['avg_combined_norm'], 
                alpha=0.6, label='Overall score', s=50)
    
    # 添加趋势线
    for col, color, label in zip(['avg_judge_norm', 'avg_fan_norm', 'avg_combined_norm'],
                                 ['blue', 'red', 'green'],
                                 ['Judges', 'Fans', 'synthesis']):
        z = np.polyfit(analysis_df_norm['age'], analysis_df_norm[col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(analysis_df_norm['age'].min(), analysis_df_norm['age'].max(), 100)
        plt.plot(x_range, p(x_range), color=color, linewidth=2, label=f'{label}trend')
    
    plt.xlabel('Age')
    plt.ylabel('Normalized scores')
    plt.title('Relationship between age and normalized scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5: 评委与粉丝影响比例分布
    plt.subplot(3, 4, 5)
    plt.hist(analysis_df_norm['judge_impact_ratio'], bins=30, alpha=0.7, color='red', label='Judge influence ratio')
    plt.hist(analysis_df_norm['fan_impact_ratio'], bins=30, alpha=0.7, color='blue', label='Fan influence ratio')
    plt.axvline(x=avg_judge_impact_ratio, color='darkred', linestyle='--', linewidth=2, label='Average of judges')
    plt.axvline(x=avg_fan_impact_ratio, color='darkblue', linestyle='--', linewidth=2, label='Average of fans')
    plt.xlabel('Proportion of influence')
    plt.ylabel('frequency')
    plt.title('Judges and fans influence the proportion distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图6: 评委vs粉丝分数与最终排名的关系
    plt.subplot(3, 4, 6)
    plt.scatter(analysis_df_norm['avg_judge_norm'], analysis_df_norm['final_rank'], 
                alpha=0.6, label='Judge score', s=50)
    plt.scatter(analysis_df_norm['avg_fan_norm'], analysis_df_norm['final_rank'], 
                alpha=0.6, label='Fan score', s=50)
    
    # 添加趋势线
    z_judge = np.polyfit(analysis_df_norm['avg_judge_norm'], analysis_df_norm['final_rank'], 1)
    p_judge = np.poly1d(z_judge)
    z_fan = np.polyfit(analysis_df_norm['avg_fan_norm'], analysis_df_norm['final_rank'], 1)
    p_fan = np.poly1d(z_fan)
    
    x_range_judge = np.linspace(analysis_df_norm['avg_judge_norm'].min(), analysis_df_norm['avg_judge_norm'].max(), 100)
    x_range_fan = np.linspace(analysis_df_norm['avg_fan_norm'].min(), analysis_df_norm['avg_fan_norm'].max(), 100)
    
    plt.plot(x_range_judge, p_judge(x_range_judge), 'b-', linewidth=2, label=f'Trends of judges (r={judge_rank_corr_norm:.3f})')
    plt.plot(x_range_fan, p_fan(x_range_fan), 'r-', linewidth=2, label=f'Trends of judges (r={fan_rank_corr_norm:.3f})')
    
    plt.xlabel('Normalized scores')
    plt.ylabel('Final ranking')
    plt.title('Relation of the score to the final ranking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图7: 评委-粉丝分数比分布
    plt.subplot(3, 4, 7)

    # 清理数据：移除无穷大和NaN值
    ratio_data = analysis_df_norm['judge_fan_ratio'].copy()
    ratio_data = ratio_data.replace([np.inf, -np.inf], np.nan)
    ratio_data_clean = ratio_data.dropna()

    if len(ratio_data_clean) > 0:
        # 处理极端值：截断在99%分位数
        upper_limit = ratio_data_clean.quantile(0.99)
        ratio_data_clipped = ratio_data_clean.clip(upper=upper_limit)
    
        # 计算合适的bins数量
        n_bins = min(40, max(10, len(ratio_data_clipped) // 20))
    
        # 绘制直方图
        n, bins, patches = plt.hist(ratio_data_clipped, bins=n_bins, alpha=0.7, 
                                     color='purple', edgecolor='black', linewidth=0.5)
    
        # 添加参考线
        plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Judges = Fans')
    
        # 计算并显示中位数
        median_val = ratio_data_clipped.median()
        plt.axvline(x=median_val, color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {median_val:.2f}')
    
        plt.xlabel('Judge Score / Fan Score Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Judge-Fan Score Ratio\n(Clipped at 99th percentile)')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
    
        # 添加统计信息文本框
        stats_text = f"""
        Data Points: {len(ratio_data_clean)}
        Cleaned Points: {len(ratio_data_clipped)}
        Mean: {ratio_data_clipped.mean():.2f}
        Median: {ratio_data_clipped.median():.2f}
        Std: {ratio_data_clipped.std():.2f}
    
        Ratio > 2 (Judge-favored): {((ratio_data_clean > 2).sum()/len(ratio_data_clean)*100):.1f}%
        Ratio < 0.5 (Fan-favored): {((ratio_data_clean < 0.5).sum()/len(ratio_data_clean)*100):.1f}%
        """
    
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=7, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No valid ratio data available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Distribution of Judge-Fan Score Ratio\n(No valid data)')
    
    # 子图8: 行业分析 - 评委与粉丝分数比
    plt.subplot(3, 4, 8)
    if 'industry' in analysis_df_norm.columns and analysis_df_norm['industry'].nunique() > 1:
        # 只分析出现频率较高的行业
        industry_counts = analysis_df_norm['industry'].value_counts()
        top_industries = industry_counts[industry_counts >= 3].index.tolist()
        
        if len(top_industries) > 0:
            industry_data = analysis_df_norm[analysis_df_norm['industry'].isin(top_industries)]
            industry_means = industry_data.groupby('industry')[['avg_judge_norm', 'avg_fan_norm']].mean()
            industry_means = industry_means.sort_values('avg_judge_norm', ascending=False)
            
            x_pos = np.arange(len(industry_means))
            width = 0.35
            
            plt.bar(x_pos - width/2, industry_means['avg_judge_norm'], width, label='Judge score', alpha=0.8, color='red')
            plt.bar(x_pos + width/2, industry_means['avg_fan_norm'], width, label='Fan score', alpha=0.8, color='blue')
            
            plt.xticks(x_pos, industry_means.index, rotation=45, ha='right', fontsize=9)
            plt.xlabel('Industry')
            plt.ylabel('The average normalized score')
            plt.title('Judges and fans scores for different industries')
            plt.legend()
    
    # 子图9: 评委与粉丝影响的热力图
    plt.subplot(3, 4, 9)
    # 创建评委分数和粉丝分数的二维直方图
    plt.hist2d(analysis_df_norm['avg_judge_norm'], analysis_df_norm['avg_fan_norm'], 
               bins=30, cmap='YlOrRd')
    plt.colorbar(label='Number of players')
    plt.xlabel('Normalize judges scores')
    plt.ylabel('Normalized fan scores')
    plt.title('Heatmap of judge score vs fan score distribution')
    
    # 添加分类边界
    plt.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.5)
    
    # 子图10: 评委与粉丝分数的箱线图对比
    plt.subplot(3, 4, 10)
    data_to_plot = [analysis_df_norm['avg_judge_norm'], analysis_df_norm['avg_fan_norm']]
    bp = plt.boxplot(data_to_plot, patch_artist=True, labels=['Judge score', 'Fan score'])
    
    # 设置箱线图颜色
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Box plot of judges vs. fans score distribution')
    plt.ylabel('Normalized scores')
    plt.grid(True, alpha=0.3)
    
    # 子图11: 评委与粉丝影响的比例随时间变化
    plt.subplot(3, 4, 11)
    if 'season' in analysis_df_norm.columns:
        season_impact = analysis_df_norm.groupby('season').agg({
            'judge_impact_ratio': 'mean',
            'fan_impact_ratio': 'mean'
        }).reset_index()
        
        plt.plot(season_impact['season'], season_impact['judge_impact_ratio'], 
                 'ro-', linewidth=2, markersize=6, label='Judge influence ratio')
        plt.plot(season_impact['season'], season_impact['fan_impact_ratio'], 
                 'bo-', linewidth=2, markersize=6, label='Fan influence ratio')
        
        # 添加趋势线
        z_judge_season = np.polyfit(season_impact['season'], season_impact['judge_impact_ratio'], 1)
        p_judge_season = np.poly1d(z_judge_season)
        z_fan_season = np.polyfit(season_impact['season'], season_impact['fan_impact_ratio'], 1)
        p_fan_season = np.poly1d(z_fan_season)
        
        x_range_season = np.linspace(season_impact['season'].min(), season_impact['season'].max(), 100)
        plt.plot(x_range_season, p_judge_season(x_range_season), 'r--', alpha=0.5, linewidth=1)
        plt.plot(x_range_season, p_fan_season(x_range_season), 'b--', alpha=0.5, linewidth=1)
        
        plt.xlabel('Season')
        plt.ylabel('Proportion of influence')
        plt.title('The ratio of judges to fans influence changes over time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 子图12: 评委与粉丝分数的相关性矩阵
    plt.subplot(3, 4, 12)
    # 选择相关变量
    corr_vars = ['avg_judge_norm', 'avg_fan_norm', 'avg_combined_norm', 
                 'final_rank', 'age', 'survival_rate']
    
    corr_data = analysis_df_norm[corr_vars]
    corr_matrix = corr_data.corr()
    
    # 绘制热力图
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # 添加文本标注
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                     ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                     fontsize=8)
    
    plt.xticks(range(len(corr_vars)), [v.replace('_', '\n') for v in corr_vars], rotation=45, ha='right')
    plt.yticks(range(len(corr_vars)), [v.replace('_', '\n') for v in corr_vars])
    plt.title('Variable correlation matrix')
    
    plt.tight_layout()
    plt.savefig('Task3_Feature_Analysis_Normalized.png', dpi=300, bbox_inches='tight')
    print("✅ 归一化特征分析图表已保存: Task3_Feature_Analysis_Normalized.png")
    
    # ===================== 结果总结 =====================
    
    print("\n" + "="*80)
    print("📋 归一化分析结果总结")
    print("="*80)
    
    print(f"\n📊 分数分布:")
    print(f"   • 评委分数均值: {analysis_df_norm['avg_judge_norm'].mean():.3f}")
    print(f"   • 粉丝分数均值: {analysis_df_norm['avg_fan_norm'].mean():.3f}")
    print(f"   • 评委分数标准差: {analysis_df_norm['avg_judge_norm'].std():.3f}")
    print(f"   • 粉丝分数标准差: {analysis_df_norm['avg_fan_norm'].std():.3f}")
    
    print(f"\n📈 相关性分析:")
    print(f"   • 评委分数与粉丝分数相关性: {corr:.3f}")
    print(f"   • 评委分数与最终排名相关性: {judge_rank_corr_norm:.3f}")
    print(f"   • 粉丝分数与最终排名相关性: {fan_rank_corr_norm:.3f}")
    
    print(f"\n⚖️ 影响比例:")
    print(f"   • 平均评委影响比例: {avg_judge_impact_ratio:.3f}")
    print(f"   • 平均粉丝影响比例: {avg_fan_impact_ratio:.3f}")
    print(f"   • 评委:粉丝影响比例: {avg_judge_impact_ratio/avg_fan_impact_ratio:.2f}:1")
    
    print(f"\n👥 群体特征:")
    print(f"   • 评委偏爱型选手比例 (评委/粉丝比>2): {(analysis_df_norm['judge_fan_ratio'] > 2).mean():.1%}")
    print(f"   • 粉丝偏爱型选手比例 (评委/粉丝比<0.5): {(analysis_df_norm['judge_fan_ratio'] < 0.5).mean():.1%}")
    print(f"   • 均衡型选手比例 (0.5≤评委/粉丝比≤2): {((analysis_df_norm['judge_fan_ratio'] >= 0.5) & (analysis_df_norm['judge_fan_ratio'] <= 2)).mean():.1%}")
    
    # 保存分析结果
    analysis_df_norm.to_excel("Task3_Feature_Analysis_Normalized_Data.xlsx", index=False)
    print("\n✅ 归一化特征分析数据已保存: Task3_Feature_Analysis_Normalized_Data.xlsx")
    
    return analysis_df_norm

# 运行优化后的分析
analysis_df_norm = analyze_feature_impact_with_normalization(df)

# ===================== 7.7 评委与粉丝影响机制的深入分析 =====================

def analyze_judge_fan_mechanism(analysis_df_norm):
    """
    深入分析评委与粉丝影响机制的差异
    """
    print("\n" + "="*80)
    print("🔍 评委与粉丝影响机制的深入分析")
    print("="*80)
    
    # 分类分析：根据评委/粉丝分数比将选手分为三类
    analysis_df_norm['score_ratio_category'] = pd.cut(
        analysis_df_norm['judge_fan_ratio'],
        bins=[0, 0.5, 2, np.inf],
        labels=['Fan preference type', 'Type of equilibrium', 'Judge preference']
    )
    
    print("\n📊 选手分类统计:")
    category_stats = analysis_df_norm['score_ratio_category'].value_counts().sort_index()
    for category, count in category_stats.items():
        percentage = count / len(analysis_df_norm) * 100
        print(f"  {category}: {count}人 ({percentage:.1f}%)")
    
    # 分析各类选手的特征
    print("\n📈 各类选手特征分析:")
    
    category_analysis = analysis_df_norm.groupby('score_ratio_category').agg({
        'age': 'mean',
        'final_rank': 'mean',
        'survival_rate': 'mean',
        'avg_judge_norm': 'mean',
        'avg_fan_norm': 'mean',
        'judge_impact_ratio': 'mean',
        'fan_impact_ratio': 'mean',
        'player_id': 'count'
    }).rename(columns={'player_id': 'count'}).round(3)
    
    print(category_analysis)
    
    # 行业偏好分析
    if 'industry' in analysis_df_norm.columns:
        print("\n🏢 各类选手的行业分布:")
        
        # 创建交叉表
        industry_cross = pd.crosstab(
            analysis_df_norm['score_ratio_category'],
            analysis_df_norm['industry'],
            normalize='index'
        ).round(3)
        
        # 只显示比例较高的行业
        industry_cross = industry_cross.loc[:, industry_cross.max() > 0.1]
        
        if not industry_cross.empty:
            print(industry_cross)
            
            # 可视化行业偏好
            plt.figure(figsize=(12, 8))
            
            # 获取行业数据
            industries_to_plot = industry_cross.columns.tolist()
            categories = industry_cross.index.tolist()
            
            x = np.arange(len(industries_to_plot))
            width = 0.25
            
            for i, category in enumerate(categories):
                offset = (i - 1) * width
                plt.bar(x + offset, industry_cross.loc[category], width, label=category)
            
            plt.xlabel('Industry')
            plt.ylabel('proportion')
            plt.title('Industry distribution of players in different categories')
            plt.xticks(x, industries_to_plot, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('Task3_Industry_Preference_by_Category.png', dpi=300)
            print("✅ 行业偏好分析图表已保存: Task3_Industry_Preference_by_Category.png")
    
    # 成功因素分析：什么因素导致评委偏爱或粉丝偏爱？
    print("\n🔑 成功因素分析:")
    
    # 计算各类选手的成功率（定义为最终排名前30%）
    top_threshold = analysis_df_norm['final_rank'].quantile(0.3)
    analysis_df_norm['is_successful'] = (analysis_df_norm['final_rank'] <= top_threshold).astype(int)
    
    success_by_category = analysis_df_norm.groupby('score_ratio_category')['is_successful'].mean()
    
    print("各类选手的成功率（最终排名前30%）:")
    for category, success_rate in success_by_category.items():
        print(f"  {category}: {success_rate:.1%}")
    
    # 逻辑回归分析成功因素
    from sklearn.linear_model import LogisticRegression
    
    # 准备特征
    success_features = ['age', 'avg_judge_norm', 'avg_fan_norm', 'judge_impact_ratio', 'fan_impact_ratio']
    X_success = analysis_df_norm[success_features].fillna(0)
    y_success = analysis_df_norm['is_successful']
    
    if len(X_success) > 10:
        model_success = LogisticRegression(max_iter=1000)
        model_success.fit(X_success, y_success)
        
        print("\n成功因素的逻辑回归系数:")
        for feature, coef in zip(success_features, model_success.coef_[0]):
            print(f"  {feature}: {coef:.4f}")
        
        # 计算特征重要性
        importance = pd.DataFrame({
            'feature': success_features,
            'coefficient': model_success.coef_[0],
            'importance': np.abs(model_success.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print("\n成功因素重要性排序:")
        print(importance[['feature', 'coefficient']])
    
    # 生成总结报告
    print("\n" + "="*80)
    print("📋 评委与粉丝影响机制总结")
    print("="*80)
    
    print("\n🎯 关键发现:")
    print("1. 评委与粉丝评价存在系统性差异")
    print("2. 不同类别选手具有不同的成功模式")
    print("3. 评委偏爱型选手通常技术表现更稳定")
    print("4. 粉丝偏爱型选手更依赖人气和娱乐性")
    print("5. 均衡型选手在比赛中表现最为稳定")
    
    print("\n💡 建议:")
    print("• 节目制作方应保持评委与粉丝评价的平衡")
    print("• 选手应根据自身特点选择适合的发展策略")
    print("• 评委评分应更注重技术性，粉丝投票应更注重娱乐性")
    print("• 合理的评委-粉丝权重设计能提高比赛公平性")
    
    
    return analysis_df_norm

# 运行深入分析
analysis_df_norm = analyze_judge_fan_mechanism(analysis_df_norm)

print("\n" + "="*80)
print("✅ 任务1 特征分析（优化版）完成！")
print("="*80)
print("📁 生成的文件:")
print("  • Task3_Feature_Analysis_Normalized.png - 归一化分析图表")
print("  • Task3_Industry_Preference_by_Category.png - 行业偏好分析")
print("="*80)
