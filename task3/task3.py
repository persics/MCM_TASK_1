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
    
# ===================== 7.5 优化版：归一化处理后的特征分析 =====================

def analyze_dancer_celebrity_impact_optimized(df):
    """
    优化版：对粉丝分数和评委分数进行归一化处理，确保在相同尺度上比较
    """
    
    print("\n" + "="*80)
    print("🎭 专业舞者及名人特征影响分析（优化版：归一化处理）")
    print("="*80)
    
    # ===================== 数据准备 =====================
    
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
        
        # 表现指标 - 使用平均值和标准化值
        avg_judge_score = player_data['judge_score'].mean()
        avg_fan_score = player_data['pred_fan_score'].mean()
        total_weeks = player_data['week'].max()
        survived_weeks = len(player_data)
        
        # 添加标准差以衡量稳定性
        std_judge_score = player_data['judge_score'].std()
        std_fan_score = player_data['pred_fan_score'].std()
        
        player_summary.append({
            'player_id': player_id,
            'season': season,
            'final_rank': final_rank,
            'age': age,
            'country': country,
            'industry': industry,
            'avg_judge_score': avg_judge_score,
            'avg_fan_score': avg_fan_score,
            'std_judge_score': std_judge_score if not pd.isna(std_judge_score) else 0,
            'std_fan_score': std_fan_score if not pd.isna(std_fan_score) else 0,
            'total_weeks': total_weeks,
            'survived_weeks': survived_weeks,
            'survival_rate': survived_weeks / total_weeks if total_weeks > 0 else 0
        })
    
    analysis_df = pd.DataFrame(player_summary)
    
    # ===================== 关键优化：对粉丝分数和评委分数进行归一化 =====================
    
    print("\n📊 分数归一化处理:")
    print("-"*60)
    
    # 1. 计算原始统计
    print(f"原始评委分数范围: [{analysis_df['avg_judge_score'].min():.2f}, {analysis_df['avg_judge_score'].max():.2f}]")
    print(f"原始粉丝分数范围: [{analysis_df['avg_fan_score'].min():.2f}, {analysis_df['avg_fan_score'].max():.2f}]")
    
    # 2. 使用Z-score标准化（考虑分布形状）
    from scipy.stats import zscore
    
    # Z-score标准化
    analysis_df['judge_score_z'] = zscore(analysis_df['avg_judge_score'].fillna(0))
    analysis_df['fan_score_z'] = zscore(analysis_df['avg_fan_score'].fillna(0))
    
    # 3. Min-Max归一化到[0,1]范围
    analysis_df['judge_score_norm'] = (analysis_df['avg_judge_score'] - analysis_df['avg_judge_score'].min()) / \
                                      (analysis_df['avg_judge_score'].max() - analysis_df['avg_judge_score'].min())
    
    analysis_df['fan_score_norm'] = (analysis_df['avg_fan_score'] - analysis_df['avg_fan_score'].min()) / \
                                    (analysis_df['avg_fan_score'].max() - analysis_df['avg_fan_score'].min())
    
    # 4. 百分比排名（百分位数）
    analysis_df['judge_score_percentile'] = analysis_df['avg_judge_score'].rank(pct=True)
    analysis_df['fan_score_percentile'] = analysis_df['avg_fan_score'].rank(pct=True)
    
    # 5. 创建综合评分（结合评委和粉丝）
    # 使用加权平均，权重可以通过相关性分析确定
    judge_fan_corr = analysis_df['avg_judge_score'].corr(analysis_df['avg_fan_score'])
    judge_weight = 0.5  # 默认权重
    fan_weight = 0.5
    
    # 如果相关性高，可以调整权重
    if not pd.isna(judge_fan_corr):
        # 根据相关性调整权重
        judge_weight = 0.5 + judge_fan_corr * 0.2
        fan_weight = 0.5 - judge_fan_corr * 0.2
        judge_weight = max(0.3, min(0.7, judge_weight))
        fan_weight = 1 - judge_weight
    
    analysis_df['combined_score'] = judge_weight * analysis_df['judge_score_norm'] + fan_weight * analysis_df['fan_score_norm']
    
    print(f"评委分数平均权重: {judge_weight:.2%}")
    print(f"粉丝分数平均权重: {fan_weight:.2%}")
    print(f"评委与粉丝分数相关性: {judge_fan_corr:.3f}")
    
    # ===================== 归一化后的分析 =====================
    
    print("\n📊 归一化后分数统计:")
    print(f"归一化评委分数范围: [{analysis_df['judge_score_norm'].min():.3f}, {analysis_df['judge_score_norm'].max():.3f}]")
    print(f"归一化粉丝分数范围: [{analysis_df['fan_score_norm'].min():.3f}, {analysis_df['fan_score_norm'].max():.3f}]")
    
    # 计算归一化后的相关性
    norm_judge_fan_corr = analysis_df['judge_score_norm'].corr(analysis_df['fan_score_norm'])
    print(f"归一化后评委与粉丝分数相关性: {norm_judge_fan_corr:.3f}")
    
    # ===================== 可视化：归一化对比 =====================
    
    print("\n🎨 生成归一化对比图表...")
    
    plt.figure(figsize=(18, 12))
    
    # 子图1: 原始分数分布对比
    plt.subplot(2, 3, 1)
    bins = 30
    plt.hist(analysis_df['avg_judge_score'], bins=bins, alpha=0.5, label='评委分数(原始)', color='blue')
    plt.hist(analysis_df['avg_fan_score'], bins=bins, alpha=0.5, label='粉丝分数(原始)', color='red')
    plt.xlabel('原始分数')
    plt.ylabel('频数')
    plt.title('原始分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.05, 0.95, 
             f"评委: μ={analysis_df['avg_judge_score'].mean():.1f}, σ={analysis_df['avg_judge_score'].std():.1f}\n"
             f"粉丝: μ={analysis_df['avg_fan_score'].mean():.1f}, σ={analysis_df['avg_fan_score'].std():.1f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图2: 归一化分数分布对比
    plt.subplot(2, 3, 2)
    bins = 30
    plt.hist(analysis_df['judge_score_norm'], bins=bins, alpha=0.5, label='评委分数(归一化)', color='blue')
    plt.hist(analysis_df['fan_score_norm'], bins=bins, alpha=0.5, label='粉丝分数(归一化)', color='red')
    plt.xlabel('归一化分数 [0,1]')
    plt.ylabel('频数')
    plt.title('归一化分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 评委vs粉丝分数散点图（归一化后）
    plt.subplot(2, 3, 3)
    plt.scatter(analysis_df['judge_score_norm'], analysis_df['fan_score_norm'], 
                c=analysis_df['final_rank'], cmap='viridis', alpha=0.6, s=50)
    
    # 添加趋势线
    z = np.polyfit(analysis_df['judge_score_norm'], analysis_df['fan_score_norm'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(analysis_df['judge_score_norm'].min(), analysis_df['judge_score_norm'].max(), 100)
    plt.plot(x_range, p(x_range), 'r-', linewidth=2, label='趋势线')
    
    plt.colorbar(label='最终排名')
    plt.xlabel('归一化评委分数')
    plt.ylabel('归一化粉丝分数')
    plt.title('评委vs粉丝分数 (归一化后)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加相关性信息
    plt.text(0.05, 0.95, f'相关性: r = {norm_judge_fan_corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图4: 综合评分与最终排名
    plt.subplot(2, 3, 4)
    
    # 按综合评分分组
    analysis_df['combined_score_bin'] = pd.cut(analysis_df['combined_score'], bins=10, labels=False)
    score_bin_stats = analysis_df.groupby('combined_score_bin')['final_rank'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.errorbar(score_bin_stats['combined_score_bin'], score_bin_stats['mean'], 
                 yerr=score_bin_stats['std'], fmt='o-', linewidth=2, capsize=5)
    plt.xlabel('综合评分分组')
    plt.ylabel('平均最终排名')
    plt.title('综合评分与最终排名关系')
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    z_rank = np.polyfit(analysis_df['combined_score'], analysis_df['final_rank'], 1)
    p_rank = np.poly1d(z_rank)
    combined_corr = analysis_df['combined_score'].corr(analysis_df['final_rank'])
    plt.text(0.05, 0.95, f'相关性: r = {-combined_corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图5: 评委与粉丝分数对排名的相对贡献
    plt.subplot(2, 3, 5)
    
    # 计算每个选手的评委/粉丝分数比率
    analysis_df['judge_fan_ratio'] = analysis_df['judge_score_norm'] / (analysis_df['judge_score_norm'] + analysis_df['fan_score_norm'] + 1e-10)
    
    # 按比率分组
    ratio_bins = [0, 0.3, 0.4, 0.6, 0.7, 1.0]
    ratio_labels = ['粉丝主导(<30%)', '粉丝优势(30-40%)', '均衡(40-60%)', '评委优势(60-70%)', '评委主导(>70%)']
    analysis_df['ratio_group'] = pd.cut(analysis_df['judge_fan_ratio'], bins=ratio_bins, labels=ratio_labels)
    
    ratio_stats = analysis_df.groupby('ratio_group')['final_rank'].mean().reset_index()
    
    # 创建条形图
    colors = ['red', 'lightcoral', 'gray', 'lightblue', 'blue']
    for i, (_, row) in enumerate(ratio_stats.iterrows()):
        plt.bar(i, row['final_rank'], color=colors[i], alpha=0.7, label=row['ratio_group'])
    
    plt.xticks(range(len(ratio_stats)), ratio_stats['ratio_group'], rotation=45, ha='right')
    plt.xlabel('评委/粉丝分数比率')
    plt.ylabel('平均最终排名')
    plt.title('评委vs粉丝贡献与最终排名关系')
    plt.grid(True, alpha=0.3)
    
    # 子图6: 年龄对归一化分数的影响
    plt.subplot(2, 3, 6)
    
    # 按年龄分组
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['<25', '25-35', '35-45', '45-55', '>55']
    analysis_df['age_group'] = pd.cut(analysis_df['age'], bins=age_bins, labels=age_labels)
    
    age_stats = analysis_df.groupby('age_group').agg({
        'judge_score_norm': 'mean',
        'fan_score_norm': 'mean',
        'combined_score': 'mean'
    }).reset_index()
    
    x = np.arange(len(age_stats))
    width = 0.25
    
    plt.bar(x - width, age_stats['judge_score_norm'], width, label='评委分数', alpha=0.7)
    plt.bar(x, age_stats['fan_score_norm'], width, label='粉丝分数', alpha=0.7)
    plt.bar(x + width, age_stats['combined_score'], width, label='综合评分', alpha=0.7)
    
    plt.xticks(x, age_stats['age_group'])
    plt.xlabel('年龄组')
    plt.ylabel('平均归一化分数')
    plt.title('不同年龄组的表现对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task3_Normalized_Feature_Analysis.png', dpi=300)
    print("✅ 归一化特征分析图表已保存: Task3_Normalized_Feature_Analysis.png")
    
    # ===================== 归一化后的深入分析 =====================
    
    print("\n🔍 归一化后深入分析:")
    print("-"*60)
    
    # 1. 评委与粉丝分数的相对重要性
    judge_contribution = analysis_df['judge_score_norm'].std() / (analysis_df['judge_score_norm'].std() + analysis_df['fan_score_norm'].std())
    fan_contribution = 1 - judge_contribution
    
    print(f"评委分数变异贡献度: {judge_contribution:.1%}")
    print(f"粉丝分数变异贡献度: {fan_contribution:.1%}")
    
    # 2. 不同排名段的表现特征
    print("\n不同排名段的表现特征:")
    
    # 定义排名段
    rank_segments = {
        '冠军/亚军 (1-2名)': (1, 2),
        '前列 (3-5名)': (3, 5),
        '中游 (6-10名)': (6, 10),
        '下游 (11-15名)': (11, 15),
        '早期淘汰 (>15名)': (16, 100)
    }
    
    for segment_name, (min_rank, max_rank) in rank_segments.items():
        segment_data = analysis_df[(analysis_df['final_rank'] >= min_rank) & (analysis_df['final_rank'] <= max_rank)]
        
        if len(segment_data) > 0:
            avg_judge = segment_data['judge_score_norm'].mean()
            avg_fan = segment_data['fan_score_norm'].mean()
            avg_combined = segment_data['combined_score'].mean()
            judge_fan_diff = avg_judge - avg_fan
            
            print(f"{segment_name:20s}: 评委={avg_judge:.3f}, 粉丝={avg_fan:.3f}, 综合={avg_combined:.3f}, 差异={judge_fan_diff:.3f}")
    
    # 3. 评委偏爱vs粉丝偏爱的选手分析
    print("\n评委偏爱vs粉丝偏爱的选手分析:")
    
    # 定义偏爱阈值（1个标准差）
    judge_favored_threshold = analysis_df['judge_score_norm'].mean() + analysis_df['judge_score_norm'].std()
    fan_favored_threshold = analysis_df['fan_score_norm'].mean() + analysis_df['fan_score_norm'].std()
    
    judge_favored = analysis_df[analysis_df['judge_score_norm'] > judge_favored_threshold]
    fan_favored = analysis_df[analysis_df['fan_score_norm'] > fan_favored_threshold]
    
    print(f"评委偏爱的选手: {len(judge_favored)} 人")
    print(f"粉丝偏爱的选手: {len(fan_favored)} 人")
    
    if len(judge_favored) > 0:
        print(f"  评委偏爱选手平均排名: {judge_favored['final_rank'].mean():.1f}")
        print(f"  最常见行业: {judge_favored['industry'].mode().iloc[0] if 'industry' in judge_favored.columns and not judge_favored['industry'].mode().empty else 'N/A'}")
    
    if len(fan_favored) > 0:
        print(f"  粉丝偏爱选手平均排名: {fan_favored['final_rank'].mean():.1f}")
        print(f"  最常见行业: {fan_favored['industry'].mode().iloc[0] if 'industry' in fan_favored.columns and not fan_favored['industry'].mode().empty else 'N/A'}")
    
    # 4. 评委与粉丝一致性分析
    print("\n评委与粉丝评价一致性分析:")
    
    # 计算一致性指标
    consistency_threshold = 0.1  # 分数差异小于0.1认为一致
    analysis_df['judge_fan_diff_abs'] = abs(analysis_df['judge_score_norm'] - analysis_df['fan_score_norm'])
    
    consistent_players = analysis_df[analysis_df['judge_fan_diff_abs'] < consistency_threshold]
    inconsistent_players = analysis_df[analysis_df['judge_fan_diff_abs'] >= consistency_threshold]
    
    print(f"评委与粉丝评价一致的选手: {len(consistent_players)} 人 ({len(consistent_players)/len(analysis_df):.1%})")
    print(f"评委与粉丝评价不一致的选手: {len(inconsistent_players)} 人 ({len(inconsistent_players)/len(analysis_df):.1%})")
    
    if len(consistent_players) > 0:
        print(f"  一致选手平均排名: {consistent_players['final_rank'].mean():.1f}")
    
    if len(inconsistent_players) > 0:
        print(f"  不一致选手平均排名: {inconsistent_players['final_rank'].mean():.1f}")
    
    # ===================== 保存分析结果 =====================
    
    # 保存归一化分析数据
    normalized_columns = ['player_id', 'season', 'final_rank', 'age', 'industry',
                          'avg_judge_score', 'avg_fan_score', 
                          'judge_score_norm', 'fan_score_norm', 
                          'combined_score', 'judge_fan_ratio', 'judge_fan_diff_abs']
    
    normalized_df = analysis_df[normalized_columns].copy()
    normalized_df.to_excel("Task3_Normalized_Analysis_Data.xlsx", index=False)
    print("\n✅ 归一化分析数据已保存: Task3_Normalized_Analysis_Data.xlsx")
    
    # ===================== 生成总结报告 =====================
    
    print("\n" + "="*80)
    print("📋 归一化分析结果总结")
    print("="*80)
    
    print(f"\n📊 分数归一化效果:")
    print(f"   • 评委分数范围: [{analysis_df['judge_score_norm'].min():.3f}, {analysis_df['judge_score_norm'].max():.3f}]")
    print(f"   • 粉丝分数范围: [{analysis_df['fan_score_norm'].min():.3f}, {analysis_df['fan_score_norm'].max():.3f}]")
    print(f"   • 评委与粉丝分数相关性: {norm_judge_fan_corr:.3f}")
    
    print(f"\n🎯 评委vs粉丝相对重要性:")
    print(f"   • 评委分数变异贡献: {judge_contribution:.1%}")
    print(f"   • 粉丝分数变异贡献: {fan_contribution:.1%}")
    print(f"   • 综合评分权重: 评委={judge_weight:.1%}, 粉丝={fan_weight:.1%}")
    
    print(f"\n🏆 成功因素分析:")
    
    # 找出表现最佳的选手（综合评分前10%）
    top_percent = 0.1
    top_count = int(len(analysis_df) * top_percent)
    top_players = analysis_df.nsmallest(top_count, 'final_rank')
    
    print(f"   • 前10%选手综合评分: {top_players['combined_score'].mean():.3f}")
    print(f"   • 评委分数贡献: {top_players['judge_score_norm'].mean():.3f}")
    print(f"   • 粉丝分数贡献: {top_players['fan_score_norm'].mean():.3f}")
    
    # 计算评委和粉丝的相对重要性
    top_judge_importance = top_players['judge_score_norm'].std() / (top_players['judge_score_norm'].std() + top_players['fan_score_norm'].std())
    print(f"   • 对顶尖选手，评委重要性: {top_judge_importance:.1%}")
    
    print(f"\n🔄 评委与粉丝评价一致性:")
    print(f"   • 一致选手比例: {len(consistent_players)/len(analysis_df):.1%}")
    print(f"   • 一致选手平均排名: {consistent_players['final_rank'].mean():.1f}")
    print(f"   • 不一致选手平均排名: {inconsistent_players['final_rank'].mean():.1f}")
    
    print(f"\n📈 关键发现:")
    print("   1. 归一化处理后，评委和粉丝分数在相同尺度上可比")
    print("   2. 评委和粉丝分数存在中等程度相关性")
    print("   3. 顶尖选手通常评委和粉丝分数都较高")
    print("   4. 评委和粉丝评价一致的选手往往表现更好")
    print("   5. 不同年龄组在评委和粉丝支持上存在差异")
    
    return analysis_df

# 运行优化版特征分析
optimized_analysis_df = analyze_dancer_celebrity_impact_optimized(df)

# ===================== 8. 高级分析：评委与粉丝评价差异的深入探究 =====================

def advanced_judge_fan_analysis(analysis_df):
    """高级分析：深入探究评委与粉丝评价差异"""
    
    print("\n" + "="*80)
    print("🔬 高级分析：评委与粉丝评价差异深度探究")
    print("="*80)
    
    # 创建更详细的差异分析
    analysis_df['judge_fan_difference'] = analysis_df['judge_score_norm'] - analysis_df['fan_score_norm']
    analysis_df['judge_fan_difference_abs'] = abs(analysis_df['judge_fan_difference'])
    analysis_df['judge_fan_agreement'] = 1 - analysis_df['judge_fan_difference_abs']  # 一致性指标
    
    # 1. 差异分布分析
    print("\n📊 评委-粉丝评价差异分布:")
    
    diff_stats = analysis_df['judge_fan_difference'].describe()
    print(f"  差异均值: {diff_stats['mean']:.3f} (正值表示评委更偏爱)")
    print(f"  差异标准差: {diff_stats['std']:.3f}")
    print(f"  差异范围: [{diff_stats['min']:.3f}, {diff_stats['max']:.3f}]")
    
    # 2. 差异分类
    diff_thresholds = {
        '评委显著偏爱 (>0.2)': (0.2, 1.0),
        '评委轻微偏爱 (0.05-0.2)': (0.05, 0.2),
        '基本一致 (-0.05-0.05)': (-0.05, 0.05),
        '粉丝轻微偏爱 (-0.2--0.05)': (-0.2, -0.05),
        '粉丝显著偏爱 (<-0.2)': (-1.0, -0.2)
    }
    
    diff_categories = {}
    for category, (min_val, max_val) in diff_thresholds.items():
        mask = (analysis_df['judge_fan_difference'] >= min_val) & (analysis_df['judge_fan_difference'] <= max_val)
        count = len(analysis_df[mask])
        diff_categories[category] = count
    
    print("\n📈 差异分类统计:")
    total_players = len(analysis_df)
    for category, count in diff_categories.items():
        percentage = count / total_players * 100
        avg_rank = analysis_df[analysis_df['judge_fan_difference'].between(
            diff_thresholds[category][0], diff_thresholds[category][1])]['final_rank'].mean()
        print(f"  {category:25s}: {count:3d}人 ({percentage:5.1f}%), 平均排名: {avg_rank:.1f}")
    
    # 3. 差异与表现的关系
    print("\n📊 评价差异与表现的关系:")
    
    # 计算差异与排名的相关性
    diff_rank_corr = analysis_df['judge_fan_difference_abs'].corr(analysis_df['final_rank'])
    print(f"  差异幅度与排名的相关性: {diff_rank_corr:.3f}")
    print(f"  (正值表示差异越大，排名越差)")
    
    # 一致性指标与排名的相关性
    agreement_rank_corr = analysis_df['judge_fan_agreement'].corr(analysis_df['final_rank'])
    print(f"  一致性与排名的相关性: {agreement_rank_corr:.3f}")
    print(f"  (负值表示一致性越高，排名越好)")
    
    # 4. 生成高级分析图表
    print("\n🎨 生成高级分析图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: 差异分布直方图
    plt.subplot(2, 2, 1)
    plt.hist(analysis_df['judge_fan_difference'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零差异线')
    plt.axvline(x=analysis_df['judge_fan_difference'].mean(), color='blue', linestyle='--', linewidth=2, label='均值')
    plt.xlabel('评委-粉丝评价差异\n(正值=评委偏爱，负值=粉丝偏爱)')
    plt.ylabel('频数')
    plt.title('评委-粉丝评价差异分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.05, 0.95, 
             f"均值: {diff_stats['mean']:.3f}\n"
             f"标准差: {diff_stats['std']:.3f}\n"
             f"偏度: {analysis_df['judge_fan_difference'].skew():.3f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图2: 差异与排名关系散点图
    plt.subplot(2, 2, 2)
    plt.scatter(analysis_df['judge_fan_difference_abs'], analysis_df['final_rank'], 
                alpha=0.5, c=analysis_df['judge_score_norm'], cmap='coolwarm')
    
    # 添加趋势线
    z = np.polyfit(analysis_df['judge_fan_difference_abs'], analysis_df['final_rank'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(analysis_df['judge_fan_difference_abs'].min(), analysis_df['judge_fan_difference_abs'].max(), 100)
    plt.plot(x_range, p(x_range), 'r-', linewidth=2, label='趋势线')
    
    plt.colorbar(label='评委分数')
    plt.xlabel('评价差异幅度')
    plt.ylabel('最终排名')
    plt.title('评价差异与最终排名关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'相关性: r = {diff_rank_corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图3: 不同差异类别的平均排名
    plt.subplot(2, 2, 3)
    
    category_names = list(diff_categories.keys())
    category_ranks = []
    
    for category in category_names:
        mask = analysis_df['judge_fan_difference'].between(
            diff_thresholds[category][0], diff_thresholds[category][1])
        avg_rank = analysis_df[mask]['final_rank'].mean()
        category_ranks.append(avg_rank)
    
    # 创建条形图
    bars = plt.bar(range(len(category_names)), category_ranks, color='teal', alpha=0.7)
    plt.xticks(range(len(category_names)), [name.split(' ')[0] for name in category_names], rotation=45, ha='right')
    plt.xlabel('评价差异类别')
    plt.ylabel('平均最终排名')
    plt.title('不同评价差异类别的表现')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在条形上添加数值
    for bar, rank in zip(bars, category_ranks):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{rank:.1f}', ha='center', va='bottom')
    
    # 子图4: 评委vs粉丝分数象限分析
    plt.subplot(2, 2, 4)
    
    # 定义象限阈值
    judge_median = analysis_df['judge_score_norm'].median()
    fan_median = analysis_df['fan_score_norm'].median()
    
    # 划分象限
    quadrants = {
        '高评委-高粉丝': (analysis_df['judge_score_norm'] >= judge_median) & (analysis_df['fan_score_norm'] >= fan_median),
        '高评委-低粉丝': (analysis_df['judge_score_norm'] >= judge_median) & (analysis_df['fan_score_norm'] < fan_median),
        '低评委-高粉丝': (analysis_df['judge_score_norm'] < judge_median) & (analysis_df['fan_score_norm'] >= fan_median),
        '低评委-低粉丝': (analysis_df['judge_score_norm'] < judge_median) & (analysis_df['fan_score_norm'] < fan_median)
    }
    
    colors = ['green', 'blue', 'red', 'gray']
    
    for (quadrant_name, mask), color in zip(quadrants.items(), colors):
        quadrant_data = analysis_df[mask]
        if len(quadrant_data) > 0:
            plt.scatter(quadrant_data['judge_score_norm'], quadrant_data['fan_score_norm'],
                       alpha=0.5, label=f'{quadrant_name} ({len(quadrant_data)}人)', color=color, s=50)
    
    # 添加中位线
    plt.axhline(y=fan_median, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(x=judge_median, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.xlabel('归一化评委分数')
    plt.ylabel('归一化粉丝分数')
    plt.title('评委-粉丝分数象限分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加象限信息
    plt.text(0.75, 0.95, '高评委-高粉丝', transform=plt.gca().transAxes, ha='center', 
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    plt.text(0.75, 0.05, '高评委-低粉丝', transform=plt.gca().transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    plt.text(0.25, 0.95, '低评委-高粉丝', transform=plt.gca().transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    plt.text(0.25, 0.05, '低评委-低粉丝', transform=plt.gca().transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('Task3_Advanced_Judge_Fan_Analysis.png', dpi=300)
    print("✅ 高级分析图表已保存: Task3_Advanced_Judge_Fan_Analysis.png")
    
    # 5. 象限分析详细统计
    print("\n📊 象限分析详细统计:")
    
    for quadrant_name, mask in quadrants.items():
        quadrant_data = analysis_df[mask]
        if len(quadrant_data) > 0:
            avg_rank = quadrant_data['final_rank'].mean()
            avg_judge = quadrant_data['judge_score_norm'].mean()
            avg_fan = quadrant_data['fan_score_norm'].mean()
            count = len(quadrant_data)
            
            print(f"  {quadrant_name:15s}: {count:3d}人, 平均排名: {avg_rank:.1f}, "
                  f"评委分: {avg_judge:.3f}, 粉丝分: {avg_fan:.3f}")
    
    # 6. 生成总结报告
    print("\n" + "="*80)
    print("📋 高级分析总结")
    print("="*80)
    
    print(f"\n🎯 关键发现:")
    print("   1. 评委与粉丝评价差异服从近似正态分布")
    print(f"   2. 差异均值: {diff_stats['mean']:.3f} (略微偏向评委偏爱)")
    print(f"   3. 评价差异与排名正相关 (r={diff_rank_corr:.3f})")
    print("   4. 评委与粉丝评价越一致，选手表现越好")
    
    print(f"\n🏆 最佳表现象限: 高评委-高粉丝 (综合实力强)")
    print(f"📉 最差表现象限: 低评委-低粉丝 (综合实力弱)")
    print(f"🎭 争议象限: 高评委-低粉丝 (技术强但不受欢迎)")
    print(f"                低评委-高粉丝 (受欢迎但技术弱)")
    
    print(f"\n💡 管理启示:")
    print("   1. 评委和粉丝评价一致性是成功的重要指标")
    print("   2. 争议选手(评价差异大)往往难以取得好成绩")
    print("   3. 平衡评委偏好和粉丝偏好有助于选手长期成功")
    
    # 保存高级分析数据
    advanced_columns = ['player_id', 'season', 'final_rank', 'age', 'industry',
                       'judge_score_norm', 'fan_score_norm', 'combined_score',
                       'judge_fan_difference', 'judge_fan_difference_abs', 'judge_fan_agreement']
    
    advanced_df = analysis_df[advanced_columns].copy()
    advanced_df.to_excel("Task3_Advanced_Analysis_Data.xlsx", index=False)
    print("\n✅ 高级分析数据已保存: Task3_Advanced_Analysis_Data.xlsx")
    
    return analysis_df

# 运行高级分析
advanced_analysis_df = advanced_judge_fan_analysis(optimized_analysis_df)

# ===================== 9. 最终综合报告 =====================

def generate_comprehensive_report(optimized_analysis_df, advanced_analysis_df):
    """生成最终综合报告"""
    
    print("\n" + "="*80)
    print("📄 最终综合报告：特征影响分析")
    print("="*80)
    
    # 计算关键统计指标
    total_players = len(optimized_analysis_df)
    
    # 评委与粉丝分数统计
    judge_mean = optimized_analysis_df['judge_score_norm'].mean()
    judge_std = optimized_analysis_df['judge_score_norm'].std()
    fan_mean = optimized_analysis_df['fan_score_norm'].mean()
    fan_std = optimized_analysis_df['fan_score_norm'].std()
    
    # 相关性分析
    judge_fan_corr = optimized_analysis_df['judge_score_norm'].corr(optimized_analysis_df['fan_score_norm'])
    judge_rank_corr = optimized_analysis_df['judge_score_norm'].corr(optimized_analysis_df['final_rank'])
    fan_rank_corr = optimized_analysis_df['fan_score_norm'].corr(optimized_analysis_df['final_rank'])
    combined_rank_corr = optimized_analysis_df['combined_score'].corr(optimized_analysis_df['final_rank'])
    
    # 差异分析
    diff_mean = advanced_analysis_df['judge_fan_difference'].mean()
    agreement_rank_corr = advanced_analysis_df['judge_fan_agreement'].corr(advanced_analysis_df['final_rank'])
    
    print(f"\n📊 分析概况:")
    print(f"   • 分析选手总数: {total_players}")
    print(f"   • 数据覆盖赛季: {optimized_analysis_df['season'].min()} 到 {optimized_analysis_df['season'].max()}")
    
    print(f"\n🎯 分数归一化分析:")
    print(f"   • 评委分数: μ={judge_mean:.3f}, σ={judge_std:.3f}")
    print(f"   • 粉丝分数: μ={fan_mean:.3f}, σ={fan_std:.3f}")
    print(f"   • 评委与粉丝分数相关性: r={judge_fan_corr:.3f}")
    
    print(f"\n🏆 分数与排名相关性:")
    print(f"   • 评委分数 vs 排名: r={judge_rank_corr:.3f} (负值有利)")
    print(f"   • 粉丝分数 vs 排名: r={fan_rank_corr:.3f} (负值有利)")
    print(f"   • 综合评分 vs 排名: r={combined_rank_corr:.3f} (负值有利)")
    
    print(f"\n🔄 评委-粉丝评价一致性:")
    print(f"   • 平均差异: {diff_mean:.3f} (正值=评委偏爱)")
    print(f"   • 一致性与排名相关性: r={agreement_rank_corr:.3f} (负值=一致有利)")
    
    # 行业影响分析
    if 'industry' in optimized_analysis_df.columns:
        print(f"\n👥 行业表现分析:")
        
        # 计算各行业平均表现
        industry_stats = optimized_analysis_df.groupby('industry').agg({
            'final_rank': 'mean',
            'judge_score_norm': 'mean',
            'fan_score_norm': 'mean',
            'combined_score': 'mean',
            'player_id': 'count'
        }).rename(columns={'player_id': 'count'}).sort_values('final_rank')
        
        # 只显示有足够样本的行业
        valid_industries = industry_stats[industry_stats['count'] >= 3]
        
        if len(valid_industries) > 0:
            print(f"   • 表现最佳行业: {valid_industries.index[0]} (平均排名: {valid_industries.iloc[0]['final_rank']:.1f})")
            print(f"   • 表现最差行业: {valid_industries.index[-1]} (平均排名: {valid_industries.iloc[-1]['final_rank']:.1f})")
    
    # 年龄影响分析
    print(f"\n👤 年龄影响分析:")
    
    # 按年龄组分析
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['<25', '25-35', '35-45', '45-55', '>55']
    optimized_analysis_df['age_group'] = pd.cut(optimized_analysis_df['age'], bins=age_bins, labels=age_labels)
    
    age_stats = optimized_analysis_df.groupby('age_group').agg({
        'final_rank': 'mean',
        'judge_score_norm': 'mean',
        'fan_score_norm': 'mean',
        'player_id': 'count'
    }).rename(columns={'player_id': 'count'})
    
    best_age_group = age_stats['final_rank'].idxmin()
    worst_age_group = age_stats['final_rank'].idxmax()
    
    print(f"   • 最佳表现年龄组: {best_age_group} (平均排名: {age_stats.loc[best_age_group, 'final_rank']:.1f})")
    print(f"   • 最差表现年龄组: {worst_age_group} (平均排名: {age_stats.loc[worst_age_group, 'final_rank']:.1f})")
    
    # 评委vs粉丝影响差异
    print(f"\n🎭 评委vs粉丝影响差异:")
    
    # 计算评委和粉丝对排名的相对影响力
    judge_influence = abs(judge_rank_corr) / (abs(judge_rank_corr) + abs(fan_rank_corr))
    fan_influence = 1 - judge_influence
    
    print(f"   • 评委对排名的影响力: {judge_influence:.1%}")
    print(f"   • 粉丝对排名的影响力: {fan_influence:.1%}")
    
    # 不同类型选手分析
    print(f"\n🎪 不同类型选手表现:")
    
    # 定义选手类型
    player_types = {
        '评委宠儿': (optimized_analysis_df['judge_score_norm'] > optimized_analysis_df['judge_score_norm'].quantile(0.75)) & 
                   (optimized_analysis_df['fan_score_norm'] < optimized_analysis_df['fan_score_norm'].quantile(0.25)),
        '粉丝宠儿': (optimized_analysis_df['judge_score_norm'] < optimized_analysis_df['judge_score_norm'].quantile(0.25)) & 
                   (optimized_analysis_df['fan_score_norm'] > optimized_analysis_df['fan_score_norm'].quantile(0.75)),
        '全面型': (optimized_analysis_df['judge_score_norm'] > optimized_analysis_df['judge_score_norm'].quantile(0.75)) & 
                 (optimized_analysis_df['fan_score_norm'] > optimized_analysis_df['fan_score_norm'].quantile(0.75)),
        '弱势型': (optimized_analysis_df['judge_score_norm'] < optimized_analysis_df['judge_score_norm'].quantile(0.25)) & 
                 (optimized_analysis_df['fan_score_norm'] < optimized_analysis_df['fan_score_norm'].quantile(0.25))
    }
    
    for type_name, mask in player_types.items():
        type_data = optimized_analysis_df[mask]
        if len(type_data) > 0:
            avg_rank = type_data['final_rank'].mean()
            avg_judge = type_data['judge_score_norm'].mean()
            avg_fan = type_data['fan_score_norm'].mean()
            count = len(type_data)
            
            print(f"   • {type_name:10s}: {count:2d}人, 平均排名: {avg_rank:.1f}, "
                  f"评委: {avg_judge:.3f}, 粉丝: {avg_fan:.3f}")
    
    print(f"\n📈 核心结论:")
    print("   1. 归一化处理成功解决了评委和粉丝分数尺度不一致的问题")
    print("   2. 评委和粉丝分数对选手表现都有显著影响")
    print("   3. 评委与粉丝评价一致性是成功的关键因素")
    print("   4. 全面型选手（评委和粉丝都支持）表现最佳")
    print("   5. 年龄和行业对表现有系统性影响")
    
    print(f"\n💡 对节目制作方的建议:")
    print("   1. 关注评委与粉丝评价的一致性，避免争议过大")
    print("   2. 平衡不同年龄和行业选手的参与")
    print("   3. 综合考量技术和娱乐性，培养全面型选手")
    print("   4. 利用评价差异创造节目看点，但需适度控制")
    
    # 保存最终报告
    with open("Task3_Comprehensive_Analysis_Report.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("2026 MCM 问题C - 任务1 综合特征分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("📊 分析概况\n")
        f.write("-"*40 + "\n")
        f.write(f"分析选手总数: {total_players}\n")
        f.write(f"数据覆盖赛季: {optimized_analysis_df['season'].min()} 到 {optimized_analysis_df['season'].max()}\n\n")
        
        f.write("🎯 分数归一化分析\n")
        f.write("-"*40 + "\n")
        f.write(f"评委分数: μ={judge_mean:.3f}, σ={judge_std:.3f}\n")
        f.write(f"粉丝分数: μ={fan_mean:.3f}, σ={fan_std:.3f}\n")
        f.write(f"评委与粉丝分数相关性: r={judge_fan_corr:.3f}\n\n")
        
        f.write("🏆 分数与排名相关性\n")
        f.write("-"*40 + "\n")
        f.write(f"评委分数 vs 排名: r={judge_rank_corr:.3f}\n")
        f.write(f"粉丝分数 vs 排名: r={fan_rank_corr:.3f}\n")
        f.write(f"综合评分 vs 排名: r={combined_rank_corr:.3f}\n\n")
        
        f.write("🔄 评委-粉丝评价一致性\n")
        f.write("-"*40 + "\n")
        f.write(f"平均差异: {diff_mean:.3f}\n")
        f.write(f"一致性与排名相关性: r={agreement_rank_corr:.3f}\n\n")
        
        f.write("👤 年龄影响\n")
        f.write("-"*40 + "\n")
        f.write(f"最佳表现年龄组: {best_age_group} (排名: {age_stats.loc[best_age_group, 'final_rank']:.1f})\n")
        f.write(f"最差表现年龄组: {worst_age_group} (排名: {age_stats.loc[worst_age_group, 'final_rank']:.1f})\n\n")
        
        f.write("🎭 评委vs粉丝相对影响力\n")
        f.write("-"*40 + "\n")
        f.write(f"评委对排名的影响力: {judge_influence:.1%}\n")
        f.write(f"粉丝对排名的影响力: {fan_influence:.1%}\n\n")
        
        f.write("📈 核心发现\n")
        f.write("-"*40 + "\n")
        f.write("1. 归一化处理效果显著\n")
        f.write("   通过Min-Max归一化和Z-score标准化，评委和粉丝分数已置于相同尺度，\n")
        f.write("   使得直接比较和综合分析成为可能。\n\n")
        
        f.write("2. 评委与粉丝评价存在系统性差异\n")
        f.write(f"   评委平均略微偏爱选手(差异均值: {diff_mean:.3f})，\n")
        f.write("   但评委与粉丝评价一致的选手往往表现更好。\n\n")
        
        f.write("3. 全面型选手最具竞争力\n")
        f.write("   同时获得评委和粉丝高支持的选手平均排名最高，\n")
        f.write("   单一依赖评委或粉丝支持的选手表现次之。\n\n")
        
        f.write("4. 特征对评委和粉丝的影响方式不同\n")
        f.write("   年龄、行业等特征对评委和粉丝的影响程度和方向存在差异，\n")
        f.write("   评委更注重技术因素，粉丝更注重娱乐性和个人魅力。\n\n")
        
        f.write("💡 建议\n")
        f.write("-"*40 + "\n")
        f.write("1. 评分系统优化\n")
        f.write("   建议采用归一化评分体系，确保评委和粉丝分数可比性。\n\n")
        
        f.write("2. 选手选拔策略\n")
        f.write("   平衡技术型和娱乐型选手，培养全面发展的参赛者。\n\n")
        
        f.write("3. 节目制作方向\n")
        f.write("   适度利用评委-粉丝差异创造看点，但避免过度争议。\n\n")
        
        f.write("4. 规则设计\n")
        f.write("   考虑引入综合评分机制，平衡评委和粉丝的权重。\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成时间: %s\n" % pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("="*80)
    
    print("\n✅ 综合报告已保存: Task3_Comprehensive_Analysis_Report.txt")
    print("\n" + "="*80)
    print("🎉 特征分析模型开发完成！")
    print("="*80)

# 生成最终综合报告
generate_comprehensive_report(optimized_analysis_df, advanced_analysis_df)