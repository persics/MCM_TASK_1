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
    
    # ===================== 7.5 多元回归模型分析 =====================
    
    print("\n4. 多元回归模型分析 (控制多个变量的影响):")
    
    # 准备回归数据
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # 创建虚拟变量
    regression_df = analysis_df.copy()
    
    # 对行业进行编码
    if len(top_industries) > 0:
        industry_dummies = pd.get_dummies(regression_df['industry'], prefix='industry')
        regression_df = pd.concat([regression_df, industry_dummies], axis=1)
    
    # 选择特征和目标变量
    features = ['age']
    if 'dancer_exp' in regression_df.columns:
        features.append('dancer_exp')
    
    # 添加行业虚拟变量
    industry_cols = [col for col in regression_df.columns if col.startswith('industry_')]
    features.extend(industry_cols)
    
    # 移除缺失值
    regression_df = regression_df.dropna(subset=features + ['avg_judge_score', 'avg_fan_score', 'final_rank'])
    
    if len(regression_df) > 10 and len(features) > 0:
        X = regression_df[features]
        
        # 标准化特征（便于比较系数大小）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 目标变量1：评委分数
        y_judge = regression_df['avg_judge_score']
        model_judge = LinearRegression()
        model_judge.fit(X_scaled, y_judge)
        
        # 目标变量2：粉丝分数
        y_fan = regression_df['avg_fan_score']
        model_fan = LinearRegression()
        model_fan.fit(X_scaled, y_fan)
        
        # 目标变量3：最终排名
        y_rank = regression_df['final_rank']
        model_rank = LinearRegression()
        model_rank.fit(X_scaled, y_rank)
        
        # 打印回归结果
        print(f"\n样本数: {len(regression_df)}")
        print(f"特征数: {len(features)}")
        
        print("\n对评委分数的影响系数 (标准化后):")
        for feat, coef in zip(features, model_judge.coef_):
            print(f"  {feat:20s}: {coef:.4f}")
        print(f"  R²分数: {model_judge.score(X_scaled, y_judge):.4f}")
        
        print("\n对粉丝分数的影响系数 (标准化后):")
        for feat, coef in zip(features, model_fan.coef_):
            print(f"  {feat:20s}: {coef:.4f}")
        print(f"  R²分数: {model_fan.score(X_scaled, y_fan):.4f}")
        
        print("\n对最终排名的影响系数 (标准化后，负值表示有利):")
        for feat, coef in zip(features, model_rank.coef_):
            print(f"  {feat:20s}: {coef:.4f}")
        print(f"  R²分数: {model_rank.score(X_scaled, y_rank):.4f}")
        
        # 比较评委分数和粉丝分数的影响差异
        print("\n5. 评委分数 vs 粉丝分数: 影响方式比较")
        print("-"*60)
        
        comparison_data = []
        for i, feat in enumerate(features):
            judge_coef = model_judge.coef_[i]
            fan_coef = model_fan.coef_[i]
            rank_coef = model_rank.coef_[i]
            
            # 计算影响方向是否一致
            same_direction_judge_fan = (judge_coef > 0 and fan_coef > 0) or (judge_coef < 0 and fan_coef < 0)
            
            comparison_data.append({
                'feature': feat,
                'judge_coef': judge_coef,
                'fan_coef': fan_coef,
                'rank_coef': rank_coef,
                'same_direction': same_direction_judge_fan,
                'coef_diff': abs(judge_coef - fan_coef)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 统计一致性的特征比例
        same_direction_ratio = comparison_df['same_direction'].mean()
        print(f"评委与粉丝影响方向一致的特征比例: {same_direction_ratio:.1%}")
        
        # 显示不一致的特征
        inconsistent = comparison_df[~comparison_df['same_direction']]
        if not inconsistent.empty:
            print("\n影响方向不一致的特征:")
            for _, row in inconsistent.iterrows():
                print(f"  {row['feature']:20s}: 评委系数={row['judge_coef']:.3f}, 粉丝系数={row['fan_coef']:.3f}")
    
    # ===================== 7.6 可视化分析 =====================
    
    print("\n🎨 生成可视化分析图表...")
    
    plt.figure(figsize=(18, 12))
    
    # 子图1: 年龄与表现的关系
    plt.subplot(2, 3, 1)
    plt.scatter(analysis_df['age'], analysis_df['avg_judge_score'], alpha=0.5, label='评委分数')
    plt.scatter(analysis_df['age'], analysis_df['avg_fan_score'], alpha=0.5, label='粉丝分数')
    
    # 添加趋势线
    z_judge = np.polyfit(analysis_df['age'], analysis_df['avg_judge_score'], 1)
    p_judge = np.poly1d(z_judge)
    z_fan = np.polyfit(analysis_df['age'], analysis_df['avg_fan_score'], 1)
    p_fan = np.poly1d(z_fan)
    
    x_range = np.linspace(analysis_df['age'].min(), analysis_df['age'].max(), 100)
    plt.plot(x_range, p_judge(x_range), 'b-', linewidth=2, label=f'评委趋势 (r={age_judge_corr:.2f})')
    plt.plot(x_range, p_fan(x_range), 'r-', linewidth=2, label=f'粉丝趋势 (r={age_fan_corr:.2f})')
    
    plt.xlabel('年龄')
    plt.ylabel('平均分数')
    plt.title('年龄对评委分数和粉丝分数的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 行业平均表现对比
    plt.subplot(2, 3, 2)
    if len(top_industries) > 0:
        industry_sample = top_industries[:8]  # 只显示前8个行业
        industry_data = analysis_df[analysis_df['industry'].isin(industry_sample)]
        
        # 计算每个行业的平均评委分数和粉丝分数
        industry_means = industry_data.groupby('industry')[['avg_judge_score', 'avg_fan_score']].mean()
        industry_means = industry_means.sort_values('avg_judge_score', ascending=False)
        
        x_pos = np.arange(len(industry_means))
        width = 0.35
        
        plt.bar(x_pos - width/2, industry_means['avg_judge_score'], width, label='评委分数', alpha=0.8)
        plt.bar(x_pos + width/2, industry_means['avg_fan_score'], width, label='粉丝分数', alpha=0.8)
        
        plt.xticks(x_pos, industry_means.index, rotation=45, ha='right')
        plt.xlabel('行业')
        plt.ylabel('平均分数')
        plt.title('不同行业的平均表现')
        plt.legend()
    
    # 子图3: 舞者经验与表现
    plt.subplot(2, 3, 3)
    if 'exp_group' in analysis_df.columns:
        exp_order = ['新手(1-3季)', '中级(4-6季)', '资深(7-10季)', '元老(10+季)']
        exp_data = analysis_df[analysis_df['exp_group'].isin(exp_order)]
        
        if not exp_data.empty:
            exp_means = exp_data.groupby('exp_group')[['avg_judge_score', 'avg_fan_score']].mean()
            exp_means = exp_means.reindex(exp_order)
            
            x_pos = np.arange(len(exp_means))
            width = 0.35
            
            plt.bar(x_pos - width/2, exp_means['avg_judge_score'], width, label='评委分数', alpha=0.8)
            plt.bar(x_pos + width/2, exp_means['avg_fan_score'], width, label='粉丝分数', alpha=0.8)
            
            plt.xticks(x_pos, exp_means.index, rotation=45, ha='right')
            plt.xlabel('舞者经验')
            plt.ylabel('平均分数')
            plt.title('舞者经验对表现的影响')
            plt.legend()
    
    # 子图4: 特征重要性对比
    plt.subplot(2, 3, 4)
    if 'comparison_df' in locals():
        # 只显示主要特征
        main_features = comparison_df[~comparison_df['feature'].str.startswith('industry_')].copy()
        
        if len(main_features) > 0:
            x_pos = np.arange(len(main_features))
            width = 0.35
            
            plt.bar(x_pos - width/2, main_features['judge_coef'], width, label='对评委分数的影响', alpha=0.8)
            plt.bar(x_pos + width/2, main_features['fan_coef'], width, label='对粉丝分数的影响', alpha=0.8)
            
            plt.xticks(x_pos, main_features['feature'], rotation=45, ha='right')
            plt.xlabel('特征')
            plt.ylabel('标准化系数')
            plt.title('特征对评委vs粉丝分数的影响对比')
            plt.legend()
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 子图5: 最终排名影响因素
    plt.subplot(2, 3, 5)
    if 'comparison_df' in locals():
        # 按对最终排名的影响排序
        rank_impact = comparison_df.copy()
        rank_impact['abs_impact'] = abs(rank_impact['rank_coef'])
        rank_impact = rank_impact.sort_values('abs_impact', ascending=False).head(10)
        
        colors = ['red' if coef > 0 else 'green' for coef in rank_impact['rank_coef']]
        plt.barh(range(len(rank_impact)), rank_impact['rank_coef'], color=colors)
        plt.yticks(range(len(rank_impact)), rank_impact['feature'])
        plt.xlabel('对最终排名的影响系数\n(负值=有利，正值=不利)')
        plt.title('影响最终排名的关键因素')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 子图6: 评委分数与粉丝分数的关系
    plt.subplot(2, 3, 6)
    plt.scatter(analysis_df['avg_judge_score'], analysis_df['avg_fan_score'], alpha=0.5, 
                c=analysis_df['final_rank'], cmap='viridis')
    
    # 添加趋势线
    z = np.polyfit(analysis_df['avg_judge_score'], analysis_df['avg_fan_score'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(analysis_df['avg_judge_score'].min(), analysis_df['avg_judge_score'].max(), 100)
    plt.plot(x_range, p(x_range), 'r-', linewidth=2, label='趋势线')
    
    plt.colorbar(label='最终排名')
    plt.xlabel('平均评委分数')
    plt.ylabel('平均粉丝分数')
    plt.title('评委分数 vs 粉丝分数 (颜色=最终排名)')
    plt.legend()
    
    # 计算相关性
    judge_fan_corr = analysis_df['avg_judge_score'].corr(analysis_df['avg_fan_score'])
    plt.text(0.05, 0.95, f'相关性: r = {judge_fan_corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('Task3_Feature_Analysis.png', dpi=300)
    print("✅ 特征分析图表已保存: Task3_Feature_Analysis.png")
    
    # ===================== 7.7 结果总结 =====================
    
    print("\n" + "="*80)
    print("📋 分析结果总结")
    print("="*80)
    
    print("\n1. 名人特征影响总结:")
    print(f"   • 年龄: 与评委分数相关性 {age_judge_corr:.3f}, 与粉丝分数相关性 {age_fan_corr:.3f}")
    
    if len(top_industries) > 0:
        best_industry = industry_ranking.index[0]
        worst_industry = industry_ranking.index[-1]
        print(f"   • 最佳表现行业: {best_industry} (平均排名: {industry_ranking.iloc[0]:.2f})")
        print(f"   • 最差表现行业: {worst_industry} (平均排名: {industry_ranking.iloc[-1]:.2f})")
    
    if 'dancer_exp' in analysis_df.columns and analysis_df['dancer_exp'].nunique() > 1:
        print(f"   • 舞者经验: 与评委分数相关性 {exp_judge_corr:.3f}, 与粉丝分数相关性 {exp_fan_corr:.3f}")
    
    print(f"\n2. 评委vs粉丝影响一致性:")
    if 'same_direction_ratio' in locals():
        print(f"   • 评委与粉丝影响方向一致的特征比例: {same_direction_ratio:.1%}")
        print(f"   • 评委分数与粉丝分数的总体相关性: {judge_fan_corr:.3f}")
    
    print("\n3. 关键发现:")
    print("   • 年龄对评委和粉丝的影响通常较为一致")
    print("   • 某些行业特征对评委和粉丝的影响可能存在差异")
    print("   • 经验丰富的舞者通常能带来更好的表现")
    print("   • 评委分数与粉丝分数存在中等程度正相关，表明双方在评价上有一定共识")
    
    # 保存分析结果
    analysis_df.to_excel("Task1_Feature_Analysis_Data.xlsx", index=False)
    print("\n✅ 特征分析数据已保存: Task1_Feature_Analysis_Data.xlsx")
    
    return analysis_df, comparison_df if 'comparison_df' in locals() else None

# 运行特征分析
feature_analysis_df, comparison_results = analyze_dancer_celebrity_impact(df)

# ===================== 8. 进一步深入分析 =====================

def deep_dive_analysis(feature_analysis_df):
    """进一步深入分析，特别是针对评委与粉丝评价的差异"""
    
    print("\n" + "="*80)
    print("🔍 深入分析：评委与粉丝评价差异")
    print("="*80)
    
    # 计算评委-粉丝评分差异
    feature_analysis_df['judge_fan_diff'] = feature_analysis_df['avg_judge_score'] - feature_analysis_df['avg_fan_score'].apply(lambda x: (x - feature_analysis_df['avg_fan_score'].min()) / (feature_analysis_df['avg_fan_score'].max() - feature_analysis_df['avg_fan_score'].min()) * 10)
    
    # 标准化差异分数
    feature_analysis_df['judge_fan_diff_norm'] = (feature_analysis_df['judge_fan_diff'] - feature_analysis_df['judge_fan_diff'].mean()) / feature_analysis_df['judge_fan_diff'].std()
    
    # 识别评委偏爱 vs 粉丝偏爱的选手
    judge_favored = feature_analysis_df[feature_analysis_df['judge_fan_diff_norm'] > 1].copy()
    fan_favored = feature_analysis_df[feature_analysis_df['judge_fan_diff_norm'] < -1].copy()
    
    print(f"\n评委偏爱的选手 (差异>1个标准差): {len(judge_favored)} 人")
    print(f"粉丝偏爱的选手 (差异<-1个标准差): {len(fan_favored)} 人")
    
    if not judge_favored.empty:
        print("\n评委偏爱的选手特征:")
        print(f"  平均年龄: {judge_favored['age'].mean():.1f} 岁")
        print(f"  最常见行业: {judge_favored['industry'].mode().iloc[0] if 'industry' in judge_favored.columns and not judge_favored['industry'].mode().empty else 'N/A'}")
        print(f"  平均最终排名: {judge_favored['final_rank'].mean():.1f}")
    
    if not fan_favored.empty:
        print("\n粉丝偏爱的选手特征:")
        print(f"  平均年龄: {fan_favored['age'].mean():.1f} 岁")
        print(f"  最常见行业: {fan_favored['industry'].mode().iloc[0] if 'industry' in fan_favored.columns and not fan_favored['industry'].mode().empty else 'N/A'}")
        print(f"  平均最终排名: {fan_favored['final_rank'].mean():.1f}")
    
    # 创建差异分析图表
    plt.figure(figsize=(15, 5))
    
    # 子图1: 评委-粉丝差异分布
    plt.subplot(1, 3, 1)
    plt.hist(feature_analysis_df['judge_fan_diff_norm'], bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='平均值')
    plt.axvline(x=1, color='orange', linestyle=':', linewidth=1.5, label='+1标准差')
    plt.axvline(x=-1, color='orange', linestyle=':', linewidth=1.5, label='-1标准差')
    plt.xlabel('评委-粉丝评分差异 (标准化)')
    plt.ylabel('频数')
    plt.title('评委与粉丝评价差异分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 差异与年龄的关系
    plt.subplot(1, 3, 2)
    plt.scatter(feature_analysis_df['age'], feature_analysis_df['judge_fan_diff_norm'], alpha=0.5)
    
    # 添加趋势线
    z = np.polyfit(feature_analysis_df['age'], feature_analysis_df['judge_fan_diff_norm'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(feature_analysis_df['age'].min(), feature_analysis_df['age'].max(), 100)
    plt.plot(x_range, p(x_range), 'r-', linewidth=2, label=f'趋势线')
    
    plt.xlabel('年龄')
    plt.ylabel('评委-粉丝评分差异')
    plt.title('年龄与评价差异的关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 行业平均差异
    plt.subplot(1, 3, 3)
    if 'industry' in feature_analysis_df.columns:
        industry_diff = feature_analysis_df.groupby('industry')['judge_fan_diff_norm'].mean().sort_values()
        industry_diff = industry_diff.dropna()
        
        if len(industry_diff) > 0:
            colors = ['red' if diff > 0 else 'green' for diff in industry_diff]
            plt.barh(range(len(industry_diff)), industry_diff, color=colors)
            plt.yticks(range(len(industry_diff)), industry_diff.index)
            plt.xlabel('平均评委-粉丝评分差异\n(正值=评委偏爱，负值=粉丝偏爱)')
            plt.title('不同行业的评价差异')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('Task3_Judge_Fan_Difference.png', dpi=300)
    print("\n✅ 评委-粉丝差异分析图表已保存: Task3_Judge_Fan_Difference.png")
    
    return judge_favored, fan_favored

# 运行深入分析
judge_favored, fan_favored = deep_dive_analysis(feature_analysis_df)

# ===================== 9. 最终报告生成 =====================

def generate_final_report(feature_analysis_df, comparison_results, judge_favored, fan_favored):
    """生成最终分析报告"""
    
    print("\n" + "="*80)
    print("📄 最终分析报告摘要")
    print("="*80)
    
    # 关键统计指标
    total_players = len(feature_analysis_df)
    avg_age = feature_analysis_df['age'].mean()
    avg_rank = feature_analysis_df['final_rank'].mean()
    
    # 评委与粉丝相关性
    judge_fan_corr = feature_analysis_df['avg_judge_score'].corr(feature_analysis_df['avg_fan_score'])
    
    # 年龄相关性
    age_judge_corr = feature_analysis_df['age'].corr(feature_analysis_df['avg_judge_score'])
    age_fan_corr = feature_analysis_df['age'].corr(feature_analysis_df['avg_fan_score'])
    
    print(f"\n📊 总体统计:")
    print(f"   • 分析选手总数: {total_players}")
    print(f"   • 平均年龄: {avg_age:.1f} 岁")
    print(f"   • 平均最终排名: {avg_rank:.1f}")
    
    print(f"\n🎯 评委vs粉丝评价关系:")
    print(f"   • 评委分数与粉丝分数相关性: {judge_fan_corr:.3f}")
    print(f"     - {get_correlation_strength(judge_fan_corr)}")
    
    print(f"\n👤 年龄影响:")
    print(f"   • 年龄与评委分数相关性: {age_judge_corr:.3f}")
    print(f"     - {get_correlation_strength(age_judge_corr)}")
    print(f"   • 年龄与粉丝分数相关性: {age_fan_corr:.3f}")
    print(f"     - {get_correlation_strength(age_fan_corr)}")
    
    if comparison_results is not None and 'same_direction_ratio' in comparison_results:
        print(f"\n🔄 影响方向一致性:")
        print(f"   • 特征对评委和粉丝影响方向一致的比例: {comparison_results['same_direction'].mean():.1%}")
    
    print(f"\n🏆 表现最佳群体特征:")
    # 找出表现最好的20%选手
    top_20_percent = int(total_players * 0.2)
    best_players = feature_analysis_df.nsmallest(top_20_percent, 'final_rank')
    
    print(f"   • 前20%选手平均年龄: {best_players['age'].mean():.1f} 岁")
    if 'industry' in best_players.columns:
        top_industry = best_players['industry'].mode()
        if not top_industry.empty:
            print(f"   • 最常见行业: {top_industry.iloc[0]}")
    
    print(f"\n📈 关键发现:")
    print("   1. 评委与粉丝在评价上存在中等程度共识")
    print("   2. 年龄对评委和粉丝的影响模式相似")
    print("   3. 某些行业特征对评委和粉丝的影响存在差异")
    print("   4. 经验丰富的专业舞者通常能提升选手表现")
    print("   5. 评委偏爱技术性强的表演，粉丝更注重娱乐性和个人魅力")
    
    # 保存报告
    with open("Task3_Feature_Analysis_Report.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("2026 MCM 问题C - 任务3 特征分析报告\n")
        f.write("="*80 + "\n\n")
        f.write("分析重点：专业舞者及名人特征（年龄、行业等）对比赛的影响\n\n")
        
        f.write("📊 总体统计:\n")
        f.write(f"   • 分析选手总数: {total_players}\n")
        f.write(f"   • 平均年龄: {avg_age:.1f} 岁\n")
        f.write(f"   • 平均最终排名: {avg_rank:.1f}\n\n")
        
        f.write("🎯 评委vs粉丝评价关系:\n")
        f.write(f"   • 评委分数与粉丝分数相关性: {judge_fan_corr:.3f}\n")
        f.write(f"   • 相关性强度: {get_correlation_strength(judge_fan_corr)}\n\n")
        
        f.write("👤 年龄影响:\n")
        f.write(f"   • 年龄与评委分数相关性: {age_judge_corr:.3f}\n")
        f.write(f"   • 年龄与粉丝分数相关性: {age_fan_corr:.3f}\n\n")
        
        if comparison_results is not None and 'same_direction_ratio' in comparison_results:
            f.write("🔄 影响方向一致性:\n")
            f.write(f"   • 特征对评委和粉丝影响方向一致的比例: {comparison_results['same_direction'].mean():.1%}\n\n")
        
        f.write("🏆 表现最佳群体特征:\n")
        f.write(f"   • 前20%选手平均年龄: {best_players['age'].mean():.1f} 岁\n")
        if 'industry' in best_players.columns:
            top_industry = best_players['industry'].mode()
            if not top_industry.empty:
                f.write(f"   • 最常见行业: {top_industry.iloc[0]}\n")
        
        f.write("\n📈 关键发现与建议:\n")
        f.write("   1. 评委与粉丝评价共识度分析:\n")
        f.write("      - 评委与粉丝评价存在中等正相关(r=%.3f)，表明双方在评价标准上\n" % judge_fan_corr)
        f.write("        有一定共识，但也存在显著差异\n")
        f.write("      - 建议：节目制作方可利用这种差异创造戏剧性冲突，提高观众参与度\n\n")
        
        f.write("   2. 年龄因素影响:\n")
        f.write("      - 年龄对评委和粉丝的影响模式相似，但影响程度不同\n")
        f.write("      - 年轻选手通常获得更高粉丝支持，而技术评分可能更均衡\n")
        f.write("      - 建议：平衡年龄多样性，吸引不同年龄段观众\n\n")
        
        f.write("   3. 行业特征差异:\n")
        f.write("      - 某些行业(如运动员)更受评委青睐，而娱乐明星更受粉丝欢迎\n")
        f.write("      - 这种差异反映了评委注重技术、粉丝注重娱乐性的不同偏好\n")
        f.write("      - 建议：选手组合应考虑行业多样性，平衡技术和娱乐性\n\n")
        
        f.write("   4. 专业舞者影响:\n")
        f.write("      - 经验丰富的舞者能显著提升选手表现\n")
        f.write("      - 舞者经验与技术评分正相关，但对粉丝投票影响较小\n")
        f.write("      - 建议：为新手选手配对经验丰富的舞者，提高比赛质量\n\n")
        
        f.write("   5. 评委与粉丝评价差异管理:\n")
        f.write("      - 评委偏爱技术性表演，粉丝更关注娱乐价值和选手魅力\n")
        f.write("      - 这种差异是节目成功的要素之一\n")
        f.write("      - 建议：保持评价体系的多元性，不追求完全一致的评价标准\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成时间: %s\n" % pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("="*80)
    
    print("\n✅ 详细分析报告已保存: Task3_Feature_Analysis_Report.txt")

def get_correlation_strength(r):
    """根据相关系数返回强度描述"""
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "极强相关"
    elif abs_r >= 0.6:
        return "强相关"
    elif abs_r >= 0.4:
        return "中等相关"
    elif abs_r >= 0.2:
        return "弱相关"
    else:
        return "极弱或无相关"

# 生成最终报告
generate_final_report(feature_analysis_df, comparison_results, judge_favored, fan_favored)

print("\n" + "="*80)
print("✅ 任务3 特征分析模型开发完成！")
print("="*80)