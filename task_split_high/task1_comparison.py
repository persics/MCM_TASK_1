# ===================== 1. 基础设置 =====================
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = np.int_

import emcee
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# ===================== 7. 核心对比算法 (Task 2) =====================
def compare_methods(df):
    print("\n" + "="*80)
    print("开始执行 Task 2: 对比淘汰机制")
    print("="*80)
    
    print("⚔️ 正在对比 [排名法] vs [百分比法] ...")
    
    # 生成模拟粉丝票数（基于预测的粉丝生存指数）
    # 使用指数变换将 log-odds 转换为正数
    df['pred_fan_votes'] = np.exp(df['pred_fan_score'])
    
    results = []
    
    for s in sorted(df['season'].unique()):
        for w in sorted(df[df['season']==s]['week'].unique()):
            mask = (df['season']==s) & (df['week']==w)
            sub = df[mask].copy()
            
            # 如果该周只有1人或数据不足，跳过
            if len(sub) < 2: continue
            
            # --- 算法 A: 排名法 (Rank Method) ---
            # 规则：评委排名(1=最高分) + 粉丝排名(1=最高票)。总和最大的淘汰(因为Rank N是最后一名)。
            # 注意：题目附录示例中，Rachel Hunter 评委Rank 2，粉丝Rank 4，总和6 被淘汰。
            # 这意味着 Rank 1 = 最好。Rank N = 最差。Sum 越大越危险。
            
            # rankdata 默认是从小到大排名 (低分=1)。我们需要反转：高分=1。
            # method='min' 意味着并列第一都算1。method='average' 意味着并列第一算1.5。通常选average。
            sub['rank_judge'] = rankdata(-sub['judge_score'], method='average') 
            sub['rank_fan'] = rankdata(-sub['pred_fan_votes'], method='average')
            sub['rank_sum'] = sub['rank_judge'] + sub['rank_fan']
            
            # 淘汰者：Rank Sum 最大的
            elim_rank_idx = sub['rank_sum'].idxmax()
            elim_player_rank = sub.loc[elim_rank_idx, 'player_id']
            
            # --- 算法 B: 百分比法 (Percentage Method) ---
            # 规则：评委分数占比 + 粉丝投票占比。总和最小的淘汰。
            total_score = sub['judge_score'].sum()
            total_votes = sub['pred_fan_votes'].sum()
            
            sub['pct_judge'] = (sub['judge_score'] / total_score) * 100
            sub['pct_fan'] = (sub['pred_fan_votes'] / total_votes) * 100
            sub['pct_sum'] = sub['pct_judge'] + sub['pct_fan']
            
            # 淘汰者：Pct Sum 最小的
            elim_pct_idx = sub['pct_sum'].idxmin()
            elim_player_pct = sub.loc[elim_pct_idx, 'player_id']
            
            # --- 获取实际淘汰者 ---
            actual_elim_rows = sub[sub['actual_eliminate'] == 1]
            actual_elim_player = actual_elim_rows['player_id'].values[0] if len(actual_elim_rows) > 0 else None
            
            if actual_elim_player:
                results.append({
                    'season': s,
                    'week': w,
                    'elim_rank': elim_player_rank,
                    'elim_pct': elim_player_pct,
                    'actual': actual_elim_player,
                    'match_rank': 1 if elim_player_rank == actual_elim_player else 0,
                    'match_pct': 1 if elim_player_pct == actual_elim_player else 0,
                    'methods_agree': 1 if elim_player_rank == elim_player_pct else 0,
                    # 记录此时粉丝投票最低的人是否被淘汰，用于分析"粉丝保护力"
                    'fan_lowest_saved_by_rank': 1 if (sub.loc[sub['pred_fan_votes'].idxmin(), 'player_id'] != elim_player_rank) else 0,
                    'fan_lowest_saved_by_pct': 1 if (sub.loc[sub['pred_fan_votes'].idxmin(), 'player_id'] != elim_player_pct) else 0
                })

    return pd.DataFrame(results)

# ===================== 8. 分析与可视化 (Task 2) =====================
def analyze_and_plot(res_df):
    # 1. 总体准确率对比
    acc_rank = res_df['match_rank'].mean()
    acc_pct = res_df['match_pct'].mean()
    agreement = res_df['methods_agree'].mean()
    
    print("\n" + "="*40)
    print("📊 两种方法对比结果摘要")
    print("="*40)
    print(f"总样本周数: {len(res_df)}")
    print(f"排名法 (Rank) 匹配历史结果率: {acc_rank:.2%}")
    print(f"百分比法 (Pct) 匹配历史结果率: {acc_pct:.2%}")
    print(f"两种方法达成一致的频率:      {agreement:.2%}")
    
    # 2. 哪种方法更保护"粉丝票数低"的选手？（反向即：哪种更依赖粉丝）
    # 如果该方法淘汰了粉丝票最低的人，说明它顺从粉丝意愿。
    # 如果该方法由评委分救回了粉丝票最低的人，说明它受评委影响大。
    
    # 计算：粉丝票最低者被淘汰的概率 (越高说明越偏向粉丝)
    # 注意：saved = 1 意味着没被淘汰。 eliminated = 1 - saved.
    fan_influence_rank = 1 - res_df['fan_lowest_saved_by_rank'].mean()
    fan_influence_pct = 1 - res_df['fan_lowest_saved_by_pct'].mean()
    
    print("\n⚖️ 权重偏向性分析")
    print(f"排名法淘汰粉丝票最低者的概率: {fan_influence_rank:.2%} (数值越大越听粉丝的)")
    print(f"百分比法淘汰粉丝票最低者的概率: {fan_influence_pct:.2%}")
    
    if fan_influence_pct > fan_influence_rank:
        print(">> 结论: 百分比法通常赋予粉丝投票更大的权重（或对粉丝票数差异更敏感）。")
    else:
        print(">> 结论: 排名法通常赋予粉丝投票更大的权重。")


    # --- 绘图 ---
    fig = plt.figure(figsize=(14, 10))
    

    # 图1: 不同赛季的匹配率
    ax1 = fig.add_subplot(2, 2, 1)
    season_acc = res_df.groupby('season')[['match_rank', 'match_pct']].mean()
    season_acc.plot(kind='bar', ax=ax1, width=0.8, color=['#4c72b0', '#dd8452'])
    ax1.set_title('Comparison of elimination prediction accuracy in each season (Rank vs Pct)')
    ax1.set_ylabel('Rate of agreement with historical results')
    ax1.set_ylim(0, 1.1)
    ax1.legend(["Method of ranking", "Method of percentage"])
    ax1.grid(axis='y', alpha=0.3)


    # 图2: 方法一致性随时间变化
    ax2 = fig.add_subplot(2, 2, 2)
    # 移动平均
    res_df['agree_ma'] = res_df['methods_agree'].rolling(window=10).mean()
    ax2.plot(res_df.index, res_df['agree_ma'], color='green', linewidth=2)
    ax2.set_title('Consistency of the two methods (moving average)')
    ax2.set_ylabel('Agreement rate (1= perfect agreement)')
    ax2.set_xlabel('Week of Competition (Timeline)')
    ax2.grid(True, alpha=0.3)
    

    # 图3: 差异案例分析 - 当两者不一致时，发生了什么？
    ax3 = fig.add_subplot(2, 1, 2)
    diff_mask = res_df['methods_agree'] == 0
    if diff_mask.sum() > 0:
        diff_data = res_df[diff_mask]
        # 统计在不一致时，谁赢了？(谁匹配了历史)
        rank_wins = diff_data['match_rank'].sum()
        pct_wins = diff_data['match_pct'].sum()
        neither_wins = len(diff_data) - rank_wins - pct_wins
        
        labels = ['The ranking method is correct', 'Percentage method is correct', 'None of them are correct']
        sizes = [rank_wins, pct_wins, neither_wins]
        colors = ['#4c72b0', '#dd8452', '#999999']
        
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax3.set_title(f'When the two approaches diverge ({len(diff_data)}次)，Who fits the historical truth?')
    else:
        ax3.text(0.5, 0.5, "The results of the two methods are completely consistent in all samples", ha='center')


    plt.tight_layout()
    plt.savefig('Task2_Method_Comparison.png', dpi=300)
    print("\n✅ 图表已保存: Task2_Method_Comparison.png")
    


    # 导出Excel
    res_df.to_excel("Task2_Detailed_Comparison.xlsx", index=False)
    print("✅ 详细数据已保存: Task2_Detailed_Comparison.xlsx")
    
    return res_df

# 执行 Task 2 对比
res_df = compare_methods(df)
if not res_df.empty:
    analyze_and_plot(res_df)

print("\n" + "="*80)
print("🎉 所有任务完成！")
print("="*80)
print("📁 生成的文件:")
print("  - Task1_High_Accuracy_Report.png (预测可视化)")
print("  - Task1_Weekly_Accuracy.xlsx (每周准确率)")
print("  - Task1_Optimized_Result.xlsx (预测结果)")
print("  - Task2_Method_Comparison.png (机制对比)")
print("  - Task2_Detailed_Comparison.xlsx (详细对比数据)")
print("="*80)