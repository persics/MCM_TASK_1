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
    
    # ===================== 计算每周准确率 =====================
    weekly_accuracies = []
    
    for s in df['season'].unique():
        season_df = df[df['season'] == s]
        weeks = sorted(season_df['week'].unique())
        
        for w in weeks:
            week_df = season_df[season_df['week'] == w]
            
            # 检查是否有实际的淘汰数据
            if week_df['actual_eliminate'].isin([0, 1]).any():
                # 计算本周准确率
                week_acc = accuracy_score(week_df['actual_eliminate'], week_df['est_eliminate'])
                
                # 统计信息
                week_samples = len(week_df)
                week_actual_elim = week_df['actual_eliminate'].sum()
                week_pred_elim = week_df['est_eliminate'].sum()
                
                weekly_accuracies.append({
                    'season': s,
                    'week': w,
                    'accuracy': week_acc,
                    'samples': week_samples,
                    'actual_eliminations': int(week_actual_elim),
                    'predicted_eliminations': int(week_pred_elim),
                    'correct_predictions': int((week_df['actual_eliminate'] == week_df['est_eliminate']).sum()),
                    'incorrect_predictions': int((week_df['actual_eliminate'] != week_df['est_eliminate']).sum())
                })
    
    weekly_acc_df = pd.DataFrame(weekly_accuracies)
    
    # 打印每周准确率
    if not weekly_acc_df.empty:
        print("\n📈 每周预测准确率:")
        print("-" * 80)
        
        # 按赛季分组显示
        for season in sorted(weekly_acc_df['season'].unique()):
            season_weeks = weekly_acc_df[weekly_acc_df['season'] == season].sort_values('week')
            print(f"\n赛季 {season}:")
            print(f"{'周次':<6} {'准确率':<10} {'样本数':<8} {'实际淘汰':<10} {'预测淘汰':<10} {'正确预测':<10} {'错误预测':<10}")
            print("-" * 80)
            
            season_accuracy_sum = 0
            week_count = 0
            
            for _, row in season_weeks.iterrows():
                print(f"{row['week']:<6} {row['accuracy']:<10.2%} {row['samples']:<8} {row['actual_eliminations']:<10} "
                      f"{row['predicted_eliminations']:<10} {row['correct_predictions']:<10} {row['incorrect_predictions']:<10}")
                
                season_accuracy_sum += row['accuracy']
                week_count += 1
            
            if week_count > 0:
                season_avg = season_accuracy_sum / week_count
                print(f"赛季 {season} 平均准确率: {season_avg:.2%}")
        
        # 计算总体平均每周准确率
        avg_weekly_acc = weekly_acc_df['accuracy'].mean()
        print(f"\n📊 平均每周准确率: {avg_weekly_acc:.2%}")
        
        # 按周次统计平均准确率
        print("\n📊 按周次统计的平均准确率:")
        week_avg_stats = weekly_acc_df.groupby('week')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        for _, row in week_avg_stats.iterrows():
            print(f"第 {int(row['week']):2d} 周: {row['mean']:.2%} (±{row['std']:.3f}, 样本数: {int(row['count'])})")
    
    # ===================== 绘图部分 =====================
    plt.figure(figsize=(20, 12))
    
    # 子图1: 概率分布
    plt.subplot(2, 3, 1)
    sns.histplot(df[df['actual_eliminate']==1]['final_elim_prob'], color='red', label='Actual eliminators', kde=True, bins=20)
    sns.histplot(df[df['actual_eliminate']==0]['final_elim_prob'], color='green', label='The actual finalist', kde=True, bins=20, alpha=0.3)
    plt.title("Predicting the elimination probability distribution (higher red-green separation is better)")
    plt.legend()
    
    # 子图2: 各赛季预测准确率
    plt.subplot(2, 3, 2)
    season_acc = df.groupby('season').apply(lambda x: accuracy_score(x['actual_eliminate'], x['est_eliminate'])).reset_index()
    season_acc.columns = ['season', 'acc']
    sns.barplot(x='season', y='acc', data=season_acc, palette='viridis')
    plt.axhline(y=acc, color='r', linestyle='--', label='Overall average accuracy')
    plt.title("Prediction accuracy by season")
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    
    # 子图3: 每周准确率热力图
    plt.subplot(2, 3, 3)
    if not weekly_acc_df.empty:
        # 创建热力图数据
        heatmap_data = weekly_acc_df.pivot(index='season', columns='week', values='accuracy')
        
        # 绘制热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'accuracy'}, vmin=0, vmax=1)
        plt.title("Heatmap of weekly prediction accuracy")
        plt.xlabel("Time of week")
        plt.ylabel("Season")
    else:
        plt.text(0.5, 0.5, "Weekly accuracy data is not available", ha='center', va='center')
        plt.title("Heatmap of weekly prediction accuracy")
    
    # 子图4: 每周平均准确率趋势
    plt.subplot(2, 3, 4)
    if not weekly_acc_df.empty:
        week_avg_acc = weekly_acc_df.groupby('week')['accuracy'].agg(['mean', 'std']).reset_index()
        plt.errorbar(week_avg_acc['week'], week_avg_acc['mean'], 
                    yerr=week_avg_acc['std'], fmt='bo-', linewidth=2, 
                    markersize=8, capsize=5, capthick=2)
        plt.fill_between(week_avg_acc['week'], 
                        week_avg_acc['mean'] - week_avg_acc['std'],
                        week_avg_acc['mean'] + week_avg_acc['std'],
                        alpha=0.2)
        plt.axhline(y=avg_weekly_acc, color='r', linestyle='--', label=f'average: {avg_weekly_acc:.2%}')
        plt.xlabel("Time of week")
        plt.ylabel("Average precision")
        plt.title("Trend of average weekly accuracy (with error bars)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 子图5: 样本数量分布
    plt.subplot(2, 3, 5)
    if not weekly_acc_df.empty:
        plt.bar(range(len(weekly_acc_df)), weekly_acc_df['samples'], alpha=0.7)
        plt.xlabel("Index of data points (sorted by season and week)")
        plt.ylabel("Number of samples")
        plt.title("Distribution of the number of samples in each week")
        plt.text(0.05, 0.95, f"Total number of samples: {len(valid_df)}", 
                transform=plt.gca().transAxes, verticalalignment='top')
    
    # 子图6: 准确率与样本量关系
    plt.subplot(2, 3, 6)
    if not weekly_acc_df.empty and len(weekly_acc_df) > 5:
        plt.scatter(weekly_acc_df['samples'], weekly_acc_df['accuracy'], 
                   c=weekly_acc_df['week'], cmap='viridis', s=100, alpha=0.7)
        plt.xlabel("Number of samples")
        plt.ylabel("accuracy")
        plt.title("Accuracy versus sample size")
        plt.colorbar(label='Time of week')
        
        # 添加趋势线
        z = np.polyfit(weekly_acc_df['samples'], weekly_acc_df['accuracy'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(weekly_acc_df['samples'].min(), weekly_acc_df['samples'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.5, label='Trend line')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('Task1_High_Accuracy_Report.png', dpi=300)
    print("\n✅ 图表已保存: Task1_High_Accuracy_Report.png")
    
    # 保存每周准确率到Excel
    if not weekly_acc_df.empty:
        # 添加更多统计信息
        weekly_acc_df['error_rate'] = 1 - weekly_acc_df['accuracy']
        weekly_acc_df['prediction_correctness'] = weekly_acc_df['correct_predictions'] / weekly_acc_df['samples']
        
        weekly_acc_df.to_excel("Task1_Weekly_Accuracy.xlsx", index=False)
        print("✅ 每周准确率数据已保存: Task1_Weekly_Accuracy.xlsx")
    
    return season_acc, weekly_acc_df

season_stats, weekly_stats = check_performance(df)

# 导出预测结果
df.to_excel("Task1_Optimized_Result.xlsx", index=False)
print("✅ 预测结果已保存: Task1_Optimized_Result.xlsx")

# 打印汇总统计
print("\n" + "="*80)
print("🎯 预测性能汇总")
print("="*80)

if not weekly_stats.empty:
    # 按周次统计
    print("\n按周次统计:")
    week_summary = weekly_stats.groupby('week').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'samples': 'sum'
    }).round(4)
    print(week_summary)
    
    # 按赛季统计
    print("\n按赛季统计:")
    season_summary = weekly_stats.groupby('season').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'samples': 'sum',
        'correct_predictions': 'sum',
        'incorrect_predictions': 'sum'
    }).round(4)
    print(season_summary)